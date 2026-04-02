#!/usr/bin/env python3
"""
Cross-model latent communication test:
  Qwen3-4B (sender/thinker) → Gemma-2-2B (receiver/solver) 

Qwen does the latent thinking, its hidden states are projected
into Gemma's embedding space, and Gemma generates the answer.
"""

import mlx.core as mx
import mlx_lm
import time
from mlx_lm.models.cache import make_prompt_cache
from cross_align import compute_cross_alignment, apply_cross_realignment
from prompts import build_prompt
from utils import extract_boxed_answer, normalize_answer, extract_gold


QUESTIONS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "gold": "18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "gold": "3",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "gold": "5",
    },
]

AGENTS = ["planner", "critic", "refiner"]


def collect_sender_hidden_states(model, tokenizer, prompt_text, cache,
                                 latent_steps, W_a_self, target_norm_self):
    """Run prompt through sender model and collect self-aligned hidden states.
    
    Each hidden state is projected back to the sender's embedding space
    via self-alignment before collection. This is necessary because the
    cross-alignment matrix maps embedding↔embedding, not hidden↔embedding.
    
    Returns: [1, num_collected, D_sender] in sender's embedding space.
    """
    from latent_comm import apply_realignment
    
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = mx.array([tokens])
    
    # Forward through sender's transformer (no lm_head)
    hidden = model.model(input_ids, cache=cache)
    mx.eval(hidden, *[c.state for c in cache])
    
    # Self-align the last hidden state back to embedding space
    last_hidden = hidden[:, -1:, :]
    aligned = apply_realignment(last_hidden, W_a_self, target_norm_self)
    collected = [aligned]
    
    # Latent thought loop — collect self-aligned states
    for step in range(latent_steps):
        aligned = apply_realignment(last_hidden, W_a_self, target_norm_self)
        hidden = model.model(None, cache=cache, input_embeddings=aligned)
        mx.eval(hidden, *[c.state for c in cache])
        last_hidden = hidden[:, -1:, :]
        # Collect the ALIGNED version (in sender's embedding space)
        aligned_step = apply_realignment(last_hidden, W_a_self, target_norm_self)
        collected.append(aligned_step)
    
    # Stack into [1, num_collected, D_sender]
    all_hidden = mx.concatenate(collected, axis=1)
    mx.eval(all_hidden)
    return all_hidden


def gemma_forward_with_embeddings(model, input_embeddings, cache):
    """Run Gemma-2 forward pass with pre-computed embeddings.
    
    Gemma-2's MLX model doesn't have input_embeddings parameter,
    so we manually replicate the forward pass, bypassing embed_tokens.
    Gemma-2 scales embeddings by sqrt(hidden_size) — we apply this.
    Also applies Gemma-2's logit soft-capping.
    """
    from mlx_lm.models.base import create_attention_mask
    
    h = input_embeddings * (model.model.args.hidden_size ** 0.5)
    
    if cache is None:
        cache = [None] * len(model.model.layers)
    
    mask = create_attention_mask(h, cache[0], return_array=True)
    
    for layer, c in zip(model.model.layers, cache):
        h = layer(h, mask, c)
    
    h = model.model.norm(h)
    
    # Tied embeddings + logit soft-capping (Gemma-2 specific)
    out = model.model.embed_tokens.as_linear(h)
    out = mx.tanh(out / model.final_logit_softcapping)
    out = out * model.final_logit_softcapping
    
    return out


def generate_with_cross_latents(receiver_model, receiver_tokenizer,
                                 projected_embeddings, question,
                                 max_tokens=1024, temperature=0.6):
    """Feed projected latent embeddings to receiver and generate answer.
    
    1. Create fresh cache for receiver
    2. Feed projected embeddings (builds receiver's KV cache)
    3. Feed the judger prompt tokens
    4. Generate text autoregressively
    """
    from mlx_lm.sample_utils import make_sampler
    
    cache = make_prompt_cache(receiver_model)
    
    # Step 1: Feed projected latent thoughts through Gemma
    # This builds the receiver's KV cache with the sender's "reasoning"
    logits = gemma_forward_with_embeddings(
        receiver_model, projected_embeddings, cache
    )
    mx.eval(logits, *[c.state for c in cache])
    print(f"    Latent prefill done, cache offset: {cache[0].offset}")
    
    # Step 2: Feed judger prompt as normal tokens
    judger_prompt = (
        f"<start_of_turn>user\n"
        f"You have been given latent reasoning context about the following question. "
        f"Use it to solve the problem step by step.\n\n"
        f"Question: {question}\n\n"
        f"Put your final numerical answer inside \\boxed{{}}.\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    tokens = receiver_tokenizer.encode(judger_prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])
    
    # Normal forward pass for the text prompt (uses standard model path)
    logits = receiver_model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])
    print(f"    Judger prompt prefilled, cache offset: {cache[0].offset}")
    
    # Step 3: Autoregressive generation
    sampler = make_sampler(temp=temperature, top_p=0.95)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = receiver_tokenizer.eos_token_id
    
    for _ in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        next_input = mx.array([[token_id]])
        next_logits = receiver_model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])
    
    return receiver_tokenizer.decode(generated_tokens)


def main():
    sender_name = "mlx-community/Qwen3-4B-4bit"
    receiver_name = "mlx-community/gemma-2-2b-it-4bit"
    # Use 8-bit for alignment computation
    align_sender = "mlx-community/Qwen3-4B-8bit"
    align_receiver = "mlx-community/gemma-2-2b-it-8bit"
    latent_steps = 10
    
    # Step 1: Compute cross-alignment with 8-bit models
    print("=== Computing cross-alignment matrix (8-bit models) ===")
    t0 = time.time()
    model_a_8, tok_a_8 = mlx_lm.load(align_sender)
    model_b_8, tok_b_8 = mlx_lm.load(align_receiver)
    alignment = compute_cross_alignment(model_a_8, tok_a_8, model_b_8, tok_b_8)
    W_ab = alignment["W_ab"]
    mean_a = alignment["mean_a"]
    mean_b = alignment["mean_b"]
    target_norm_b = alignment["target_norm_b"]
    # Free 8-bit models
    del model_a_8, model_b_8, tok_a_8, tok_b_8
    mx.clear_cache()
    print(f"Alignment computed in {time.time()-t0:.1f}s\n")
    
    # Step 2: Load inference models (4-bit)
    print(f"=== Loading inference models ===")
    print(f"Sender: {sender_name}")
    sender_model, sender_tok = mlx_lm.load(sender_name)
    
    # Compute sender's self-alignment (hidden→embedding for Qwen)
    from latent_comm import compute_alignment
    W_a_self, target_norm_self = compute_alignment(sender_model)
    print(f"Sender self-alignment ready")
    
    print(f"Receiver: {receiver_name}")
    receiver_model, receiver_tok = mlx_lm.load(receiver_name)
    print()
    
    # Step 3: Run test questions
    for i, item in enumerate(QUESTIONS):
        question = item["question"]
        gold = item["gold"]
        print(f"==================== Question #{i+1} ====================")
        print(f"Q: {question[:100]}...")
        print(f"Gold: {gold}")
        
        t0 = time.time()
        
        # Sender (Qwen) does latent thinking
        all_agent_hidden = []
        sender_cache = make_prompt_cache(sender_model)
        
        for role in AGENTS:
            prompt = build_prompt(sender_tok, role, question)
            hidden = collect_sender_hidden_states(
                sender_model, sender_tok, prompt,
                sender_cache, latent_steps,
                W_a_self, target_norm_self,
            )
            all_agent_hidden.append(hidden)
            print(f"  [{role}] collected {hidden.shape[1]} hidden states")
        
        # Concatenate all agents' hidden states
        sender_hidden = mx.concatenate(all_agent_hidden, axis=1)
        mx.eval(sender_hidden)
        print(f"  Total sender hidden: {sender_hidden.shape}")
        
        # Project into Gemma's embedding space
        projected = apply_cross_realignment(
            sender_hidden, W_ab, mean_a, mean_b, target_norm_b
        )
        mx.eval(projected)
        print(f"  Projected to receiver space: {projected.shape}")
        
        # Gemma generates answer using Qwen's latent thoughts
        print(f"  Generating with receiver (Gemma)...")
        output = generate_with_cross_latents(
            receiver_model, receiver_tok,
            projected, question,
            max_tokens=512, temperature=0.6,
        )
        
        elapsed = time.time() - t0
        pred = normalize_answer(extract_boxed_answer(output))
        ok = (pred == gold) if (pred and gold) else False
        
        print(f"\n[Gemma output]\n{output[:500]}\n")
        print(f"Result: Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
        print()


if __name__ == "__main__":
    main()
