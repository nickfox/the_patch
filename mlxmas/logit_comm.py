#!/usr/bin/env python3
"""
Cross-model latent communication via logit distribution transfer.

Zero-shot approach: Qwen generates autoregressively, we capture logit
distributions at each step, remap through shared vocabulary, convert
to soft embeddings in Gemma's space. No training required.
"""

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask
from mlx_lm.sample_utils import make_sampler

from mlxmas.cross_align import find_shared_tokens


def build_vocab_map(sender_tok, receiver_tok):
    """Build index mapping between sender and receiver vocabularies.

    Returns:
        sender_ids: mx.array of sender token IDs for shared tokens
        receiver_ids: mx.array of receiver token IDs for shared tokens
        n_shared: number of shared tokens
    """
    shared = find_shared_tokens(sender_tok, receiver_tok)
    sender_ids = mx.array([s[1] for s in shared])
    receiver_ids = mx.array([s[2] for s in shared])
    print(f"  Vocabulary map: {len(shared)} shared tokens", flush=True)
    return sender_ids, receiver_ids, len(shared)


def precompute_receiver_embeds(receiver_model, receiver_ids, scale):
    """Get dequantized, scaled receiver embeddings for shared tokens.

    Args:
        receiver_model: Gemma model
        receiver_ids: mx.array of receiver token IDs for shared tokens
        scale: embedding scale factor (sqrt(hidden_dim) for Gemma)

    Returns:
        [n_shared, hidden_dim] scaled embedding matrix
    """
    batch_size = 8192
    parts = []
    n = receiver_ids.shape[0]
    for i in range(0, n, batch_size):
        batch_ids = receiver_ids[i:i + batch_size]
        embeds = receiver_model.model.embed_tokens(batch_ids[None])[0]
        parts.append(embeds)
    shared_embeds = mx.concatenate(parts, axis=0) * scale
    mx.eval(shared_embeds)
    return shared_embeds


def sender_generate_with_logit_capture(sender_model, sender_tok, question,
                                       sender_ids, shared_receiver_embeds,
                                       max_tokens=2048, temperature=0.6,
                                       softmax_temp=0.1):
    """Qwen generates response, captures logit distributions at each step.

    At each generation step:
    1. Sample a token for Qwen's own continuation
    2. Capture full logit distribution at shared token positions
    3. Convert to soft embedding via weighted sum of receiver embeddings

    Returns:
        soft_embeds: [1, seq_len, hidden_dim] or None
        n_tokens: number of tokens generated
        avg_shared_mass: average probability mass on shared tokens
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f"Think carefully about the following question step by step.\n\n"
            f"Question: {question}\n\n"
            f"Reason through this problem thoroughly."
        )},
    ]
    prompt = sender_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = sender_tok.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    # Prefill
    cache = make_prompt_cache(sender_model)
    logits = sender_model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    # Generation loop with logit capture
    sampler = make_sampler(temp=temperature, top_p=0.95)
    next_logits = logits[:, -1, :]
    eos_token = sender_tok.eos_token_id
    soft_embed_list = []
    shared_mass_total = 0.0

    for i in range(max_tokens):
        # Sample for sender's own continuation
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break

        # Capture logit distribution for receiver
        probs = mx.softmax(next_logits, axis=-1)[0]
        probs_shared = probs[sender_ids]
        shared_mass = mx.sum(probs_shared).item()
        shared_mass_total += shared_mass

        # Sharpen distribution so soft embed stays near real token embeddings
        probs_sharp = probs_shared ** (1.0 / softmax_temp)
        probs_sharp = probs_sharp / (mx.sum(probs_sharp) + 1e-10)
        soft_embed = probs_sharp[None, :] @ shared_receiver_embeds
        soft_embed_list.append(soft_embed)
        mx.eval(soft_embed)

        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        # Continue sender generation
        next_input = mx.array([[token_id]])
        next_logits = sender_model(next_input, cache=cache)[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    n_tokens = len(soft_embed_list)
    if n_tokens == 0:
        return None, 0, 0.0

    avg_shared_mass = shared_mass_total / n_tokens
    soft_embeds = mx.concatenate(soft_embed_list, axis=0)
    soft_embeds = soft_embeds[None, :, :]
    mx.eval(soft_embeds)

    return soft_embeds, n_tokens, avg_shared_mass


def sender_generate_hard_tokens(sender_model, sender_tok, question,
                                sender_ids, receiver_ids,
                                max_tokens=2048, temperature=0.6):
    """Qwen generates response, maps each step to the top shared receiver token.

    At each step: argmax over shared token probabilities → receiver token ID.

    Returns:
        receiver_token_ids: list of int (Gemma token IDs)
        n_tokens: number of tokens mapped
        avg_shared_mass: average probability mass on shared tokens
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f"Think carefully about the following question step by step.\n\n"
            f"Question: {question}\n\n"
            f"Reason through this problem thoroughly."
        )},
    ]
    prompt = sender_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = sender_tok.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    # Prefill
    cache = make_prompt_cache(sender_model)
    logits = sender_model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    # Generation loop
    sampler = make_sampler(temp=temperature, top_p=0.95)
    next_logits = logits[:, -1, :]
    eos_token = sender_tok.eos_token_id
    receiver_token_ids = []
    shared_mass_total = 0.0

    for i in range(max_tokens):
        # Sample for sender's own continuation
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break

        # Map to receiver: argmax over shared probabilities
        probs = mx.softmax(next_logits, axis=-1)[0]
        probs_shared = probs[sender_ids]
        shared_mass = mx.sum(probs_shared).item()
        shared_mass_total += shared_mass

        top_shared_idx = mx.argmax(probs_shared).item()
        receiver_token_ids.append(receiver_ids[top_shared_idx].item())

        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        # Continue sender generation
        next_input = mx.array([[token_id]])
        next_logits = sender_model(next_input, cache=cache)[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    n_tokens = len(receiver_token_ids)
    avg_shared_mass = shared_mass_total / n_tokens if n_tokens > 0 else 0.0

    return receiver_token_ids, n_tokens, avg_shared_mass


def receiver_generate_from_tokens(receiver_model, receiver_tok, token_ids,
                                  max_tokens=2048, temperature=0.6):
    """Gemma generates answer from a translated token sequence.

    Builds: [instruction | translated_tokens | model_turn] → normal forward pass.
    """
    instruction = (
        "<start_of_turn>user\n"
        "Using the reasoning context provided, solve the problem step by step. "
        "Put your final numerical answer inside \\boxed{}.\n\n"
    )
    model_turn = "<end_of_turn>\n<start_of_turn>model\n"

    inst_tokens = receiver_tok.encode(instruction, add_special_tokens=True)
    turn_tokens = receiver_tok.encode(model_turn, add_special_tokens=False)
    all_tokens = inst_tokens + token_ids + turn_tokens
    input_ids = mx.array([all_tokens])

    # Normal forward pass with cache
    cache = make_prompt_cache(receiver_model)
    logits = receiver_model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    # Autoregressive generation
    sampler = make_sampler(temp=temperature, top_p=0.95)
    next_logits = logits[:, -1, :]
    generated_tokens = []
    eos_token = receiver_tok.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        tok_id = token.item()
        if tok_id == eos_token:
            break
        generated_tokens.append(tok_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        next_input = mx.array([[tok_id]])
        next_logits = receiver_model(next_input, cache=cache)[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return receiver_tok.decode(generated_tokens)


def receiver_generate_from_soft_embeds(receiver_model, receiver_tok, soft_embeds,
                                       scale, max_tokens=2048, temperature=0.6):
    """Gemma generates answer from soft embedding context.

    Args:
        receiver_model: Gemma model
        receiver_tok: Gemma tokenizer
        soft_embeds: [1, seq_len, hidden_dim] soft embeddings from sender
        scale: embedding scale factor
        max_tokens: max tokens for Gemma to generate
        temperature: sampling temperature

    Returns:
        generated text string
    """
    # Build instruction prompt
    instruction = (
        "<start_of_turn>user\n"
        "Using the reasoning context provided, solve the problem step by step. "
        "Put your final numerical answer inside \\boxed{}.\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    )
    inst_tokens = receiver_tok.encode(instruction, add_special_tokens=True)
    inst_embeds = receiver_model.model.embed_tokens(mx.array([inst_tokens])) * scale
    mx.eval(inst_embeds)

    # Concatenate: [instruction | soft_embeds]
    inputs_embeds = mx.concatenate([inst_embeds, soft_embeds], axis=1)

    # Single forward pass with cache
    cache = make_prompt_cache(receiver_model)
    h = inputs_embeds
    mask = create_attention_mask(h, cache[0], return_array=True)
    for layer, c in zip(receiver_model.model.layers, cache):
        h = layer(h, mask, c)
    h = receiver_model.model.norm(h)
    logits = receiver_model.model.embed_tokens.as_linear(h)

    if hasattr(receiver_model, 'final_logit_softcapping') and receiver_model.final_logit_softcapping:
        logits = mx.tanh(logits / receiver_model.final_logit_softcapping)
        logits = logits * receiver_model.final_logit_softcapping

    mx.eval(logits, *[c.state for c in cache])

    # Autoregressive generation
    sampler = make_sampler(temp=temperature, top_p=0.95)
    next_logits = logits[:, -1, :]
    generated_tokens = []
    eos_token = receiver_tok.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        next_input = mx.array([[token_id]])
        next_logits = receiver_model(next_input, cache=cache)[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return receiver_tok.decode(generated_tokens)
