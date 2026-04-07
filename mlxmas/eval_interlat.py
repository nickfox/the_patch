#!/usr/bin/env python3
"""
Eval Interlat MHA adapter: Qwen sender → MHA adapter → Gemma receiver (LoRA).

Loads both models, the trained adapter, and LoRA weights.
Runs GSM8K test questions with pure latent communication (no question to receiver).

Usage:
    python -m mlxmas.eval_interlat \
        --adapter-path mlxmas/data/interlat_adapter/ \
        --max-samples 5
"""

import argparse
import json
import os
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.base import create_attention_mask
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from mlxmas.mha_adapter import load_adapter
from mlxmas.train_interlat import add_special_tokens, gemma_forward_with_embeds
from mlxmas.utils import extract_boxed_answer, normalize_answer


def load_gsm8k_test(max_samples=-1):
    """Load GSM8K test set, shuffled."""
    import re
    import random
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for item in ds:
        gold_match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", item["answer"])
        gold = gold_match.group(1) if gold_match else None
        items.append({"question": item["question"].strip(), "gold": gold})
    random.shuffle(items)
    if max_samples > 0:
        items = items[:max_samples]
    return items


def extract_sender_hidden(model, tokenizer, question):
    """Single forward pass through Qwen, capture last-layer hidden states."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f"Think carefully about the following question step by step.\n\n"
            f"Question: {question}\n\n"
            f"Reason through this problem thoroughly."
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array([tokens])

    h = model.model.embed_tokens(input_ids)
    mask = create_attention_mask(h, return_array=True)
    for layer in model.model.layers:
        h = layer(h, mask, cache=None)
    normed = model.model.norm(h)
    mx.eval(normed)
    return normed  # [1, seq_len, D]


def generate_from_embeds(model, tokenizer, inputs_embeds,
                         max_tokens=2048, temperature=0.6):
    """Forward pass from embeddings, then autoregressive generation."""
    mask = create_attention_mask(inputs_embeds, return_array=True)
    h = inputs_embeds
    for layer in model.model.layers:
        h = layer(h, mask, cache=None)
    h = model.model.norm(h)
    logits = model.model.embed_tokens.as_linear(h)

    if hasattr(model, 'final_logit_softcapping') and model.final_logit_softcapping:
        logits = mx.tanh(logits / model.final_logit_softcapping)
        logits = logits * model.final_logit_softcapping

    mx.eval(logits)

    # Now switch to autoregressive with cache
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    # Repopulate cache with a prefill using the embeddings
    h2 = inputs_embeds
    mask2 = create_attention_mask(h2, cache[0], return_array=True)
    for layer, c in zip(model.model.layers, cache):
        h2 = layer(h2, mask2, c)
    mx.eval(*[c.state for c in cache])

    # Generate tokens
    sampler = make_sampler(temp=temperature, top_p=0.95)
    next_logits = logits[:, -1, :]
    generated_tokens = []
    eos_token = tokenizer.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        next_input = mx.array([[token_id]])
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return tokenizer.decode(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="Eval Interlat adapter")
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--receiver", default="mlx-community/gemma-2-2b-it-8bit")
    parser.add_argument("--adapter-path", default="mlxmas/data/interlat_adapter/")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    now = datetime.now().strftime("%-I:%M%p %-m-%-d-%y").lower()
    print(f"Started: {now}", flush=True)

    # Load config
    config_path = os.path.join(args.adapter_path, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    print(f"Adapter config: {json.dumps(config, indent=2)}", flush=True)

    bop_id = config["bop_id"]
    eop_id = config["eop_id"]

    # Load sender (Qwen)
    print(f"\nLoading sender: {args.sender}", flush=True)
    sender_model, sender_tok = mlx_lm.load(args.sender)

    # Load receiver (Gemma)
    print(f"Loading receiver: {args.receiver}", flush=True)
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)

    # Resize embeddings (same as training)
    add_special_tokens(receiver_model, receiver_tok)

    # Apply LoRA structure
    lora_config = {
        "rank": config["lora_rank"],
        "dropout": 0.0,
        "scale": config["lora_scale"],
    }
    linear_to_lora_layers(receiver_model, config["lora_layers"], lora_config)

    # Load trained LoRA weights
    lora_path = os.path.join(args.adapter_path, "lora_adapters.safetensors")
    lora_weights = mx.load(lora_path)
    receiver_model.load_weights(list(lora_weights.items()), strict=False)
    mx.eval(receiver_model.parameters())
    print(f"  Loaded LoRA weights from {lora_path}", flush=True)

    # Load trained MHA adapter
    adapter_path = os.path.join(args.adapter_path, "mha_adapter.safetensors")
    adapter = load_adapter(
        adapter_path,
        sender_dim=config["sender_dim"],
        receiver_dim=config["receiver_dim"],
        num_heads=config["num_heads"],
    )
    print(f"  Loaded MHA adapter from {adapter_path}", flush=True)

    # Load test questions
    print(f"\nLoading GSM8K test...", flush=True)
    items = load_gsm8k_test(max_samples=args.max_samples)
    print(f"Loaded {len(items)} questions.\n", flush=True)

    # Run eval
    correct = 0
    total = len(items)
    t0 = time.time()
    scale = receiver_model.model.args.hidden_size ** 0.5

    for i, item in enumerate(items):
        q_start = time.time()
        print(f"==================== Problem #{i+1}/{total} ====================",
              flush=True)
        print(f"Q: {item['question'][:120]}...", flush=True)

        # Sender: extract last-layer hidden states
        sender_hidden = extract_sender_hidden(
            sender_model, sender_tok, item["question"]
        )
        print(f"  [sender] hidden: {sender_hidden.shape}", flush=True)

        # Adapter: project sender → receiver space
        adapter_out = adapter(sender_hidden)
        mx.eval(adapter_out)

        # Assemble: [user_prompt | <bop> | adapter_out | <eop>]
        user_prompt = (
            "<start_of_turn>user\n"
            "Using the reasoning context provided, solve the problem step by step. "
            "Put your final numerical answer inside \\boxed{}.\n"
            "<end_of_turn>\n<start_of_turn>model\n"
        )
        user_tokens = receiver_tok.encode(user_prompt, add_special_tokens=True)
        user_embeds = receiver_model.model.embed_tokens(mx.array([user_tokens])) * scale
        bop_embed = receiver_model.model.embed_tokens(mx.array([[bop_id]])) * scale
        eop_embed = receiver_model.model.embed_tokens(mx.array([[eop_id]])) * scale

        inputs_embeds = mx.concatenate([
            user_embeds, bop_embed, adapter_out, eop_embed
        ], axis=1)

        # Generate
        output = generate_from_embeds(
            receiver_model, receiver_tok, inputs_embeds,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )

        pred = normalize_answer(extract_boxed_answer(output))
        gold = item["gold"]
        ok = False
        if pred and gold:
            try:
                ok = float(pred) == float(gold)
            except (ValueError, TypeError):
                ok = (pred == gold)
        if ok:
            correct += 1

        elapsed = time.time() - q_start
        total_elapsed = time.time() - t0
        el_m, el_s = divmod(int(total_elapsed), 60)
        remaining = (total - i - 1) * (total_elapsed / (i + 1)) if i > 0 else 0
        eta_m, eta_s = divmod(int(remaining), 60)
        now = datetime.now().strftime("%-I:%M%p %-m-%-d-%y").lower()

        print(f"\n[Gemma output]\n{output[:500]}\n", flush=True)
        print(f"  {now} [{i+1}/{total}] Pred={pred} | Gold={gold} | OK={ok} | "
              f"Time={elapsed:.1f}s elapsed={el_m}m{el_s:02d}s ETA={eta_m}m{eta_s:02d}s",
              flush=True)
        print(f"  Running: {correct}/{i+1} = {100*correct/(i+1):.0f}%\n", flush=True)

    total_time = time.time() - t0
    acc = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: {correct}/{total} = {acc*100:.0f}% accuracy", flush=True)
    print(f"{'='*60}", flush=True)

    print(json.dumps({
        "method": "interlat_mha",
        "sender": args.sender,
        "receiver": args.receiver,
        "adapter_path": args.adapter_path,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "total_time_sec": round(total_time, 2),
        "time_per_sample_sec": round(total_time / total, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
