#!/usr/bin/env python3
"""
Collect Qwen sender hidden states for Interlat training.

Runs Qwen on GSM8K training questions, generates CoT solutions,
captures last-layer hidden states per generated token. Saves alongside
generated text (the "plan") and gold answers.

Qwen is the only model loaded (~4GB). No memory pressure.

Usage:
    python -m mlxmas.collect_sender_hidden \
        --sender mlx-community/Qwen3-4B-8bit \
        --n-prompts 500 \
        --output mlxmas/data/sender_hidden_states.npz
"""

import argparse
import os
import time

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.base import create_attention_mask


def extract_prefill_hidden(model, tokenizer, prompt):
    """Single forward pass on prompt, capture last-layer hidden states.

    Returns: hidden_states [1, seq_len, D]
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array([tokens])

    h = model.model.embed_tokens(input_ids)
    mask = create_attention_mask(h, return_array=True)

    for layer in model.model.layers:
        h = layer(h, mask, cache=None)

    normed = model.model.norm(h)  # [1, seq_len, D]
    mx.eval(normed)
    return normed


def load_gsm8k_train(n_prompts=500):
    """Load GSM8K training questions."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    items = []
    for item in ds:
        items.append({
            "question": item["question"].strip(),
            "answer": item["answer"],
        })
        if len(items) >= n_prompts:
            break
    return items


def extract_gold(solution_text):
    """Extract the numeric answer from GSM8K solution text."""
    lines = solution_text.strip().split("\n")
    last = lines[-1]
    answer = last.replace("####", "").strip()
    answer = answer.replace(",", "").replace("$", "")
    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Collect Qwen sender hidden states for Interlat training"
    )
    parser.add_argument("--sender", type=str,
                        default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--n-prompts", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="mlxmas/data/sender_hidden_states.npz")
    args = parser.parse_args()

    from datetime import datetime
    start_stamp = datetime.now().strftime("%-I:%M%p %-m-%-d-%y").lower()
    print(f"Started: {start_stamp}", flush=True)

    print(f"Loading sender: {args.sender}", flush=True)
    model, tokenizer = mlx_lm.load(args.sender)
    print(f"Model loaded. D={model.args.hidden_size}", flush=True)

    print(f"Loading GSM8K train ({args.n_prompts} prompts)...", flush=True)
    items = load_gsm8k_train(args.n_prompts)
    print(f"Loaded {len(items)} items.\n", flush=True)

    all_hidden = []
    all_questions = []
    all_gold = []
    t0 = time.time()

    for i, item in enumerate(items):
        question = item["question"]
        gold = extract_gold(item["answer"])

        # Build sender prompt
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

        hidden = extract_prefill_hidden(model, tokenizer, prompt)

        h_np = np.array(hidden[0].astype(mx.float32))  # [seq_len, D]
        all_hidden.append(h_np)
        all_questions.append(question)
        all_gold.append(gold)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta_sec = (len(items) - i - 1) / rate if rate > 0 else 0
        eta_m, eta_s = divmod(int(eta_sec), 60)
        el_m, el_s = divmod(int(elapsed), 60)
        if (i + 1) % 25 == 0 or i == 0:
            from datetime import datetime
            now = datetime.now().strftime("%-I:%M%p %-m-%-d-%y").lower()
            print(f"  {now} [{i+1}/{len(items)}] tokens={h_np.shape[0]} "
                  f"elapsed={el_m}m{el_s:02d}s ETA={eta_m}m{eta_s:02d}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nCollected {len(all_hidden)} samples in {elapsed:.1f}s", flush=True)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        hidden_states=np.array(all_hidden, dtype=object),
        questions=np.array(all_questions, dtype=object),
        gold_answers=np.array(all_gold, dtype=object),
        n_samples=len(all_hidden),
        sender_model=args.sender,
    )
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved to {args.output} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
