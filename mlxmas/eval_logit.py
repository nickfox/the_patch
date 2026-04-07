#!/usr/bin/env python3
"""
Eval cross-model latent communication via logit distribution transfer.

Zero-shot: Qwen generates response, logit distributions remapped through
shared vocabulary, converted to soft embeddings, fed to Gemma.
No training, no adapter.

Usage:
    python -m mlxmas.eval_logit --max-samples 5
"""

import argparse
import json
import re
import random
import time
from datetime import datetime

import mlx.core as mx
import mlx_lm

from mlxmas.logit_comm import (
    build_vocab_map,
    precompute_receiver_embeds,
    sender_generate_with_logit_capture,
    sender_generate_hard_tokens,
    receiver_generate_from_tokens,
    receiver_generate_from_soft_embeds,
)
from mlxmas.utils import extract_boxed_answer, normalize_answer


def load_gsm8k_test(max_samples=-1):
    """Load GSM8K test set, shuffled."""
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


def main():
    parser = argparse.ArgumentParser(description="Eval logit distribution transfer")
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--receiver", default="mlx-community/gemma-2-2b-it-8bit")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--sender-max-tokens", type=int, default=2048,
                        help="Max tokens for sender (Qwen) generation")
    parser.add_argument("--receiver-max-tokens", type=int, default=2048,
                        help="Max tokens for receiver (Gemma) generation")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--softmax-temp", type=float, default=0.1,
                        help="Temperature for sharpening shared token distribution (lower=sharper)")
    parser.add_argument("--hard-tokens", action="store_true",
                        help="Use hard token mapping (argmax) instead of soft embeddings")
    args = parser.parse_args()

    now = datetime.now().strftime("%-I:%M%p %-m-%-d-%y").lower()
    print(f"Started: {now}", flush=True)

    # Load sender
    print(f"\nLoading sender: {args.sender}", flush=True)
    sender_model, sender_tok = mlx_lm.load(args.sender)

    # Load receiver
    print(f"Loading receiver: {args.receiver}", flush=True)
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)

    # Build vocabulary map
    print(f"\nBuilding vocabulary map...", flush=True)
    sender_ids, receiver_ids, n_shared = build_vocab_map(sender_tok, receiver_tok)

    # Precompute receiver embeddings for shared tokens (soft mode only)
    scale = receiver_model.model.args.hidden_size ** 0.5
    shared_receiver_embeds = None
    if not args.hard_tokens:
        print(f"Precomputing receiver embeddings (scale={scale:.1f})...", flush=True)
        shared_receiver_embeds = precompute_receiver_embeds(
            receiver_model, receiver_ids, scale
        )
        print(f"  shared_receiver_embeds: {shared_receiver_embeds.shape}", flush=True)

    mode = "hard_tokens" if args.hard_tokens else f"soft_temp={args.softmax_temp}"
    print(f"  Mode: {mode}", flush=True)

    # Load test questions
    print(f"\nLoading GSM8K test...", flush=True)
    items = load_gsm8k_test(max_samples=args.max_samples)
    print(f"Loaded {len(items)} questions.\n", flush=True)

    # Run eval
    correct = 0
    total = len(items)
    t0 = time.time()

    for i, item in enumerate(items):
        q_start = time.time()
        print(f"==================== Problem #{i+1}/{total} ====================",
              flush=True)
        print(f"Q: {item['question'][:120]}...", flush=True)

        if args.hard_tokens:
            # Hard token mode: argmax through vocab map
            token_ids, n_tokens, avg_shared_mass = sender_generate_hard_tokens(
                sender_model, sender_tok, item["question"],
                sender_ids, receiver_ids,
                max_tokens=args.sender_max_tokens,
                temperature=args.temperature,
            )
            print(f"  [sender] {n_tokens} hard tokens, avg shared mass: {avg_shared_mass:.1%}",
                  flush=True)

            if n_tokens == 0:
                print(f"  [sender] no tokens generated, skipping", flush=True)
                continue

            # Show what Gemma receives
            translated = receiver_tok.decode(token_ids)
            print(f"  [translated] {translated[:200]}...", flush=True)

            output = receiver_generate_from_tokens(
                receiver_model, receiver_tok, token_ids,
                max_tokens=args.receiver_max_tokens,
                temperature=args.temperature,
            )
        else:
            # Soft embedding mode
            soft_embeds, n_tokens, avg_shared_mass = sender_generate_with_logit_capture(
                sender_model, sender_tok, item["question"],
                sender_ids, shared_receiver_embeds,
                max_tokens=args.sender_max_tokens,
                temperature=args.temperature,
                softmax_temp=args.softmax_temp,
            )
            print(f"  [sender] {n_tokens} soft tokens, avg shared mass: {avg_shared_mass:.1%}",
                  flush=True)

            if soft_embeds is None:
                print(f"  [sender] no tokens generated, skipping", flush=True)
                continue

            output = receiver_generate_from_soft_embeds(
                receiver_model, receiver_tok, soft_embeds,
                scale=scale,
                max_tokens=args.receiver_max_tokens,
                temperature=args.temperature,
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
        "method": "logit_transfer",
        "sender": args.sender,
        "receiver": args.receiver,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "n_shared_tokens": n_shared,
        "total_time_sec": round(total_time, 2),
        "time_per_sample_sec": round(total_time / total, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
