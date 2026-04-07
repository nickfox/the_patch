#!/usr/bin/env python3
"""
Gemma→Qwen cross-model latent communication test.

Simplified pipeline (no LatentMAS subagent loop):
  Gemma reads question → extract hidden states at layer G
  → Procrustes projection into Qwen's space
  → inject into Qwen at layer Q
  → Qwen generates answer

One forward pass per model. No latent thinking loop, no self-alignment,
no subagents.
"""

import argparse
import json
import os
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from mlxmas.contextual_procrustes import load_alignment
from mlxmas.cross_align import apply_cross_realignment
from mlxmas.cross_comm import (
    extract_gemma_all_tokens,
    generate_with_qwen_from_layer,
)
from mlxmas.utils import extract_boxed_answer, normalize_answer


SENDER_NAME = "mlx-community/gemma-2-2b-it-8bit"
RECEIVER_NAME = "mlx-community/Qwen3-4B-8bit"


def load_gsm8k_test(max_samples=-1):
    """Load GSM8K test set questions with gold answers."""
    import re
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = []
    for item in ds:
        gold_match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", item["answer"])
        gold = gold_match.group(1) if gold_match else None
        questions.append({
            "question": item["question"].strip(),
            "gold": gold,
        })
        if max_samples > 0 and len(questions) >= max_samples:
            break
    return questions


def qwen_solve_direct(model, tokenizer, question, max_tokens=2048, temperature=0.6):
    """Qwen solves the question directly — no cross-model, no LatentMAS.

    Used as baseline to measure whether injected latent context helps or hurts.
    """
    from mlx_lm.sample_utils import make_sampler

    prompt = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Solve the following math problem step by step.\n\n"
        f"Question: {question}\n\n"
        f"Put your final numerical answer inside \\boxed{{}}.\n"
        f"<|im_end|>\n<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)
    logits = model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    sampler = make_sampler(temp=temperature, top_p=0.95)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = tokenizer.eos_token_id

    for _ in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        next_input = mx.array([[token_id]])
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return tokenizer.decode(generated_tokens)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Gemma→Qwen cross-model latent communication test"
    )
    parser.add_argument("--procrustes", type=str, default=None,
                        help="Path to Procrustes alignment .npz file")
    parser.add_argument("--sender-layer", type=int, default=10,
                        help="Gemma sender layer to extract from (default 10)")
    parser.add_argument("--receiver-layer", type=int, default=12,
                        help="Qwen receiver layer to inject at (default 12)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Number of GSM8K test questions (default 50, -1 for all)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--qwen-only", action="store_true",
                        help="Run Qwen standalone baseline (no cross-model)")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load questions
    questions = load_gsm8k_test(max_samples=args.max_samples)
    print(f"Loaded {len(questions)} GSM8K test questions\n")

    if args.qwen_only:
        # =========================================================
        # Qwen standalone baseline — no cross-model, no LatentMAS
        # =========================================================
        print(f"=== Qwen Standalone Baseline ===")
        print(f"Model: {RECEIVER_NAME}")
        print(f"No cross-model injection, direct prompting only\n", flush=True)

        qwen_model, qwen_tok = mlx_lm.load(RECEIVER_NAME)

        correct = 0
        total_start = time.time()
        for i, item in enumerate(questions):
            question = item["question"]
            gold = item["gold"]
            print(f"===== Question #{i+1} =====", flush=True)
            print(f"Q: {question[:100]}...", flush=True)
            print(f"Gold: {gold}", flush=True)

            t0 = time.time()
            output = qwen_solve_direct(
                qwen_model, qwen_tok, question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            elapsed = time.time() - t0

            pred = normalize_answer(extract_boxed_answer(output))
            ok = False
            if pred and gold:
                try:
                    ok = float(pred) == float(gold)
                except (ValueError, TypeError):
                    ok = (pred == gold)
            if ok:
                correct += 1

            print(f"\n[Qwen output]\n{output[:500]}\n")
            print(f"Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
            print(f"Running: {correct}/{i+1} = {100*correct/(i+1):.0f}%\n")

        total_time = time.time() - total_start
        total = len(questions)
        acc = correct / total if total > 0 else 0.0
        print(f"\n{'='*60}")
        print(f"QWEN STANDALONE: {correct}/{total} = {acc*100:.0f}%")
        print(f"{'='*60}")
        print(json.dumps({
            "method": "qwen_standalone",
            "model": RECEIVER_NAME,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "total_time_sec": round(total_time, 2),
            "time_per_sample_sec": round(total_time / total, 2),
        }, ensure_ascii=False))
        return

    # =========================================================
    # Gemma→Qwen cross-model pipeline
    # =========================================================
    sender_layer = args.sender_layer
    receiver_layer = args.receiver_layer

    # Load Procrustes alignment
    procrustes_path = args.procrustes
    if procrustes_path is None:
        procrustes_path = os.path.join(
            os.path.dirname(__file__), "data",
            f"procrustes_gemma_S{sender_layer}_qwen_R{receiver_layer}_multitoken.npz"
        )

    if not os.path.exists(procrustes_path):
        print(f"Procrustes alignment file not found: {procrustes_path}")
        print(f"Run calibration first:")
        print(f"  python -m mlxmas.contextual_procrustes \\")
        print(f"    --multitoken \\")
        print(f"    --model_a {SENDER_NAME} \\")
        print(f"    --model_b {RECEIVER_NAME} \\")
        print(f"    --sender_layer {sender_layer} \\")
        print(f"    --receiver_layer {receiver_layer} \\")
        print(f"    --n_calibration 2000 --n_positions 20 \\")
        print(f"    --output {procrustes_path}")
        return

    print(f"=== Gemma→Qwen Cross-Model Pipeline ===")
    print(f"Sender: {SENDER_NAME} (layer {sender_layer})")
    print(f"Receiver: {RECEIVER_NAME} (layer {receiver_layer})")
    print(f"Procrustes: {procrustes_path}\n")

    alignment = load_alignment(procrustes_path)
    W_ab = alignment["W_ortho"]
    mean_a = alignment["mean_a"]
    mean_b = alignment["mean_b"]
    target_norm_b = alignment["target_norm_b"]

    if "heldout_cosine" in alignment:
        print(f"  Held-out cosine: {alignment['heldout_cosine']:.4f}, "
              f"train cosine: {alignment['train_cosine']:.4f}")
    print(f"  W shape: {W_ab.shape}\n")

    # Load models
    print(f"Loading sender: {SENDER_NAME}")
    gemma_model, gemma_tok = mlx_lm.load(SENDER_NAME)
    print(f"Loading receiver: {RECEIVER_NAME}")
    qwen_model, qwen_tok = mlx_lm.load(RECEIVER_NAME)
    print()

    # Run test questions
    correct = 0
    total_start = time.time()
    for i, item in enumerate(questions):
        question = item["question"]
        gold = item["gold"]
        print(f"==================== Question #{i+1} ====================", flush=True)
        print(f"Q: {question[:100]}...", flush=True)
        print(f"Gold: {gold}", flush=True)

        t0 = time.time()

        # 1. Gemma reads the question — extract at sender layer
        gemma_hidden = extract_gemma_all_tokens(
            gemma_model, gemma_tok, question, layer=sender_layer
        )
        mx.eval(gemma_hidden)
        print(f"  Gemma extracted: {gemma_hidden.shape} at layer {sender_layer}")

        # 2. Project into Qwen's space
        projected = apply_cross_realignment(
            gemma_hidden, W_ab, mean_a, mean_b, target_norm_b
        )
        mx.eval(projected)
        print(f"  Projected to Qwen space: {projected.shape}")

        # 3. Inject into Qwen and generate
        print(f"  Generating with Qwen (inject at layer {receiver_layer})...")
        output = generate_with_qwen_from_layer(
            qwen_model, qwen_tok,
            projected,
            start_layer=receiver_layer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        elapsed = time.time() - t0
        pred = normalize_answer(extract_boxed_answer(output))
        ok = False
        if pred and gold:
            try:
                ok = float(pred) == float(gold)
            except (ValueError, TypeError):
                ok = (pred == gold)
        if ok:
            correct += 1

        print(f"\n[Qwen output]\n{output[:500]}\n")
        print(f"Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
        print(f"Running: {correct}/{i+1} = {100*correct/(i+1):.0f}%\n")

    total_time = time.time() - total_start
    total = len(questions)
    acc = correct / total if total > 0 else 0.0
    method_name = f"gemma_S{sender_layer}_qwen_R{receiver_layer}"

    print(f"\n{'='*60}")
    print(f"FINAL: {correct}/{total} = {acc*100:.0f}% accuracy")
    print(f"Method: {method_name}")
    print(f"Baselines:")
    print(f"  Qwen→Gemma Procrustes S13→R12: 3/5 (60%) on first 5")
    print(f"  Vocab LS baseline (L0→L0):      1/5 (20%)")
    print(f"{'='*60}")

    print(json.dumps({
        "method": method_name,
        "sender": SENDER_NAME,
        "receiver": RECEIVER_NAME,
        "sender_layer": sender_layer,
        "receiver_layer": receiver_layer,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "total_time_sec": round(total_time, 2),
        "time_per_sample_sec": round(total_time / total, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
