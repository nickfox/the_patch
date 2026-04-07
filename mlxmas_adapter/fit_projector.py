#!/usr/bin/env python3
"""
Fit and save the barycentric ridge self-projector for a sender model.

Usage:
    cd /Users/nickfox137/Documents/mlx_latentmas/mlxmas_adapter
    source ../venv/bin/activate
    python fit_projector.py --sender mlx-community/Qwen3-4B-8bit \
        --n-prompts 200 --output data/qwen_self_projector.npz
"""

import argparse
import os
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from self_projector import fit_self_projector


def main():
    parser = argparse.ArgumentParser(
        description="Fit barycentric ridge self-projector"
    )
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit",
                        help="Sender model")
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of calibration prompts from GSM8K train")
    parser.add_argument("--k", type=int, default=64,
                        help="Top-k for barycentric teacher")
    parser.add_argument("--tau", type=float, default=0.7,
                        help="Softmax temperature")
    parser.add_argument("--lam", type=float, default=0.01,
                        help="Ridge regularization fraction")
    parser.add_argument("--pos-chunk", type=int, default=8,
                        help="Chunk size for vocab projection")
    parser.add_argument("--output", default="data/qwen_self_projector.npz",
                        help="Output path for projector")
    args = parser.parse_args()

    # Load calibration data
    print("Loading GSM8K calibration data...")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    prompts = [item["question"].strip() for item in ds]
    print(f"  {len(prompts)} total prompts available, using {args.n_prompts}")

    # Load sender model
    print(f"\nLoading sender: {args.sender}")
    t0 = time.time()
    sender_model, sender_tok = mlx_lm.load(args.sender)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  D={sender_model.args.hidden_size}, V={sender_model.args.vocab_size}")

    # Fit projector
    print(f"\n=== Fitting self-projector ===")
    t0 = time.time()
    projector, E_raw = fit_self_projector(
        sender_model, sender_tok, prompts,
        k=args.k, tau=args.tau, lam=args.lam,
        n_prompts=args.n_prompts, pos_chunk=args.pos_chunk,
    )
    fit_time = time.time() - t0
    print(f"Fit completed in {fit_time:.1f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    projector.save(args.output)

    # Quick validation on held-out prompts
    print(f"\n=== Validation (held-out prompts) ===")
    val_prompts = prompts[args.n_prompts:args.n_prompts + 5]

    cos_sims = []
    for vp in val_prompts:
        tokens = sender_tok.encode(vp, add_special_tokens=True)
        input_ids = mx.array([tokens])
        cache = make_prompt_cache(sender_model)
        z = sender_model.model(input_ids, cache=cache)
        mx.eval(z)

        z_2d = z[0]  # [seq, D]
        y_fast = projector.project_fast(z_2d)
        y_exact = projector.project_exact(z_2d, E_raw)
        mx.eval(y_fast, y_exact)

        cos = mx.mean(mx.sum(y_fast * y_exact, axis=-1)).item()
        cos_sims.append(cos)

    mean_cos = sum(cos_sims) / len(cos_sims)
    print(f"Held-out projector fidelity (fast vs exact): {mean_cos:.4f}")
    if mean_cos > 0.85:
        print("  PASS: fidelity > 0.85")
    else:
        print(f"  WARNING: fidelity {mean_cos:.4f} < 0.85")


if __name__ == "__main__":
    main()
