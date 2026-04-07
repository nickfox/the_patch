#!/usr/bin/env python3
"""
Test 1: Full CKA Heatmap — Qwen3-4B × Gemma-2-2B

Runs GSM8K prompts through both models, extracts final-token hidden
states at EVERY layer, computes linear CKA for all (Qwen layer, Gemma layer)
pairs. Outputs a 36×26 heatmap showing where cross-model representations
are most alignable.

This is the primary empirical test for choosing sender/receiver layers.
Bright spots = high alignment = best candidates for cross-model injection.

Usage:
    python -m mlxmas.test1_cka_heatmap
    python -m mlxmas.test1_cka_heatmap --n_prompts 500
"""

import argparse
import time
import json
import os

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask


def load_gsm8k_prompts(n=200):
    """Load GSM8K training prompts for calibration."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    prompts = []
    for item in ds:
        prompts.append(item["question"].strip())
        if len(prompts) >= n:
            break
    return prompts


def extract_all_layer_states(model, tokenizer, prompt_text, is_gemma=False):
    """Run prompt through model, return final-token hidden state at EVERY layer.

    Manually iterates through transformer layers to capture intermediate states.
    Returns dict: {layer_index: np.array of shape [D]}.
    """
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)

    # Get embeddings
    h = model.model.embed_tokens(input_ids)
    if is_gemma:
        h = h * (model.model.args.hidden_size ** 0.5)

    mask = create_attention_mask(h, cache[0], return_array=True)

    states = {}
    for i, (layer, c) in enumerate(zip(model.model.layers, cache)):
        h = layer(h, mask, c)
        # Capture final-token hidden state (raw, no norm)
        states[i] = np.array(h[0, -1, :].astype(mx.float32))

    mx.eval(h)  # ensure all evals complete
    return states


def collect_all_states(model, tokenizer, prompts, is_gemma=False, label=""):
    """Collect hidden states at every layer for all prompts.

    Returns: dict of {layer_index: np.array of shape [N, D]}
    """
    n_layers = len(model.model.layers)
    by_layer = {i: [] for i in range(n_layers)}

    print(f"  Extracting all-layer states for {len(prompts)} prompts ({label})...")
    t0 = time.time()

    for idx, prompt in enumerate(prompts):
        states = extract_all_layer_states(model, tokenizer, prompt, is_gemma)
        for i in range(n_layers):
            by_layer[i].append(states[i])

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(prompts) - idx - 1) / rate
            print(f"    {idx+1}/{len(prompts)} ({100*(idx+1)/len(prompts):.0f}%) "
                  f"| {rate:.1f}/s | ETA {eta:.0f}s")

    # Stack into matrices
    result = {}
    for i in range(n_layers):
        result[i] = np.stack(by_layer[i], axis=0)  # [N, D]

    elapsed = time.time() - t0
    print(f"  Done: {len(prompts)} prompts, {n_layers} layers in {elapsed:.1f}s")
    return result


def linear_cka(X, Y):
    """Compute linear CKA between two representation matrices.

    X: [N, D1], Y: [N, D2] — N samples, different dimensions OK.
    Returns scalar CKA value in [0, 1].

    Reference: Kornblith et al., "Similarity of Neural Network
    Representations Revisited," ICML 2019.
    """
    # Center both matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    XTX = X @ X.T  # [N, N]
    YTY = Y @ Y.T  # [N, N]

    # HSIC numerator and denominators
    hsic_xy = np.sum(XTX * YTY)  # Frobenius inner product
    hsic_xx = np.sum(XTX * XTX)
    hsic_yy = np.sum(YTY * YTY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def main():
    parser = argparse.ArgumentParser(description="CKA heatmap: Qwen × Gemma layer alignment")
    parser.add_argument("--n_prompts", type=int, default=200,
                        help="Number of GSM8K prompts (default 200)")
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--receiver", default="mlx-community/gemma-2-2b-it-8bit")
    parser.add_argument("--output", default="cka_heatmap.json")
    args = parser.parse_args()

    total_start = time.time()

    # Load prompts
    print(f"Loading {args.n_prompts} GSM8K prompts...")
    prompts = load_gsm8k_prompts(args.n_prompts)

    # Load models
    print(f"\nLoading sender: {args.sender}")
    sender_model, sender_tok = mlx_lm.load(args.sender)
    sender_n_layers = len(sender_model.model.layers)
    print(f"  {sender_n_layers} layers")

    print(f"Loading receiver: {args.receiver}")
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)
    receiver_n_layers = len(receiver_model.model.layers)
    print(f"  {receiver_n_layers} layers")

    # Collect all-layer states
    print(f"\n=== Sender extraction ===")
    sender_states = collect_all_states(
        sender_model, sender_tok, prompts, is_gemma=False, label="Qwen"
    )

    print(f"\n=== Receiver extraction ===")
    receiver_states = collect_all_states(
        receiver_model, receiver_tok, prompts, is_gemma=True, label="Gemma"
    )

    # Compute full CKA heatmap
    print(f"\n=== Computing CKA heatmap ({sender_n_layers} × {receiver_n_layers}) ===")
    heatmap = np.zeros((sender_n_layers, receiver_n_layers))
    t0 = time.time()

    for s in range(sender_n_layers):
        for r in range(receiver_n_layers):
            heatmap[s, r] = linear_cka(sender_states[s], receiver_states[r])
        if (s + 1) % 4 == 0:
            print(f"  Sender layer {s+1}/{sender_n_layers} done")

    elapsed = time.time() - t0
    print(f"  CKA computed in {elapsed:.1f}s")

    # Find top pairs
    flat = []
    for s in range(sender_n_layers):
        for r in range(receiver_n_layers):
            flat.append({"sender_layer": s, "receiver_layer": r, "cka": float(heatmap[s, r])})
    flat.sort(key=lambda x: x["cka"], reverse=True)

    print(f"\n=== Top 20 layer pairs by CKA ===")
    for i, entry in enumerate(flat[:20]):
        s_pct = 100 * entry["sender_layer"] / sender_n_layers
        r_pct = 100 * entry["receiver_layer"] / receiver_n_layers
        print(f"  {i+1:2d}. Qwen L{entry['sender_layer']:2d} ({s_pct:.0f}%) → "
              f"Gemma L{entry['receiver_layer']:2d} ({r_pct:.0f}%)  "
              f"CKA={entry['cka']:.4f}")

    # Save results
    result = {
        "sender": args.sender,
        "receiver": args.receiver,
        "sender_n_layers": sender_n_layers,
        "receiver_n_layers": receiver_n_layers,
        "n_prompts": len(prompts),
        "heatmap": heatmap.tolist(),
        "top_pairs": flat[:50],
        "total_time_sec": round(time.time() - total_start, 1),
    }

    output_path = os.path.join(os.path.dirname(__file__), "data", args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")

    total_time = time.time() - total_start
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
