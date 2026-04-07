#!/usr/bin/env python3
"""
Contextual Semi-Orthogonal Procrustes Alignment

Computes cross-model alignment matrix using paired contextual
hidden states (Layer-N) instead of vocabulary embeddings (Layer-0).

Key insight: the alignment matrix should be computed on the same
type of object we actually transmit — contextual reasoning states
from the final transformer layer.

Usage:
    python contextual_procrustes.py
    python contextual_procrustes.py --n_calibration 2000
"""

import argparse
import time
import os
import json

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache


def load_calibration_prompts(n=2000):
    """Load GSM8K training prompts as calibration data. Use n=-1 for all."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    prompts = []
    for item in ds:
        prompts.append(item["question"].strip())
        if n > 0 and len(prompts) >= n:
            break
    return prompts


def extract_final_hidden_state(model, tokenizer, prompt_text):
    """Run prompt through model, return last-layer hidden state of final token.

    Returns: [1, D] array — the contextual representation of this prompt.
    Uses model.model (transformer only, no lm_head) to get hidden states.
    """
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)
    # Forward through transformer layers only (no lm_head)
    hidden = model.model(input_ids, cache=cache)
    mx.eval(hidden)

    # Extract final token's hidden state: [1, D]
    final_state = hidden[:, -1, :]
    return final_state


def extract_hidden_at_layers(model, tokenizer, prompt_text, layers, is_gemma=False):
    """Run prompt through model, return hidden states at specified layers.

    Manually iterates through model.model.layers to capture intermediate
    hidden states. Does NOT apply final norm or lm_head — raw residual-stream states.

    Args:
        model: MLX model (Qwen3 or Gemma-2)
        tokenizer: tokenizer
        prompt_text: input prompt
        layers: list of int, which layer outputs to capture (0-indexed)
        is_gemma: whether this is a Gemma model (applies sqrt(hidden_size) scaling)

    Returns:
        dict mapping layer_index -> [1, D] final-token hidden state
    """
    from mlx_lm.models.base import create_attention_mask

    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)

    # Embed tokens
    h = model.model.embed_tokens(input_ids)
    if is_gemma:
        h = h * (model.model.args.hidden_size ** 0.5)

    mask = create_attention_mask(h, cache[0], return_array=True)

    captured = {}
    for i, (layer, c) in enumerate(zip(model.model.layers, cache)):
        h = layer(h, mask, c)
        if i in layers:
            captured[i] = h[:, -1, :].astype(mx.float32)

    mx.eval(*captured.values())
    return captured


def extract_all_tokens_at_layer(model, tokenizer, prompt_text, layer, is_gemma=False):
    """Extract hidden states at ALL token positions at a specific layer.

    Returns: numpy array of shape [seq_len, D] (float32).
    """
    import numpy as np
    from mlx_lm.models.base import create_attention_mask

    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)
    h = model.model.embed_tokens(input_ids)
    if is_gemma:
        h = h * (model.model.args.hidden_size ** 0.5)

    mask = create_attention_mask(h, cache[0], return_array=True)

    for i, (lay, c) in enumerate(zip(model.model.layers, cache)):
        h = lay(h, mask, c)
        if i == layer:
            states = h[0].astype(mx.float32)  # [seq_len, D]
            mx.eval(states)
            result = np.array(states)
            # Run remaining layers to completion for cleanup
            for j in range(i + 1, len(model.model.layers)):
                h = model.model.layers[j](h, mask, cache[j])
            mx.eval(h)
            return result

    raise ValueError(f"Layer {layer} not reached")


def calibrate_multitoken_incremental(
    model_a, tok_a, model_b, tok_b, prompts,
    sender_layer, receiver_layer, is_gemma_a=False, is_gemma_b=True,
    heldout_frac=0.2, n_positions=20,
):
    """Two-pass incremental Procrustes calibration with multi-token extraction.

    For each prompt, extracts hidden states at ALL token positions at the
    target layer. Resamples both sequences to n_positions evenly-spaced
    points (handling different tokenizer lengths). This gives
    N = len(prompts) * n_positions paired vectors.

    Pass 1: Compute running means (sum / count).
    Pass 2: Accumulate cross-covariance C = (H_a - mean_a).T @ (H_b - mean_b)
             incrementally without materializing the full N × D matrix.

    Args:
        model_a, tok_a: sender model and tokenizer
        model_b, tok_b: receiver model and tokenizer
        prompts: list of calibration prompt strings
        sender_layer: sender layer to extract at
        receiver_layer: receiver layer to extract at
        is_gemma_a: whether model_a is Gemma (applies sqrt scaling)
        is_gemma_b: whether model_b is Gemma (applies sqrt scaling)
        heldout_frac: fraction of prompts to hold out
        n_positions: number of evenly-spaced positions to sample per prompt

    Returns:
        dict with W_ortho, means, norms, train/heldout cosine, n_vectors
    """
    import numpy as np

    N = len(prompts)
    n_heldout = max(1, int(N * heldout_frac))
    n_train = N - n_heldout
    train_prompts = prompts[:n_train]
    held_prompts = prompts[n_train:]

    D_a = model_a.model.embed_tokens(mx.array([[0]])).shape[-1]
    D_b = model_b.model.embed_tokens(mx.array([[0]])).shape[-1]
    mx.eval(mx.array(0))  # flush

    print(f"  D_a={D_a}, D_b={D_b}, n_positions={n_positions}")
    print(f"  Train: {n_train} prompts → ~{n_train * n_positions} vectors")
    print(f"  Held-out: {len(held_prompts)} prompts")

    def resample(arr, n_pos):
        """Resample [seq_len, D] to [n_pos, D] via nearest-neighbor."""
        seq_len = arr.shape[0]
        if seq_len <= n_pos:
            return arr
        indices = np.linspace(0, seq_len - 1, n_pos).astype(int)
        return arr[indices]

    # === Pass 1: Compute means ===
    print(f"\n  Pass 1: Computing means...")
    sum_a = np.zeros(D_a, dtype=np.float64)
    sum_b = np.zeros(D_b, dtype=np.float64)
    total_vectors = 0
    t0 = time.time()

    for i, prompt in enumerate(train_prompts):
        ha = extract_all_tokens_at_layer(model_a, tok_a, prompt, sender_layer, is_gemma=is_gemma_a)
        hb = extract_all_tokens_at_layer(model_b, tok_b, prompt, receiver_layer, is_gemma=is_gemma_b)
        ha = resample(ha, n_positions)
        hb = resample(hb, n_positions)
        n = min(ha.shape[0], hb.shape[0])
        ha, hb = ha[:n], hb[:n]

        sum_a += ha.sum(axis=0).astype(np.float64)
        sum_b += hb.sum(axis=0).astype(np.float64)
        total_vectors += n

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_train - i - 1) / rate
            print(f"    {i+1}/{n_train} ({rate:.1f}/s, ETA {eta:.0f}s) "
                  f"vectors={total_vectors}", flush=True)

    mean_a = (sum_a / total_vectors).astype(np.float32)
    mean_b = (sum_b / total_vectors).astype(np.float32)
    elapsed = time.time() - t0
    print(f"    Done: {total_vectors} vectors in {elapsed:.1f}s")

    # === Pass 2: Accumulate cross-covariance ===
    print(f"\n  Pass 2: Computing cross-covariance...")
    C = np.zeros((D_a, D_b), dtype=np.float64)
    norm_sum_a = 0.0
    norm_sum_b = 0.0
    t0 = time.time()

    for i, prompt in enumerate(train_prompts):
        ha = extract_all_tokens_at_layer(model_a, tok_a, prompt, sender_layer, is_gemma=is_gemma_a)
        hb = extract_all_tokens_at_layer(model_b, tok_b, prompt, receiver_layer, is_gemma=is_gemma_b)
        ha = resample(ha, n_positions)
        hb = resample(hb, n_positions)
        n = min(ha.shape[0], hb.shape[0])
        ha, hb = ha[:n], hb[:n]

        ha_c = ha.astype(np.float64) - mean_a.astype(np.float64)
        hb_c = hb.astype(np.float64) - mean_b.astype(np.float64)
        C += ha_c.T @ hb_c

        norm_sum_a += np.linalg.norm(ha, axis=1).sum()
        norm_sum_b += np.linalg.norm(hb, axis=1).sum()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_train - i - 1) / rate
            print(f"    {i+1}/{n_train} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    target_norm_a = norm_sum_a / total_vectors
    target_norm_b = norm_sum_b / total_vectors

    # === SVD ===
    print(f"\n  SVD on [{D_a}, {D_b}] cross-covariance...")
    t0 = time.time()
    C_mx = mx.array(C.astype(np.float32))
    U, S, Vt = mx.linalg.svd(C_mx, stream=mx.cpu)
    mx.eval(U, S, Vt)
    k = min(D_a, D_b)
    W_ortho = U[:, :k] @ Vt[:k, :]  # [D_a, D_b]
    mx.eval(W_ortho)
    print(f"    SVD done in {time.time()-t0:.2f}s, W: [{D_a}, {D_b}]")

    # === Train cosine ===
    print(f"\n  Computing train cosine (sampling 500 vectors)...")
    cos_nums, cos_denoms = [], []
    sample_count = 0
    for i, prompt in enumerate(train_prompts[:50]):
        ha = extract_all_tokens_at_layer(model_a, tok_a, prompt, sender_layer, is_gemma=is_gemma_a)
        hb = extract_all_tokens_at_layer(model_b, tok_b, prompt, receiver_layer, is_gemma=is_gemma_b)
        ha = resample(ha, n_positions)
        hb = resample(hb, n_positions)
        n = min(ha.shape[0], hb.shape[0])

        ha_c = mx.array(ha[:n].astype(np.float32)) - mx.array(mean_a)
        hb_c = mx.array(hb[:n].astype(np.float32)) - mx.array(mean_b)
        hb_hat = ha_c @ W_ortho
        cos = mx.sum(hb_hat * hb_c, axis=1) / (
            mx.linalg.norm(hb_hat, axis=1) * mx.linalg.norm(hb_c, axis=1) + 1e-8
        )
        mx.eval(cos)
        cos_nums.extend(np.array(cos.astype(mx.float32)).tolist())
        sample_count += n

    train_cos = float(np.mean(cos_nums))

    # === Held-out cosine ===
    print(f"  Computing held-out cosine ({len(held_prompts)} prompts)...")
    cos_held = []
    for prompt in held_prompts:
        ha = extract_all_tokens_at_layer(model_a, tok_a, prompt, sender_layer, is_gemma=is_gemma_a)
        hb = extract_all_tokens_at_layer(model_b, tok_b, prompt, receiver_layer, is_gemma=is_gemma_b)
        ha = resample(ha, n_positions)
        hb = resample(hb, n_positions)
        n = min(ha.shape[0], hb.shape[0])

        ha_c = mx.array(ha[:n].astype(np.float32)) - mx.array(mean_a)
        hb_c = mx.array(hb[:n].astype(np.float32)) - mx.array(mean_b)
        hb_hat = ha_c @ W_ortho
        cos = mx.sum(hb_hat * hb_c, axis=1) / (
            mx.linalg.norm(hb_hat, axis=1) * mx.linalg.norm(hb_c, axis=1) + 1e-8
        )
        mx.eval(cos)
        cos_held.extend(np.array(cos.astype(mx.float32)).tolist())

    heldout_cos = float(np.mean(cos_held))

    top_5_sv = S[:5].tolist()
    sv_sum = mx.sum(S).item()
    top_5_pct = sum(top_5_sv) / sv_sum * 100

    print(f"\n  === Multi-token Procrustes Results ===")
    print(f"    Total training vectors: {total_vectors}")
    print(f"    N/D ratio: {total_vectors/D_a:.1f}")
    print(f"    Train cosine:    {train_cos:.4f}")
    print(f"    Held-out cosine: {heldout_cos:.4f}")
    print(f"    Overfit gap:     {train_cos - heldout_cos:.4f}")
    print(f"    Target norm A:   {target_norm_a:.4f}")
    print(f"    Target norm B:   {target_norm_b:.4f}")
    print(f"    Top-5 SV explain {top_5_pct:.1f}% of variance")

    return {
        "W_ortho": W_ortho,
        "mean_a": mx.array(mean_a.reshape(1, -1)),
        "mean_b": mx.array(mean_b.reshape(1, -1)),
        "target_norm_a": mx.array(target_norm_a),
        "target_norm_b": mx.array(target_norm_b),
        "train_cosine": train_cos,
        "heldout_cosine": heldout_cos,
        "n_train_vectors": total_vectors,
        "n_train_prompts": n_train,
        "n_heldout_prompts": len(held_prompts),
        "sender_layer": sender_layer,
        "receiver_layer": receiver_layer,
    }


def collect_contextual_pairs(model_a, tok_a, model_b, tok_b,
                              prompts, is_gemma_a=False, is_gemma_b=True,
                              sender_layer=None, receiver_layer=None):
    """Collect paired contextual hidden states from both models.

    Each prompt produces one paired anchor: (hidden_a, hidden_b).

    When sender_layer/receiver_layer are None, extracts final-layer hidden
    states (original behavior). When specified, extracts at those specific
    intermediate layers using extract_hidden_at_layers().

    Args:
        model_a: sender model (Qwen)
        tok_a: sender tokenizer
        model_b: receiver model (Gemma)
        tok_b: receiver tokenizer
        prompts: list of calibration prompt strings
        is_gemma_b: whether model_b is Gemma (applies sqrt scaling)
        sender_layer: specific Qwen layer to extract at (None = final)
        receiver_layer: specific Gemma layer to extract at (None = final)

    Returns:
        H_a: [N, D_a] contextual states from model A
        H_b: [N, D_b] contextual states from model B
    """
    N = len(prompts)
    states_a = []
    states_b = []

    use_layers = sender_layer is not None or receiver_layer is not None
    if use_layers:
        print(f"Collecting {N} pairs at sender_layer={sender_layer}, "
              f"receiver_layer={receiver_layer}...")
    else:
        print(f"Collecting {N} contextual pairs (final layer)...")
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        if sender_layer is not None:
            captured = extract_hidden_at_layers(
                model_a, tok_a, prompt, [sender_layer], is_gemma=is_gemma_a
            )
            states_a.append(captured[sender_layer])
        else:
            states_a.append(extract_final_hidden_state(model_a, tok_a, prompt))

        if receiver_layer is not None:
            captured = extract_hidden_at_layers(
                model_b, tok_b, prompt, [receiver_layer], is_gemma=is_gemma_b
            )
            states_b.append(captured[receiver_layer])
        else:
            states_b.append(extract_final_hidden_state(model_b, tok_b, prompt))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  {i+1}/{N} pairs collected ({rate:.1f}/s, ETA {eta:.0f}s)")

    # Stack into matrices
    H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)  # [N, D_a]
    H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)  # [N, D_b]
    mx.eval(H_a, H_b)

    elapsed = time.time() - t0
    print(f"  Done: {N} pairs in {elapsed:.1f}s ({N/elapsed:.1f} pairs/s)")
    print(f"  H_a: {H_a.shape}, H_b: {H_b.shape}")

    return H_a, H_b


def compute_procrustes(H_a, H_b):
    """Compute semi-orthogonal Procrustes alignment from contextual pairs.

    Given paired matrices H_a [N, D_a] and H_b [N, D_b]:
    1. Mean-center both
    2. Cross-covariance C = H_a^T @ H_b
    3. SVD: C = U Sigma V^T
    4. Semi-orthogonal solution: W = U @ V^T

    W^T @ W = I guarantees isometric projection (preserves distances/angles).

    Returns:
        W_ortho: [D_a, D_b] semi-orthogonal alignment matrix
        mean_a: [1, D_a] sender mean
        mean_b: [1, D_b] receiver mean
        target_norm_b: scalar, average norm of receiver states
        quality metrics dict
    """
    print("Computing Procrustes alignment...")

    # Mean-center
    mean_a = mx.mean(H_a, axis=0, keepdims=True)
    mean_b = mx.mean(H_b, axis=0, keepdims=True)
    H_a_c = H_a - mean_a
    H_b_c = H_b - mean_b
    mx.eval(H_a_c, H_b_c, mean_a, mean_b)

    # Cross-covariance matrix
    C = H_a_c.T @ H_b_c  # [D_a, D_b]
    mx.eval(C)
    print(f"  Cross-covariance: {C.shape}")

    # SVD — needs CPU stream in MLX
    t0 = time.time()
    U, S, Vt = mx.linalg.svd(C, stream=mx.cpu)
    mx.eval(U, S, Vt)
    print(f"  SVD computed in {time.time()-t0:.2f}s")
    print(f"  U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

    # Semi-orthogonal Procrustes: W = U @ V^T
    # k = min(D_a, D_b) handles both D_a > D_b and D_a < D_b
    D_a = H_a.shape[1]
    D_b = H_b.shape[1]
    k = min(D_a, D_b)
    W_ortho = U[:, :k] @ Vt[:k, :]  # [D_a, D_b]
    mx.eval(W_ortho)
    print(f"  W_ortho: {W_ortho.shape}")

    # Quality metrics on the calibration data
    H_b_hat = H_a_c @ W_ortho  # projected centered states
    mx.eval(H_b_hat)

    # Cosine similarity (centered)
    cos_sim = mx.mean(
        mx.sum(H_b_hat * H_b_c, axis=1) /
        (mx.linalg.norm(H_b_hat, axis=1) * mx.linalg.norm(H_b_c, axis=1) + 1e-8)
    ).item()

    # Relative reconstruction error (centered)
    recon_err = mx.mean(mx.linalg.norm(H_b_hat - H_b_c, axis=1)).item()
    mean_norm_b = mx.mean(mx.linalg.norm(H_b_c, axis=1)).item()
    rel_error = recon_err / mean_norm_b

    # Target norm for runtime norm-matching
    target_norm_b = mx.mean(mx.linalg.norm(H_b, axis=1))
    target_norm_a = mx.mean(mx.linalg.norm(H_a, axis=1))
    mx.eval(target_norm_b, target_norm_a)

    # Singular value spectrum (how much structure is shared)
    top_5_sv = S[:5].tolist()
    sv_sum = mx.sum(S).item()
    top_5_pct = sum(top_5_sv) / sv_sum * 100

    print(f"\n  Contextual Procrustes Quality:")
    print(f"    Cosine similarity: {cos_sim:.4f}")
    print(f"    Relative error:    {rel_error:.4f}")
    print(f"    Target norm A:     {target_norm_a.item():.4f}")
    print(f"    Target norm B:     {target_norm_b.item():.4f}")
    print(f"    Top-5 singular values: {[f'{v:.2f}' for v in top_5_sv]}")
    print(f"    Top-5 explain {top_5_pct:.1f}% of total variance")

    return {
        "W_ortho": W_ortho,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "target_norm_a": target_norm_a,
        "target_norm_b": target_norm_b,
        "cos_sim": cos_sim,
        "rel_error": rel_error,
        "n_pairs": H_a.shape[0],
    }


def compute_procrustes_with_heldout(H_a, H_b, heldout_frac=0.2, center=True):
    """Compute Procrustes with train/held-out split for overfitting detection.

    Args:
        H_a: [N, D_a] sender states
        H_b: [N, D_b] receiver states
        heldout_frac: fraction of data to hold out (default 20%)
        center: whether to mean-center before SVD (test both for intermediate layers)

    Returns:
        dict with W_ortho, means, norms, train_cosine, heldout_cosine, center flag
    """
    N = H_a.shape[0]
    n_heldout = max(1, int(N * heldout_frac))
    n_train = N - n_heldout

    H_a_train, H_a_held = H_a[:n_train], H_a[n_train:]
    H_b_train, H_b_held = H_b[:n_train], H_b[n_train:]

    # Compute means from training set
    if center:
        mean_a = mx.mean(H_a_train, axis=0, keepdims=True)
        mean_b = mx.mean(H_b_train, axis=0, keepdims=True)
    else:
        mean_a = mx.zeros((1, H_a.shape[1]))
        mean_b = mx.zeros((1, H_b.shape[1]))

    H_a_c = H_a_train - mean_a
    H_b_c = H_b_train - mean_b
    mx.eval(H_a_c, H_b_c, mean_a, mean_b)

    # Cross-covariance and SVD
    C = H_a_c.T @ H_b_c
    mx.eval(C)
    U, S, Vt = mx.linalg.svd(C, stream=mx.cpu)
    mx.eval(U, S, Vt)

    D_b = H_b.shape[1]
    k = min(H_a.shape[1], D_b)
    W_ortho = U[:, :k] @ Vt[:k, :]
    mx.eval(W_ortho)

    # Train cosine
    H_b_hat_train = H_a_c @ W_ortho
    train_cos = mx.mean(
        mx.sum(H_b_hat_train * H_b_c, axis=1) /
        (mx.linalg.norm(H_b_hat_train, axis=1) * mx.linalg.norm(H_b_c, axis=1) + 1e-8)
    ).item()

    # Held-out cosine
    H_a_held_c = H_a_held - mean_a
    H_b_held_c = H_b_held - mean_b
    H_b_hat_held = H_a_held_c @ W_ortho
    mx.eval(H_b_hat_held)
    heldout_cos = mx.mean(
        mx.sum(H_b_hat_held * H_b_held_c, axis=1) /
        (mx.linalg.norm(H_b_hat_held, axis=1) * mx.linalg.norm(H_b_held_c, axis=1) + 1e-8)
    ).item()

    # Target norms from training set (uncentered)
    target_norm_a = mx.mean(mx.linalg.norm(H_a_train, axis=1))
    target_norm_b = mx.mean(mx.linalg.norm(H_b_train, axis=1))
    mx.eval(target_norm_a, target_norm_b)

    overfit_gap = train_cos - heldout_cos
    if overfit_gap > 0.05:
        print(f"  WARNING: overfitting detected — train={train_cos:.4f}, "
              f"heldout={heldout_cos:.4f}, gap={overfit_gap:.4f}")

    return {
        "W_ortho": W_ortho,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "target_norm_a": target_norm_a,
        "target_norm_b": target_norm_b,
        "train_cosine": train_cos,
        "heldout_cosine": heldout_cos,
        "center": center,
        "n_train": n_train,
        "n_heldout": n_heldout,
    }


def save_alignment(result, path):
    """Save alignment matrix and metadata to disk.

    Handles both old-style (cos_sim, n_pairs) and new-style
    (train_cosine, heldout_cosine, sender_layer, receiver_layer) results.
    """
    import numpy as np
    data = {
        "W_ortho": np.array(result["W_ortho"]),
        "mean_a": np.array(result["mean_a"]),
        "mean_b": np.array(result["mean_b"]),
        "target_norm_a": float(result["target_norm_a"].item()),
        "target_norm_b": float(result["target_norm_b"].item()),
    }
    # Old-style fields
    if "cos_sim" in result:
        data["cos_sim"] = result["cos_sim"]
        data["rel_error"] = result["rel_error"]
        data["n_pairs"] = result["n_pairs"]
    # Heldout metrics (may be added from a separate heldout run)
    for key in ("train_cosine", "heldout_cosine", "n_train", "n_heldout", "center"):
        if key in result:
            data[key] = result[key]
    # Layer info
    if "sender_layer" in result:
        data["sender_layer"] = result["sender_layer"]
        data["receiver_layer"] = result["receiver_layer"]
    np.savez(path, **data)
    print(f"Saved alignment to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_alignment(path):
    """Load saved alignment matrix.

    Returns dict with W_ortho, means, norms, and any available metadata
    (cos_sim, train/heldout cosine, layer info).
    """
    import numpy as np
    data = np.load(path)
    result = {
        "W_ortho": mx.array(data["W_ortho"]),
        "mean_a": mx.array(data["mean_a"]),
        "mean_b": mx.array(data["mean_b"]),
        "target_norm_a": mx.array(data["target_norm_a"]),
        "target_norm_b": mx.array(data["target_norm_b"]),
    }
    # Old-style
    if "cos_sim" in data:
        result["cos_sim"] = float(data["cos_sim"])
        result["rel_error"] = float(data["rel_error"])
        result["n_pairs"] = int(data["n_pairs"])
    # Heldout metrics
    for key in ("train_cosine", "heldout_cosine"):
        if key in data:
            result[key] = float(data[key])
    for key in ("n_train", "n_heldout"):
        if key in data:
            result[key] = int(data[key])
    if "center" in data:
        result["center"] = bool(data["center"])
    # Layer info
    if "sender_layer" in data:
        result["sender_layer"] = int(data["sender_layer"])
        result["receiver_layer"] = int(data["receiver_layer"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Contextual Procrustes Alignment")
    parser.add_argument("--model_a", default="mlx-community/Qwen3-4B-8bit",
                        help="Sender model")
    parser.add_argument("--model_b", default="mlx-community/gemma-2-2b-it-8bit",
                        help="Receiver model")
    parser.add_argument("--n_calibration", type=int, default=500,
                        help="Number of calibration prompts (default 500, -1 for all)")
    parser.add_argument("--sender_layer", type=int, default=None,
                        help="Sender layer to extract at (None = final layer)")
    parser.add_argument("--receiver_layer", type=int, default=None,
                        help="Receiver layer to extract at (None = final layer)")
    parser.add_argument("--multitoken", action="store_true",
                        help="Extract all token positions (not just final token). "
                             "Dramatically increases calibration data for stable Procrustes.")
    parser.add_argument("--n_positions", type=int, default=20,
                        help="Positions to sample per prompt in multitoken mode (default 20)")
    parser.add_argument("--heldout_frac", type=float, default=0.2,
                        help="Fraction of data to hold out (default 0.2)")
    parser.add_argument("--output", default=None,
                        help="Output file (default: procrustes_S{s}_R{r}.npz)")
    args = parser.parse_args()

    # Default output name based on layers
    if args.output is None:
        s_tag = f"S{args.sender_layer}" if args.sender_layer is not None else "Sfinal"
        r_tag = f"R{args.receiver_layer}" if args.receiver_layer is not None else "Rfinal"
        args.output = os.path.join(
            os.path.dirname(__file__), "data", f"procrustes_{s_tag}_{r_tag}.npz"
        )

    total_start = time.time()

    # Load calibration prompts
    print(f"Loading {args.n_calibration} calibration prompts...")
    prompts = load_calibration_prompts(args.n_calibration)
    print(f"Loaded {len(prompts)} prompts")

    # Load models
    print(f"\nLoading sender: {args.model_a}")
    model_a, tok_a = mlx_lm.load(args.model_a)
    print(f"Loading receiver: {args.model_b}")
    model_b, tok_b = mlx_lm.load(args.model_b)

    # Detect Gemma in both models
    is_gemma_a = "gemma" in args.model_a.lower()
    is_gemma_b = "gemma" in args.model_b.lower()

    if args.multitoken and args.sender_layer is not None and args.receiver_layer is not None:
        # Multi-token incremental calibration
        print(f"\n=== Multi-token calibration: S{args.sender_layer}→R{args.receiver_layer} ===")
        print(f"  {len(prompts)} prompts × {args.n_positions} positions")
        print(f"  is_gemma_a={is_gemma_a}, is_gemma_b={is_gemma_b}")
        result = calibrate_multitoken_incremental(
            model_a, tok_a, model_b, tok_b, prompts,
            sender_layer=args.sender_layer,
            receiver_layer=args.receiver_layer,
            is_gemma_a=is_gemma_a,
            is_gemma_b=is_gemma_b,
            heldout_frac=args.heldout_frac,
            n_positions=args.n_positions,
        )
        save_alignment(result, args.output)
        total = time.time() - total_start
        print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
        return

    # Original single-token path
    # Collect contextual pairs at specified layers
    H_a, H_b = collect_contextual_pairs(
        model_a, tok_a, model_b, tok_b, prompts,
        is_gemma_a=is_gemma_a, is_gemma_b=is_gemma_b,
        sender_layer=args.sender_layer, receiver_layer=args.receiver_layer,
    )

    # Compute Procrustes
    use_layers = args.sender_layer is not None or args.receiver_layer is not None
    if use_layers and args.heldout_frac > 0:
        # Report held-out metrics for transparency, then fit on full data
        print(f"\nComputing held-out metrics ({args.heldout_frac*100:.0f}% split)...")
        heldout_result = compute_procrustes_with_heldout(
            H_a, H_b, heldout_frac=args.heldout_frac
        )
        print(f"  Train cosine:    {heldout_result['train_cosine']:.4f}")
        print(f"  Held-out cosine: {heldout_result['heldout_cosine']:.4f}")
        print(f"  Overfit gap:     {heldout_result['train_cosine'] - heldout_result['heldout_cosine']:.4f}")

    # Fit on full data for the saved alignment matrix
    print(f"\nComputing final Procrustes on all {H_a.shape[0]} pairs...")
    result = compute_procrustes(H_a, H_b)
    if use_layers:
        result["sender_layer"] = args.sender_layer
        result["receiver_layer"] = args.receiver_layer
        if args.heldout_frac > 0:
            result["heldout_cosine"] = heldout_result["heldout_cosine"]
            result["train_cosine"] = heldout_result["train_cosine"]

    # Save
    save_alignment(result, args.output)

    total = time.time() - total_start
    print(f"\nTotal time: {total:.1f}s")
    print(f"\nComparison with vocabulary-based alignment:")
    print(f"  Vocabulary LS cosine:           0.6833")
    if "cos_sim" in result:
        print(f"  Contextual Procrustes cosine:   {result['cos_sim']:.4f}")
    if "heldout_cosine" in result:
        print(f"  Contextual Procrustes (held-out): {result['heldout_cosine']:.4f}")


if __name__ == "__main__":
    main()
