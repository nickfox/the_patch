#!/usr/bin/env python3
"""
Cross-model alignment matrix: Qwen3-4B ↔ Gemma-2-2B

Uses shared vocabulary tokens as anchor points to compute a
least-squares projection between the two models' embedding spaces. 

This enables latent communication between different model families
without training — the cross-alignment matrix maps hidden states
from the sender's space into the receiver's embedding manifold.
"""

import mlx.core as mx
import mlx_lm
import time
import numpy as np


def find_shared_tokens(tok_a, tok_b, max_tokens=50000):
    """Find tokens that exist in both tokenizers.
    
    Returns list of (token_str, id_a, id_b) tuples.
    """
    shared = []
    vocab_a = tok_a.get_vocab()
    vocab_b = tok_b.get_vocab()

    # Find tokens present in both vocabularies
    common_keys = set(vocab_a.keys()) & set(vocab_b.keys())
    
    for token_str in sorted(common_keys)[:max_tokens]:
        id_a = vocab_a[token_str]
        id_b = vocab_b[token_str]
        shared.append((token_str, id_a, id_b))
    
    return shared


def compute_cross_alignment(model_a, tok_a, model_b, tok_b):
    """Compute cross-model alignment matrix using shared vocabulary anchors.
    
    Returns W_cross [D_a, D_b] that maps:
        sender (model_a) hidden states → receiver (model_b) embedding space
    
    Also returns W_reverse [D_b, D_a] for the reverse direction.
    """
    print("Finding shared vocabulary tokens...")
    shared = find_shared_tokens(tok_a, tok_b)
    print(f"Found {len(shared)} shared tokens")
    
    if len(shared) < 100:
        raise ValueError(f"Only {len(shared)} shared tokens — need at least 100 for stable alignment")

    # Get actual embedding dimensions by passing a test token
    test = mx.array([[0]])
    D_a = model_a.model.embed_tokens(test).shape[-1]
    D_b = model_b.model.embed_tokens(test).shape[-1]
    print(f"Sender dim: {D_a}, Receiver dim: {D_b}")
    
    # Embed shared tokens through the actual embedding layers
    # (handles dequantization for quantized models)
    # Process in batches to avoid memory issues
    batch_size = 4096
    E_a_parts = []
    E_b_parts = []
    
    for i in range(0, len(shared), batch_size):
        batch = shared[i:i+batch_size]
        ids_a = mx.array([[s[1] for s in batch]])  # [1, batch]
        ids_b = mx.array([[s[2] for s in batch]])
        ea = model_a.model.embed_tokens(ids_a)[0]  # [batch, D_a]
        eb = model_b.model.embed_tokens(ids_b)[0]  # [batch, D_b]
        mx.eval(ea, eb)
        E_a_parts.append(ea)
        E_b_parts.append(eb)
    
    E_a = mx.concatenate(E_a_parts, axis=0).astype(mx.float32)  # [N_shared, D_a]
    E_b = mx.concatenate(E_b_parts, axis=0).astype(mx.float32)  # [N_shared, D_b]
    mx.eval(E_a, E_b)
    
    N = E_a.shape[0]
    print(f"Using {N} anchor pairs for alignment")

    # Store raw means for the centered realignment function
    mean_a = mx.mean(E_a, axis=0, keepdims=True)
    mean_b = mx.mean(E_b, axis=0, keepdims=True)
    mx.eval(mean_a, mean_b)

    # L2-normalize embeddings — projects onto unit hypersphere
    # so alignment captures angular correspondence only
    E_a_n = E_a / (mx.linalg.norm(E_a, axis=1, keepdims=True) + 1e-8)
    E_b_n = E_b / (mx.linalg.norm(E_b, axis=1, keepdims=True) + 1e-8)
    mx.eval(E_a_n, E_b_n)

    # Forward alignment: A→B (normalized)
    print("Computing A→B alignment matrix (L2-normalized)...")
    gram_a = E_a_n.T @ E_a_n + 1e-5 * mx.eye(D_a)
    rhs_ab = E_a_n.T @ E_b_n
    mx.eval(gram_a, rhs_ab)
    W_ab = mx.linalg.solve(gram_a, rhs_ab, stream=mx.cpu)  # [D_a, D_b]
    mx.eval(W_ab)
    
    # Reverse alignment: B→A (normalized)
    print("Computing B→A alignment matrix (L2-normalized)...")
    gram_b = E_b_n.T @ E_b_n + 1e-5 * mx.eye(D_b)
    rhs_ba = E_b_n.T @ E_a_n
    mx.eval(gram_b, rhs_ba)
    W_ba = mx.linalg.solve(gram_b, rhs_ba, stream=mx.cpu)  # [D_b, D_a]
    mx.eval(W_ba)
    
    # Target norms for each receiver
    target_norm_b = mx.mean(mx.linalg.norm(E_b, axis=1))
    target_norm_a = mx.mean(mx.linalg.norm(E_a, axis=1))
    mx.eval(target_norm_b, target_norm_a)

    # Measure reconstruction quality (on normalized data)
    E_b_hat = E_a_n @ W_ab
    mx.eval(E_b_hat)
    recon_error_ab = mx.mean(mx.linalg.norm(E_b_hat - E_b_n, axis=1)).item()
    mean_norm_b = mx.mean(mx.linalg.norm(E_b_n, axis=1)).item()
    relative_error_ab = recon_error_ab / mean_norm_b
    
    E_a_hat = E_b_n @ W_ba
    mx.eval(E_a_hat)
    recon_error_ba = mx.mean(mx.linalg.norm(E_a_hat - E_a_n, axis=1)).item()
    mean_norm_a = mx.mean(mx.linalg.norm(E_a_n, axis=1)).item()
    relative_error_ba = recon_error_ba / mean_norm_a
    
    # Cosine similarity of reconstructed vs true (normalized)
    cos_sim_ab = mx.mean(
        mx.sum(E_b_hat * E_b_n, axis=1) /
        (mx.linalg.norm(E_b_hat, axis=1) * mx.linalg.norm(E_b_n, axis=1) + 1e-8)
    ).item()
    cos_sim_ba = mx.mean(
        mx.sum(E_a_hat * E_a_n, axis=1) /
        (mx.linalg.norm(E_a_hat, axis=1) * mx.linalg.norm(E_a_n, axis=1) + 1e-8)
    ).item()

    print(f"\nAlignment Quality:")
    print(f"  A→B: relative_error={relative_error_ab:.4f}, cosine_sim={cos_sim_ab:.4f}")
    print(f"  B→A: relative_error={relative_error_ba:.4f}, cosine_sim={cos_sim_ba:.4f}")
    print(f"  target_norm_a={target_norm_a.item():.2f}, target_norm_b={target_norm_b.item():.2f}")
    
    return {
        "W_ab": W_ab,               # [D_a, D_b] sender A → receiver B
        "W_ba": W_ba,               # [D_b, D_a] sender B → receiver A
        "mean_a": mean_a,           # [1, D_a] sender embedding centroid
        "mean_b": mean_b,           # [1, D_b] receiver embedding centroid
        "target_norm_a": target_norm_a,
        "target_norm_b": target_norm_b,
        "n_shared": N,
        "cos_sim_ab": cos_sim_ab,
        "cos_sim_ba": cos_sim_ba,
        "relative_error_ab": relative_error_ab,
        "relative_error_ba": relative_error_ba,
    }


def apply_cross_realignment(hidden: mx.array, W_cross: mx.array,
                             mean_sender: mx.array, mean_receiver: mx.array,
                             target_norm: mx.array) -> mx.array:
    """Project hidden state from sender's space into receiver's embedding space.

    W_cross was computed on L2-normalized embeddings. We L2-normalize input
    for directional accuracy, but preserve the original per-vector norms
    scaled to the receiver's operating range. This maintains the dynamic
    range that the receiver's attention mechanism needs.
    """
    h = hidden.astype(mx.float32)
    # Capture original norms (the dynamic range IS information)
    orig_norms = mx.linalg.norm(h, axis=-1, keepdims=True)
    # L2-normalize to match W_cross's training space
    h_norm = h / (orig_norms + 1e-8)
    # Project direction in normalized space
    projected = h_norm @ W_cross
    # Restore per-vector dynamic range, scaled from sender's norm space
    # to receiver's norm space
    mean_sender_norm = mx.mean(orig_norms)
    scale = orig_norms / (mean_sender_norm + 1e-8) * target_norm
    proj_dir = projected / (mx.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8)
    projected = proj_dir * scale
    return projected.astype(hidden.dtype)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-model alignment matrix")
    parser.add_argument("--model_a", default="mlx-community/Qwen3-4B-4bit",
                        help="Sender model")
    parser.add_argument("--model_b", default="mlx-community/gemma-2-2b-it-4bit",
                        help="Receiver model")
    args = parser.parse_args()
    
    print(f"Loading sender: {args.model_a}")
    t0 = time.time()
    model_a, tok_a = mlx_lm.load(args.model_a)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    print(f"Loading receiver: {args.model_b}")
    t0 = time.time()
    model_b, tok_b = mlx_lm.load(args.model_b)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    result = compute_cross_alignment(model_a, tok_a, model_b, tok_b)
    
    print(f"\nCross-alignment matrix shapes:")
    print(f"  W_ab (sender→receiver): {result['W_ab'].shape}")
    print(f"  W_ba (receiver→sender): {result['W_ba'].shape}")
    print(f"\nShared anchors: {result['n_shared']}")
    print(f"Cosine similarity A→B: {result['cos_sim_ab']:.4f}")
    print(f"Cosine similarity B→A: {result['cos_sim_ba']:.4f}")
