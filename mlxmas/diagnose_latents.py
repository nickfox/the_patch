#!/usr/bin/env python3
"""
Latent channel signal analysis.

Captures the projected vectors being injected into Gemma and
analyzes whether they carry any semantic signal.

Tests:
1. Logit decode — what tokens does Gemma think the vectors represent?
2. Distribution comparison — projected vs real Gemma layer activations
3. Nearest embedding — closest vocabulary token per position
4. Norm analysis — are the vectors in the right magnitude range?
"""

import mlx.core as mx
import mlx_lm
import numpy as np
import time
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache


QUESTION = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh "
    "duck egg. How much in dollars does she make every day at the farmers' market?"
)


def get_embedding_matrix(model, chunk_size=4096):
    """Get dequantized embedding matrix [V, D]."""
    V = model.args.vocab_size
    chunks = []
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        ids = mx.array([list(range(start, end))])
        emb = model.model.embed_tokens(ids)[0]
        chunks.append(emb)
        mx.eval(emb)
    E = mx.concatenate(chunks, axis=0).astype(mx.float32)
    mx.eval(E)
    return E


def capture_real_gemma_activations(model, tokenizer, text, layer):
    """Run text through Gemma, capture raw activations at a specific layer.

    Returns [1, seq_len, D] raw residual stream (no final norm).
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = mx.array([tokens])
    cache = make_prompt_cache(model)

    h = model.model.embed_tokens(input_ids)
    h = h * (model.model.args.hidden_size ** 0.5)
    mask = create_attention_mask(h, cache[0], return_array=True)

    for i, (lay, c) in enumerate(zip(model.model.layers, cache)):
        h = lay(h, mask, c)
        if i == layer:
            captured = h.astype(mx.float32)
            mx.eval(captured)
            # finish remaining layers for cleanup
            for j in range(i + 1, len(model.model.layers)):
                h = model.model.layers[j](h, mask, cache[j])
            mx.eval(h)
            return captured
    raise ValueError(f"Layer {layer} not reached")


def logit_decode(model, tokenizer, vectors):
    """Feed vectors through Gemma's norm + lm_head, decode top tokens.

    Args:
        vectors: [1, seq, D] — the projected vectors
    Returns:
        list of (position, top5_tokens_with_probs)
    """
    h = model.model.norm(vectors)
    out = model.model.embed_tokens.as_linear(h)
    if hasattr(model, 'final_logit_softcapping'):
        out = mx.tanh(out / model.final_logit_softcapping)
        out = out * model.final_logit_softcapping
    mx.eval(out)

    results = []
    logits_np = np.array(out[0].astype(mx.float32))  # [seq, V]

    for pos in range(logits_np.shape[0]):
        probs = np.exp(logits_np[pos] - logits_np[pos].max())
        probs = probs / probs.sum()
        top_idx = np.argsort(probs)[-5:][::-1]
        top_tokens = []
        for idx in top_idx:
            tok_str = tokenizer.decode([int(idx)])
            top_tokens.append((tok_str, float(probs[idx])))
        results.append((pos, top_tokens))
    return results


def nearest_embeddings(E, tokenizer, vectors, top_k=3):
    """Find closest vocabulary embeddings to each projected vector.

    Args:
        E: [V, D] dequantized embedding matrix
        vectors: [1, seq, D] projected vectors
    Returns:
        list of (position, [(token, cosine_sim), ...])
    """
    vecs = vectors[0].astype(mx.float32)  # [seq, D]
    v_norm = vecs / (mx.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-8)
    e_norm = E / (mx.linalg.norm(E, axis=-1, keepdims=True) + 1e-8)
    mx.eval(v_norm, e_norm)

    results = []
    for pos in range(vecs.shape[0]):
        cos = v_norm[pos] @ e_norm.T  # [V]
        mx.eval(cos)
        cos_np = np.array(cos.astype(mx.float32))
        top_idx = np.argsort(cos_np)[-top_k:][::-1]
        neighbors = []
        for idx in top_idx:
            tok_str = tokenizer.decode([int(idx)])
            neighbors.append((tok_str, float(cos_np[idx])))
        results.append((pos, neighbors))
    return results


def distribution_comparison(projected, real):
    """Compare statistical properties of projected vs real activations.

    Args:
        projected: [1, seq_p, D] — Procrustes-projected vectors
        real: [1, seq_r, D] — real Gemma layer activations
    """
    p = np.array(projected[0].astype(mx.float32))
    r = np.array(real[0].astype(mx.float32))

    p_norms = np.linalg.norm(p, axis=1)
    r_norms = np.linalg.norm(r, axis=1)

    p_var = np.var(p, axis=0)  # per-dimension variance
    r_var = np.var(r, axis=0)

    # dimension-wise correlation of variance profiles
    var_corr = np.corrcoef(p_var, r_var)[0, 1]

    # cosine between mean vectors
    p_mean = p.mean(axis=0)
    r_mean = r.mean(axis=0)
    mean_cos = float(np.dot(p_mean, r_mean) /
                      (np.linalg.norm(p_mean) * np.linalg.norm(r_mean) + 1e-8))

    # top outlier dimensions
    p_max_dims = np.argsort(p_var)[-5:][::-1]
    r_max_dims = np.argsort(r_var)[-5:][::-1]

    print(f"\n=== Distribution Comparison ===")
    print(f"  Projected: {p.shape[0]} vectors, Real: {r.shape[0]} vectors")
    print(f"  Norm range — projected: [{p_norms.min():.1f}, {p_norms.max():.1f}], "
          f"mean={p_norms.mean():.1f}")
    print(f"  Norm range — real:      [{r_norms.min():.1f}, {r_norms.max():.1f}], "
          f"mean={r_norms.mean():.1f}")
    print(f"  Variance profile correlation: {var_corr:.4f}")
    print(f"    (1.0 = same dimensions are active, 0.0 = unrelated)")
    print(f"  Mean vector cosine: {mean_cos:.4f}")
    print(f"  Top-5 high-variance dims — projected: {p_max_dims.tolist()}")
    print(f"  Top-5 high-variance dims — real:      {r_max_dims.tolist()}")
    overlap = len(set(p_max_dims) & set(r_max_dims))
    print(f"  Overlap in top-5 variance dims: {overlap}/5")

    return {
        "p_norm_mean": float(p_norms.mean()),
        "r_norm_mean": float(r_norms.mean()),
        "var_corr": float(var_corr),
        "mean_cos": mean_cos,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose latent channel signal")
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--receiver", default="mlx-community/gemma-2-9b-it-8bit")
    parser.add_argument("--adapter-type", choices=["procrustes", "mlp", "cca"],
                        default="procrustes", help="Alignment method")
    parser.add_argument("--adapter-path", required=True,
                        help="Path to alignment/adapter .npz")
    parser.add_argument("--sender-layer", type=int, default=13)
    parser.add_argument("--receiver-layer", type=int, default=22)
    args = parser.parse_args()

    # Load models
    print(f"Loading sender: {args.sender}", flush=True)
    sender_model, sender_tok = mlx_lm.load(args.sender)

    print(f"Loading receiver: {args.receiver}", flush=True)
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)

    # Load alignment / adapter
    print(f"Loading {args.adapter_type} from: {args.adapter_path}", flush=True)
    if args.adapter_type == "mlp":
        from mlxmas.mlp_adapter import load_mlp_adapter, apply_mlp_adapter
        mlp_adapter, mlp_meta = load_mlp_adapter(args.adapter_path)
        print(f"  Val cosine: {mlp_meta['val_cosine']:.4f}")
        print(f"  Val MSE: {mlp_meta['val_mse']:.6f}")
    elif args.adapter_type == "cca":
        from mlxmas.cca_adapter import load_cca, apply_cca
        cca_Wa, cca_Wb, cca_mu_x, cca_mu_y, cca_target_norm, cca_meta = load_cca(args.adapter_path)
        print(f"  K={cca_meta['K']}, top correlation: {cca_meta['top_correlation']:.4f}, "
              f"mean: {cca_meta['mean_correlation']:.4f}")
    else:
        from mlxmas.contextual_procrustes import load_alignment
        from mlxmas.cross_align import apply_cross_realignment
        alignment = load_alignment(args.adapter_path)
        W_ab = alignment["W_ortho"]
        mean_a = alignment["mean_a"]
        mean_b = alignment["mean_b"]
        target_norm_b = alignment["target_norm_b"]
        print(f"  W shape: {W_ab.shape}")
        if "heldout_cosine" in alignment:
            print(f"  Held-out cosine: {alignment['heldout_cosine']:.4f}")

    # === Build projected vectors (same pipeline as test_cross.py) ===
    print(f"\n=== Sender Processing ===", flush=True)
    print(f"Question: {QUESTION[:80]}...")

    tokens = sender_tok.encode(QUESTION, add_special_tokens=True)
    input_ids = mx.array([tokens])
    sender_cache = make_prompt_cache(sender_model)
    h = sender_model.model.embed_tokens(input_ids)
    mask = create_attention_mask(h, sender_cache[0], return_array=True)
    for i, (layer, c) in enumerate(zip(sender_model.model.layers, sender_cache)):
        h = layer(h, mask, c)
        if i == args.sender_layer:
            sender_hidden = h.astype(mx.float32)
            break
    mx.eval(sender_hidden)
    print(f"  Sender hidden: {sender_hidden.shape}")

    if args.adapter_type == "mlp":
        projected = apply_mlp_adapter(sender_hidden, mlp_adapter)
    elif args.adapter_type == "cca":
        projected = apply_cca(sender_hidden, cca_Wa, cca_Wb, cca_mu_x, cca_mu_y, cca_target_norm)
    else:
        projected = apply_cross_realignment(
            sender_hidden, W_ab, mean_a, mean_b, target_norm_b
        )
    mx.eval(projected)
    print(f"  Projected: {projected.shape}")

    # === Test 1: Logit Decode ===
    print(f"\n=== Test 1: Logit Decode ===")
    print(f"  What tokens does Gemma think these vectors represent?")
    print(f"  (If signal: question-related words. If noise: random tokens.)\n")

    decoded = logit_decode(receiver_model, receiver_tok, projected)
    for pos, top5 in decoded:
        tok_strs = [f"'{t}' ({p:.3f})" for t, p in top5]
        print(f"  pos {pos:2d}: {', '.join(tok_strs)}")

    # === Test 2: Distribution Comparison ===
    print(f"\n=== Test 2: Distribution Comparison ===")
    print(f"  Comparing projected vectors vs real Gemma layer-{args.receiver_layer} "
          f"activations on the same question.\n")

    real_activations = capture_real_gemma_activations(
        receiver_model, receiver_tok, QUESTION, args.receiver_layer
    )
    print(f"  Real activations: {real_activations.shape}")
    dist_stats = distribution_comparison(projected, real_activations)

    # === Test 3: Nearest Embeddings ===
    print(f"\n=== Test 3: Nearest Vocabulary Embeddings ===")
    print(f"  Loading dequantized embedding matrix...", flush=True)
    E_recv = get_embedding_matrix(receiver_model)
    print(f"  E shape: {E_recv.shape}")
    print(f"  Closest vocab token per projected position:\n")

    neighbors = nearest_embeddings(E_recv, receiver_tok, projected, top_k=3)
    for pos, top3 in neighbors:
        tok_strs = [f"'{t}' ({s:.3f})" for t, s in top3]
        print(f"  pos {pos:2d}: {', '.join(tok_strs)}")

    # === Test 4: Logit Decode on REAL activations (control) ===
    print(f"\n=== Test 4: Control — Logit Decode on REAL Gemma L{args.receiver_layer} ===")
    print(f"  Same question, but using Gemma's own activations.\n")

    decoded_real = logit_decode(receiver_model, receiver_tok, real_activations)
    tokens_in = receiver_tok.encode(QUESTION, add_special_tokens=True)
    token_strs = [receiver_tok.decode([t]) for t in tokens_in]

    for pos, top5 in decoded_real:
        input_tok = token_strs[pos] if pos < len(token_strs) else "?"
        tok_strs_d = [f"'{t}' ({p:.3f})" for t, p in top5]
        print(f"  pos {pos:2d} [input='{input_tok}']: {', '.join(tok_strs_d)}")

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Projected vectors: {projected.shape}")
    print(f"  Real activations:  {real_activations.shape}")
    print(f"  Norm ratio (projected/real): "
          f"{dist_stats['p_norm_mean']/dist_stats['r_norm_mean']:.2f}")
    print(f"  Variance profile correlation: {dist_stats['var_corr']:.4f}")
    print(f"  Mean vector cosine: {dist_stats['mean_cos']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
