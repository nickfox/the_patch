"""
Canonical Correlation Analysis (CCA) adapter for cross-model latent communication.

Instead of learning a full nonlinear mapping (MLP) or a global rotation (Procrustes),
CCA finds the SUBSPACE where the two models' representations are maximally correlated.
Only the shared dimensions are kept; model-specific noise is discarded.

The canonical correlations are diagnostic: they tell us the dimensionality and
strength of shared structure between Qwen L13 and Gemma L22. If the top correlations
are low, there's fundamentally little shared structure at these layers.

Fitting is closed-form (eigendecomposition + SVD). No training.
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np


def _matrix_inv_sqrt(C):
    """Compute C^{-1/2} via eigendecomposition.

    C must be symmetric positive semi-definite (regularized covariance).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Clip small/negative eigenvalues for numerical stability
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    inv_sqrt = 1.0 / np.sqrt(eigenvalues)
    return (eigenvectors * inv_sqrt[np.newaxis, :]) @ eigenvectors.T


def fit_cca(X, Y, K=512, reg=1e-4):
    """Compute CCA projections from paired states.

    Args:
        X: [N, D_a] sender states (numpy float32/float64)
        Y: [N, D_b] receiver states (numpy float32/float64)
        K: number of canonical dimensions to keep
        reg: regularization for covariance matrices

    Returns:
        W_a: [D_a, K] sender projection
        W_b: [D_b, K] receiver projection
        correlations: [K] canonical correlations (sorted descending)
        mu_x: [D_a] sender mean
        mu_y: [D_b] receiver mean
        target_norm: float — mean norm of receiver states
    """
    N, D_a = X.shape
    D_b = Y.shape[1]
    K = min(K, D_a, D_b)

    # Promote to float64 for numerical stability
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Center
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    print(f"  Computing covariance matrices...")
    t0 = time.time()

    # Within-covariance (regularized)
    C_xx = (Xc.T @ Xc) / N + reg * np.eye(D_a)
    C_yy = (Yc.T @ Yc) / N + reg * np.eye(D_b)

    # Cross-covariance
    C_xy = (Xc.T @ Yc) / N

    print(f"    Covariances: {time.time() - t0:.1f}s")
    print(f"  Computing inverse square roots...")
    t0 = time.time()

    # Whitened cross-covariance
    C_xx_inv_sqrt = _matrix_inv_sqrt(C_xx)
    C_yy_inv_sqrt = _matrix_inv_sqrt(C_yy)

    print(f"    Inv sqrt: {time.time() - t0:.1f}s")
    print(f"  Computing SVD of whitened cross-covariance [{D_a}, {D_b}]...")
    t0 = time.time()

    T = C_xx_inv_sqrt @ C_xy @ C_yy_inv_sqrt

    # SVD — T is [D_a, D_b], use full_matrices=False for efficiency
    U, S, Vt = np.linalg.svd(T, full_matrices=False)

    print(f"    SVD: {time.time() - t0:.1f}s")

    # Take top K canonical directions
    W_a = (C_xx_inv_sqrt @ U[:, :K]).astype(np.float32)    # [D_a, K]
    W_b = (C_yy_inv_sqrt @ Vt[:K, :].T).astype(np.float32)  # [D_b, K]
    correlations = S[:K].astype(np.float32)

    # Target norm: mean norm of receiver states
    target_norm = float(np.mean(np.linalg.norm(Y, axis=1)))

    # Scale W_b so that (Xc @ W_a) @ W_b.T has the same magnitude as Yc.
    # Without this, the CCA reconstruction is ~100x smaller than mu_y and
    # every projected vector collapses to the mean.
    recon = (Xc @ W_a.astype(np.float64)) @ W_b.astype(np.float64).T
    scale = float(np.std(Yc) / np.std(recon))
    W_b = (W_b * scale).astype(np.float32)
    print(f"  W_b scale factor: {scale:.2f} "
          f"(real_std={np.std(Yc):.2f}, recon_std={np.std(recon):.2f})")

    return W_a, W_b, correlations, mu_x.astype(np.float32), mu_y.astype(np.float32), target_norm


def save_cca(path, W_a, W_b, correlations, mu_x, mu_y, target_norm,
             sender_layer, receiver_layer):
    """Save CCA projections to .npz."""
    K = W_a.shape[1]
    np.savez(path,
             W_a=W_a, W_b=W_b,
             correlations=correlations,
             mu_x=mu_x, mu_y=mu_y,
             target_norm=target_norm,
             sender_layer=sender_layer,
             receiver_layer=receiver_layer,
             K=K)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved CCA adapter to {path} ({size_mb:.1f} MB)")


def load_cca(path):
    """Load CCA projections from .npz.

    Returns: (W_a, W_b, mu_x, mu_y, target_norm, metadata)
        All projection matrices returned as MLX arrays for runtime use.
    """
    data = np.load(path)
    W_a = mx.array(data["W_a"])         # [D_a, K]
    W_b = mx.array(data["W_b"])         # [D_b, K]
    mu_x = mx.array(data["mu_x"])       # [D_a]
    mu_y = mx.array(data["mu_y"])       # [D_b]
    target_norm = float(data["target_norm"])
    correlations = data["correlations"]

    metadata = {
        "sender_layer": int(data["sender_layer"]),
        "receiver_layer": int(data["receiver_layer"]),
        "K": int(data["K"]),
        "top_correlation": float(correlations[0]),
        "mean_correlation": float(correlations.mean()),
        "correlations": correlations,
    }
    return W_a, W_b, mu_x, mu_y, target_norm, metadata


def apply_cca(hidden, W_a, W_b, mu_x, mu_y, target_norm):
    """Project sender hidden states through CCA into receiver space.

    Args:
        hidden: [B, seq, D_a] or [B, D_a] sender hidden states (MLX)
        W_a: [D_a, K] sender projection (MLX)
        W_b: [D_b, K] receiver projection (MLX)
        mu_x: [D_a] sender mean (MLX)
        mu_y: [D_b] receiver mean (MLX)
        target_norm: float — mean norm of receiver calibration states

    Returns: projected states in receiver space
    """
    h = hidden.astype(mx.float32)
    # Project: sender space -> shared space -> receiver space
    shared = (h - mu_x) @ W_a          # [..., K]
    projected = shared @ W_b.T + mu_y  # [..., D_b]

    return projected


def main():
    parser = argparse.ArgumentParser(
        description="Fit CCA on paired hidden states"
    )
    parser.add_argument("--paired-data", required=True,
                        help="Path to paired_states .npz from collect_paired_states")
    parser.add_argument("--K", type=int, default=512,
                        help="Number of canonical dimensions (default 512)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep K values: 64, 128, 256, 512, 1024")
    parser.add_argument("--reg", type=float, default=1e-4,
                        help="Regularization for covariance matrices")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load paired states
    print(f"Loading paired states from {args.paired_data}...")
    data = np.load(args.paired_data)
    H_sender = data["sender_states"]
    H_receiver = data["receiver_states"]
    sender_layer = int(data["sender_layer"])
    receiver_layer = int(data["receiver_layer"])
    N, D_a = H_sender.shape
    D_b = H_receiver.shape[1]
    print(f"  {N} paired vectors, D_sender={D_a}, D_receiver={D_b}")

    if args.sweep:
        K_values = [64, 128, 256, 512, 1024]
    else:
        K_values = [args.K]

    for K in K_values:
        print(f"\n{'='*60}")
        print(f"Fitting CCA with K={K}")
        print(f"{'='*60}")

        t0 = time.time()
        W_a, W_b, correlations, mu_x, mu_y, target_norm = fit_cca(
            H_sender, H_receiver, K=K, reg=args.reg
        )
        elapsed = time.time() - t0

        # Print canonical correlations
        print(f"\n  Canonical correlations (K={K}):")
        print(f"    Top 10:  {correlations[:10]}")
        print(f"    Top 1:   {correlations[0]:.4f}")
        print(f"    Top 10 mean: {correlations[:10].mean():.4f}")
        print(f"    Top 50 mean: {correlations[:50].mean():.4f}")
        print(f"    Overall mean: {correlations.mean():.4f}")
        print(f"    Min:     {correlations[-1]:.4f}")
        print(f"    Dims with corr > 0.9: {(correlations > 0.9).sum()}")
        print(f"    Dims with corr > 0.8: {(correlations > 0.8).sum()}")
        print(f"    Dims with corr > 0.5: {(correlations > 0.5).sum()}")
        print(f"    Target norm: {target_norm:.2f}")
        print(f"    Fit time: {elapsed:.1f}s")

        # Save
        if args.output:
            out_path = args.output
        else:
            out_path = os.path.join(
                os.path.dirname(__file__), "data",
                f"cca_adapter_S{sender_layer}_R{receiver_layer}_K{K}.npz"
            )
        save_cca(out_path, W_a, W_b, correlations, mu_x, mu_y,
                 target_norm, sender_layer, receiver_layer)

    # If sweep, print comparison table
    if args.sweep:
        print(f"\n{'='*60}")
        print(f"Sweep Summary")
        print(f"{'='*60}")
        print(f"  {'K':>6}  {'Top-1':>8}  {'Top-10':>8}  {'Mean':>8}  {'Corr>0.5':>10}")
        for K in K_values:
            out_path = os.path.join(
                os.path.dirname(__file__), "data",
                f"cca_adapter_S{sender_layer}_R{receiver_layer}_K{K}.npz"
            )
            d = np.load(out_path)
            c = d["correlations"]
            print(f"  {K:>6}  {c[0]:>8.4f}  {c[:10].mean():>8.4f}  "
                  f"{c.mean():>8.4f}  {(c > 0.5).sum():>10}")


if __name__ == "__main__":
    main()
