#!/usr/bin/env python3
"""
Residual Adapter for Cross-Model Latent Communication

Architecture: Frozen Procrustes base + trainable 2-layer MLP residual
    f(x) = (x - mean_a) @ W_procrustes + mean_b + MLP(x - mean_a)

The MLP learns the nonlinear manifold warping that Procrustes can't capture.
Initialized near zero so training starts at the linear baseline.

Usage:
    # Step 1: Collect paired vectors
    python -m mlxmas.residual_adapter collect --n_prompts 2000

    # Step 2: Train adapter
    python -m mlxmas.residual_adapter train --data mlxmas/paired_vectors_S13_R12.npz

    # Step 3: Evaluate
    python -m mlxmas.residual_adapter eval --adapter mlxmas/adapter_S13_R12.npz
"""

import argparse
import time
import os
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ============================================================
# Module
# ============================================================

class ResidualAdapter(nn.Module):
    """Frozen linear base + trainable nonlinear residual.

    The linear path is the Procrustes rotation (fixed).
    The MLP path learns corrections for nonlinear manifold warping.
    """

    def __init__(self, D_in=2560, D_out=2304, bottleneck=512):
        super().__init__()
        self.down = nn.Linear(D_in, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, D_out, bias=False)
        # Init up weights near zero so adapter starts at Procrustes baseline
        self.up.weight = self.up.weight * 0.01

    def __call__(self, x_centered, W_base):
        """Forward pass.

        Args:
            x_centered: [B, D_in] mean-centered sender states
            W_base: [D_in, D_out] frozen Procrustes matrix (not a parameter)
        """
        linear = x_centered @ W_base
        residual = self.up(nn.silu(self.down(x_centered)))
        return linear + residual


# ============================================================
# Data collection — save raw paired vectors
# ============================================================

def collect_paired_vectors(n_prompts=2000, sender_layer=13, receiver_layer=12,
                           n_positions=20, output=None):
    """Collect and save raw paired vectors for adapter training.

    Extracts all-token hidden states at the target layers from both models,
    resamples to n_positions per prompt, and saves the paired matrices.
    """
    import mlx_lm
    from mlxmas.contextual_procrustes import (
        load_calibration_prompts, extract_all_tokens_at_layer
    )

    if output is None:
        output = os.path.join(
            os.path.dirname(__file__), "data",
            f"paired_vectors_S{sender_layer}_R{receiver_layer}.npz"
        )

    print(f"Collecting paired vectors: S{sender_layer}→R{receiver_layer}")
    print(f"  {n_prompts} prompts × {n_positions} positions")

    prompts = load_calibration_prompts(n_prompts)
    print(f"  Loaded {len(prompts)} prompts")


    print("Loading sender: mlx-community/Qwen3-4B-8bit")
    sender_model, sender_tok = mlx_lm.load("mlx-community/Qwen3-4B-8bit")
    print("Loading receiver: mlx-community/gemma-2-2b-it-8bit")
    receiver_model, receiver_tok = mlx_lm.load("mlx-community/gemma-2-2b-it-8bit")

    def resample(arr, n_pos):
        seq_len = arr.shape[0]
        if seq_len <= n_pos:
            return arr
        indices = np.linspace(0, seq_len - 1, n_pos).astype(int)
        return arr[indices]

    all_a, all_b = [], []
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        ha = extract_all_tokens_at_layer(
            sender_model, sender_tok, prompt, sender_layer, is_gemma=False
        )
        hb = extract_all_tokens_at_layer(
            receiver_model, receiver_tok, prompt, receiver_layer, is_gemma=True
        )
        ha = resample(ha, n_positions)
        hb = resample(hb, n_positions)
        n = min(ha.shape[0], hb.shape[0])
        all_a.append(ha[:n].astype(np.float32))
        all_b.append(hb[:n].astype(np.float32))


        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            total_vecs = sum(a.shape[0] for a in all_a)
            print(f"    {i+1}/{len(prompts)} ({rate:.1f}/s, ETA {eta:.0f}s) "
                  f"vectors={total_vecs}")

    H_a = np.concatenate(all_a, axis=0)
    H_b = np.concatenate(all_b, axis=0)

    elapsed = time.time() - t0
    print(f"\n  Collected {H_a.shape[0]} paired vectors in {elapsed:.1f}s")
    print(f"  H_a: {H_a.shape} ({H_a.nbytes / 1e6:.1f} MB)")
    print(f"  H_b: {H_b.shape} ({H_b.nbytes / 1e6:.1f} MB)")

    np.savez(output, H_a=H_a, H_b=H_b,
             sender_layer=sender_layer, receiver_layer=receiver_layer,
             n_prompts=len(prompts), n_positions=n_positions)
    total_mb = os.path.getsize(output) / 1e6
    print(f"  Saved to {output} ({total_mb:.1f} MB)")
    return output


# ============================================================
# Training
# ============================================================

def train_adapter(data_path, procrustes_path=None, bottleneck=512,
                  lr=1e-3, weight_decay=0.01, epochs=100, batch_size=512,
                  heldout_frac=0.2, output=None):
    """Train the residual adapter on precomputed paired vectors.

    Args:
        data_path: path to paired_vectors npz
        procrustes_path: path to Procrustes npz (for frozen W_base + means)
        bottleneck: MLP hidden dimension
        lr: learning rate
        weight_decay: AdamW weight decay
        epochs: training epochs
        batch_size: mini-batch size
        heldout_frac: validation split fraction
        output: path to save trained adapter
    """
    # Load paired vectors
    print(f"Loading paired vectors from {data_path}...")
    data = np.load(data_path)
    H_a = data["H_a"]  # [N, D_a]
    H_b = data["H_b"]  # [N, D_b]
    sender_layer = int(data["sender_layer"])
    receiver_layer = int(data["receiver_layer"])
    N, D_a = H_a.shape
    D_b = H_b.shape[1]
    print(f"  {N} vectors, D_a={D_a}, D_b={D_b}")


    # Load Procrustes for frozen base + means
    if procrustes_path is None:
        procrustes_path = os.path.join(
            os.path.dirname(__file__), "data",
            f"procrustes_S{sender_layer}_R{receiver_layer}_multitoken.npz"
        )
    print(f"Loading Procrustes base from {procrustes_path}...")
    proc = np.load(procrustes_path)
    W_base = mx.array(proc["W_ortho"])  # [D_a, D_b]
    mean_a = proc["mean_a"].reshape(1, -1).astype(np.float32)
    mean_b = proc["mean_b"].reshape(1, -1).astype(np.float32)
    target_norm_b = float(proc["target_norm_b"])
    print(f"  W_base: {W_base.shape}, target_norm_b: {target_norm_b:.2f}")

    # Center the data using Procrustes means
    H_a_c = H_a - mean_a  # [N, D_a]
    H_b_c = H_b - mean_b  # [N, D_b]

    # Train/val split — keep as numpy, convert to MLX at batch time
    n_val = max(1, int(N * heldout_frac))
    n_train = N - n_val
    perm = np.random.RandomState(42).permutation(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    Ha_train = H_a_c[train_idx]  # numpy [n_train, D_a]
    Hb_train = H_b_c[train_idx]  # numpy [n_train, D_b]
    X_val = mx.array(H_a_c[val_idx])
    Y_val = mx.array(H_b_c[val_idx])
    print(f"  Train: {n_train}, Val: {n_val}")


    # Procrustes-only baseline cosine (before training)
    baseline_pred = X_val @ W_base
    baseline_cos = mx.mean(
        mx.sum(baseline_pred * Y_val, axis=1) /
        (mx.linalg.norm(baseline_pred, axis=1) * mx.linalg.norm(Y_val, axis=1) + 1e-8)
    ).item()
    print(f"\n  Procrustes-only baseline val cosine: {baseline_cos:.4f}")

    # Create adapter
    adapter = ResidualAdapter(D_in=D_a, D_out=D_b, bottleneck=bottleneck)
    mx.eval(adapter.parameters())

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(model, x_batch, y_batch):
        pred = model(x_batch, W_base)
        # Cosine loss: 1 - mean(cosine_similarity)
        cos_sim = mx.sum(pred * y_batch, axis=1) / (
            mx.linalg.norm(pred, axis=1) * mx.linalg.norm(y_batch, axis=1) + 1e-8
        )
        cos_loss = 1.0 - mx.mean(cos_sim)
        # MSE loss for magnitude matching
        mse_loss = mx.mean(mx.sum((pred - y_batch) ** 2, axis=1))
        # Normalize MSE by dimension to keep it on similar scale as cosine
        mse_loss = mse_loss / D_b
        return cos_loss + 0.1 * mse_loss, cos_sim

    loss_and_grad = nn.value_and_grad(adapter, lambda m, x, y: loss_fn(m, x, y)[0])


    # Training loop
    print(f"\n=== Training: {epochs} epochs, batch_size={batch_size}, lr={lr} ===")
    best_val_cos = baseline_cos
    best_down_w = None
    best_up_w = None
    patience = 15
    patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        # Shuffle training data
        perm_epoch = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm_epoch[start:end]
            x_batch = mx.array(Ha_train[idx])
            y_batch = mx.array(Hb_train[idx])

            loss, grads = loss_and_grad(adapter, x_batch, y_batch)
            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1


        # Validation cosine
        val_pred = adapter(X_val, W_base)
        val_cos = mx.mean(
            mx.sum(val_pred * Y_val, axis=1) /
            (mx.linalg.norm(val_pred, axis=1) * mx.linalg.norm(Y_val, axis=1) + 1e-8)
        ).item()
        mx.eval(val_pred)

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            improvement = val_cos - baseline_cos
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}  "
                  f"val_cos={val_cos:.4f} ({improvement:+.4f} vs baseline)  "
                  f"[{elapsed:.0f}s]")

        # Early stopping
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_down_w = mx.array(adapter.down.weight)
            best_up_w = mx.array(adapter.up.weight)
            mx.eval(best_down_w, best_up_w)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break


    # Restore best weights
    if best_down_w is not None:
        adapter.down.weight = best_down_w
        adapter.up.weight = best_up_w
        mx.eval(adapter.parameters())

    # Final metrics
    final_pred = adapter(X_val, W_base)
    final_cos = mx.mean(
        mx.sum(final_pred * Y_val, axis=1) /
        (mx.linalg.norm(final_pred, axis=1) * mx.linalg.norm(Y_val, axis=1) + 1e-8)
    ).item()

    # Also compute train cosine (sample 5000)
    n_sample = min(5000, n_train)
    X_train_sample = mx.array(Ha_train[:n_sample])
    Y_train_sample = mx.array(Hb_train[:n_sample])
    train_pred = adapter(X_train_sample, W_base)
    train_cos = mx.mean(
        mx.sum(train_pred * Y_train_sample, axis=1) /
        (mx.linalg.norm(train_pred, axis=1) * mx.linalg.norm(Y_train_sample, axis=1) + 1e-8)
    ).item()

    total_time = time.time() - t0
    print(f"\n=== Results ===")
    print(f"  Procrustes baseline val cosine: {baseline_cos:.4f}")
    print(f"  Adapter train cosine:           {train_cos:.4f}")
    print(f"  Adapter val cosine:             {final_cos:.4f}")
    print(f"  Improvement over Procrustes:    {final_cos - baseline_cos:+.4f}")
    print(f"  Overfit gap:                    {train_cos - final_cos:.4f}")
    print(f"  Training time:                  {total_time:.1f}s")


    # Save adapter
    if output is None:
        output = os.path.join(
            os.path.dirname(__file__), "data",
            f"adapter_S{sender_layer}_R{receiver_layer}.npz"
        )

    # Extract weights for saving
    down_w = np.array(adapter.down.weight)
    up_w = np.array(adapter.up.weight)

    np.savez(output,
             down_weight=down_w,
             up_weight=up_w,
             W_base=np.array(W_base),
             mean_a=mean_a,
             mean_b=mean_b,
             target_norm_b=target_norm_b,
             sender_layer=sender_layer,
             receiver_layer=receiver_layer,
             baseline_cosine=baseline_cos,
             adapter_val_cosine=final_cos,
             adapter_train_cosine=train_cos,
             bottleneck=bottleneck)

    print(f"  Saved to {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
    return output


# ============================================================
# Load and apply
# ============================================================

def load_adapter(path):
    """Load a trained adapter from disk.

    Returns: (adapter, W_base, mean_a, mean_b, target_norm_b, metadata)
    """
    data = np.load(path)
    D_a = data["W_base"].shape[0]
    D_b = data["W_base"].shape[1]
    bottleneck = int(data["bottleneck"])

    adapter = ResidualAdapter(D_in=D_a, D_out=D_b, bottleneck=bottleneck)
    adapter.down.weight = mx.array(data["down_weight"])
    adapter.up.weight = mx.array(data["up_weight"])
    mx.eval(adapter.parameters())

    W_base = mx.array(data["W_base"])
    mean_a = mx.array(data["mean_a"].reshape(1, -1))
    mean_b = mx.array(data["mean_b"].reshape(1, -1))
    target_norm_b = float(data["target_norm_b"])

    metadata = {
        "sender_layer": int(data["sender_layer"]),
        "receiver_layer": int(data["receiver_layer"]),
        "baseline_cosine": float(data["baseline_cosine"]),
        "adapter_val_cosine": float(data["adapter_val_cosine"]),
    }
    return adapter, W_base, mean_a, mean_b, target_norm_b, metadata


def apply_adapter(hidden, adapter, W_base, mean_a, mean_b, target_norm_b):
    """Project a sender hidden state through the adapter.

    Args:
        hidden: [1, seq, D_a] or [B, D_a] sender hidden states
        adapter: trained ResidualAdapter
        W_base: [D_a, D_b] frozen Procrustes
        mean_a, mean_b: [1, D_a] and [1, D_b] centroids
        target_norm_b: scalar target norm

    Returns: projected states in receiver space, norm-matched
    """
    h = hidden.astype(mx.float32)
    h_c = h - mean_a
    projected = adapter(h_c, W_base) + mean_b
    # Norm-match
    norm = mx.linalg.norm(projected, axis=-1, keepdims=True)
    norm = mx.maximum(norm, mx.array(1e-6))
    projected = projected * (target_norm_b / norm)
    return projected.astype(hidden.dtype)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Residual adapter for cross-model latent communication")
    sub = parser.add_subparsers(dest="command", required=True)

    # Collect
    p_collect = sub.add_parser("collect", help="Collect paired vectors")
    p_collect.add_argument("--n_prompts", type=int, default=2000)
    p_collect.add_argument("--sender_layer", type=int, default=13)
    p_collect.add_argument("--receiver_layer", type=int, default=12)
    p_collect.add_argument("--n_positions", type=int, default=20)
    p_collect.add_argument("--output", default=None)

    # Train
    p_train = sub.add_parser("train", help="Train adapter")
    p_train.add_argument("--data", required=True, help="Path to paired vectors npz")
    p_train.add_argument("--procrustes", default=None, help="Path to Procrustes npz")
    p_train.add_argument("--bottleneck", type=int, default=512)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--output", default=None)

    args = parser.parse_args()


    if args.command == "collect":
        collect_paired_vectors(
            n_prompts=args.n_prompts,
            sender_layer=args.sender_layer,
            receiver_layer=args.receiver_layer,
            n_positions=args.n_positions,
            output=args.output,
        )

    elif args.command == "train":
        train_adapter(
            data_path=args.data,
            procrustes_path=args.procrustes,
            bottleneck=args.bottleneck,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output=args.output,
        )


if __name__ == "__main__":
    main()
