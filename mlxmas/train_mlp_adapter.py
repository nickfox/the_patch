"""
Train the MLP adapter on precomputed paired states.

Loads paired (Qwen L13, Gemma L22) states from .npz,
trains a 2-layer MLP to minimize MSE between
MLP(sender_states) and receiver_states.
"""

import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlxmas.mlp_adapter import MLPAdapter, save_mlp_adapter


def train(data_path, lr=1e-3, weight_decay=0.01, epochs=200,
          batch_size=256, heldout_frac=0.2, output=None):
    """Train the MLP adapter on precomputed paired states."""

    # Load paired states
    print(f"Loading paired states from {data_path}...")
    data = np.load(data_path)
    H_sender = data["sender_states"]    # [N, D_sender]
    H_receiver = data["receiver_states"]  # [N, D_receiver]
    sender_layer = int(data["sender_layer"])
    receiver_layer = int(data["receiver_layer"])

    N, D_sender = H_sender.shape
    D_receiver = H_receiver.shape[1]
    print(f"  {N} paired vectors, D_sender={D_sender}, D_receiver={D_receiver}")

    if output is None:
        output = os.path.join(
            os.path.dirname(__file__), "data",
            f"mlp_adapter_S{sender_layer}_R{receiver_layer}.npz"
        )

    # Train/val split
    n_val = max(1, int(N * heldout_frac))
    n_train = N - n_val
    perm = np.random.RandomState(42).permutation(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train_np = H_sender[train_idx]
    Y_train_np = H_receiver[train_idx]
    X_val = mx.array(H_sender[val_idx])
    Y_val = mx.array(H_receiver[val_idx])
    print(f"  Train: {n_train}, Val: {n_val}")

    # Mean-baseline diagnostic: what cosine do you get by always outputting
    # the mean target vector? If the trained adapter doesn't beat this, it
    # just learned the mean (MSE mean-regression).
    mean_target = mx.mean(Y_val, axis=0, keepdims=True)  # [1, D_b]
    mean_expanded = mx.broadcast_to(mean_target, Y_val.shape)
    mean_baseline_cos = mx.mean(
        mx.sum(mean_expanded * Y_val, axis=-1) /
        (mx.linalg.norm(mean_expanded, axis=-1) * mx.linalg.norm(Y_val, axis=-1) + 1e-8)
    ).item()
    print(f"  Mean-baseline val cosine: {mean_baseline_cos:.4f} "
          f"(adapter must beat this to have learned anything beyond the mean)")

    # Create adapter
    adapter = MLPAdapter(input_dim=D_sender, hidden_dim=D_sender,
                         output_dim=D_receiver)
    mx.eval(adapter.parameters())
    print(f"  Adapter params: ~{sum(np.array(p).size for k, p in nn.utils.tree_flatten(adapter.parameters()))/ 1e6:.1f}M")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(model, x_batch, y_batch):
        pred = model(x_batch)
        # Cosine loss: prioritize directional accuracy
        cos_sim = mx.sum(pred * y_batch, axis=-1) / (
            mx.linalg.norm(pred, axis=-1) * mx.linalg.norm(y_batch, axis=-1) + 1e-8
        )
        cos_loss = 1.0 - mx.mean(cos_sim)
        # MSE loss for magnitude matching (normalized by dimension)
        mse_loss = mx.mean(mx.sum((pred - y_batch) ** 2, axis=-1)) / D_receiver
        return cos_loss + 0.1 * mse_loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    # Training loop
    print(f"\n=== Training: {epochs} epochs, batch_size={batch_size}, lr={lr} ===")
    best_val_cos = -1.0
    best_weights = None
    patience = 15
    patience_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        perm_epoch = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm_epoch[start:end]
            x_batch = mx.array(X_train_np[idx])
            y_batch = mx.array(Y_train_np[idx])

            loss, grads = loss_and_grad(adapter, x_batch, y_batch)
            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_pred = adapter(X_val)
        val_mse = mx.mean(mx.sum((val_pred - Y_val) ** 2, axis=-1)).item()

        # Validation cosine (for comparison with Procrustes)
        val_cos = mx.mean(
            mx.sum(val_pred * Y_val, axis=-1) /
            (mx.linalg.norm(val_pred, axis=-1) * mx.linalg.norm(Y_val, axis=-1) + 1e-8)
        ).item()

        elapsed = time.time() - t0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | train_mse={avg_train_loss:.4f} "
                  f"val_mse={val_mse:.4f} val_cos={val_cos:.4f} "
                  f"[{elapsed:.0f}s]")

        # Early stopping on val cosine (higher is better)
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_val_mse = val_mse
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_weights = nn.utils.tree_map(
                lambda x: mx.array(np.array(x)), adapter.parameters()
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best was epoch {best_epoch})")
                break

    # Restore best weights
    if best_weights is not None:
        adapter.update(best_weights)
        mx.eval(adapter.parameters())

    total_time = time.time() - t0
    print(f"\n=== Training complete ===")
    print(f"  Mean-baseline cosine: {mean_baseline_cos:.4f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val cosine: {best_val_cos:.4f} "
          f"(delta over mean-baseline: {best_val_cos - mean_baseline_cos:+.4f})")
    print(f"  Best val MSE:    {best_val_mse:.6f}")
    print(f"  Best train loss: {best_train_loss:.6f}")
    print(f"  Total time: {total_time:.1f}s")

    # Save
    save_mlp_adapter(adapter, output, sender_layer, receiver_layer,
                     best_val_mse, best_val_cos, best_train_loss)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP adapter on paired hidden states"
    )
    parser.add_argument("--paired-data", required=True,
                        help="Path to paired_states .npz from collect_paired_states")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--heldout-frac", type=float, default=0.2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    train(args.paired_data, lr=args.lr, weight_decay=args.weight_decay,
          epochs=args.epochs, batch_size=args.batch_size,
          heldout_frac=args.heldout_frac, output=args.output)


if __name__ == "__main__":
    main()
