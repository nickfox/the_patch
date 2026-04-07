"""
MLP Adapter for cross-model latent communication.

Replaces Procrustes alignment entirely. A 2-layer MLP learns the nonlinear
mapping from Qwen layer-13 hidden states to Gemma layer-22 hidden states.

No centering, no norm matching — the MLP learns correct magnitudes and
directional mapping through MSE training on paired states.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MLPAdapter(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=2560, output_dim=3584):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.linear2(nn.silu(self.linear1(x)))


def save_mlp_adapter(adapter, path, sender_layer, receiver_layer,
                     val_mse, val_cosine, train_mse):
    """Save trained MLP adapter weights and metadata to .npz."""
    l1_w = np.array(adapter.linear1.weight)
    l1_b = np.array(adapter.linear1.bias)
    l2_w = np.array(adapter.linear2.weight)
    l2_b = np.array(adapter.linear2.bias)

    input_dim = l1_w.shape[1]
    hidden_dim = l1_w.shape[0]
    output_dim = l2_w.shape[0]

    np.savez(path,
             linear1_weight=l1_w, linear1_bias=l1_b,
             linear2_weight=l2_w, linear2_bias=l2_b,
             input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
             sender_layer=sender_layer, receiver_layer=receiver_layer,
             val_mse=val_mse, val_cosine=val_cosine, train_mse=train_mse)

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved MLP adapter to {path} ({size_mb:.1f} MB)")


def load_mlp_adapter(path):
    """Load a trained MLP adapter from disk.

    Returns: (adapter, metadata)
        adapter: MLPAdapter with loaded weights
        metadata: dict with layer indices and training metrics
    """
    data = np.load(path)
    input_dim = int(data["input_dim"])
    hidden_dim = int(data["hidden_dim"])
    output_dim = int(data["output_dim"])

    adapter = MLPAdapter(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim)
    adapter.linear1.weight = mx.array(data["linear1_weight"])
    adapter.linear1.bias = mx.array(data["linear1_bias"])
    adapter.linear2.weight = mx.array(data["linear2_weight"])
    adapter.linear2.bias = mx.array(data["linear2_bias"])
    mx.eval(adapter.parameters())

    metadata = {
        "sender_layer": int(data["sender_layer"]),
        "receiver_layer": int(data["receiver_layer"]),
        "val_mse": float(data["val_mse"]),
        "val_cosine": float(data["val_cosine"]),
        "train_mse": float(data["train_mse"]),
    }
    return adapter, metadata


def apply_mlp_adapter(hidden, adapter):
    """Project sender hidden states through the MLP adapter.

    No centering, no norm matching. The adapter output goes directly
    into gemma_forward_from_layer() at the target receiver layer.

    Args:
        hidden: [B, seq, D_sender] or [B, D_sender] sender hidden states
        adapter: trained MLPAdapter

    Returns: projected states in receiver space
    """
    h = hidden.astype(mx.float32)
    return adapter(h)
