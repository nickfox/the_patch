"""
MHA Adapter for Interlat-style cross-model latent communication.

Linear projection + LayerNorm + single-head self-attention + LayerNorm + output projection.
No AdaptiveProjection — direct signal path to avoid mode collapse.

Input:  [batch, seq_len, 2560]  (Qwen last-layer hidden states)
Output: [batch, seq_len, 2304]  (Gemma embedding-space vectors)
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class MHAAdapter(nn.Module):
    """Adapter: Linear project + self-attention + output projection.

    Args:
        sender_dim: sender hidden dimension (Qwen = 2560)
        receiver_dim: receiver hidden dimension (Gemma = 2304)
        num_heads: number of attention heads
    """

    def __init__(self, sender_dim=2560, receiver_dim=2304, num_heads=1):
        super().__init__()
        self.input_projector = nn.Linear(sender_dim, receiver_dim, bias=True)
        self.pre_ln = nn.LayerNorm(receiver_dim, eps=1e-6)
        self.hidden_mha = nn.MultiHeadAttention(
            dims=receiver_dim,
            num_heads=num_heads,
        )
        self.post_ln = nn.LayerNorm(receiver_dim, eps=1e-6)
        self.output_projector = nn.Linear(receiver_dim, receiver_dim, bias=True)

    def __call__(self, x):
        """
        Args:
            x: [batch, seq_len, sender_dim] sender hidden states
        Returns:
            [batch, seq_len, receiver_dim] processed states for injection
        """
        x = self.input_projector(x)
        normed = self.pre_ln(x)
        attn_out = self.hidden_mha(normed, normed, normed)
        out = self.post_ln(normed + attn_out)
        out = self.output_projector(out)
        return mx.clip(out, -10.0, 10.0)


def save_adapter(adapter, path):
    """Save adapter weights to safetensors."""
    weights = dict(tree_flatten(adapter.parameters()))
    mx.save_safetensors(path, weights)


def load_adapter(path, sender_dim=2560, receiver_dim=2304, num_heads=1):
    """Load adapter weights from safetensors."""
    adapter = MHAAdapter(sender_dim, receiver_dim, num_heads)
    weights = mx.load(path)
    adapter.load_weights(list(weights.items()))
    mx.eval(adapter.parameters())
    return adapter
