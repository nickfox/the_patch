#!/usr/bin/env python3
"""
Barycentric Ridge Self-Projector for cross-model latent communication.

Maps sender hidden states (transformer output space) to embedding-like vectors
via a two-stage approach:
  1. Exact barycentric projection (teacher): LM head -> top-k -> softmax -> weighted embedding sum
  2. Ridge regression (fast runtime): single matrix multiply approximation

Replaces the broken W_a = identity path for tied-embedding models.
"""

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache


def get_embedding_matrix(model, chunk_size=4096):
    """Get dequantized embedding matrix [V, D] from a quantized model.

    8-bit quantized MLX models store embedding weights in packed format.
    Must dequantize by passing token IDs through the embedding layer.
    Do NOT read model.model.embed_tokens.weight directly.
    """
    V = model.args.vocab_size
    chunks = []
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        ids = mx.array([list(range(start, end))])
        emb = model.model.embed_tokens(ids)[0]  # [chunk, D]
        chunks.append(emb)
        mx.eval(emb)
    E = mx.concatenate(chunks, axis=0)  # [V, D]
    mx.eval(E)
    return E


class BarycentricRidgeSelfProjector:
    """Fitted projector: sender hidden states -> embedding-like vectors."""

    def __init__(self, P, mu_z, mu_y, k=64, tau=0.7, lam=1e-2):
        """
        Args:
            P: [D, D] ridge regression matrix
            mu_z: [D] mean of hidden states (for centering)
            mu_y: [D] mean of barycentric targets (for de-centering)
            k: top-k used during fitting
            tau: softmax temperature used during fitting
            lam: ridge regularization fraction used during fitting
        """
        self.P = P
        self.mu_z = mu_z
        self.mu_y = mu_y
        self.k = k
        self.tau = tau
        self.lam = lam

    def project_fast(self, z):
        """Fast ridge approximation. Single matrix multiply.

        Args:
            z: [..., D] post-final-norm hidden states from sender

        Returns:
            [..., D] L2-normalized embedding-space vectors
        """
        z32 = z.astype(mx.float32)
        y = (z32 - self.mu_z) @ self.P + self.mu_y
        y = y / (mx.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
        return y

    def project_exact(self, z, E_raw, k=64, tau=0.7, pos_chunk=8):
        """Exact barycentric projection via LM head + softmax + weighted sum.

        Args:
            z: [N, D] post-final-norm hidden states (2D only)
            E_raw: [V, D] dequantized sender embedding matrix
            k: top-k for logit selection
            tau: temperature for softmax sharpening
            pos_chunk: chunk size for vocab projection (memory bound)

        Returns:
            [N, D] L2-normalized embedding-space vectors
        """
        z_flat = z.reshape(-1, z.shape[-1]).astype(mx.float32)  # [N, D]
        N = z_flat.shape[0]

        results = []
        for start in range(0, N, pos_chunk):
            end = min(start + pos_chunk, N)
            z_chunk = z_flat[start:end]  # [C, D]

            # logits: z @ E_raw.T -> [C, V]
            logits = z_chunk @ E_raw.T

            # top-k: argpartition for indices, then gather values
            # mx.topk returns values only — no indices
            idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]  # [C, k]
            vals = mx.take_along_axis(logits, idx, axis=-1)  # [C, k]

            # softmax over top-k values (top-k then softmax, not softmax then top-k)
            probs = mx.softmax(vals / tau, axis=-1)  # [C, k]

            # gather top-k embeddings, cast to float32 for precision
            e_top = E_raw[idx].astype(mx.float32)  # [C, k, D]

            # L2-normalize gathered embedding rows
            e_top = e_top / (mx.linalg.norm(e_top, axis=-1, keepdims=True) + 1e-8)

            # weighted sum of normalized embeddings
            y_chunk = (probs[..., None] * e_top).sum(axis=-2)  # [C, D]

            # L2-normalize output
            y_chunk = y_chunk / (mx.linalg.norm(y_chunk, axis=-1, keepdims=True) + 1e-8)

            results.append(y_chunk)
            mx.eval(y_chunk)

        y = mx.concatenate(results, axis=0)  # [N, D]
        return y

    def save(self, path):
        """Save projector to .npz file."""
        np.savez(
            path,
            P=np.array(self.P),
            mu_z=np.array(self.mu_z),
            mu_y=np.array(self.mu_y),
            k=np.array(self.k),
            tau=np.array(self.tau),
            lam=np.array(self.lam),
        )
        print(f"Projector saved to {path}")

    @classmethod
    def load(cls, path):
        """Load projector from .npz file."""
        data = np.load(path)
        P = mx.array(data["P"])
        mu_z = mx.array(data["mu_z"])
        mu_y = mx.array(data["mu_y"])
        k = int(data["k"])
        tau = float(data["tau"])
        lam = float(data["lam"])
        return cls(P, mu_z, mu_y, k=k, tau=tau, lam=lam)


class BarycentricRidgeFitter:
    """Offline fitter. Accumulates sufficient statistics, then solves ridge."""

    def __init__(self, E_raw, k=64, tau=0.7, lam=1e-2, pos_chunk=8):
        """
        Args:
            E_raw: [V, D] dequantized sender embedding matrix
            k: top-k for barycentric teacher
            tau: temperature for softmax
            lam: ridge regularization fraction
            pos_chunk: chunk size for vocab projection
        """
        self.E_raw = E_raw
        self.k = k
        self.tau = tau
        self.lam = lam
        self.pos_chunk = pos_chunk

        self.D = E_raw.shape[-1]
        # Temporary projector instance for calling project_exact (P unused)
        self._teacher = BarycentricRidgeSelfProjector(None, None, None)

        # Sufficient statistics (float32)
        self.S_zz = mx.zeros((self.D, self.D), dtype=mx.float32)
        self.S_zy = mx.zeros((self.D, self.D), dtype=mx.float32)
        self.sum_z = mx.zeros((self.D,), dtype=mx.float32)
        self.sum_y = mx.zeros((self.D,), dtype=mx.float32)
        self.n = 0

    def accumulate(self, z_batch):
        """Feed a batch of post-final-norm hidden states.

        Computes barycentric teacher targets and accumulates sufficient stats.

        Args:
            z_batch: [N, D] hidden states (2D)
        """
        z = z_batch.reshape(-1, self.D).astype(mx.float32)
        N = z.shape[0]

        # Compute teacher targets
        y = self._teacher.project_exact(
            z, self.E_raw, k=self.k, tau=self.tau, pos_chunk=self.pos_chunk
        )
        y = y.astype(mx.float32)

        # Accumulate sufficient statistics
        self.S_zz = self.S_zz + z.T @ z
        self.S_zy = self.S_zy + z.T @ y
        self.sum_z = self.sum_z + z.sum(axis=0)
        self.sum_y = self.sum_y + y.sum(axis=0)
        self.n += N

        mx.eval(self.S_zz, self.S_zy, self.sum_z, self.sum_y)

    def finalize(self):
        """Solve ridge regression. Returns a BarycentricRidgeSelfProjector.

        Trace-scaled regularization:
          S_zz_c = S_zz / n - outer(mu_z, mu_z)
          alpha = lam * trace(S_zz_c) / D
          P = solve(S_zz_c + alpha * I, S_zy_c)
        """
        n = self.n
        if n == 0:
            raise ValueError("No data accumulated")

        mu_z = self.sum_z / n
        mu_y = self.sum_y / n

        S_zz_c = self.S_zz / n - mx.outer(mu_z, mu_z)
        S_zy_c = self.S_zy / n - mx.outer(mu_z, mu_y)

        alpha = self.lam * mx.trace(S_zz_c) / self.D
        mx.eval(alpha)
        print(f"Ridge alpha = {alpha.item():.6f} "
              f"(lam={self.lam}, trace/D={mx.trace(S_zz_c).item() / self.D:.4f})")

        A = S_zz_c + alpha * mx.eye(self.D)
        mx.eval(A, S_zy_c)
        P = mx.linalg.solve(A, S_zy_c, stream=mx.cpu)
        mx.eval(P)

        return BarycentricRidgeSelfProjector(
            P, mu_z, mu_y, k=self.k, tau=self.tau, lam=self.lam
        )


def fit_self_projector(sender_model, sender_tok, prompts,
                       k=64, tau=0.7, lam=1e-2, n_prompts=200, pos_chunk=8):
    """Fit a barycentric ridge self-projector from calibration prompts.

    Args:
        sender_model: MLX model (e.g., Qwen3-4B-8bit)
        sender_tok: tokenizer
        prompts: list of calibration strings
        k: top-k for barycentric teacher
        tau: softmax temperature
        lam: ridge regularization fraction
        n_prompts: how many prompts to use
        pos_chunk: chunk size for vocab projection

    Returns:
        (BarycentricRidgeSelfProjector, E_raw)
    """
    prompts = prompts[:n_prompts]

    print("Dequantizing embedding matrix...")
    E_raw = get_embedding_matrix(sender_model)
    print(f"  E_raw shape: {E_raw.shape}")

    fitter = BarycentricRidgeFitter(E_raw, k=k, tau=tau, lam=lam, pos_chunk=pos_chunk)

    total_tokens = 0
    for i, prompt in enumerate(prompts):
        tokens = sender_tok.encode(prompt, add_special_tokens=True)
        input_ids = mx.array([tokens])

        cache = make_prompt_cache(sender_model)
        z = sender_model.model(input_ids, cache=cache)  # [1, seq, D] post-norm
        mx.eval(z)

        # Accumulate as [seq_len, D]
        fitter.accumulate(z[0])
        total_tokens += len(tokens)

        if (i + 1) % 50 == 0 or i == len(prompts) - 1:
            print(f"  Processed {i + 1}/{len(prompts)} prompts, "
                  f"{total_tokens} tokens, {fitter.n} positions")

    print(f"\nFitting ridge regression ({fitter.n} positions, D={fitter.D})...")
    projector = fitter.finalize()
    print("Projector ready.")

    return projector, E_raw
