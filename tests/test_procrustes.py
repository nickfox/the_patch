"""Tests for Procrustes alignment in mlxmas.contextual_procrustes.

Uses real MLX operations on real tensors extracted from real models.
"""

import os
import tempfile

import mlx.core as mx

from mlxmas.contextual_procrustes import (
    compute_procrustes,
    compute_procrustes_with_heldout,
    extract_final_hidden_state,
    save_alignment,
    load_alignment,
)


class TestComputeProcrustes:

    def test_with_real_model_states(self, sender_model_and_tok, receiver_model_and_tok):
        """Compute Procrustes on real hidden states from both models."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        prompts = [
            "What is 2 + 2?",
            "How many legs does a dog have?",
            "What is the capital of France?",
            "If I have 5 apples and eat 2, how many remain?",
            "What is 10 times 3?",
        ]

        states_a, states_b = [], []
        for p in prompts:
            h_a = extract_final_hidden_state(sender, tok_s, p)
            h_b = extract_final_hidden_state(receiver, tok_r, p)
            states_a.append(h_a)
            states_b.append(h_b)

        H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)
        H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)
        mx.eval(H_a, H_b)

        result = compute_procrustes(H_a, H_b)

        # W_ortho should map [D_a] -> [D_b]
        D_a = sender.args.hidden_size
        D_b = receiver.args.hidden_size
        assert result["W_ortho"].shape == (D_a, D_b)
        assert result["mean_a"].shape == (1, D_a)
        assert result["mean_b"].shape == (1, D_b)
        assert result["n_pairs"] == 5

    def test_orthogonality(self, sender_model_and_tok, receiver_model_and_tok):
        """W^T @ W should be close to identity (semi-orthogonal)."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        prompts = [f"What is {i} + {i+1}?" for i in range(10)]
        states_a, states_b = [], []
        for p in prompts:
            states_a.append(extract_final_hidden_state(sender, tok_s, p))
            states_b.append(extract_final_hidden_state(receiver, tok_r, p))

        H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)
        H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)
        mx.eval(H_a, H_b)

        result = compute_procrustes(H_a, H_b)
        W = result["W_ortho"]
        D_b = H_b.shape[1]

        WtW = W.T @ W
        mx.eval(WtW)
        identity = mx.eye(D_b)
        diff = mx.max(mx.abs(WtW - identity)).item()
        assert diff < 0.05, f"W^T@W deviates from I by {diff}"


class TestComputeProcrustesWithHeldout:

    def test_train_heldout_split(self, sender_model_and_tok, receiver_model_and_tok):
        """Should split data and report metrics on both sets."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        prompts = [f"What is {i} times {i+2}?" for i in range(10)]
        states_a, states_b = [], []
        for p in prompts:
            states_a.append(extract_final_hidden_state(sender, tok_s, p))
            states_b.append(extract_final_hidden_state(receiver, tok_r, p))

        H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)
        H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)
        mx.eval(H_a, H_b)

        result = compute_procrustes_with_heldout(H_a, H_b, heldout_frac=0.2)
        assert result["n_train"] == 8
        assert result["n_heldout"] == 2
        assert "train_cosine" in result
        assert "heldout_cosine" in result
        assert result["center"] is True

    def test_uncentered(self, sender_model_and_tok, receiver_model_and_tok):
        """center=False should produce zero means."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        prompts = [f"Question {i}" for i in range(6)]
        states_a, states_b = [], []
        for p in prompts:
            states_a.append(extract_final_hidden_state(sender, tok_s, p))
            states_b.append(extract_final_hidden_state(receiver, tok_r, p))

        H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)
        H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)
        mx.eval(H_a, H_b)

        result = compute_procrustes_with_heldout(H_a, H_b, center=False)
        assert result["center"] is False
        assert mx.max(mx.abs(result["mean_a"])).item() < 1e-6
        assert mx.max(mx.abs(result["mean_b"])).item() < 1e-6

    def test_centered_vs_uncentered_differ(self, sender_model_and_tok, receiver_model_and_tok):
        """Centered and uncentered should produce different W matrices."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        prompts = [f"Solve: {i} + {i*2} = ?" for i in range(8)]
        states_a, states_b = [], []
        for p in prompts:
            states_a.append(extract_final_hidden_state(sender, tok_s, p))
            states_b.append(extract_final_hidden_state(receiver, tok_r, p))

        H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)
        H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)
        mx.eval(H_a, H_b)

        r_c = compute_procrustes_with_heldout(H_a, H_b, center=True)
        r_u = compute_procrustes_with_heldout(H_a, H_b, center=False)

        diff = mx.max(mx.abs(r_c["W_ortho"] - r_u["W_ortho"])).item()
        assert diff > 0.001, "Centered and uncentered should produce different W"


class TestSaveLoadRoundtrip:

    def test_roundtrip(self, sender_model_and_tok, receiver_model_and_tok):
        """save_alignment -> load_alignment should produce identical values."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        prompts = [f"What is {i}?" for i in range(5)]
        states_a, states_b = [], []
        for p in prompts:
            states_a.append(extract_final_hidden_state(sender, tok_s, p))
            states_b.append(extract_final_hidden_state(receiver, tok_r, p))

        H_a = mx.concatenate(states_a, axis=0).astype(mx.float32)
        H_b = mx.concatenate(states_b, axis=0).astype(mx.float32)
        mx.eval(H_a, H_b)

        original = compute_procrustes(H_a, H_b)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            save_alignment(original, path)
            loaded = load_alignment(path)

            diff_W = mx.max(mx.abs(original["W_ortho"] - loaded["W_ortho"])).item()
            assert diff_W < 1e-5, f"W_ortho differs by {diff_W}"

            diff_a = mx.max(mx.abs(original["mean_a"] - loaded["mean_a"])).item()
            assert diff_a < 1e-5

            assert abs(original["cos_sim"] - loaded["cos_sim"]) < 1e-5
            assert original["n_pairs"] == loaded["n_pairs"]
        finally:
            os.unlink(path)
