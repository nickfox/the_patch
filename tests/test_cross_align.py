"""Tests for mlxmas.cross_align — vocabulary-based alignment functions."""

import mlx.core as mx

from mlxmas.cross_align import apply_cross_realignment, compute_cross_alignment


class TestApplyCrossRealignment:

    def test_output_shape_2d(self, sender_model_and_tok, receiver_model_and_tok):
        """[1, D_sender] input -> [1, D_receiver] output."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok
        D_s = sender.args.hidden_size
        D_r = receiver.args.hidden_size

        # Use real embeddings from sender as hidden state
        tokens = tok_s.encode("Test", add_special_tokens=False)
        input_ids = mx.array([tokens])
        hidden = sender.model(input_ids)
        mx.eval(hidden)
        h = hidden[:, -1, :]  # [1, D_s]

        # Create alignment matrix at correct dimensions
        W_cross = mx.random.normal((D_s, D_r)) * 0.01
        mean_s = mx.zeros((1, D_s))
        mean_r = mx.zeros((1, D_r))
        target_norm = mx.array(10.0)
        mx.eval(W_cross, mean_s, mean_r, target_norm)

        out = apply_cross_realignment(h, W_cross, mean_s, mean_r, target_norm)
        mx.eval(out)
        assert out.shape == (1, D_r)

    def test_output_shape_3d(self, sender_model_and_tok, receiver_model_and_tok):
        """[1, seq, D_sender] input -> [1, seq, D_receiver] output."""
        sender, tok_s = sender_model_and_tok
        receiver, _ = receiver_model_and_tok
        D_s = sender.args.hidden_size
        D_r = receiver.args.hidden_size

        tokens = tok_s.encode("Hello world test", add_special_tokens=False)
        input_ids = mx.array([tokens])
        hidden = sender.model(input_ids)
        mx.eval(hidden)
        # hidden is [1, seq, D_s]

        W_cross = mx.random.normal((D_s, D_r)) * 0.01
        mean_s = mx.zeros((1, D_s))
        mean_r = mx.zeros((1, D_r))
        target_norm = mx.array(10.0)
        mx.eval(W_cross, mean_s, mean_r, target_norm)

        out = apply_cross_realignment(hidden, W_cross, mean_s, mean_r, target_norm)
        mx.eval(out)
        seq_len = len(tokens)
        assert out.shape == (1, seq_len, D_r)

    def test_norm_matching(self, sender_model_and_tok, receiver_model_and_tok):
        """Output vectors should have norm close to target_norm."""
        sender, tok_s = sender_model_and_tok
        receiver, _ = receiver_model_and_tok
        D_s = sender.args.hidden_size
        D_r = receiver.args.hidden_size

        tokens = tok_s.encode("Norm test", add_special_tokens=False)
        input_ids = mx.array([tokens])
        hidden = sender.model(input_ids)
        mx.eval(hidden)
        h = hidden[:, -1:, :]  # [1, 1, D_s]

        W_cross = mx.random.normal((D_s, D_r)) * 0.01
        mean_s = mx.zeros((1, D_s))
        mean_r = mx.zeros((1, D_r))
        target_norm = mx.array(7.5)
        mx.eval(W_cross, mean_s, mean_r, target_norm)

        out = apply_cross_realignment(h, W_cross, mean_s, mean_r, target_norm)
        mx.eval(out)

        out_norm = mx.linalg.norm(out).item()
        assert abs(out_norm - 7.5) < 0.1, f"Expected norm ~7.5, got {out_norm}"
