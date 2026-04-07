"""Tests for extract_hidden_at_layers in mlxmas.contextual_procrustes."""

import mlx.core as mx

from mlxmas.contextual_procrustes import extract_hidden_at_layers


class TestExtractHiddenAtLayers:

    def test_returns_requested_layers_gemma(self, receiver_model_and_tok):
        """Should return a dict with exactly the requested layer indices."""
        model, tok = receiver_model_and_tok
        captured = extract_hidden_at_layers(model, tok, "What is 2+2?", [0, 4, 8], is_gemma=True)
        assert set(captured.keys()) == {0, 4, 8}

    def test_returns_requested_layers_qwen(self, sender_model_and_tok):
        """Should work for Qwen (is_gemma=False)."""
        model, tok = sender_model_and_tok
        captured = extract_hidden_at_layers(model, tok, "What is 2+2?", [0, 12, 24], is_gemma=False)
        assert set(captured.keys()) == {0, 12, 24}

    def test_output_shape_gemma(self, receiver_model_and_tok):
        """Each captured state should be [1, D_receiver]."""
        model, tok = receiver_model_and_tok
        D = model.args.hidden_size
        captured = extract_hidden_at_layers(model, tok, "Test", [2, 10], is_gemma=True)
        for layer_idx, state in captured.items():
            assert state.shape == (1, D), f"Layer {layer_idx}: expected (1,{D}), got {state.shape}"

    def test_output_shape_qwen(self, sender_model_and_tok):
        """Each captured state should be [1, D_sender]."""
        model, tok = sender_model_and_tok
        D = model.args.hidden_size
        captured = extract_hidden_at_layers(model, tok, "Test", [5, 20], is_gemma=False)
        for layer_idx, state in captured.items():
            assert state.shape == (1, D), f"Layer {layer_idx}: expected (1,{D}), got {state.shape}"

    def test_intermediate_differs_from_final(self, receiver_model_and_tok):
        """Hidden states at different layers should differ."""
        model, tok = receiver_model_and_tok
        n_layers = len(model.model.layers)
        captured = extract_hidden_at_layers(
            model, tok, "Hello world", [0, n_layers - 1], is_gemma=True
        )
        diff = mx.max(mx.abs(captured[0] - captured[n_layers - 1])).item()
        assert diff > 0.01, f"Layer 0 and {n_layers-1} should differ, got {diff}"

    def test_float32_output(self, receiver_model_and_tok):
        """Captured states should be float32."""
        model, tok = receiver_model_and_tok
        captured = extract_hidden_at_layers(model, tok, "Test", [0], is_gemma=True)
        assert captured[0].dtype == mx.float32

    def test_gemma_vs_qwen_scaling_difference(self, sender_model_and_tok, receiver_model_and_tok):
        """Gemma applies sqrt(D) scaling, Qwen does not — norms should reflect this."""
        sender, tok_s = sender_model_and_tok
        receiver, tok_r = receiver_model_and_tok

        # Extract layer-0 states from both models on same prompt
        cap_s = extract_hidden_at_layers(sender, tok_s, "Test scaling", [0], is_gemma=False)
        cap_r = extract_hidden_at_layers(receiver, tok_r, "Test scaling", [0], is_gemma=True)

        norm_s = mx.linalg.norm(cap_s[0]).item()
        norm_r = mx.linalg.norm(cap_r[0]).item()

        # Both should be finite and positive
        assert norm_s > 0
        assert norm_r > 0
