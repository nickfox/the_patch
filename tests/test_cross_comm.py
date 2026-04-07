"""Tests for mlxmas.cross_comm — Gemma forward pass functions."""

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlxmas.cross_comm import gemma_forward_with_embeddings, gemma_forward_from_layer


class TestForwardFromLayerZero:
    """start_layer=0 should match the baseline function exactly."""

    def test_output_matches_baseline(self, receiver_model_and_tok):
        model, tok = receiver_model_and_tok

        # Create real embeddings from a short token sequence
        tokens = tok.encode("Hello world", add_special_tokens=False)
        input_ids = mx.array([tokens])
        embeds = model.model.embed_tokens(input_ids)
        mx.eval(embeds)

        cache0 = make_prompt_cache(model)
        cache1 = make_prompt_cache(model)

        out_baseline = gemma_forward_with_embeddings(model, embeds, cache0)
        mx.eval(out_baseline)
        out_new = gemma_forward_from_layer(model, embeds, cache1, start_layer=0)
        mx.eval(out_new)

        diff = mx.max(mx.abs(out_baseline - out_new)).item()
        assert diff < 1e-5, f"Outputs differ by {diff}"

    def test_output_shape(self, receiver_model_and_tok):
        model, tok = receiver_model_and_tok
        tokens = tok.encode("Test prompt", add_special_tokens=False)
        input_ids = mx.array([tokens])
        embeds = model.model.embed_tokens(input_ids)
        mx.eval(embeds)

        cache = make_prompt_cache(model)
        out = gemma_forward_from_layer(model, embeds, cache, start_layer=0)
        mx.eval(out)

        seq_len = len(tokens)
        vocab_size = model.args.vocab_size
        assert out.shape == (1, seq_len, vocab_size), f"Got {out.shape}"


class TestForwardFromIntermediateLayer:
    """start_layer > 0 should skip early layers and scaling."""

    def test_skips_early_layers(self, receiver_model_and_tok):
        model, tok = receiver_model_and_tok
        D = model.args.hidden_size

        # Create a synthetic hidden state at the right dimension
        h = mx.random.normal((1, 3, D))
        mx.eval(h)

        cache = make_prompt_cache(model)
        out = gemma_forward_from_layer(model, h, cache, start_layer=4)
        mx.eval(out)

        # Layers 0-3 should have offset=0 (skipped)
        for i in range(4):
            assert cache[i].offset == 0, f"cache[{i}].offset = {cache[i].offset}, expected 0"
        # Layers 4+ should have offset=3 (processed 3 tokens)
        assert cache[4].offset == 3, f"cache[4].offset = {cache[4].offset}, expected 3"

    def test_intermediate_produces_different_output(self, receiver_model_and_tok):
        """start_layer=0 vs start_layer=4 should produce different logits."""
        model, tok = receiver_model_and_tok
        D = model.args.hidden_size

        h = mx.random.normal((1, 3, D))
        mx.eval(h)

        cache0 = make_prompt_cache(model)
        out0 = gemma_forward_from_layer(model, h, cache0, start_layer=0)
        mx.eval(out0)

        cache4 = make_prompt_cache(model)
        out4 = gemma_forward_from_layer(model, h, cache4, start_layer=4)
        mx.eval(out4)

        diff = mx.max(mx.abs(out0 - out4)).item()
        assert diff > 0.1, f"Expected significant difference, got {diff}"


class TestCacheOffsetConsistency:
    """Cache offset behavior with intermediate injection."""

    def test_offset_after_injection(self, receiver_model_and_tok):
        model, _ = receiver_model_and_tok
        D = model.args.hidden_size
        start = 6

        h = mx.random.normal((1, 5, D))
        mx.eval(h)

        cache = make_prompt_cache(model)
        out = gemma_forward_from_layer(model, h, cache, start_layer=start)
        # Only eval states for layers that were actually used
        mx.eval(out, *[cache[i].state for i in range(start, len(model.model.layers))])

        n_layers = len(model.model.layers)
        # Layers before start_layer: offset=0 (skipped, keys=None)
        for i in range(start):
            assert cache[i].offset == 0
        # Layers from start_layer onward: offset=5
        for i in range(start, n_layers):
            assert cache[i].offset == 5

    def test_sequential_injections_accumulate(self, receiver_model_and_tok):
        model, _ = receiver_model_and_tok
        D = model.args.hidden_size
        start = 4
        n_layers = len(model.model.layers)

        cache = make_prompt_cache(model)

        # First injection: 3 tokens
        h1 = mx.random.normal((1, 3, D))
        mx.eval(h1)
        out1 = gemma_forward_from_layer(model, h1, cache, start_layer=start)
        mx.eval(out1, *[cache[i].state for i in range(start, n_layers)])
        assert cache[start].offset == 3

        # Second injection: 2 tokens
        h2 = mx.random.normal((1, 2, D))
        mx.eval(h2)
        out2 = gemma_forward_from_layer(model, h2, cache, start_layer=start)
        mx.eval(out2, *[cache[i].state for i in range(start, n_layers)])
        assert cache[start].offset == 5  # 3 + 2
