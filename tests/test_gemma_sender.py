"""
Comprehensive tests for the Gemma→Qwen pipeline.

All tests use real 8-bit models and real data.
No mocks, no placeholders, no dummy data.

Models:
  Sender:   mlx-community/gemma-2-2b-it-8bit
  Receiver: mlx-community/Qwen3-4B-8bit
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

from mlxmas.cross_comm import (
    extract_gemma_all_tokens,
    generate_with_qwen_from_layer,
    qwen_forward_from_layer,
    _qwen_forward_per_layer_mask,
)

REAL_QUESTION = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh "
    "duck egg. How much in dollars does she make every day at the farmers' market?"
)


# ============================================================
# 1. qwen_forward_from_layer: start_layer=0 must match model()
# ============================================================

class TestQwenForwardFromLayerZero:
    """start_layer=0 must produce bitwise-identical logits to model()."""

    def test_logits_match_standard_forward(self, qwen_model_and_tok):
        """THE critical correctness test. If this fails, nothing else matters."""
        model, tok = qwen_model_and_tok

        tokens = tok.encode("What is 2 + 2?", add_special_tokens=True)
        input_ids = mx.array([tokens])

        # Ground truth: standard model forward
        cache_ref = make_prompt_cache(model)
        logits_ref = model(input_ids, cache=cache_ref)
        mx.eval(logits_ref)

        # Our function: embed manually, then forward from layer 0
        cache_new = make_prompt_cache(model)
        embeds = model.model.embed_tokens(input_ids)
        mx.eval(embeds)
        logits_new = qwen_forward_from_layer(model, embeds, cache_new, start_layer=0)
        mx.eval(logits_new)

        diff = mx.max(mx.abs(logits_ref - logits_new)).item()
        assert diff < 1e-4, f"Logits differ by {diff} — must be bitwise-close"

    def test_cache_offsets_match(self, qwen_model_and_tok):
        """Cache offsets must be identical between model() and our function."""
        model, tok = qwen_model_and_tok

        tokens = tok.encode("Test cache offsets", add_special_tokens=True)
        input_ids = mx.array([tokens])
        seq_len = len(tokens)

        cache_ref = make_prompt_cache(model)
        logits_ref = model(input_ids, cache=cache_ref)
        mx.eval(logits_ref, *[c.state for c in cache_ref])

        cache_new = make_prompt_cache(model)
        embeds = model.model.embed_tokens(input_ids)
        logits_new = qwen_forward_from_layer(model, embeds, cache_new, start_layer=0)
        mx.eval(logits_new, *[c.state for c in cache_new])

        for i in range(len(model.model.layers)):
            assert cache_ref[i].offset == cache_new[i].offset, (
                f"Layer {i}: ref offset {cache_ref[i].offset} != "
                f"new offset {cache_new[i].offset}"
            )
            assert cache_ref[i].offset == seq_len

    def test_output_shape(self, qwen_model_and_tok):
        model, tok = qwen_model_and_tok
        tokens = tok.encode("Shape test", add_special_tokens=True)
        input_ids = mx.array([tokens])
        embeds = model.model.embed_tokens(input_ids)
        mx.eval(embeds)

        cache = make_prompt_cache(model)
        out = qwen_forward_from_layer(model, embeds, cache, start_layer=0)
        mx.eval(out)

        seq_len = len(tokens)
        vocab_size = model.args.vocab_size
        assert out.shape == (1, seq_len, vocab_size), f"Got {out.shape}"


# ============================================================
# 2. qwen_forward_from_layer: intermediate layer injection
# ============================================================

class TestQwenForwardFromIntermediateLayer:
    """start_layer > 0 skips early layers, caches are correct."""

    def test_skips_early_layers(self, qwen_model_and_tok):
        model, tok = qwen_model_and_tok
        D = model.args.hidden_size
        start = 12  # Spec: Qwen receiver at layer 12

        h = mx.random.normal((1, 5, D))
        mx.eval(h)

        cache = make_prompt_cache(model)
        out = qwen_forward_from_layer(model, h, cache, start_layer=start)
        mx.eval(out, *[c.state for c in cache[start:]])

        # Layers 0..11 must have offset=0 (never touched)
        for i in range(start):
            assert cache[i].offset == 0, (
                f"cache[{i}].offset = {cache[i].offset}, expected 0"
            )
        # Layers 12..35 must have offset=5
        n_layers = len(model.model.layers)
        for i in range(start, n_layers):
            assert cache[i].offset == 5, (
                f"cache[{i}].offset = {cache[i].offset}, expected 5"
            )

    def test_intermediate_vs_layer0_differ(self, qwen_model_and_tok):
        """start_layer=0 vs start_layer=12 must produce different logits."""
        model, tok = qwen_model_and_tok
        D = model.args.hidden_size

        h = mx.random.normal((1, 3, D))
        mx.eval(h)

        cache0 = make_prompt_cache(model)
        out0 = qwen_forward_from_layer(model, h, cache0, start_layer=0)
        mx.eval(out0)

        cache12 = make_prompt_cache(model)
        out12 = qwen_forward_from_layer(model, h, cache12, start_layer=12)
        mx.eval(out12)

        diff = mx.max(mx.abs(out0 - out12)).item()
        assert diff > 0.1, f"Expected significant difference, got {diff}"

    def test_output_shape_intermediate(self, qwen_model_and_tok):
        model, tok = qwen_model_and_tok
        D = model.args.hidden_size
        h = mx.random.normal((1, 7, D))
        mx.eval(h)

        cache = make_prompt_cache(model)
        out = qwen_forward_from_layer(model, h, cache, start_layer=9)
        mx.eval(out)

        vocab_size = model.args.vocab_size
        assert out.shape == (1, 7, vocab_size), f"Got {out.shape}"


# ============================================================
# 3. _qwen_forward_per_layer_mask: uniform cache must match standard
# ============================================================

class TestQwenPerLayerMask:
    """When all caches have the same offset, per-layer mask == single mask."""

    def test_matches_standard_forward_uniform_cache(self, qwen_model_and_tok):
        """With uniform cache offsets, per-layer mask must match model()."""
        model, tok = qwen_model_and_tok

        tokens = tok.encode("Per-layer mask test", add_special_tokens=True)
        input_ids = mx.array([tokens])

        # Reference: standard forward
        cache_ref = make_prompt_cache(model)
        logits_ref = model(input_ids, cache=cache_ref)
        mx.eval(logits_ref, *[c.state for c in cache_ref])

        # Per-layer mask with fresh (uniform offset=0) cache
        cache_plm = make_prompt_cache(model)
        h = model.model.embed_tokens(input_ids)
        # Qwen does NOT scale by sqrt(hidden_size)
        logits_plm = _qwen_forward_per_layer_mask(model, h, cache_plm)
        mx.eval(logits_plm, *[c.state for c in cache_plm])

        diff = mx.max(mx.abs(logits_ref - logits_plm)).item()
        assert diff < 1e-4, (
            f"Per-layer mask differs from standard by {diff}"
        )

    def test_handles_mixed_cache_offsets(self, qwen_model_and_tok):
        """After injection at layer 12, per-layer mask handles offset mismatch."""
        model, tok = qwen_model_and_tok
        D = model.args.hidden_size
        start = 12

        cache = make_prompt_cache(model)

        # Inject at layer 12 — layers 0..11 at offset 0, layers 12..35 at offset 5
        h_inject = mx.random.normal((1, 5, D))
        mx.eval(h_inject)
        logits_inject = qwen_forward_from_layer(
            model, h_inject, cache, start_layer=start
        )
        mx.eval(logits_inject, *[c.state for c in cache[start:]])

        # Now feed a new token through all layers with per-layer masks
        h_new = model.model.embed_tokens(mx.array([[0]]))
        mx.eval(h_new)

        # This must not crash — the key test is that it handles
        # layers 0..11 (offset=0) and layers 12..35 (offset=5)
        logits_new = _qwen_forward_per_layer_mask(model, h_new, cache)
        mx.eval(logits_new, *[c.state for c in cache])

        assert logits_new.shape[0] == 1
        assert logits_new.shape[1] == 1
        assert logits_new.shape[2] == model.args.vocab_size


# ============================================================
# 4. extract_gemma_all_tokens: shape, dtype, values
# ============================================================

class TestExtractGemmaAllTokens:
    """extract_gemma_all_tokens must return correct hidden states."""

    def test_output_shape(self, gemma_model_and_tok):
        model, tok = gemma_model_and_tok
        prompt = "Hello world"
        tokens = tok.encode(prompt, add_special_tokens=True)
        expected_seq_len = len(tokens)
        D = model.args.hidden_size  # 2304 for Gemma-2-2B

        result = extract_gemma_all_tokens(model, tok, prompt, layer=10)
        mx.eval(result)

        assert result.shape == (1, expected_seq_len, D), (
            f"Expected (1, {expected_seq_len}, {D}), got {result.shape}"
        )

    def test_output_dtype_float32(self, gemma_model_and_tok):
        model, tok = gemma_model_and_tok
        result = extract_gemma_all_tokens(model, tok, "Test", layer=10)
        mx.eval(result)
        assert result.dtype == mx.float32

    def test_values_nonzero(self, gemma_model_and_tok):
        """Real hidden states must have non-trivial norms."""
        model, tok = gemma_model_and_tok
        result = extract_gemma_all_tokens(model, tok, REAL_QUESTION, layer=10)
        mx.eval(result)

        norms = mx.linalg.norm(result[0], axis=1)
        mx.eval(norms)
        mean_norm = mx.mean(norms).item()
        assert mean_norm > 1.0, f"Mean norm {mean_norm} too small — states look dead"

    def test_matches_manual_layer_extraction(self, gemma_model_and_tok):
        """Cross-validate: extract_gemma_all_tokens must match manual iteration."""
        model, tok = gemma_model_and_tok
        prompt = "Two plus two"

        # Our function
        result_fn = extract_gemma_all_tokens(model, tok, prompt, layer=10)
        mx.eval(result_fn)

        # Manual extraction (ground truth)
        tokens = tok.encode(prompt, add_special_tokens=True)
        input_ids = mx.array([tokens])
        cache = make_prompt_cache(model)
        h = model.model.embed_tokens(input_ids)
        h = h * (model.model.args.hidden_size ** 0.5)
        mask = create_attention_mask(h, cache[0], return_array=True)

        for i, (layer, c) in enumerate(zip(model.model.layers, cache)):
            h = layer(h, mask, c)
            if i == 10:
                manual = h.astype(mx.float32)
                break
        mx.eval(manual)

        diff = mx.max(mx.abs(result_fn - manual)).item()
        assert diff < 1e-5, f"Function vs manual differ by {diff}"

    def test_different_layers_produce_different_states(self, gemma_model_and_tok):
        """Layer 5 vs layer 15 must produce different hidden states."""
        model, tok = gemma_model_and_tok
        prompt = "Different layers"

        h5 = extract_gemma_all_tokens(model, tok, prompt, layer=5)
        h15 = extract_gemma_all_tokens(model, tok, prompt, layer=15)
        mx.eval(h5, h15)

        diff = mx.max(mx.abs(h5 - h15)).item()
        assert diff > 0.1, f"Layers 5 and 15 too similar: max diff {diff}"


# ============================================================
# 5. generate_with_qwen_from_layer: text generation
# ============================================================

class TestGenerateWithQwenFromLayer:
    """Full generation pipeline must produce real text."""

    def test_generates_nonempty_text_layer0(self, qwen_model_and_tok):
        """Baseline: start_layer=0 with real embeddings must generate text."""
        model, tok = qwen_model_and_tok

        # Create real embeddings from a simple prompt
        prompt = "What is 2 + 2?"
        tokens = tok.encode(prompt, add_special_tokens=True)
        input_ids = mx.array([tokens])
        embeds = model.model.embed_tokens(input_ids)
        mx.eval(embeds)

        output = generate_with_qwen_from_layer(
            model, tok, embeds,
            start_layer=0, max_tokens=2048, temperature=0.6,
        )
        assert len(output) > 0, "Generated empty string"
        assert len(output) > 5, f"Output suspiciously short: '{output}'"

    def test_generates_text_intermediate_layer(self, qwen_model_and_tok):
        """Injection at layer 12 with random hidden states must not crash."""
        model, tok = qwen_model_and_tok
        D = model.args.hidden_size

        h = mx.random.normal((1, 10, D))
        mx.eval(h)

        output = generate_with_qwen_from_layer(
            model, tok, h,
            start_layer=12, max_tokens=2048, temperature=0.6,
        )
        # Random states won't produce sensible text, but must not crash
        assert isinstance(output, str)


# ============================================================
# 6. End-to-end: Gemma extract → project → Qwen generate
# ============================================================

class TestEndToEnd:
    """Full Gemma→Qwen pipeline with real Procrustes (if available)."""

    def test_extract_project_generate(self, gemma_model_and_tok, qwen_model_and_tok):
        """Gemma extract → identity projection → Qwen inject → generate.

        Uses identity projection (no Procrustes) to test the pipeline
        plumbing independently of alignment quality. The Gemma D=2304
        doesn't match Qwen D=2560, so we pad with zeros to test the
        mechanics. This is NOT expected to produce correct answers —
        it validates that the functions compose without errors.
        """
        gemma_model, gemma_tok = gemma_model_and_tok
        qwen_model, qwen_tok = qwen_model_and_tok

        D_gemma = gemma_model.args.hidden_size  # 2304
        D_qwen = qwen_model.args.hidden_size    # 2560

        # 1. Gemma extract
        gemma_hidden = extract_gemma_all_tokens(
            gemma_model, gemma_tok, REAL_QUESTION, layer=10
        )
        mx.eval(gemma_hidden)
        assert gemma_hidden.shape[2] == D_gemma

        # 2. Pad to Qwen dimension (crude test projection)
        pad_width = D_qwen - D_gemma
        padded = mx.concatenate([
            gemma_hidden,
            mx.zeros((1, gemma_hidden.shape[1], pad_width)),
        ], axis=2)
        mx.eval(padded)
        assert padded.shape[2] == D_qwen

        # 3. Inject into Qwen and generate
        output = generate_with_qwen_from_layer(
            qwen_model, qwen_tok, padded,
            start_layer=12, max_tokens=2048, temperature=0.6,
        )
        assert isinstance(output, str)
        assert len(output) > 0, "Pipeline produced empty output"


# ============================================================
# 7. contextual_procrustes: is_gemma_a parameter wiring
# ============================================================

class TestProcrustesIsGemmaFix:
    """Verify the is_gemma_a parameter is wired through correctly."""

    def test_calibrate_signature_has_is_gemma_a(self):
        """calibrate_multitoken_incremental must accept is_gemma_a."""
        import inspect
        from mlxmas.contextual_procrustes import calibrate_multitoken_incremental
        sig = inspect.signature(calibrate_multitoken_incremental)
        assert "is_gemma_a" in sig.parameters, "is_gemma_a parameter missing"
        assert "is_gemma_b" in sig.parameters, "is_gemma_b parameter missing"

    def test_collect_contextual_pairs_signature(self):
        """collect_contextual_pairs must accept is_gemma_a."""
        import inspect
        from mlxmas.contextual_procrustes import collect_contextual_pairs
        sig = inspect.signature(collect_contextual_pairs)
        assert "is_gemma_a" in sig.parameters, "is_gemma_a parameter missing"

    def test_extract_with_is_gemma_flag(self, gemma_model_and_tok):
        """extract_all_tokens_at_layer with is_gemma=True must apply sqrt scaling.

        Validates that the is_gemma flag actually changes the extracted states.
        """
        from mlxmas.contextual_procrustes import extract_all_tokens_at_layer
        model, tok = gemma_model_and_tok
        prompt = "Test scaling"

        # With is_gemma=True (applies sqrt(hidden_size) scaling)
        h_gemma = extract_all_tokens_at_layer(model, tok, prompt, 10, is_gemma=True)

        # With is_gemma=False (no scaling — wrong for Gemma)
        h_no_scale = extract_all_tokens_at_layer(model, tok, prompt, 10, is_gemma=False)

        # These must differ because sqrt(2304) = 48 is a large factor
        diff = np.max(np.abs(h_gemma - h_no_scale))
        assert diff > 1.0, (
            f"is_gemma flag had no effect — diff {diff}. "
            f"The sqrt scaling is not being applied."
        )

    def test_main_detects_gemma_in_both_models(self):
        """The main() function must detect 'gemma' in both model_a and model_b."""
        import ast
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "mlxmas", "contextual_procrustes.py"
        )
        with open(path) as f:
            source = f.read()

        # Verify both is_gemma_a and is_gemma_b are set in main()
        assert 'is_gemma_a = "gemma" in args.model_a.lower()' in source, (
            "main() does not detect gemma in model_a"
        )
        assert 'is_gemma_b = "gemma" in args.model_b.lower()' in source, (
            "main() does not detect gemma in model_b"
        )
        # Verify the old single-variable pattern is gone
        assert 'is_gemma = "gemma" in args.model_b.lower()' not in source, (
            "Old single-variable is_gemma pattern still present"
        )


# ============================================================
# 8. Qwen model properties validation
# ============================================================

class TestQwenModelProperties:
    """Validate assumptions about Qwen3-4B-8bit architecture."""

    def test_tied_embeddings(self, qwen_model_and_tok):
        model, _ = qwen_model_and_tok
        assert model.args.tie_word_embeddings is True, (
            "Qwen3-4B must have tied embeddings"
        )

    def test_num_layers(self, qwen_model_and_tok):
        model, _ = qwen_model_and_tok
        n_layers = len(model.model.layers)
        assert n_layers == 36, f"Expected 36 layers, got {n_layers}"

    def test_hidden_size(self, qwen_model_and_tok):
        model, _ = qwen_model_and_tok
        assert model.args.hidden_size == 2560, (
            f"Expected hidden_size=2560, got {model.args.hidden_size}"
        )


class TestGemmaModelProperties:
    """Validate assumptions about Gemma-2-2B-it-8bit architecture."""

    def test_num_layers(self, gemma_model_and_tok):
        model, _ = gemma_model_and_tok
        n_layers = len(model.model.layers)
        assert n_layers == 26, f"Expected 26 layers, got {n_layers}"

    def test_hidden_size(self, gemma_model_and_tok):
        model, _ = gemma_model_and_tok
        assert model.args.hidden_size == 2304, (
            f"Expected hidden_size=2304, got {model.args.hidden_size}"
        )

    def test_has_logit_softcapping(self, gemma_model_and_tok):
        model, _ = gemma_model_and_tok
        assert hasattr(model, "final_logit_softcapping"), (
            "Gemma-2 must have final_logit_softcapping"
        )
