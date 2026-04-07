"""
Cross-model latent communication.

Sender does a single forward pass → hidden states are projected
into receiver's space → receiver generates from latents only.

Gemma-2's MLX implementation doesn't support input_embeddings,
so we manually replicate the forward pass.
"""

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache


def gemma_forward_with_embeddings(model, input_embeddings, cache):
    """Run Gemma-2 forward pass with pre-computed embeddings.

    Gemma-2's MLX model doesn't have input_embeddings parameter,
    so we manually replicate the forward pass, bypassing embed_tokens.
    Gemma-2 scales embeddings by sqrt(hidden_size) — we apply this.
    Also applies Gemma-2's logit soft-capping.
    """
    h = input_embeddings * (model.model.args.hidden_size ** 0.5)

    if cache is None:
        cache = [None] * len(model.model.layers)

    mask = create_attention_mask(h, cache[0], return_array=True)

    for layer, c in zip(model.model.layers, cache):
        h = layer(h, mask, c)

    h = model.model.norm(h)

    # Tied embeddings + logit soft-capping (Gemma-2 specific)
    out = model.model.embed_tokens.as_linear(h)
    out = mx.tanh(out / model.final_logit_softcapping)
    out = out * model.final_logit_softcapping

    return out


def gemma_forward_from_layer(model, hidden_states, cache, start_layer=0):
    """Run Gemma forward pass starting from an intermediate layer.

    Args:
        model: Gemma-2 MLX model (the full Model, not model.model)
        hidden_states: [B, seq, D] tensor to inject
        cache: list of KVCache objects (one per layer, all 26)
        start_layer: which Gemma layer to begin processing at

    Returns:
        logits: [B, seq, vocab] output logits

    When start_layer == 0: applies sqrt(hidden_size) embedding scaling
    (standard Gemma behavior for raw embeddings).

    When start_layer > 0: skips scaling — intermediate representations
    are already at operating scale. Uses cache[start_layer] for attention
    mask computation since cache[0..start_layer-1] are empty.

    IMPORTANT: When start_layer > 0, ALL subsequent operations on this
    cache must also use start_layer > 0 to avoid cache offset mismatch.
    """
    if start_layer == 0:
        h = hidden_states * (model.model.args.hidden_size ** 0.5)
    else:
        h = hidden_states

    if cache is None:
        cache = [None] * len(model.model.layers)

    mask = create_attention_mask(h, cache[start_layer], return_array=True)

    for layer, c in zip(model.model.layers[start_layer:], cache[start_layer:]):
        h = layer(h, mask, c)

    h = model.model.norm(h)

    out = model.model.embed_tokens.as_linear(h)
    out = mx.tanh(out / model.final_logit_softcapping)
    out = out * model.final_logit_softcapping

    return out


def _gemma_forward_per_layer_mask(model, h, cache):
    """Gemma forward pass with per-layer attention masks.

    Required when cache offsets differ across layers (e.g., after
    intermediate-layer injection where layers 0..k-1 have offset=0
    but layers k..end have offset=N).

    Args:
        model: Gemma-2 MLX model
        h: [B, seq, D] already embedded and scaled (sqrt(hidden_size) applied)
        cache: list of KVCache (one per layer, all 26)

    Returns:
        logits: [B, seq, vocab]
    """
    for layer, c in zip(model.model.layers, cache):
        mask = create_attention_mask(h, c, return_array=True)
        h = layer(h, mask, c)

    h = model.model.norm(h)
    out = model.model.embed_tokens.as_linear(h)
    out = mx.tanh(out / model.final_logit_softcapping)
    out = out * model.final_logit_softcapping
    return out


def generate_with_cross_latents_from_layer(
    receiver_model, receiver_tokenizer,
    projected_embeddings,
    start_layer=0, max_tokens=2048, temperature=0.6,
):
    """Feed projected latent embeddings to receiver and generate answer.

    Pure latent-only: the receiver gets NO question text, only the
    projected latent states.

    When start_layer > 0:
    - Latent states are injected at layers [start_layer..end] only
    - Subsequent tokens go through ALL 26 layers with per-layer masks
      (layers 0..start_layer-1 process tokens independently; layers
      start_layer..end attend to both injected states and tokens)

    Args:
        receiver_model: Gemma-2 MLX model
        receiver_tokenizer: Gemma tokenizer
        projected_embeddings: [B, seq, D] projected latent states
        start_layer: which Gemma layer to inject at (0 = baseline)
        max_tokens: max tokens to generate
        temperature: sampling temperature
    """
    from mlx_lm.sample_utils import make_sampler

    cache = make_prompt_cache(receiver_model)

    # Step 1: Feed projected latent thoughts at start_layer
    logits = gemma_forward_from_layer(
        receiver_model, projected_embeddings, cache, start_layer
    )
    # Only eval cache entries that were written to
    mx.eval(logits, *[c.state for c in cache[start_layer:]])
    print(f"    Latent prefill done, cache offset: {cache[start_layer].offset}", flush=True)

    # Step 2: Feed judger prompt through all layers
    judger_prompt = (
        f"<start_of_turn>user\n"
        f"Using the reasoning context provided, solve the problem step by step. "
        f"Put your final numerical answer inside \\boxed{{}}.\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    tokens = receiver_tokenizer.encode(judger_prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    if start_layer == 0:
        logits = receiver_model(input_ids, cache=cache)
    else:
        # Per-layer masks: layers 0..start_layer-1 have empty caches,
        # layers start_layer..end have injected states in cache
        h = receiver_model.model.embed_tokens(input_ids)
        h = h * (receiver_model.model.args.hidden_size ** 0.5)
        logits = _gemma_forward_per_layer_mask(receiver_model, h, cache)
    mx.eval(logits, *[c.state for c in cache])
    print(f"    Judger prompt prefilled, cache offset: {cache[start_layer].offset}", flush=True)

    # Step 3: Autoregressive generation (all layers, per-layer masks)
    sampler = make_sampler(temp=temperature, top_p=0.95)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = receiver_tokenizer.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        next_input = mx.array([[token_id]])
        if start_layer == 0:
            next_logits = receiver_model(next_input, cache=cache)
        else:
            h = receiver_model.model.embed_tokens(next_input)
            h = h * (receiver_model.model.args.hidden_size ** 0.5)
            next_logits = _gemma_forward_per_layer_mask(receiver_model, h, cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return receiver_tokenizer.decode(generated_tokens)



# ============================================================
# Gemma→Qwen pipeline: Gemma sender, Qwen receiver
# ============================================================


def extract_gemma_all_tokens(model, tokenizer, prompt_text, layer=10):
    """Run Gemma forward, capture all-token hidden states at a specific layer.

    Manually iterates through Gemma layers. Applies sqrt(hidden_size)
    scaling at the start (standard Gemma embedding behavior). Captures
    raw residual-stream states — no RMSNorm, no lm_head, no logit
    soft-capping.

    The full forward pass completes (all 26 layers) even though we only
    capture at one layer, to keep memory clean.

    Args:
        model: Gemma-2 MLX model (the full Model, not model.model)
        tokenizer: Gemma tokenizer
        prompt_text: the question to process
        layer: which Gemma layer to extract at (0-indexed)

    Returns:
        [1, seq_len, D] raw residual-stream hidden states (D=2304 for Gemma-2-2B)
    """
    from mlx_lm.models.base import create_attention_mask

    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)
    h = model.model.embed_tokens(input_ids)
    h = h * (model.model.args.hidden_size ** 0.5)

    mask = create_attention_mask(h, cache[0], return_array=True)

    captured = None
    for i, (layer_module, c) in enumerate(zip(model.model.layers, cache)):
        h = layer_module(h, mask, c)
        if i == layer:
            captured = h.astype(mx.float32)  # [1, seq_len, D]

    # Eval everything for clean memory
    mx.eval(h, captured)
    return captured


def qwen_forward_from_layer(model, hidden_states, cache, start_layer=0):
    """Run Qwen forward pass starting from an intermediate layer.

    Args:
        model: Qwen3 MLX model (the full Model, not model.model)
        hidden_states: [B, seq, D] tensor to inject
        cache: list of KVCache objects (one per layer, all 36)
        start_layer: which Qwen layer to begin processing at

    Returns:
        logits: [B, seq, vocab]

    Qwen3 does NOT apply sqrt(hidden_size) scaling to embeddings,
    so there is no conditional scaling here (unlike Gemma).

    When start_layer > 0: skips layers 0..start_layer-1. Uses
    cache[start_layer] for attention mask since earlier caches are empty.

    IMPORTANT: When start_layer > 0, ALL subsequent operations on this
    cache must also account for the offset mismatch (use per-layer masks).
    """
    h = hidden_states

    if cache is None:
        cache = [None] * len(model.model.layers)

    mask = create_attention_mask(h, cache[start_layer], return_array=True)

    for layer, c in zip(model.model.layers[start_layer:], cache[start_layer:]):
        h = layer(h, mask, c)

    h = model.model.norm(h)

    # Qwen3-4B has tied embeddings
    if model.args.tie_word_embeddings:
        out = model.model.embed_tokens.as_linear(h)
    else:
        out = model.lm_head(h)

    return out


def _qwen_forward_per_layer_mask(model, h, cache):
    """Qwen forward pass with per-layer attention masks.

    Required when cache offsets differ across layers (e.g., after
    intermediate-layer injection where layers 0..k-1 have offset=0
    but layers k..end have offset=N).

    Args:
        model: Qwen3 MLX model
        h: [B, seq, D] input (already embedded, no scaling needed)
        cache: list of KVCache (one per layer, all 36)

    Returns:
        logits: [B, seq, vocab]
    """
    for layer, c in zip(model.model.layers, cache):
        mask = create_attention_mask(h, c, return_array=True)
        h = layer(h, mask, c)

    h = model.model.norm(h)

    if model.args.tie_word_embeddings:
        out = model.model.embed_tokens.as_linear(h)
    else:
        out = model.lm_head(h)

    return out


def generate_with_qwen_from_layer(
    receiver_model, receiver_tokenizer,
    projected_embeddings,
    start_layer=0, max_tokens=2048, temperature=0.6,
):
    """Inject projected Gemma states into Qwen and generate answer.

    Pure latent-only: the receiver gets NO question text, only the
    projected latent states. This is a true test of whether latent
    communication carries sufficient semantic content.

    Pipeline:
    1. Create fresh cache for Qwen (36 layers)
    2. Feed projected embeddings through Qwen layers [start_layer..35]
    3. Feed the judger prompt through ALL 36 layers with per-layer masks
    4. Generate text autoregressively

    Args:
        receiver_model: Qwen3 MLX model
        receiver_tokenizer: Qwen3 tokenizer
        projected_embeddings: [B, seq, D] projected latent states
        start_layer: which Qwen layer to inject at (0 = baseline)
        max_tokens: max tokens to generate
        temperature: sampling temperature
    """
    from mlx_lm.sample_utils import make_sampler

    cache = make_prompt_cache(receiver_model)

    # Step 1: Feed projected latent thoughts at start_layer
    logits = qwen_forward_from_layer(
        receiver_model, projected_embeddings, cache, start_layer
    )
    mx.eval(logits, *[c.state for c in cache[start_layer:]])
    print(f"    Latent prefill done, cache offset: {cache[start_layer].offset}", flush=True)

    # Step 2: Feed judger prompt through all layers
    # Qwen3 chat template format
    judger_prompt = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Using the reasoning context provided, solve the problem step by step. "
        f"Put your final numerical answer inside \\boxed{{}}.\n"
        f"<|im_end|>\n<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )
    tokens = receiver_tokenizer.encode(judger_prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    if start_layer == 0:
        logits = receiver_model(input_ids, cache=cache)
    else:
        # Per-layer masks: layers 0..start_layer-1 have empty caches,
        # layers start_layer..end have injected states in cache
        h = receiver_model.model.embed_tokens(input_ids)
        # Qwen3 does NOT scale embeddings by sqrt(hidden_size)
        logits = _qwen_forward_per_layer_mask(receiver_model, h, cache)
    mx.eval(logits, *[c.state for c in cache])
    print(f"    Judger prompt prefilled, cache offset: {cache[start_layer].offset}", flush=True)

    # Step 3: Autoregressive generation (all layers, per-layer masks)
    sampler = make_sampler(temp=temperature, top_p=0.95)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = receiver_tokenizer.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)

        next_input = mx.array([[token_id]])
        if start_layer == 0:
            next_logits = receiver_model(next_input, cache=cache)
        else:
            h = receiver_model.model.embed_tokens(next_input)
            next_logits = _qwen_forward_per_layer_mask(receiver_model, h, cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return receiver_tokenizer.decode(generated_tokens)
