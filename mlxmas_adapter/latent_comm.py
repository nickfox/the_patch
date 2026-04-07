"""
LatentMAS core operations for MLX.

Three operations, zero training:
1. Alignment matrix W_a (computed once at init)
2. Latent thought loop (per agent)
3. Agent transfer via KV cache
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache


def compute_alignment(model) -> tuple:
    """Compute alignment matrix W_a and target norm.

    For tied embeddings (Qwen3-4B), W_a ≈ identity.
    Main operation is norm-matching.

    Handles quantized models by passing tokens through embed_tokens
    to get dequantized embeddings.

    Returns:
        (W_a, target_norm) where W_a is [D, D] and target_norm is scalar
    """
    # Get actual embedding dimension (handles quantized models)
    test_emb = model.model.embed_tokens(mx.array([[0]]))
    D = test_emb.shape[-1]

    # Sample embeddings through the actual embedding layer
    # (dequantizes for quantized models)
    n_samples = 10000
    sample_ids = mx.array([list(range(n_samples))])
    E = model.model.embed_tokens(sample_ids)[0].astype(mx.float32)  # [n_samples, D]
    mx.eval(E)

    # For tied embeddings, W_a ≈ identity — just compute target norm
    if model.args.tie_word_embeddings:
        W_a = mx.eye(D)
        target_norm = mx.mean(mx.linalg.norm(E, axis=1))
        mx.eval(W_a, target_norm)
        return W_a, target_norm

    # Untied: need actual solve with output embeddings
    # Sample through lm_head weights too
    W_out_sample = model.lm_head.weight[:n_samples].astype(mx.float32)
    mx.eval(W_out_sample)

    gram = W_out_sample.T @ W_out_sample
    reg = 1e-5 * mx.eye(gram.shape[0])
    gram = gram + reg
    rhs = W_out_sample.T @ E
    mx.eval(gram, rhs)

    # linalg.solve requires CPU stream in MLX
    W_a = mx.linalg.solve(gram, rhs, stream=mx.cpu)
    mx.eval(W_a)

    target_norm = mx.mean(mx.linalg.norm(E, axis=1))
    mx.eval(target_norm)

    return W_a, target_norm


def apply_realignment(hidden: mx.array, W_a: mx.array, target_norm: mx.array,
                       use_realign: bool = True) -> mx.array:
    """Project hidden state back toward input embedding manifold.

    Args:
        hidden: [D] or [1, D] last hidden state from model
        W_a: [D, D] alignment matrix
        target_norm: scalar target embedding norm
        use_realign: if False, just do norm-matching (W_a = identity)
    """
    h = hidden.astype(mx.float32)
    if use_realign:
        aligned = h @ W_a
    else:
        aligned = h
    norm = mx.linalg.norm(aligned, axis=-1, keepdims=True)
    norm = mx.maximum(norm, mx.array(1e-6))
    aligned = aligned * (target_norm / norm)
    return aligned.astype(hidden.dtype)


def latent_forward(model, tokenizer, prompt_text: str, cache: list,
                   latent_steps: int, W_a: mx.array, target_norm: mx.array,
                   use_realign: bool = True) -> list:
    """Run prompt through model and perform latent thought steps.

    1. Tokenize and forward pass to get hidden states + KV cache
    2. Loop latent_steps times: realign hidden → feed as embedding → new hidden
    3. Return updated cache with all latent thoughts stored

    Args:
        model: MLX Qwen3 model (the full Model, not model.model)
        tokenizer: tokenizer
        prompt_text: the agent's prompt string
        cache: list of KVCache objects (one per layer)
        latent_steps: number of latent thinking steps
        W_a: alignment matrix [D, D]
        target_norm: target embedding norm
        use_realign: whether to apply full realignment or just norm-match

    Returns:
        cache: updated KV cache list containing prompt + latent thoughts
    """
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = mx.array([tokens])  # [1, seq_len]

    # Forward pass through transformer layers (not lm_head)
    # model.model returns normalized hidden states [B, seq, D]
    hidden = model.model(input_ids, cache=cache)
    mx.eval(hidden, *[c.state for c in cache])

    # Extract last token hidden state: [1, D]
    last_hidden = hidden[:, -1:, :]

    # Latent thought loop
    for step in range(latent_steps):
        # Realign hidden state toward embedding manifold
        aligned = apply_realignment(last_hidden, W_a, target_norm, use_realign)

        # Feed aligned embedding through transformer (bypasses embed_tokens)
        hidden = model.model(None, cache=cache, input_embeddings=aligned)
        mx.eval(hidden, *[c.state for c in cache])

        # Update last hidden for next step
        last_hidden = hidden[:, -1:, :]

    return cache


def generate_with_cache(model, tokenizer, prompt_text: str, cache: list,
                        max_tokens: int = 2048, temperature: float = 0.6,
                        top_p: float = 0.95) -> str:
    """Generate text with pre-populated KV cache from previous agents.

    Tokenizes the judger prompt, runs it through the model with the
    accumulated cache, then autoregressively generates tokens.

    Args:
        model: MLX Qwen3 model
        tokenizer: tokenizer
        prompt_text: the judger's prompt
        cache: list of KVCache containing previous agents' thoughts
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold

    Returns:
        generated text string
    """
    from mlx_lm.sample_utils import make_sampler

    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = mx.array([tokens])  # [1, seq_len]

    # Prefill: run prompt through model with existing cache
    logits = model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    # Sample from last position
    sampler = make_sampler(temp=temperature, top_p=top_p)
    generated_tokens = []

    # Get logits for last token
    next_logits = logits[:, -1, :]

    eos_token = tokenizer.eos_token_id

    for _ in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()

        if token_id == eos_token:
            break

        generated_tokens.append(token_id)

        # Feed token back through model
        next_input = mx.array([[token_id]])
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return tokenizer.decode(generated_tokens)
