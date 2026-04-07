"""
LatentMAS core operations for MLX.

Single-pass sender→receiver via KV cache transfer.
No alignment matrix, no latent thinking loop.
"""

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def generate_with_cache(model, tokenizer, prompt_text: str, cache: list,
                        max_tokens: int = 2048, temperature: float = 0.6,
                        top_p: float = 0.95) -> str:
    """Generate text with pre-populated KV cache from sender.

    Tokenizes the receiver prompt, runs it through the model with the
    sender's cache, then autoregressively generates tokens.

    Args:
        model: MLX model
        tokenizer: tokenizer
        prompt_text: the receiver's prompt (no question text)
        cache: list of KVCache containing sender's context
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold

    Returns:
        generated text string
    """
    from mlx_lm.sample_utils import make_sampler

    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = mx.array([tokens])

    logits = model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    sampler = make_sampler(temp=temperature, top_p=top_p)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = tokenizer.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)
        next_input = mx.array([[token_id]])
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return tokenizer.decode(generated_tokens)
