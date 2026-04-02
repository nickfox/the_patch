---
name: LatentMAS Implementation Plan
description: Complete plan for training-free tensor communication using LatentMAS approach on MLX/Gemma-2-2B. Replaces all Phase 4 training.
type: project
---

# LatentMAS Implementation Plan for SoulMCP


## What: LatentMAS Core Mechanism

Paper: "Latent Collaboration in Multi-Agent Systems" (arXiv:2511.20639v2)
Code: https://github.com/Gen-Verse/LatentMAS
Key file: `models.py` — complete implementation in ~400 lines

### Three operations, zero training:

**1. Alignment matrix W_a (computed once at init)**
```python
W_in = model.get_input_embeddings().weight   # [vocab, D]
W_out = model.get_output_embeddings().weight  # [vocab, D]
gram = W_out.T @ W_out + 1e-5 * I
W_a = solve(gram, W_out.T @ W_in)  # [D, D]
target_norm = W_in.norm(dim=1).mean()
```
For Gemma-2: embeddings are tied (W_in == W_out), so W_a ≈ identity. Main operation is norm-matching.

**2. Latent thought loop (per agent)**
```python
outputs = model(input_ids, use_cache=True, output_hidden_states=True)
kv_cache = outputs.past_key_values
h = outputs.hidden_states[-1][:, -1, :]  # last token, last layer

for step in range(latent_steps):  # m ∈ {0, 10, 20, 40, 80}
    aligned = h @ W_a
    aligned = aligned * (target_norm / aligned.norm())
    outputs = model(inputs_embeds=aligned.unsqueeze(1), past_key_values=kv_cache, ...)
    kv_cache = outputs.past_key_values
    h = outputs.hidden_states[-1][:, -1, :]
```

**3. Agent transfer via KV cache**
Agent A's KV cache (all layers) → prepend to Agent B's KV cache. B attends to everything A "thought." Only final agent decodes to text tokens.

## How: MLX Implementation

### KV Cache on MLX — Proven by genlm-backend

Files the user provided: `/Users/nickfox137/Downloads/cache.py` and `/Users/nickfox137/Downloads/mlx.py` from genlm-backend library.

**Extract KV from cache** (genlm `add_to_cache`, lines 241-253):
```python
keys = [c.keys[i, :, start:end, :] for c in prompt_cache]
values = [c.values[i, :, start:end, :] for c in prompt_cache]
keys_values = mx.stack([mx.stack(keys), mx.stack(values)], axis=0)
```

**Restore KV into cache** (genlm `_process_kv`, lines 313-317):
```python
cache.keys = padded_pasts[0, i]
cache.values = padded_pasts[1, i]
cache.offset += cached_len
```

### MLX Gemma-2 KV Cache Details

Location: `/Users/nickfox137/Documents/soulmcp/venv/lib/python3.13/site-packages/mlx_lm/models/cache.py`

- `KVCache` class: stores `self.keys`, `self.values` shape [B, n_kv_heads, seq_len, head_dim], `self.offset`
- `make_prompt_cache(model)` creates list of KVCache objects (one per layer, 26 for Gemma-2)
- `cache.update_and_fetch(keys, values)` — used by attention to grow cache
- `cache.state` property — getter returns (keys, values) trimmed to offset, setter loads state
- Gemma-2 uses standard KVCache (NOT hybrid like Qwen3)

### MLX Gemma-2 Model Internals

Location: `/Users/nickfox137/Documents/soulmcp/venv/lib/python3.13/site-packages/mlx_lm/models/gemma2.py`

```python
class GemmaModel:
    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0], return_array=True)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)
```

- `embed_tokens`: nn.Embedding(256128, 2304)
- `embed_tokens.as_linear(h)`: output projection (tied weights)
- 26 layers, each takes `(x, mask, cache)`
- Attention uses `cache.offset` for RoPE position, `cache.update_and_fetch` for K/V accumulation
- `create_attention_mask(h, cache[0], return_array=True)` handles causal mask with cache offset

### Implementation Steps

**Step 1: Alignment matrix** — new file `models/latent_comm.py`
- Load Gemma-2 embed_tokens weight
- Compute W_a via ridge regression (or just use identity + norm matching since tied)
- Compute target_norm = mean row norm of embed_tokens.weight
- This is a one-time computation, ~5 seconds

**Step 2: Latent forward pass** — function in `models/latent_comm.py`
- Takes embeddings (not token IDs), passes through Gemma-2 layers with cache
- Must handle: embedding scaling (* sqrt(hidden_size)), mask creation with cache offset, RoPE offset
- Returns: last-layer hidden states, updated cache
- Key: bypass `embed_tokens` call, feed embeddings directly into layer loop

**Step 3: Latent thought generation** — function in `models/latent_comm.py`
- Input: token_ids (the question/prompt)
- Process:
  1. Create cache via `make_prompt_cache(model)` (import from mlx_lm.models.cache)
  2. Forward pass through model with cache to get initial hidden state
  3. Loop m times: align h → feed back as embedding → get new h
  4. Return: final cache (= working memory), final hidden state
- Output: list of KVCache objects containing all K/V states from input + latent steps

**Step 4: KV cache transfer** — function in `models/latent_comm.py`
- Input: Agent A's cache list, Agent B's fresh cache list
- Process: for each layer, set B's cache.keys/values to A's, set offset
- Follow genlm pattern for padding/alignment
- Output: Agent B's cache pre-loaded with A's working memory

**Step 5: Text decoding** — only the final agent does this
- Standard autoregressive generation using Gemma-2
- KV cache already contains all previous agents' "thoughts"
- Use mlx-lm's existing `generate_step` or manual loop

**Step 6: Integration test** — `tests/test_latent_comm.py`
- Encode a question → latent thoughts → decode answer
- Single agent first (no transfer), verify coherent output
- Then two agents with KV transfer

### Critical Details

1. **RoPE positions must be correct.** When feeding latent embeddings, the cache.offset must increment properly so RoPE assigns correct positions. The genlm code handles this via `cache.offset += cached_len`.

2. **Mask must include cached positions.** `create_attention_mask(h, cache[0])` uses cache[0].offset to create the right mask size. This should work automatically if cache.offset is set correctly.

3. **Embedding scaling.** Gemma-2 multiplies embeddings by sqrt(hidden_size) = sqrt(2304) ≈ 48. The aligned hidden states should either be pre-scaled or the forward loop should handle scaling. LatentMAS's norm-matching may handle this implicitly.


### What NOT to Do

- Do NOT train anything. This is inference-only code.
- Do NOT build cross-attention adapters, Talkers, or gates.
- Do NOT guess at implementations — read the LatentMAS source code and genlm-backend source code.

### Reference Files

- LatentMAS paper: `tensor-communication-1.pdf`
- LatentMAS source: https://github.com/Gen-Verse/LatentMAS (models.py is the key file)
- genlm-backend cache: `cache.py`
- genlm-backend MLX wrapper: `mlx.py`
- MLX Gemma-2 model: `gemma2.py`
- MLX KVCache: `venv/.../mlx_lm/models/cache.py`

