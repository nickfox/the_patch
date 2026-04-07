---
title: MLX LatentMAS v0.1 Architecture Document
author: Nick Fox
date: 2026-04-02
status: Working Implementation
---

# MLX LatentMAS v0.1

## Overview

MLX LatentMAS is a native Apple Silicon implementation of latent multi-agent reasoning, based on the paper "Latent Collaboration in Multi-Agent Systems" (arXiv:2511.20639). Multiple AI agents collaborate on a problem by passing hidden states through KV cache rather than generating and re-encoding text tokens. Only the final agent produces text output.

This implementation runs on MLX (Apple's machine learning framework) using Qwen3-4B as the backbone model. It was built alongside a parallel PyTorch/MPS implementation of the same algorithm for comparison.

### Key Properties

- **Training-free**: No fine-tuning, adapters, or learned parameters. The alignment matrix is computed from the model's existing embedding weights.
- **Zero inter-agent tokens**: Planner, Critic, and Refiner produce no text. Their "reasoning" exists only as KV cache states.
- **Native Apple Silicon**: Runs entirely on MLX Metal backend. No PyTorch, no CUDA translation layer.

### Verified Results

| Configuration | Accuracy | Time/Sample | Memory |
|---|---|---|---|
| MLX 4-bit, sequential, latent_steps=10 | 5/5 (100%) | 13.4s | 5.6 GB |
| MLX 4-bit, hierarchical, latent_steps=10 | 2/5 (40%) | 12.2s | 5.6 GB |
| MLX cross-model Qwen→Gemma, sequential | 1/5 (20%) | 5.2s | ~4 GB |
| PyTorch fp16/MPS, sequential, latent_steps=10 | 5/5 (100%) | 64.1s | 13.2 GB |


## Architecture

### File Structure

```
mlxmas/
├── run.py           # Entry point: model loading, evaluation loop, CLI args
├── latent_comm.py   # Core algorithm: alignment, latent thoughts, KV transfer
├── prompts.py       # Agent prompt builders (sequential + hierarchical)
├── utils.py         # Answer extraction (boxed format), normalization
├── cross_align.py   # Cross-model alignment matrix computation and projection
└── test_cross.py    # Cross-model test: Qwen→Gemma latent communication
```

Total: ~650 lines. No dependencies beyond `mlx`, `mlx-lm`, `datasets`.

### Agent Pipeline

Two modes are supported:

**Sequential** (iterative refinement):
```
Planner → Critic → Refiner → Judger
  [latent]  [latent]  [latent]  [text output]
```

**Hierarchical** (committee of experts):
```
Math Agent → Science Agent → Code Agent → Task Summarizer
  [latent]      [latent]       [latent]      [text output]
```

In both modes, the first three agents encode their prompts and perform latent thought steps. Their KV cache accumulates into a shared context. The final agent (Judger / Task Summarizer) receives this accumulated cache and generates text.


## Core Algorithm

### Operation 1: Alignment Matrix (computed once at init)

**File**: `latent_comm.py` → `compute_alignment(model)`

The alignment matrix $W_a$ maps from the model's output hidden space back to its input embedding space. This is needed because the latent thought loop feeds hidden states back as pseudo-embeddings — they must land on the right manifold for the model to interpret them.

**Computation**:
```
W_in  = embed_tokens.weight          # [vocab_size, D]  (151936 × 2560 for Qwen3-4B)
W_out = lm_head.weight               # [vocab_size, D]  (same as W_in when tied)

gram = W_out^T @ W_out + λI           # [D, D] regularized Gram matrix
W_a  = solve(gram, W_out^T @ W_in)   # [D, D] least-squares alignment

target_norm = mean(‖W_in[i]‖)        # scalar: average embedding row norm
```

**Qwen3-4B specifics**: Embeddings are tied (`tie_word_embeddings: True`), meaning `W_in == W_out`. This makes `W_a ≈ I` (identity). The dominant operation is norm-matching — rescaling the aligned vector so its magnitude matches the typical embedding norm.

**MLX detail**: `mx.linalg.solve` requires `stream=mx.cpu` because the GPU kernel is not yet implemented. This is a one-time cost (~2 seconds) at startup.


### Operation 2: Latent Thought Loop (per agent)

**File**: `latent_comm.py` → `latent_forward(model, tokenizer, prompt, cache, latent_steps, W_a, target_norm)`

Each latent agent executes the following:

**Step 2a — Encode the prompt**:
```python
tokens = tokenizer.encode(prompt_text)
input_ids = mx.array([tokens])                        # [1, seq_len]
hidden = model.model(input_ids, cache=cache)           # [1, seq_len, 2560]
```

This calls `Qwen3Model.__call__` directly (bypassing the lm_head projection), returning the final-layer normalized hidden states. The KV cache is updated in-place — every layer's key/value tensors grow by `seq_len` positions.

**Step 2b — Latent thinking loop**:
```python
last_hidden = hidden[:, -1:, :]                        # [1, 1, 2560]

for step in range(latent_steps):
    aligned = apply_realignment(last_hidden, W_a, target_norm)
    hidden = model.model(None, cache=cache, input_embeddings=aligned)
    last_hidden = hidden[:, -1:, :]
```

Each iteration:
1. Projects the last hidden state back toward the embedding manifold via $W_a$
2. Rescales to `target_norm` (norm-matching)
3. Feeds it as `input_embeddings` — Qwen3's MLX model natively accepts this parameter, bypassing `embed_tokens` entirely
4. The model processes this pseudo-token through all 36 transformer layers
5. KV cache grows by 1 position per step (RoPE positions increment correctly via `cache.offset`)

After `latent_steps` iterations, the cache contains `seq_len + latent_steps` positions: the original prompt tokens plus the latent "thoughts."


### Operation 3: Agent Transfer via KV Cache

Agents share context by reusing the same KV cache list. The cache is initialized once per question via `make_prompt_cache(model)` (creates 36 `KVCache` objects, one per transformer layer). Each agent's `latent_forward` call appends to this shared cache:

```
Agent 1 (Planner):   cache grows from 0 → ~139 positions  (prompt + 10 latent steps)
Agent 2 (Critic):    cache grows from 139 → ~278 positions
Agent 3 (Refiner):   cache grows from 278 → ~417 positions
Agent 4 (Judger):    reads all 417 positions as context, generates text
```

The Judger's prompt is tokenized and prefilled into the model with the accumulated cache. It then attends to everything all previous agents "thought" — the original prompts AND the latent thought steps — through standard causal attention. Only the Judger's autoregressive output is decoded to tokens.

### Operation 4: Text Generation (final agent only)

**File**: `latent_comm.py` → `generate_with_cache(model, tokenizer, prompt, cache, ...)`

Standard autoregressive generation:
1. Prefill the judger's prompt tokens through the full model (including lm_head) with the accumulated cache
2. Sample from the last position's logits using temperature + top-p nucleus sampling
3. Feed each sampled token back through the model, extending the cache by 1 position per step
4. Stop on EOS token or `max_tokens` limit

The `mlx_lm.sample_utils.make_sampler` handles the sampling logic (temperature scaling, top-p filtering).


## Model Architecture: Qwen3-4B on MLX

### Model Constants

| Parameter | Value |
|---|---|
| hidden_size | 2560 |
| num_hidden_layers | 36 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA, 4:1 ratio) |
| head_dim | 128 |
| vocab_size | 151936 |
| intermediate_size | 9728 |
| tie_word_embeddings | True |
| sliding_window | None (full causal attention, all layers) |
| embedding_scaling | None (unlike Gemma-2 which uses √hidden_size) |

### MLX Model Forward Pass

The Qwen3 MLX implementation (`mlx_lm/models/qwen3.py`) has a critical feature that makes LatentMAS clean to implement:

```python
class Qwen3Model(nn.Module):
    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings           # ← latent path: bypass embed_tokens
        else:
            h = self.embed_tokens(inputs)  # ← normal token path

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)
```

The `input_embeddings` parameter allows injecting arbitrary tensors into the transformer stack without going through the embedding layer. This is exactly what the latent thought loop needs — the aligned hidden states from `apply_realignment` are fed directly as `input_embeddings`.

No embedding scaling is applied (unlike Gemma-2 which multiplies by √2304 ≈ 48). The norm-matching in `apply_realignment` handles magnitude alignment.


### KV Cache Structure

Each `KVCache` object (from `mlx_lm.models.cache`) stores:
- `keys`: `[1, n_kv_heads, seq_len, head_dim]` = `[1, 8, seq_len, 128]`
- `values`: `[1, n_kv_heads, seq_len, head_dim]` = `[1, 8, seq_len, 128]`
- `offset`: integer tracking the current sequence position (used for RoPE and masking)

There are 36 KVCache objects (one per transformer layer). The `make_prompt_cache(model)` function creates them all at once.

Key properties:
- `cache.update_and_fetch(keys, values)` — called internally by attention layers to grow the cache
- `cache.offset` — auto-increments as tokens are processed; drives RoPE position encoding
- `cache.state` — getter/setter for (keys, values) tuple, used for `mx.eval()` materialization
- `create_attention_mask(h, cache[0])` — automatically builds the correct causal mask accounting for cached positions

### Qwen3-Specific Attention Features

Qwen3 applies **QK norms** (RMSNorm on queries and keys) inside each attention layer:
```python
queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1))
keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1))
```

This normalizes the attention inputs at every layer, which arguably makes latent communication more stable — the hidden state representations are being regularized throughout the stack.


## The Realignment Step

### Why It's Needed

The transformer's hidden states live in a different geometric space than its input embeddings. After 36 layers of attention and MLP transformations, the last hidden state has drifted far from the manifold where `embed_tokens` places its outputs. Feeding raw hidden states back as pseudo-embeddings produces garbage.

The alignment matrix $W_a$ computes the least-squares projection from output space back to input space:

$$W_a = (W_{out}^T W_{out} + \lambda I)^{-1} W_{out}^T W_{in}$$

For tied embeddings ($W_{in} = W_{out}$), this simplifies to approximately the identity matrix. The dominant effect is **norm-matching**: the hidden states have a different magnitude than embeddings, and rescaling to `target_norm` (the average embedding row norm) puts them in the right ballpark.

### The `--realign` Flag

- **Without `--realign`** (default): Only norm-matching is applied. The hidden state is rescaled to `target_norm` without matrix multiplication. Faster, works well for tied-embedding models.
- **With `--realign`**: The full $W_a$ matrix is applied before norm-matching. May help for models with untied embeddings where the output→input space mapping is non-trivial.

The paper treats this as a hyperparameter — some task/model combinations benefit from full realignment, others don't.


## MLX vs PyTorch Comparison

Both implementations run the same algorithm on the same model (Qwen3-4B) on the same hardware (Mac Mini M2 Pro, 32GB).

| Aspect | MLX (mlxmas/) | PyTorch (LatentMAS/) |
|---|---|---|
| Framework | mlx + mlx-lm | torch + transformers |
| Backend | Metal (native) | MPS (PyTorch → Metal bridge) |
| Dtype | 4-bit quantized / bf16 | fp16 |
| KV Cache | `KVCache` objects (list of 36) | `DynamicCache` (transformers 5.4) |
| Hidden state access | `model.model(...)` returns hidden states | `output_hidden_states=True` in forward |
| Embedding bypass | Native `input_embeddings` parameter | Native `inputs_embeds` parameter |
| Alignment solve | `mx.linalg.solve(stream=mx.cpu)` | `torch.linalg.solve` |
| Text generation | Manual sample loop | `model.generate()` |
| Speed (4-bit) | ~12s/sample | N/A (4-bit not tested) |
| Speed (fp16) | Not yet tested | ~64s/sample |
| Memory (4-bit) | 5.6 GB | N/A |
| Memory (fp16) | ~13 GB (est.) | 13.2 GB |

The MLX version is architecturally simpler. Qwen3's `input_embeddings` parameter eliminates the need for any model internals hacking. The PyTorch version had to deal with transformers 5.4's `DynamicCache` API changes, `cache_position` conflicts, and deprecated `torch_dtype` parameters.


## Cross-Model Latent Communication

### How It Works

When two different model families need to communicate in latent space, the KV cache can't be shared directly — different architectures have different head counts, head dimensions, and layer counts. Instead, we collect the sender's hidden states, project them through a cross-alignment matrix, and feed them as input embeddings to the receiver.

**Cross-alignment matrix computation:**

1. Find tokens that exist in both vocabularies (25,139 shared tokens between Qwen3 and Gemma-2)
2. Embed each shared token through both models' embedding layers — these are paired anchor points
3. Compute a least-squares projection $W_{cross}: \mathbb{R}^{D_{sender}} \rightarrow \mathbb{R}^{D_{receiver}}$ that minimizes reconstruction error on the anchor pairs
4. Apply norm-matching to the receiver's embedding scale

**File:** `mlxmas/cross_align.py`

### Verified Cross-Model Pipeline

Qwen3-4B (sender) → alignment matrix (2560→2304) → Gemma-2-2B (receiver):

```
Question → Qwen [planner → critic → refiner] → 33 self-aligned hidden states
         → cross-project to Gemma's space (2560 → 2304)
         → Gemma forward pass with projected embeddings
         → Gemma generates text answer
```

Alignment quality: 0.68 cosine similarity (linear projection, no training). The pipeline produces coherent, well-structured text output from Gemma, confirming that the projected latent states are interpretable. However, multi-step reasoning accuracy degrades (1/5 GSM8K) — the 32% directional information loss in the linear projection is enough to disrupt the detailed planning context that multi-step math problems require. Errors are logical (missing a subtraction step, dropping a multiplier), not syntactic — indicating partial but incomplete transfer of reasoning structure. A trained adapter (as in the Interlat paper) would likely improve this significantly.

### MLX Implementation Note: Gemma-2

Gemma-2's MLX model does not accept `input_embeddings`. The cross-model pipeline manually replicates the forward pass, bypassing `embed_tokens` and applying Gemma-2's embedding scaling ($\times \sqrt{2304}$) and logit soft-capping. See `gemma_forward_with_embeddings()` in `test_cross.py`.


## Relevance to SoulMCP

### Language Is Collapse

LatentMAS empirically demonstrates the principle at the core of SoulMCP v0.1.4: **tokenization is lossy compression of rich tensor states**. When agents communicate through tokens, they:

1. Project their internal state down to vocabulary space (lm_head)
2. Sample discrete tokens from that projection
3. Re-embed those tokens for the next agent (embed_tokens)

Each step loses information. The hidden state at the last layer of a 4B parameter transformer lives in a 2560-dimensional space. The vocabulary projection collapses that to a categorical distribution over 151,936 tokens, then selects one. The round-trip through tokenization→detokenization→retokenization discards the geometric structure of the original representation.

LatentMAS skips this collapse entirely. The KV cache carries the full-rank attention state forward — every layer's key and value projections for every position, preserving the relationships that the transformer computed.

### Toward Mother-Child Tensor Communication

SoulMCP v0.1.4 envisions two dockerized instances communicating via tensors rather than text. LatentMAS provides a working proof-of-concept for the communication mechanism:

- **KV cache as message**: Agent A's KV cache IS the message. It contains everything A "thought" — encoded as attention patterns, not as words.
- **Alignment as protocol**: The realignment matrix ensures the message is interpretable by the receiver. For identical models (mother and child running the same backbone), this is approximately identity — the same way two instances of the same architecture share a latent space.
- **Latent steps as deliberation time**: The `latent_steps` parameter controls how much "internal reasoning" each agent does before passing context. This maps to the heartbeat architecture's notion of processing cycles.

The key insight: **you don't need to train anything to get this working**. The model's existing embedding geometry provides the alignment. Training (as in the EBM Hybrid project) would refine and steer the latent communication — but the channel already exists.


## Usage

### Quick Start

```bash
cd /Users/nickfox137/Documents/mlx_latentmas/mlxmas
source ../venv/bin/activate

# Sequential mode (planner/critic/refiner) — recommended
python run.py --model mlx-community/Qwen3-4B-4bit --max_samples 5 --latent_steps 10

# Hierarchical mode (math/science/code specialists)
python run.py --model mlx-community/Qwen3-4B-4bit --max_samples 5 --latent_steps 10 --prompt hierarchical

# Full precision (auto-converts from HuggingFace, ~8GB download)
python run.py --model Qwen/Qwen3-4B --max_samples 5 --latent_steps 10

# With full realignment matrix
python run.py --model mlx-community/Qwen3-4B-4bit --max_samples 5 --latent_steps 10 --realign
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-4B` | HuggingFace model name (or mlx-community quantized) |
| `--task` | `gsm8k` | Evaluation task |
| `--max_samples` | `5` | Number of questions to evaluate (-1 for all) |
| `--latent_steps` | `10` | Latent thinking steps per agent (paper suggests 0–80) |
| `--max_tokens` | `1024` | Max generation length for final agent |
| `--temperature` | `0.6` | Sampling temperature |
| `--prompt` | `sequential` | Agent architecture: `sequential` or `hierarchical` |
| `--realign` | `False` | Apply full W_a matrix (vs norm-match only) |


## Dependencies

- `mlx` >= 0.31.1
- `mlx-lm` >= 0.31.1
- `datasets` (HuggingFace, for GSM8K loading)
- Python 3.13+ (tested on 3.14)
- macOS with Apple Silicon (M1/M2/M3/M4)

### Environment

```
/Users/nickfox137/Documents/mlx_latentmas/venv
```

This venv contains both PyTorch (for the LatentMAS/ directory) and MLX (for mlxmas/). Both paths share the same HuggingFace cache for model weights.

## References

- **LatentMAS paper**: Zou et al., "Latent Collaboration in Multi-Agent Systems," arXiv:2511.20639, 2025
- **LatentMAS source**: https://github.com/Gen-Verse/LatentMAS
- **MLX Qwen3 model**: `mlx_lm/models/qwen3.py` (mlx-lm 0.31.1)
- **MLX KVCache**: `mlx_lm/models/cache.py`
- **genlm-backend**: KV cache extraction/restoration patterns (reference for future multi-process communication)
- **SoulMCP architecture**: `ai_life_architecture_v0_5c.md`
- **Original plan**: `project_latentmas_plan.md` (written for Gemma-2, adapted to Qwen3-4B)

## What's Next

1. **Full precision comparison**: Run `Qwen/Qwen3-4B` (non-quantized) on MLX to get an apples-to-apples accuracy comparison with the PyTorch fp16 run.
2. **Tune latent_steps**: The paper suggests 0–80 range. Higher values may improve reasoning on harder tasks (AIME, GPQA) at the cost of longer KV cache and more compute.
3. **Cross-model pipeline**: Each model takes a different cognitive role — e.g., Qwen as planner → Gemma as critic → Model C as refiner → Model D as judger. Cross-alignment matrices already verified at 0.68 cosine similarity. Infrastructure exists in `cross_align.py` and `test_cross.py`.
4. **Cross-process KV transfer**: Serialize KV cache states and transfer between two separate MLX processes — the precursor to SoulMCP mother-child communication.
5. **EBM integration**: The latent thought loop produces hidden states at every step. These are the exact tensors the EBM Hybrid project trains on — the alignment matrix provides a natural bridge between the two projects.
