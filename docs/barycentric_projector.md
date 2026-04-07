---
title: "Barycentric Ridge Self-Projector: Implementation Spec"
author: Nick Fox
date: 2026-04-05
status: Implementation Spec
---

# Barycentric Ridge Self-Projector for Cross-Model Latent Communication

## The Problem

The cross-model latent communication pipeline has a fundamental bug in the
sender-side projection for tied-embedding models (Qwen3-4B).

**Current broken pipeline:**
1. Qwen processes question → produces hidden states `h_L` (transformer output space)
2. "Self-align" via `W_a = identity` → no-op, states remain in transformer output space
3. Cross-alignment `W_AB` (trained on static embedding vectors) receives transformer output vectors
4. Distribution mismatch → garbage output from receiver

**The root cause:** For tied-embedding models, `compute_alignment()` in `latent_comm.py`
returns `W_a = mx.eye(D)`. This treats identity as an embedding-space projector, but
transformer output states are contextual and anisotropic — they do NOT look like static
embeddings even though they share the same coordinate system.

**Evidence:** Every cross-model configuration produces ~0.68–0.72 cosine similarity
regardless of model pair, alignment method, or direction. This is the accidental
structural overlap between transformer output space and embedding space, not a quality
ceiling of the alignment.

## The Fix

Replace the identity self-alignment with a **barycentric ridge self-projector**:

1. **Exact barycentric projection (teacher):** Use the sender's own tied LM head to
   map hidden states to a probability distribution over the vocabulary, then take a
   weighted sum of the static embeddings. This forces the output onto the embedding
   manifold by construction.

2. **Ridge regression (fast runtime):** Fit a closed-form linear map from hidden states
   to the barycentric teacher targets. One matrix solve, not training. At runtime,
   a single matrix multiply replaces the exact projection.

This was designed by the dragon slayer (gpt-5.4-high) and reviewed by gemini-3.1-pro.


## Working Directory

All code lives in: `/Users/nickfox137/Documents/mlx_latentmas/mlxmas_adapter/`

Uses the venv at: `/Users/nickfox137/Documents/mlx_latentmas/venv/`

Run from the `mlxmas_adapter` directory with `source ../venv/bin/activate`.

## Models (8-bit only, no 4-bit)

- Sender: `mlx-community/Qwen3-4B-8bit` (D=2560, 36 layers, tied embeddings)
- Receiver: `mlx-community/Phi-4-mini-instruct-8bit` (D=3072, 32 layers, tied embeddings)

Memory budget: ~32GB total. Both models + KV caches fit comfortably (~9GB combined).

## Existing Files (DO NOT MODIFY unless specified)

| File | Purpose | Modify? |
|------|---------|---------|
| `cross_align.py` | Cross-model W_AB alignment + `apply_cross_realignment()` | YES — update `apply_cross_realignment` |
| `latent_comm.py` | `compute_alignment()` (the broken identity path), `apply_realignment()` | NO — superseded by new projector |
| `test_cross.py` | Test pipeline: extract → project → inject → generate | YES — rewire to use new projector |
| `utils.py` | `extract_boxed_answer()`, `normalize_answer()` | NO |
| `prompts.py` | Agent prompts (unused in simplified pipeline) | NO |
| `run.py` | Same-model LatentMAS (unused) | NO |


## New File: `self_projector.py`

This is the core new module. It contains:

### Class: `BarycentricRidgeSelfProjector`

Fitted projector that maps sender hidden states → embedding-like vectors.

**Methods:**
- `project_exact(z, E_raw, k=64, tau=0.7)` — Exact barycentric projection.
  Used for debugging and as the teacher during fitting.
  - `z`: `[..., D]` final-norm hidden states from sender
  - `E_raw`: `[V, D]` sender embedding matrix (dequantized, see note below)
  - Returns: `[..., D]` L2-normalized embedding-space vectors
  - Algorithm: `z → logits = z @ E_raw.T → topk → softmax(topk/tau) → weighted sum of L2-normalized E_raw rows`

- `project_fast(z)` — Fast ridge approximation.
  Used at runtime. Single matrix multiply.
  - `z`: `[..., D]` final-norm hidden states
  - Returns: `[..., D]` L2-normalized embedding-space vectors
  - Algorithm: `y = (z - mu_z) @ P + mu_y` then L2-normalize

- `save(path)` / `load(path)` — Save/load to `.npz` file

### Class: `BarycentricRidgeFitter`

Offline fitter. Accumulates sufficient statistics streaming, then solves once.

**Methods:**
- `accumulate(z_batch)` — Feed a batch of final-norm hidden states. Internally:
  1. Computes barycentric teacher targets via `project_exact`
  2. Accumulates `S_zz`, `S_zy`, `sum_z`, `sum_y`, `n`
  3. Calls `mx.eval()` after each chunk to prevent graph growth

- `finalize()` — Solves ridge regression, returns `BarycentricRidgeSelfProjector`
  - Centers: `mu_z = sum_z / n`, `mu_y = sum_y / n`
  - Normalizes: `S_zz_c = S_zz / n - outer(mu_z, mu_z)`
  - Trace-scaled regularization: `alpha = lam * trace(S_zz_c) / D`
  - Solves: `P = solve(S_zz_c + alpha * I, S_zy_c)`


### Top-level function: `fit_self_projector(sender_model, sender_tok, prompts, ...)`

Convenience function that:
1. Gets the dequantized embedding matrix from the sender model
2. Runs the sender on each prompt to get final-norm hidden states
3. Feeds them to the fitter
4. Returns the fitted projector

**Parameters:**
- `sender_model`: MLX model
- `sender_tok`: tokenizer
- `prompts`: list of calibration strings (GSM8K train questions are fine)
- `k`: top-k for barycentric teacher (default 64)
- `tau`: temperature for softmax sharpening (default 0.7)
- `lam`: ridge regularization fraction (default 1e-2)
- `n_prompts`: how many prompts to use (default 200, ~20k-50k tokens total)
- `pos_chunk`: chunk size for vocab projection (default 8, keeps memory bounded)

## CRITICAL: Quantized Embedding Matrix Handling

8-bit quantized MLX models store embedding weights in packed format.
`model.model.embed_tokens.weight` has shape `[V, D_packed]` where `D_packed != D`.

**You MUST dequantize by passing tokens through the embedding layer:**

```python
def get_embedding_matrix(model, chunk_size=4096):
    """Get dequantized embedding matrix [V, D] from a quantized model."""
    V = model.args.vocab_size
    chunks = []
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        ids = mx.array([list(range(start, end))])
        emb = model.model.embed_tokens(ids)[0]  # [chunk, D]
        chunks.append(emb)
        mx.eval(emb)
    E = mx.concatenate(chunks, axis=0)  # [V, D]
    mx.eval(E)
    return E
```

This is the ONLY correct way to get E_raw for quantized models. Do NOT read
`model.model.embed_tokens.weight` directly.


## CRITICAL: Final-Norm Hidden States

The projector operates on **post-final-RMSNorm** hidden states. This is the tensor
that gets multiplied by the tied embedding matrix to produce logits:

```
logits = final_norm(h_L) @ E.T
```

To get this from Qwen:
```python
cache = make_prompt_cache(model)
# model.model() returns post-norm states (this is what Qwen's forward does)
z = model.model(input_ids, cache=cache)  # [1, seq_len, D]
```

Verify: `model.model()` in MLX returns `self.norm(h)` after the layer loop.
This is already the final-normed tensor. Do NOT apply norm again.

## Calibration Data

Use GSM8K training set questions. The prompts should match the domain of the
actual test (math reasoning). Load via:

```python
from datasets import load_dataset
ds = load_dataset("gsm8k", "main", split="train")
prompts = [item["question"].strip() for item in ds][:200]
```

200 prompts at ~50-100 tokens each gives ~10k-20k calibration positions.
The dragon slayer recommends 20k-50k positions total. 200-500 prompts should suffice.

## Softmax-Before-TopK: NOT Needed

Gemini suggested applying full-vocab softmax before top-k selection.
The dragon slayer proved this is mathematically identical to top-k-on-logits
then softmax (renormalization cancels the partition function). Use the cheaper
version: top-k first, then softmax over the k values.


## Changes to Existing Files

### 1. `cross_align.py` — Update `apply_cross_realignment`

The current version L2-normalizes the input and projects through W_AB.
This is correct ONLY if the input is already in embedding space.

After the self-projector fix, the input to `apply_cross_realignment` will be
L2-normalized embedding-space vectors from `project_fast()`. The function should:

1. L2-normalize input (redundant but safe — input is already normalized)
2. Project through W_AB
3. L2-normalize output
4. Scale to receiver's target norm

The current implementation already does this correctly. No changes needed
to `apply_cross_realignment` itself.

### 2. `test_cross.py` — Rewire the Pipeline

Replace `extract_sender_all_tokens()` which currently does:
```
hidden = model.model(input_ids)  # post-norm transformer output
aligned = apply_realignment(hidden, W_a=identity, target_norm)  # no-op
return aligned  # WRONG: still in transformer space
```

With:
```
hidden = model.model(input_ids)  # post-norm transformer output
projected = self_projector.project_fast(hidden)  # → embedding space
return projected  # CORRECT: in embedding space
```

**Full updated pipeline in `main()`:**

```
# Offline (once per sender model):
#   1. Load sender
#   2. Fit self-projector from calibration data
#   3. Save to .npz
#
# Runtime (per question):
#   1. z = sender.model(question)           → post-norm hidden states
#   2. u_A = projector.project_fast(z)      → L2-normalized embedding-space
#   3. e_B = apply_cross_realignment(u_A, W_AB, ...)  → receiver embedding space
#   4. inject e_B into receiver, generate answer
```


## Execution Order

```
1. Create self_projector.py in mlxmas_adapter/
   ↓
2. Create fit_projector.py — CLI script to fit + save the projector
   Usage: python fit_projector.py --n-prompts 200 --output qwen_self_projector.npz
   ↓
3. Modify test_cross.py:
   - Remove extract_sender_all_tokens() (broken)
   - Remove the latent_comm import (W_a identity path, superseded)
   - Load the saved projector at startup
   - Use projector.project_fast(z) in the question loop
   ↓
4. Run fit_projector.py to create qwen_self_projector.npz
   ↓
5. Run test_cross.py --max-samples 5 with --mode exact (uses project_exact)
   This is the gold-standard diagnostic — if this produces coherent output,
   the sender-side projection was the problem.
   ↓
6. Run test_cross.py --max-samples 5 with --mode fast (uses project_fast)
   Compare against exact. Should be close.
   ↓
7. Run test_cross.py --max-samples 50 for full eval
```

## Three Diagnostics (from the dragon slayer)

After implementation, run these three diagnostics to isolate bottlenecks:

### Diagnostic 1: Sender Projector Fidelity
```python
cos(project_fast(z), project_exact(z))
```
Should be 0.9+. If not, ridge fit needs more data or better regularization.

### Diagnostic 2: Static W_AB Ceiling
```python
cos(normalize(E_A @ W_AB), normalize(E_B))
```
On held-out shared tokens. This is the ceiling of the current cross-map.

### Diagnostic 3: End-to-End Linear Ceiling (MOST IMPORTANT)
Refit W_AB on the actual projected vectors:
```python
u_A = project_exact(z_A)   # projected sender states
```
Fit W_AB_new: `u_A → E_B` instead of `E_A → E_B`.

If the cosine jumps past the old ~0.7, the sender mismatch was the bottleneck.


## Memory Considerations

- E_raw (full vocab dequantized): Qwen V=151936, D=2560 → ~1.5GB float32.
  Keep in model dtype (float16/bfloat16) and cast to float32 only for the
  gathered top-k rows. The code in project_exact should:
  1. Compute logits: `z @ E_raw.T` (E_raw stays in model dtype)
  2. Top-k: extract k=64 indices
  3. Gather: `E_raw[idx]` gets [C, 64, D] — small
  4. Cast gathered rows to float32 for normalization and weighted sum

- Ridge statistics: `S_zz` [2560, 2560] + `S_zy` [2560, 2560] = ~50MB float32. Fine.

- pos_chunk=8: Process 8 token positions at a time through the vocab projection.
  Keeps peak memory from the [C, V] logits tensor bounded.

## What Success Looks Like

**If the fix works (sender-side mismatch was the problem):**
- Diagnostic 1: project_fast vs project_exact cosine > 0.90
- Diagnostic 3: refitted W_AB cosine jumps above 0.75
- GSM8K with exact projection: coherent output, some correct answers
- GSM8K with fast projection: similar to exact

**If the fix doesn't help (W_AB or receiver injection is the real bottleneck):**
- Diagnostic 1 passes (projector fits well)
- Diagnostic 3 stays around 0.7 (the cross-map itself is the limit)
- Next step: train a nonlinear adapter (residual_adapter.py already exists)
  or use Procrustes on contextual hidden states instead of vocabulary LS

## Important: Do NOT Assume a 0.75 Ceiling

Gemini claimed ~0.68-0.75 is the theoretical maximum for independently trained
embedding alignment. The dragon slayer explicitly rejected this. That number
comes from cross-lingual unsupervised word translation — a different setup
(different languages, orthogonal constraints, translation retrieval evaluation).

Our setup is: same language, partially shared tokenizer, general linear map
allowed. There is no proven theoretical cap. Treat 0.7 as a symptom to
diagnose, not a wall to accept.


## Reference Implementation (from dragon slayer, with corrections applied)

The dragon slayer provided a complete MLX implementation. Two corrections from
gemini-3.1-pro were reviewed; only one was valid:

**Applied:** Trace-scaled ridge regularization. The unnormalized `S_zz` scales
with N, so a fixed `lam` provides zero regularization at large N. Fix: divide
by N first, then scale `alpha = lam * trace(S_zz_c) / D`.

**Rejected:** Softmax-before-topk. Dragon slayer proved mathematically that
topk-then-softmax is identical to full-softmax-then-topk-then-renormalize.
The cheaper version is correct.

### Key implementation details:

```python
# Getting E_raw from quantized model (MUST dequantize):
E_raw = get_embedding_matrix(sender_model)  # [V, D], pass through embed_tokens

# Getting z (MUST be post-final-norm):
z = sender_model.model(input_ids, cache=cache)  # already post-norm in MLX

# Exact barycentric projection:
logits = z @ E_raw.T                          # [N, V]
vals, idx = mx.topk(logits, k, axis=-1)       # [N, k]
probs = mx.softmax(vals / tau, axis=-1)       # [N, k]
e_top = E_raw[idx].astype(mx.float32)         # [N, k, D]
e_top = e_top / (norm(e_top) + eps)           # L2-normalize gathered rows
y = (probs[..., None] * e_top).sum(axis=-2)   # [N, D]
y = y / (norm(y) + eps)                        # L2-normalize output

# Ridge solve (trace-scaled):
S_zz_c = S_zz / n - outer(mu_z, mu_z)
S_zy_c = S_zy / n - outer(mu_z, mu_y)
alpha = lam * trace(S_zz_c) / D
P = solve(S_zz_c + alpha * I, S_zy_c)

# Fast projection at runtime:
u_A = normalize((z - mu_z) @ P + mu_y)

# Cross-model projection:
e_B = normalize(u_A @ W_AB) * target_norm_B
```


## CLI Interface

### fit_projector.py
```bash
cd /Users/nickfox137/Documents/mlx_latentmas/mlxmas_adapter
source ../venv/bin/activate

python fit_projector.py \
    --sender mlx-community/Qwen3-4B-8bit \
    --n-prompts 200 \
    --k 64 --tau 0.7 --lam 0.01 \
    --output data/qwen_self_projector.npz
```

### test_cross.py (updated)
```bash
# Exact barycentric (debug/gold-standard):
python -u test_cross.py --mode exact --max-samples 5

# Fast ridge (production):
python -u test_cross.py --mode fast --max-samples 50

# Diagnostics only (no GSM8K eval, just projector quality metrics):
python -u test_cross.py --diagnostics

# Swap receiver model:
python -u test_cross.py --receiver mlx-community/gemma-2-2b-it-8bit --max-samples 5
```

## Data Directory

Create `mlxmas_adapter/data/` for:
- `qwen_self_projector.npz` — fitted projector weights
- Future: refitted W_AB matrices

## What NOT to Do

- Do NOT use `latent_comm.compute_alignment()` for the sender self-projection.
  That function returns identity for tied-embedding models. It is superseded
  by the barycentric ridge projector for the cross-model pipeline.

- Do NOT read `model.model.embed_tokens.weight` directly for E_raw. Quantized
  models store packed weights. Always dequantize by passing token IDs through
  `model.model.embed_tokens()`.

- Do NOT apply `model.model.norm()` to the output of `model.model()`. The MLX
  Qwen3 model already applies final norm inside `model.model()`. Applying it
  twice would corrupt the states.

- Do NOT use argmax token embeddings as ridge regression targets. The causal
  shift (position t predicts token t+1) makes this fundamentally misaligned.
  Use the soft barycentric targets instead.

## References

- Dragon slayer (gpt-5.4-high): hybrid barycentric+ridge architecture, MLX implementation
- Gemini-3.1-pro: trace-scaled regularization correction, calibration data guidance
- Zou et al., "Latent Collaboration in Multi-Agent Systems," arXiv:2511.20639
- Du et al., "Enabling Agents to Communicate Entirely in Latent Space," arXiv:2511.09149
