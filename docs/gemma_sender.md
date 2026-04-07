---
title: Gemma→Qwen Simplified Cross-Model Pipeline
author: Nick Fox
date: 2026-04-04
status: Implementation Spec
---

# Gemma Sender → Qwen Receiver: Simplified Pipeline

## Motivation

The current pipeline uses Qwen as sender with a full LatentMAS multi-agent loop (planner → critic → refiner) and Gemma-2-2B as receiver. Results with the multi-token Procrustes (S13→R12, held-out cosine 0.722):

| Config | Accuracy |
|--------|----------|
| Vocab LS baseline (L0→L0) | 1/5 (20%) |
| Procrustes S13→R12 (overfit, single-token) | 1/5 (20%) |
| Procrustes S13→R12 (multi-token, 0.722 held-out) | 3/5 (60%) |

The 3/5 result is real but Gemma-2-2B (2B params) is the bottleneck — it makes arithmetic errors even with good latent context. The obvious fix: make the stronger model (Qwen3-4B, 4B params, 36 layers) the receiver.

Additionally, the LatentMAS subagent loop (planner/critic/refiner) adds complexity without clear benefit for cross-model transfer. The simplest useful pipeline is:

1. Gemma reads the question → extract hidden states at some layer
2. Project via Procrustes into Qwen's space
3. Inject into Qwen at an intermediate layer
4. Qwen solves and generates the answer

One forward pass per model. No latent thinking loop, no self-alignment, no subagents.


## Empirical Data We Already Have

### CKA Heatmap (`cka_heatmap.json`)

The heatmap is a 36×26 matrix: `heatmap[qwen_layer][gemma_layer]`. For the reverse direction (Gemma sending, Qwen receiving), we read it transposed: which Gemma layers align best with which Qwen layers.

From the existing data, the top CKA pairs (all early-to-middle):

| Qwen Layer | Gemma Layer | CKA |
|-----------|-------------|-----|
| 6 | 4 | 0.831 |
| 7 | 4 | 0.831 |
| 7 | 6 | 0.831 |
| 12 | 9 | 0.824 |
| 12 | 10 | 0.820 |
| 14 | 10 | 0.813 |
| 13 | 10 | 0.811 |

These are symmetric — CKA measures structural similarity regardless of direction. The same pairs that work for Qwen→Gemma work for Gemma→Qwen.

### Layer Skip Tests

**Gemma-2-2B** (`layer_skip_gemma-2-2b-it-8bit.json`):
- Critical layers (reasoning): 25, 22, 23, 20, 21 (all in the final 25%)
- Dispensable layers: 0-18 (all have positive delta when skipped)
- **Best sender layers:** 11-15 range — has some signal AND high CKA with Qwen

**Qwen3-4B** (`layer_skip_Qwen3-4B-8bit.json`):
- Critical layers (reasoning): 0, 1, 11, 7, 23, 6, 13, 28
- Dispensable layers: 34, 35, 29, 19 (logit lens collapse at the end)
- **Best receiver layers for injection:** 4-12 — dispensable for Qwen's reasoning, high CKA with Gemma, leaves Qwen's reasoning stack (layers 23-35) intact downstream


### Recommended Layer Pairs for Gemma→Qwen

Combining CKA alignment with layer skip data:

| Priority | Gemma Sender | Qwen Receiver | Rationale |
|----------|-------------|---------------|-----------|
| 1 | 10 | 12 | CKA 0.82, Gemma 10 not critical, Qwen 12 dispensable |
| 2 | 10 | 9 | CKA 0.82, same Gemma layer, shallower Qwen injection |
| 3 | 9 | 12 | Similar CKA band, Gemma 9 dispensable |
| 4 | 12 | 14 | Slightly deeper Gemma, high CKA with Qwen 14 |

Note: the forward (Qwen→Gemma) result showed S13→R12 was best. By symmetry, Gemma 10-12 → Qwen 9-14 is the sweet spot. Exact best pair should be confirmed by running the multi-token Procrustes calibration in reverse.

## Architecture

### The Pipeline

```
GSM8K question
    ↓
Gemma forward pass (all 26 layers)
    → capture hidden states at layer G (all token positions)
    ↓
Procrustes projection: Gemma layer G → Qwen layer Q
    ↓
Inject into Qwen at layer Q
    → Qwen processes through layers Q..35
    → Qwen generates answer text
```

No LatentMAS loop. No self-alignment. No planner/critic/refiner. Just:
- One Gemma forward pass (extract)
- One matrix multiply (project)
- One Qwen partial forward pass (inject + generate)


### What's Different from the Forward Pipeline

| Aspect | Qwen→Gemma (current) | Gemma→Qwen (new) |
|--------|----------------------|-------------------|
| Sender | Qwen3-4B (stronger) | Gemma-2-2B (weaker) |
| Receiver | Gemma-2-2B (weaker) | Qwen3-4B (stronger) |
| Sender processing | Full LatentMAS loop (3 agents × 11 steps = 33 states) | Single forward pass (all tokens at one layer) |
| Sender output | 33 final-token hidden states | All token hidden states from prompt |
| Projection | Procrustes Qwen L13 → Gemma L12 | Procrustes Gemma LG → Qwen LQ |
| Receiver injection | `gemma_forward_from_layer(start_layer=12)` | `qwen_forward_from_layer(start_layer=Q)` |
| Self-alignment needed | Yes (W_a for latent loop) | No (no latent loop) |
| Answer generator | Gemma-2-2B (2B, weak arithmetic) | Qwen3-4B (4B, stronger reasoning) |

## Implementation Tasks

### Task 1: `qwen_forward_from_layer()` in `cross_comm.py`

Analogous to `gemma_forward_from_layer()`. Parameterized Qwen forward pass starting at any layer.

```python
def qwen_forward_from_layer(model, hidden_states, cache, start_layer=0):
    """Run Qwen forward pass starting from an intermediate layer.

    Args:
        model: Qwen3 MLX model
        hidden_states: [B, seq, D] tensor to inject
        cache: list of KVCache objects (one per layer, all 36)
        start_layer: which Qwen layer to begin processing at

    Returns:
        logits: [B, seq, vocab]
    """
```


**Implementation notes:**

- Qwen3 does NOT apply sqrt(hidden_size) scaling to embeddings (unlike Gemma). So there is no conditional scaling to worry about — the only concern is: when `start_layer > 0`, skip layers 0 through `start_layer-1`.
- Use `cache[start_layer]` for attention mask, not `cache[0]` (same pattern as Gemma).
- Apply `model.model.norm(h)` after the layer loop.
- Qwen3-4B has tied embeddings, so logits = `model.model.embed_tokens.as_linear(h)`. No logit soft-capping (that's Gemma-specific).
- Handle per-layer masks for subsequent token generation (same as `_gemma_forward_per_layer_mask` but for Qwen). When cache offsets differ across layers (layers 0..Q-1 at offset 0, layers Q..35 at offset N), each layer needs its own attention mask.

**Verify:** Call with `start_layer=0` on the same inputs as `model(input_ids, cache=cache)`. Outputs must match.

### Task 2: `extract_gemma_all_tokens()` in `cross_comm.py`

Extract Gemma's hidden states at ALL token positions at a specific layer. This is simpler than the Qwen sender in the current pipeline — no latent loop, no self-alignment. Just one forward pass.

```python
def extract_gemma_all_tokens(model, tokenizer, prompt_text, layer=10):
    """Run Gemma forward, capture all-token hidden states at a specific layer.

    Args:
        model: Gemma-2 MLX model
        tokenizer: Gemma tokenizer
        prompt_text: the question to process
        layer: which Gemma layer to extract at

    Returns:
        [1, seq_len, D] raw residual-stream hidden states
    """
```


**Implementation notes:**

- Manually iterate through `model.model.layers`, capture `h` after the target layer.
- Apply Gemma's sqrt(hidden_size) scaling at the start (before layer 0).
- Return raw residual stream — no RMSNorm, no lm_head, no logit soft-capping.
- The full forward pass should complete (all 26 layers) even though we only capture at one layer, to keep memory clean.
- Return shape `[1, seq_len, D]` where D=2304 for Gemma-2-2B.

### Task 3: Reverse Procrustes Calibration

Use the existing `calibrate_multitoken_incremental()` from `contextual_procrustes.py` but swap sender and receiver:

```bash
python -m mlxmas.contextual_procrustes \
    --multitoken \
    --model_a mlx-community/gemma-2-2b-it-8bit \
    --model_b mlx-community/Qwen3-4B-8bit \
    --sender_layer 10 \
    --receiver_layer 12 \
    --n_calibration 2000 \
    --n_positions 20 \
    --output mlxmas/procrustes_gemma_S10_qwen_R12_multitoken.npz
```

**Important:** The `is_gemma` detection in the existing code checks `model_b` for "gemma". When Gemma is model_a (sender), this needs to be updated. Either:
- Add `--is_gemma_a` / `--is_gemma_b` flags, or
- Detect "gemma" in both model names and apply sqrt scaling accordingly

The `extract_all_tokens_at_layer()` function already has an `is_gemma` parameter. The issue is in `calibrate_multitoken_incremental()` which currently hardcodes `is_gemma=False` for model_a and `is_gemma=is_gemma_b` for model_b. This needs to be parameterized.

Run calibration for the top 4 layer pairs:
- Gemma 10 → Qwen 12
- Gemma 10 → Qwen 9
- Gemma 9 → Qwen 12
- Gemma 12 → Qwen 14


### Task 4: `test_gemma_sender.py` — The Test Pipeline

New test script. Clean, no subagents.

```python
def main():
    # Load models
    gemma_model, gemma_tok = mlx_lm.load("mlx-community/gemma-2-2b-it-8bit")
    qwen_model, qwen_tok = mlx_lm.load("mlx-community/Qwen3-4B-8bit")

    # Load reverse Procrustes
    alignment = load_alignment("mlxmas/procrustes_gemma_S10_qwen_R12_multitoken.npz")

    for question in questions:
        # 1. Gemma reads the question — extract at sender layer
        gemma_hidden = extract_gemma_all_tokens(
            gemma_model, gemma_tok, question, layer=sender_layer
        )

        # 2. Project into Qwen's space
        projected = apply_cross_realignment(
            gemma_hidden, W_ab, mean_a, mean_b, target_norm_b
        )

        # 3. Inject into Qwen and generate
        output = generate_with_qwen_from_layer(
            qwen_model, qwen_tok, projected, question,
            start_layer=receiver_layer
        )
```


### Task 5: `generate_with_qwen_from_layer()` in `cross_comm.py`

Analogous to `generate_with_cross_latents_from_layer()` but for Qwen as receiver.

```python
def generate_with_qwen_from_layer(
    receiver_model, receiver_tokenizer,
    projected_embeddings, question,
    start_layer=0, max_tokens=1024, temperature=0.6,
):
    """Inject projected Gemma states into Qwen and generate answer.

    1. Create fresh cache for Qwen
    2. Feed projected embeddings through Qwen layers [start_layer..35]
    3. Feed the judger prompt through ALL 36 layers with per-layer masks
    4. Generate text autoregressively
    """
```

**Judger prompt format (Qwen3 chat template):**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
You have been given latent reasoning context about the following question.
Use it to solve the problem step by step.

Question: {question}

Put your final numerical answer inside \boxed{}.
<|im_end|>
<|im_start|>assistant
```

**Cache handling:** Same pattern as Gemma — layers 0..start_layer-1 have empty caches (offset=0), layers start_layer..35 have the injected states. Subsequent token processing uses per-layer attention masks.


## Execution Order

```
1. Fix is_gemma detection in contextual_procrustes.py for Gemma-as-sender
   ↓
2. Implement qwen_forward_from_layer() in cross_comm.py
   Verify: start_layer=0 matches model(input_ids, cache=cache) bitwise
   ↓
3. Implement extract_gemma_all_tokens() in cross_comm.py
   ↓
4. Run reverse Procrustes calibration for top 4 layer pairs
   (Gemma 10→Qwen 12, Gemma 10→Qwen 9, Gemma 9→Qwen 12, Gemma 12→Qwen 14)
   ↓
5. Implement generate_with_qwen_from_layer() in cross_comm.py
   ↓
6. Write test_gemma_sender.py
   ↓
7. Run 50-question GSM8K eval, compare against:
   - Qwen→Gemma Procrustes S13→R12: 3/5 on first 5, running 50 now
   - Qwen standalone (no cross-model, just Qwen solving alone): need baseline
```

## Memory Budget

- Gemma-2-2B 8-bit: ~2.5 GB
- Qwen3-4B 8-bit: ~4.5 GB
- KV caches (both): ~2 GB worst case
- Total: ~9 GB, well within 32 GB

Both models stay loaded throughout. No load/unload needed.

## Important: Qwen Standalone Baseline

Before evaluating Gemma→Qwen, we need to know how well Qwen3-4B does **on its own** on GSM8K. If Qwen alone gets 80% on GSM8K, then Gemma→Qwen getting 60% would be a degradation, not an improvement. The cross-model pipeline is only useful if it adds value over the receiver solving alone.

Run a simple Qwen-only baseline:

```bash
# Qwen solving GSM8K alone (no cross-model, no LatentMAS, just direct prompting)
python -m mlxmas.test_gemma_sender --qwen-only --max-samples 50
```

This should be a flag in the test script — same questions, same prompt format, but Qwen gets the question directly without any injected latent context.


## Existing Code to Reference

These files contain working implementations of the patterns needed:

| Need | Reference |
|------|-----------|
| Parameterized forward from layer | `cross_comm.py: gemma_forward_from_layer()` |
| Per-layer attention masks | `cross_comm.py: _gemma_forward_per_layer_mask()` |
| All-token extraction at a layer | `contextual_procrustes.py: extract_all_tokens_at_layer()` |
| Multi-token Procrustes calibration | `contextual_procrustes.py: calibrate_multitoken_incremental()` |
| Generation with injected latents | `cross_comm.py: generate_with_cross_latents_from_layer()` |
| Cross realignment (project + norm match) | `cross_align.py: apply_cross_realignment()` |
| GSM8K test loading | `test_cross.py: load_gsm8k_test()` |
| Answer extraction | `utils.py: extract_boxed_answer(), normalize_answer()` |

## Models

All 8-bit. No 4-bit.

- Sender: `mlx-community/gemma-2-2b-it-8bit`
- Receiver: `mlx-community/Qwen3-4B-8bit`

## Future: Residual Adapter

If the linear Procrustes plateaus for Gemma→Qwen (as it did for Qwen→Gemma at ~0.72), the residual adapter infrastructure already exists in `mlxmas/residual_adapter.py`. The same architecture (frozen Procrustes base + trainable 2-layer SiLU MLP) works in either direction. The `collect` command just needs to be pointed at the reverse model pair.

## Future: Bidirectional

Once both Qwen→Gemma and Gemma→Qwen work, the two adapters (one per direction) enable full bidirectional latent communication. This is the foundation for the LatentMAS subagent loop where agents of different model families can collaborate: Gemma plans, Qwen critiques, Gemma refines, Qwen judges — all in latent space, no text decoding between agents.
