# Session Briefing — April 6, 2026 (updated 4:58pm)

## EXACT NEXT STEP

The MHA adapter mode-collapsed during training. Diagnostic shows:
- Every position produces the same two alternating vectors regardless of input
- cos(pos0, pos1) = -0.82 across ALL samples (should vary)
- All positions decode to `PreferredItem` or space — no semantic content
- Loss decreased because LoRA learned to predict answers from ANY fixed adapter output, not because adapter transmitted information

Root cause: AdaptiveProjection starts near-zero (scale=0.2, output_scale=0.1) + gradient clipping at 1.0 = adapter never develops enough signal to matter. LoRA learns to ignore it.

Three options presented to user:
1. Increase adapter initial scale so it passes real signal from start
2. Remove AdaptiveProjection entirely — just Linear + LayerNorm + MHA + Linear
3. Warm up: train LoRA first with plan text, then introduce adapter

**User has not chosen yet.**

## Training results (epoch 3 was best)
- train_loss: 2.38, val_loss: 2.33 (best)
- 1011 seconds total, 475 train / 25 val samples
- Weights saved at mlxmas/data/interlat_adapter/

## Eval results: 0/5 — adapter outputs garbage digits

## Key architecture details
- MHAAdapter: 37.7M params (Linear 2560→2304 + LN + 1-head MHA + LN + AdaptiveProjection)
- Gemma LoRA: rank 8, 16 layers, 6.6M params
- Embeddings: dequantized to float32, FROZEN (critical — was 589M trainable bug)
- Special tokens: <bop>=256000, <eop>=256001
- Injection: layer 0 (embedding level)
- Loss: CE only
- Gradient clipping: max_norm=1.0

## Files
- `mlxmas/mha_adapter.py` — adapter class
- `mlxmas/collect_sender_hidden.py` — prefill-only data collection (3 min / 500 samples)
- `mlxmas/train_interlat.py` — training loop
- `mlxmas/eval_interlat.py` — eval script (NEW)
- `mlxmas/data/sender_hidden_states.npz` — 500 samples, 500.9 MB
- `mlxmas/data/interlat_adapter/` — trained weights (mode-collapsed, useless)

## Same-model results (WORKING)
| Pipeline | Score | Time/sample |
|----------|-------|-------------|
| Qwen→Qwen (with think) | 5/5 | ~37s |
| Qwen→Qwen (--no-think) | 5/5 | ~9s |

## Standards
- Logging: `1:37pm 4-6-26 [100/500] tokens=42 elapsed=0m36s ETA=2m24s`
- max_tokens: 2048 everywhere
- Generation progress: every 50 tokens
- No question text to any receiver
- Procrustes is dead, don't suggest it
