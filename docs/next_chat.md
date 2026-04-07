---
title: "Next Chat Briefing: Cross-Model Latent Communication"
date: 2026-04-06
status: Active — Claude Code implementing approaches
---

# Project Briefing: mlx_latentmas

## Read These Files First (in order)

1. **THIS FILE** — full project state and failure history
2. `docs/beyond_procrustes.md` — Implementation spec for MLP/CCA/OT approaches
3. `mlxmas/diagnose_latents.py` — Signal analysis tool (4 tests: logit decode,
   distribution comparison, nearest embeddings, control decode on real activations)
4. `mlxmas/test_cross.py` — GSM8K eval pipeline (LatentMAS stripped, layer offset fixed)
5. `mlxmas/cross_comm.py` — Sender extraction + receiver injection pipeline
6. `mlxmas/cross_align.py` — Cross-model projection (norm-preserving version)

Optional context:
- `docs/barycentric_projector.md` — Prior failed approach (barycentric ridge)
- Check `logs/` for latest run outputs — sort by date
- Check with Claude Code for any progress since this document was written


## What This Project Is

Two independently trained LLMs communicate through projected hidden states.
Qwen3-4B (sender) processes a math question, its layer-13 hidden states are
projected into Gemma-2-9B's (receiver) layer-22 activation space, and Gemma
generates the answer. No tokens exchanged. The receiver must extract ALL
information from the injected latent states — the question text is NOT in
the receiver's prompt.

Foundation for SoulMCP v0.1.4 Mother/Child tensor communication.

## Current State (April 6, 2026)

**The latent channel carries zero usable signal.** Six approaches tested.
All produce 0/5 on GSM8K when question text is removed from receiver prompt.
Receiver responds: "Please provide me with the problem."

All prior "results" (3/3, 26/50, 66%) were the receiver reading the question
in plain text and solving it on its own. Discovered April 5-6 when the
question text was finally removed from the judger prompt.


## Complete Failure History (6 approaches, all 0/5)

| # | Approach | Key Metric | GSM8K (no Q text) | Why it failed |
|---|---|---|---|---|
| 1 | Vocab LS, L0 inject | 0.68 cosine | 0/5 | Garbage tokens after projection |
| 2 | Contextual Procrustes S13→R12 (Gemma-2B) | 0.877 cal cosine | 0/5 | Wrong directions (0.22 mean cosine) |
| 3 | Contextual Procrustes S13→R22 (Gemma-9B) | 0.80 held-out | 0/5 | Wrong directions (0.22 mean cosine) |
| 4 | Barycentric ridge projector (Qwen→Phi-4) | 0.72 cosine | 0/5 | Collapsed reasoning to lexical mixtures |
| 5 | MLP adapter (Linear→SiLU→Linear, 16M params) | 0.84 val cosine | 0/5 | MSE mean regression, only +0.089 over mean baseline |
| 6 | CCA K=512 | 0.9887 top canonical corr | 0/5 | Bottleneck collapses all vectors to receiver mean |

**The pattern:** Every point-wise projection (same transform applied to every
vector independently) collapses per-token specificity. All output vectors
converge to approximately the same point — "average Gemma L22 activation."
Gemma's attention finds nothing to latch onto.


## Critical Diagnostic Findings

**From `diagnose_latents.py` (Procrustes, after norm fix):**
- Logit decode (projected): `itſelf`, `setVerticalGroup`, `autorytatywna` — noise
- Logit decode (real Gemma L22): `ducks` (0.53), `breakfast` (0.84), `market` (0.61) — clean signal
- Variance profile correlation: **0.9274** — right dimensions are active
- Mean vector cosine: **0.2221** — wrong directions within those dimensions
- Projected norms: [69, 1314] vs real [190, 4713] — dynamic range preserved

**From MLP adapter:**
- Val cosine: 0.8416, but mean-baseline cosine is 0.7504
- Delta over baseline: only +0.089 — adapter barely beats outputting the mean vector
- All positions decode to function words (and, of, to, the)

**From CCA:**
- Top canonical correlation: 0.9887, 89 dims > 0.8 — shared structure EXISTS
- But CCA back-projection: `shared @ W_b.T + mu_y` — mu_y dominates
- All output vectors collapse to mu_y + tiny perturbation
- Norm range: [324.9, 325.7] — flat (bottleneck itself collapses variation)

**Key insight from diagnostics:** The variance profile correlation (0.93)
proves the projected vectors activate the RIGHT dimensions. The mean cosine
(0.22) proves they point in the WRONG directions. The CCA correlations
(0.99) prove shared structure exists. But no point-wise method preserves
the per-token signal needed for attention to extract information.


## Bugs Found and Fixed

1. **Question text in receiver prompt (CRITICAL).** Every prior eval included
   `Question: {question}` in the judger prompt. Receiver was solving from
   text, not latent states. Now removed.

2. **Norm flattening.** `apply_cross_realignment` forced all vectors to same
   magnitude (524.3). Real norms: 190-4713. Fixed — preserves dynamic range.

3. **Layer offset bug.** `extract_all_tokens_at_layer(layer=22)` captures
   L22 OUTPUT. `gemma_forward_from_layer(start_layer=22)` feeds INTO L22,
   double-processing. Fixed — injects at `start_layer + 1` (layer 23).

4. **LatentMAS loop stripped.** Planner/critic/refiner produced 33 recurrent
   dream states, out-of-distribution for any mapping trained on forward-pass
   data. Sender now does single forward pass: ~50-65 real token positions.

5. **Identity self-alignment for tied embeddings.** `compute_alignment()`
   returns `W_a = identity` for Qwen. Superseded by MLP/CCA adapters.

## What Claude Code Has Built

In `mlxmas/`:
- `collect_paired_states.py` — Collects paired (Qwen L13, Gemma L22) states
- `mlp_adapter.py` — MLP adapter with train/eval (failed: mean regression)
- `cca_adapter.py` — CCA adapter with fit/eval (failed: bottleneck collapse)
- `train_mlp_adapter.py` — CLI training script
- `diagnose_latents.py` — Signal analysis tool (4 tests)
- `test_cross.py` — Rewritten: single forward pass, layer offset fix,
  `--adapter-type mlp|cca|residual|procrustes`, `--mode exact|fast`


## Why Every Approach Fails the Same Way

Claude Code's analysis (confirmed by this session): "the mean direction
between Qwen L13 and Gemma L22 is easy (0.98 cosine), but per-token
deviations from that mean are not predictable from one space to the other
with any of these methods."

All three methods (Procrustes, MLP, CCA) are position-independent — they
apply the same transformation to every vector regardless of its position
in the sequence or what it encodes. They converge on the distributional mean
because that's the best position-independent answer.

The Du/Interlat paper (arXiv:2511.09149) used a multi-head attention adapter
precisely because cross-model transfer requires context-dependent, position-
aware mappings. An MHA adapter attends to the full sequence and produces
different outputs for each position based on surrounding context.

## Paths Forward (ranked)

**Path A: MHA adapter** — Context-dependent mapping. Attends to the full
sender sequence, produces position-specific outputs. This is what actually
works in the literature. Not yet implemented.

**Path B: End-to-end training on task loss** — Train the adapter to minimize
receiver's cross-entropy on correct answers, not MSE on paired states.
Learns what information the receiver NEEDS, not what the sender PRODUCES.
Requires backprop through frozen receiver.

**Path C: Same-model communication** — Skip cross-model entirely. Use two
instances of the same model family (e.g., Qwen Mother + Qwen Child) where
alignment is near-identity. This is what the Zou paper actually tests.
Focus energy on SoulMCP v0.1.4 architecture instead.

**Path D: Rethink injection** — Maybe KV cache injection at intermediate
layers can't work without sequential structure the receiver expects.
Try layer-0 injection with a trained nonlinear adapter.


## Models (8-bit ONLY)

- Sender: `mlx-community/Qwen3-4B-8bit` (D=2560, 36 layers, tied embeddings)
- Receiver: `mlx-community/gemma-2-9b-it-8bit` (D=3584, 42 layers)
- Memory: ~16GB for both models. 32GB Mac Mini M2 Pro.
- Qwen standalone GSM8K baseline: 96% (48/50)

Do NOT use 4-bit. Do NOT use Gemma-2-2B (too weak). Do NOT use Phi-4-mini
(vocab LS alignment was garbage for that pair).

## The Dragon Slayer

Nick's term for the adversarial review AI — currently gpt-5.4-high. Used
for mathematical verification and catching errors Claude misses.

Key contributions this session:
- Designed barycentric+ridge architecture (correct math, wrong approach)
- Rejected Gemini's "0.75 ceiling" claim with mathematical proof
- Proved softmax-before-topk is unnecessary (renormalization cancels)
- Recommended trace-scaled ridge regularization (applied)
- Identified MHA adapter as architecturally correct next step

## What NOT To Do

1. **Do NOT put question text in the receiver prompt.** This is the bug that
   invalidated all prior results. The receiver prompt is:
   "Using the reasoning context provided, solve the problem step by step.
   Put your final numerical answer inside \boxed{}."

2. **Do NOT flatten norms.** `apply_cross_realignment` was fixed. Do not
   reintroduce `projected * (target_norm / proj_norm)` for all vectors.

3. **Do NOT use `latent_comm.compute_alignment()` for sender self-projection.**
   Returns identity for tied-embedding models. Superseded.

4. **Do NOT read `model.model.embed_tokens.weight` directly.** Quantized
   models store packed weights. Pass tokens through `model.model.embed_tokens()`.

5. **Do NOT assume Gemma forward starts at layer 0 when injecting.** Inject
   at `start_layer + 1` because the adapter/alignment targets the OUTPUT
   of start_layer.

6. **Do NOT state conclusions without evidence.** Don't say "zero signal"
   without measuring. Don't claim theoretical ceilings without citations.
   Don't fabricate quotes from outputs.

7. **Do NOT propose fixes with high confidence unless you've verified the
   reasoning.** State uncertainty. Design experiments. Six approaches have
   failed — humility is warranted.


## What Went Wrong in This Session (DO NOT REPEAT)

1. **Question text in receiver prompt from day one.** I (Claude) wrote the
   judger prompt with `Question: {question}`. Every result was invalid.

2. **Stated conclusions without evidence.** Said "zero signal" without
   measuring. Said "0.75 is the theoretical ceiling" when I knew it was
   from a different domain. Fabricated a Claude Code quote.

3. **Proposed fixes with 100% confidence, 0% effectiveness.** Six approaches,
   each presented as the breakthrough. All failed identically.

4. **Knew relevant information but didn't volunteer it.** The cross-lingual
   alignment ceiling (~0.70) was well-documented in NLP literature I knew.
   I didn't mention it until Nick asked directly. Days wasted.

5. **Copied patterns without thinking.** Norm flattening from self-alignment.
   Question text from Zou paper. Neither was appropriate for our use case.

## Nick's Expectations

- Direct communication. No apologies. No diplomatic softening.
- When he asks "why?" he wants the real reason, not a tutorial.
- When angry, it's because real time was wasted. Acknowledge, fix, move on.
- He expects Claude to own code it wrote without deflection.
- Say "I don't know" rather than produce partial fixes.
- Never start coding without understanding the full problem.
- He has been coding since 1990. Do not explain basic concepts.


## Project File Layout

```
/Users/nickfox137/Documents/mlx_latentmas/
├── mlxmas/                          # Main codebase (active)
│   ├── cross_comm.py                # Sender extraction + receiver injection
│   ├── cross_align.py               # Cross-model projection (norm fix applied)
│   ├── contextual_procrustes.py     # Procrustes calibration
│   ├── diagnose_latents.py          # Signal analysis (4 tests)
│   ├── test_cross.py                # GSM8K eval (rewritten, clean)
│   ├── mlp_adapter.py               # MLP adapter (failed)
│   ├── cca_adapter.py               # CCA adapter (failed)
│   ├── collect_paired_states.py     # Paired state collection
│   ├── latent_comm.py               # Self-alignment (identity bug, superseded)
│   ├── residual_adapter.py          # Residual adapter (partial, older)
│   ├── prompts.py                   # Agent prompts (unused after loop strip)
│   ├── utils.py                     # extract_boxed_answer, normalize_answer
│   └── data/                        # .npz data files
├── mlxmas_adapter/                  # Simplified pipeline (mostly superseded)
│   ├── self_projector.py            # Barycentric projector (failed)
│   └── ...
├── docs/
│   ├── beyond_procrustes.md         # MLP/CCA/OT implementation spec
│   ├── barycentric_projector.md     # Barycentric ridge spec (failed)
│   ├── gemma_sender.md              # Gemma→Qwen reverse pipeline spec
│   └── next_chat.md                 # THIS FILE
├── logs/                            # Run logs (check latest by date)
├── tests/                           # Test suite
└── venv/                            # Python virtual environment
```

## Key Logs (check `ls -lt logs/` for latest)

- `logs/diagnose_latents_no_norm_flatten.log` — Best diagnostic (after norm fix)
- `logs/eval_mlp_5.log` — MLP adapter eval (0/5)
- `logs/gsm8k_9b_S13_R22_no_question.log` — First run without question text (0/5)
- `logs/cca_fit.log` — CCA canonical correlations
- `logs/diag_cca_K512.log` — CCA diagnostics

## References

- Du et al., arXiv:2511.09149 — "Enabling Agents to Communicate Entirely
  in Latent Space" (MHA adapter for cross-model transfer)
- Zou et al., arXiv:2511.20639 — "Latent Collaboration in Multi-Agent
  Systems" (LatentMAS, same-model communication)
- Huh et al. — "Platonic Representation Hypothesis" (convergence of
  intermediate representations across models)
