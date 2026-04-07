---
title: "Barycentric Ridge Self-Projector: Failure Report"
author: Nick Fox + Claude
date: 2026-04-05
status: Post-Mortem
---

# Barycentric Ridge Self-Projector — Failure Report

## Summary

The barycentric ridge self-projector was designed to fix the broken `W_a = identity` path for tied-embedding models in the Qwen→Phi-4 cross-model pipeline. The theory was sound: use the sender's LM head to build soft teacher targets on the embedding manifold, then distill into a single-matrix ridge regression for fast runtime projection. In practice, the implementation is correct but the approach made end-to-end performance **worse** than the broken baseline it was replacing.

| Pipeline | Projector Fidelity | Static W_AB | End-to-End (5 GSM8K) |
|---|---|---|---|
| Old identity path (W_a=I) | N/A | 0.72 (Qwen→Gemma) | 1/5, coherent text |
| Barycentric fast (ridge) | 0.72 | 0.78 (Qwen→Phi-4) | ~0/5, mostly garbage |
| Barycentric exact (teacher) | 1.0 (by definition) | 0.78 (Qwen→Phi-4) | 0/5, worse than fast |

The exact barycentric projection — the gold standard that the ridge was trying to approximate — produced worse output than the ridge approximation. This rules out the ridge fidelity (0.72) as the bottleneck and points to a deeper structural problem.

---

## What Was Built

All files in `mlxmas_adapter/`:

- **`self_projector.py`** — `BarycentricRidgeSelfProjector` (project_fast, project_exact), `BarycentricRidgeFitter` (accumulate, finalize), `get_embedding_matrix()`, `fit_self_projector()`
- **`fit_projector.py`** — CLI script, fit on 500 GSM8K train prompts (30,532 positions), saved to `data/qwen_self_projector.npz`
- **`test_cross.py`** — Rewired pipeline with `--mode fast|exact`, `--diagnostics`, `--projector-path`

The implementation is correct. The `mx.topk` API discrepancy (returns values only, not indices) was caught and handled via `mx.argpartition`. Ridge solve uses `stream=mx.cpu`. All `mx.eval()` calls prevent graph growth. Embedding dequantization passes tokens through `embed_tokens()`, not the packed weight matrix.

---

## Diagnostic Results

### Diagnostic 1: Projector Fidelity — cos(project_fast, project_exact)

| Prompt | Tokens | Cosine |
|---|---|---|
| GSM8K test #1 | 65 | 0.656 |
| GSM8K test #2 | 26 | 0.715 |
| GSM8K test #3 | 57 | 0.789 |
| GSM8K test #4 | 35 | 0.767 |
| GSM8K test #5 | 111 | 0.663 |
| **Mean** | | **0.718** |

Target was > 0.85. Tested: more data (500 prompts / 30k positions — no improvement over 200 / 12k), lower regularization (lam=0.001, 0.0001 — made it worse, overfitting), normalized inputs (same result), raw un-normalized targets (worse at 0.62). The ~0.7 cosine is the inherent linear approximation ceiling for the softmax-topk-normalize nonlinearity on this model.

### Diagnostic 2: Static W_AB Ceiling — cos(normalize(E_A @ W_AB), normalize(E_B))

**0.7820** on 1,000 held-out shared tokens (Qwen→Phi-4).

This is the upper bound on cross-alignment quality when given perfect embedding-space vectors. It is decent — better than the Qwen→Gemma static alignment (0.68).

### Diagnostic 3: Projected W_AB Ceiling — refit W_AB on projected vectors

**0.5381** (980 train / 245 test matched token pairs).

This is the critical failure metric. Refitting W_AB on the barycentric-projected vectors produces **worse** cross-alignment (0.54) than the static embedding alignment (0.78). The projected contextual vectors occupy a different region of embedding space than the static embeddings W_AB was trained on.

---

## Why It Failed

### Failure Mode 1: The Barycentric Projection Creates a New Distribution Mismatch

The entire theory rests on this chain:

```
hidden states (transformer space)
    → barycentric projection → embedding manifold
    → W_AB (trained on static embeddings) → receiver embedding space
```

The assumption is that barycentric-projected vectors live in the same distribution as static embeddings, so W_AB can cross-align them. This assumption is wrong.

Static embeddings are context-free: token "the" always maps to the same vector. Barycentric projections are contextual: token "the" at position 5 maps to a weighted sum of the top-64 predicted token embeddings, which depends entirely on the preceding context. Even though both live in the same vector space (R^2560), they occupy different manifold regions with different covariance structure.

Diagnostic 3 confirms this: W_AB refitted on projected vectors achieves 0.54, not 0.78. The projected vectors are harder to cross-align, not easier.

### Failure Mode 2: Exact Projection Is Worse Than Fast — Sharpness Kills Cross-Alignment

The most damning result: `--mode exact` (the teacher signal itself) produced worse end-to-end output than `--mode fast` (the ridge approximation). With fast mode, 2/5 questions produced coherent English reasoning. With exact mode, 0/5 were coherent.

This is counterintuitive until you consider what the exact projection does: softmax at tau=0.7 with top-64 is quite sharp. For positions where the model is confident, the barycentric projection collapses to approximately the embedding of the single predicted token. This produces vectors that are highly concentrated on specific regions of the embedding manifold, with a very different distribution than the uniformly-distributed static embeddings W_AB was trained on.

The ridge regression acts as an implicit smoother — its linear approximation naturally blurs the sharp teacher targets, producing vectors that are more evenly distributed and happen to cross-align better. This accidental smoothing is why fast mode outperformed exact mode.

### Failure Mode 3: The "Broken" Identity Path Wasn't Actually Broken for This Pipeline

The original diagnosis was: `W_a = identity` is a no-op, so hidden states stay in transformer output space, creating a distribution mismatch with W_AB (trained on embeddings).

This diagnosis was technically correct but practically wrong. The identity path produced hidden states that, while not on the embedding manifold, had **accidental structural overlap** with the distribution W_AB was trained on. The `apply_realignment` function's norm-matching step rescaled these vectors to embedding-like magnitudes, and the L2-normalization in `apply_cross_realignment` projected them onto the unit sphere — the same space where W_AB operates.

The identity path's ~0.70 cosine overlap (documented in the original problem statement as the "structural overlap between transformer output space and embedding space") was apparently sufficient for the cross-alignment to work. By replacing this with barycentric projection, we moved the vectors to a region where W_AB performs worse, not better.

### Failure Mode 4: Linear Ridge Cannot Approximate Softmax-TopK-Normalize

The ridge regression tries to learn: `z → softmax(z @ E.T / tau) @ normalize(E) → normalize(result)`. The first step (z @ E.T) is linear, but the softmax over 151,936 vocabulary items is a 151k-way categorical decision that varies dramatically across token positions. A single [2560, 2560] matrix cannot capture this variation.

The 0.72 cosine ceiling held regardless of:
- Data volume (12k vs 30k positions — identical)
- Regularization strength (lam=0.01, 0.001, 0.0001)
- Input normalization (raw vs L2-normalized z)
- Target normalization (normalized vs raw barycentric targets)

This suggests the linear approximation quality is a property of the model's prediction entropy on this domain, not a tuning issue.

---

## What We Learned (Relative to Prior Work)

### The project has now tested four cross-model alignment approaches:

| # | Approach | Cosine (metric) | GSM8K | Lesson |
|---|---|---|---|---|
| 1 | Vocab LS, L0→L0 inject (Qwen→Gemma) | 0.68 | 1/5 | Coherent text, logical errors |
| 2 | Contextual Procrustes, LN→LN inject at L0 | 0.74* | 0/5 | Wrong layer space — gibberish |
| 3 | Intermediate injection S13→R12 (Qwen→Gemma) | 0.877 | 3/5 | Best result. Layer matching matters most |
| 4 | **Barycentric projector + Vocab LS (Qwen→Phi-4)** | **0.72** | **~0/5** | **New distribution mismatch** |

*Cosine on calibration data, not held-out.

The pattern is clear: **every time we change the distribution of vectors entering W_AB without retraining W_AB on that distribution, performance degrades.** Approach 2 failed for the same reason — Procrustes was fit on Layer-N states but applied to Layer-0 states. Approach 4 (ours) fits W_AB on static embeddings but feeds it barycentric-projected contextual vectors.

### The lesson from Approach 3 still stands

The best result in this project's history (3/5, 0.877 cosine) came from **intermediate layer injection with contextual Procrustes** — fitting the alignment on the same type of vectors that flow through it at runtime. The barycentric projector tried to avoid this by forcing everything through embedding space, but embedding space is the wrong meeting point for contextual representations.

---

## What the Barycentric Projector Would Need to Work

For this approach to succeed, one of these would need to change:

1. **Refit W_AB on projected vectors.** Not on static embeddings, but on the actual barycentric-projected hidden states paired with receiver embeddings for the same tokens. Diagnostic 3 attempted this with only 1,225 pairs and got 0.54 — too few for a stable [2560, 3072] matrix. With 10k+ contextual pairs, the refitted W_AB might recover. But at that point, you're doing contextual Procrustes with extra steps.

2. **Use a nonlinear adapter after the projector.** The barycentric projection moves vectors to the right space but the wrong distribution. A trained residual adapter could correct the distribution mismatch. But this adds training, defeating the "closed-form, training-free" appeal of the approach.

3. **Higher-tau softmax to smooth the teacher.** Tau=0.7 produces sharp distributions that create the concentration problem in Failure Mode 2. Tau=2.0 or higher would smooth the projections but also lose the semantic specificity that makes them useful.

None of these are worth pursuing given that intermediate layer injection (Approach 3) already works better with less machinery.

---

## Files Produced

```
mlxmas_adapter/
├── self_projector.py          # Core module (correct implementation, wrong approach)
├── fit_projector.py           # CLI fitting script
├── test_cross.py              # Updated pipeline with --mode, --diagnostics
└── data/
    └── qwen_self_projector.npz  # Fitted projector (500 prompts, 30k positions)
```

---

## References

- Dragon slayer (gpt-5.4-high): designed the barycentric+ridge architecture
- Gemini-3.1-pro: trace-scaled regularization correction (applied, correct)
- Gemini-3.1-pro: softmax-before-topk suggestion (rejected, mathematically equivalent)
- Prior work in this project: `docs/no_alignment.md`, `docs/latent_mlx_v0_1.md`
- Zou et al., "Latent Collaboration in Multi-Agent Systems," arXiv:2511.20639
- Du et al., "Enabling Agents to Communicate Entirely in Latent Space," arXiv:2511.09149
