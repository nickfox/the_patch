---
title: "Beyond Procrustes: Three Approaches to Cross-Model Latent Mapping"
author: Nick Fox
date: 2026-04-06
status: Implementation Spec
---

# Beyond Procrustes: MLP Adapter, CCA, and Optimal Transport

## Why This Document Exists

We have spent days trying to make cross-model latent communication work between
Qwen3-4B (sender) and Gemma-2-9B (receiver). Every approach has failed to
transfer meaningful information through the latent channel. This document
explains WHY each prior approach failed, what the diagnostic data tells us,
and specifies three new approaches to try.

**READ THE FULL "WHY" SECTION BEFORE WRITING ANY CODE.**


## The Story So Far

### What we're trying to do

Two independently trained LLMs communicate through projected hidden states.
Qwen3-4B (sender) processes a question, its internal representations are
projected into Gemma-2-9B's (receiver) activation space, and Gemma generates
the answer. No tokens are exchanged between models. The receiver never sees
the question as text — it must extract all information from the injected
latent states.

### What actually happened

Every approach appeared to work at first. We got 3/3, then 26/50, then 66%.
Then we discovered the receiver was reading the question in plain text the
entire time — the judger prompt included `Question: {question}`. When we
removed the question text, the receiver scored 0/5 and said "Please provide
me with the problem." The latent channel was carrying zero usable signal.

### Results table

| Approach | Accuracy | What actually happened |
|---|---|---|
| Qwen standalone (no cross-model) | 96% (48/50) | Baseline — Qwen is strong |
| Vocab LS, Qwen→Gemma-2B | 3/3 then 1/5 | Receiver reading question text |
| Procrustes S13→R12, Qwen→Gemma-2B | 3/5 | Receiver reading question text |
| Procrustes S13→R22, Qwen→Gemma-9B | 66% (34/50) | Receiver reading question text |
| Barycentric projector, Qwen→Phi-4 | 0/5 | Garbage output even WITH question text |
| **Any approach, question text removed** | **0/5** | **"Please provide me with the problem"** |


## The Diagnostic Data (critical — read carefully)

We built `mlxmas/diagnose_latents.py` to look inside the actual projected
vectors being injected into Gemma. The question was Janet's ducks / eggs /
farmers market (gold answer: 18). Here is what we found:

### Test 1: Logit Decode — what does Gemma think the projected vectors mean?

We fed each projected vector through Gemma's own `norm → lm_head`. If the
vectors carry question semantics, Gemma should decode tokens like "eggs",
"ducks", "sells", "market". Instead:

```
pos 0: '<bos>' (0.617), 'InstanceState' (0.010), 'ècie' (0.007)
pos 2: ' itſelf' (0.539), ' raiſ' (0.155), ' occaf' (0.026)
pos 14: 'setVerticalGroup' (0.438), '<bos>' (0.056), 'SharedDtor' (0.049)
```

Old English long-s tokens, Java GUI fragments, Polish words. Zero relationship
to the question. This is noise.

### Test 4 (control): Logit Decode on REAL Gemma layer-22 activations

Same question, same test, but using Gemma's own activations:

```
pos 4 [input='ducks']: ' ducks' (0.527), ' and' (0.192)
pos 8 [input='6']: '6' (0.539), '2' (0.088)
pos 15 [input='three']: ' three' (0.670)
pos 17 [input='breakfast']: ' breakfast' (0.842)
pos 40 [input='market']: ' market' (0.606)
```

Real activations carry the question semantics clearly. The projected vectors
do not.


### Test 2: Distribution Comparison (before and after norm fix)

We discovered `apply_cross_realignment` was forcing every projected vector
to the same norm (524.3). Real Gemma layer-22 norms range from 190 to 4713.
This norm flattening was destroying the dynamic range. We fixed it.

| Metric | Before (flat norms) | After (dynamic norms) |
|---|---|---|
| Projected norm range | [524.3, 524.3] | [69.2, 1313.6] |
| Real Gemma norm range | [190.1, 4713.7] | [190.1, 4713.7] |
| **Variance profile correlation** | **0.4436** | **0.9274** |
| Mean vector cosine | 0.2482 | 0.2221 |

The variance profile correlation jumped from 0.44 to **0.93**. This means
the projected vectors now activate the SAME DIMENSIONS as real Gemma layer-22
activations. The distributional structure is correct.

But the DIRECTIONS within those dimensions are still wrong (mean cosine 0.22).
This is why logit decode still shows garbage — it's a direction-dependent test.

**Key insight: The channel has structural signal (right dimensions active)
but wrong directional signal (vectors point in wrong directions within those
dimensions). This is learnable — a trained adapter or CCA could correct the
directional mapping since the structural foundation is sound.**


## WHY Procrustes Fails for This Task

Procrustes computes ONE global rotation matrix from thousands of calibration
pairs. Calibration cosine was 0.80. But "on average" is the problem.

The relationship between Qwen layer 13 and Gemma layer 22 is NOT one fixed
rotation. It varies by:
- Token position and context
- Type of reasoning state (planning vs arithmetic vs word recall)
- Whether the state came from a normal forward pass or a latent loop

The Procrustes calibration was fit on states from normal text forward passes.
At runtime, the LatentMAS loop feeds hidden states back through Qwen
iteratively (10 steps × 3 agents = 30+ recurrent steps). By step 10, those
states have drifted far from anything the calibration saw. They're deep
reasoning states specific to Qwen's dynamics.

**Result:** 0.80 cosine on calibration data, 0.22 cosine on runtime data.
The rotation is correct for the space it was fit on. The runtime states
aren't in that space.

A single linear rotation cannot fix this because the mapping is:
1. Context-dependent (different per token position)
2. Nonlinear (the softmax/attention operations create complex feature
   interactions that a rotation can't capture)
3. Out-of-distribution at runtime (latent loop states ≠ forward pass states)


## Working Directory and Models

All code lives in: `/Users/nickfox137/Documents/mlx_latentmas/mlxmas/`
Venv: `/Users/nickfox137/Documents/mlx_latentmas/venv/`
Data files: `/Users/nickfox137/Documents/mlx_latentmas/mlxmas/data/`

Models (8-bit only):
- Sender: `mlx-community/Qwen3-4B-8bit` (D=2560, 36 layers)
- Receiver: `mlx-community/gemma-2-9b-it-8bit` (D=3584, 42 layers)

Existing Procrustes: `mlxmas/data/procrustes_S13_R22.npz`
- Sender layer: 13, Receiver layer: 22
- Held-out cosine: 0.7964

Memory budget: ~16GB for both models + caches. 32GB machine.

## Existing Code to Reference

| File | What it does |
|---|---|
| `cross_comm.py` | `extract_sender_at_layer()` — Qwen LatentMAS loop capturing at layer 13 |
| `cross_comm.py` | `gemma_forward_from_layer()` — Gemma forward from intermediate layer |
| `cross_comm.py` | `generate_with_cross_latents_from_layer()` — Full inject + generate pipeline |
| `cross_align.py` | `apply_cross_realignment()` — Projects sender states into receiver space |
| `contextual_procrustes.py` | `extract_all_tokens_at_layer()` — Extract hidden states at a layer |
| `contextual_procrustes.py` | `calibrate_multitoken_incremental()` — Fit Procrustes on paired states |
| `residual_adapter.py` | Existing residual MLP adapter infrastructure (partially built) |
| `diagnose_latents.py` | Signal diagnostics: logit decode, distribution comparison, nearest embeddings |
| `test_cross.py` | GSM8K eval pipeline with `--receiver` flag |


## Approach 1: Trained MLP Adapter (Highest Confidence)

### What it is

A small 2-layer neural network that learns the nonlinear mapping from Qwen
layer-13 hidden states to Gemma layer-22 hidden states. Unlike Procrustes
(one global rotation), the MLP can learn context-dependent, nonlinear feature
correspondences.

Architecture: `Linear(2560, 2560) → SiLU → Linear(2560, 3584)`
Parameters: ~16M (tiny — trains in minutes on Apple Silicon)

### Why it should work

The diagnostic showed 0.93 variance profile correlation — the right dimensions
are already active after Procrustes. The problem is directional accuracy
within those dimensions. An MLP with a nonlinear activation (SiLU) can learn
the per-dimension directional corrections that a single rotation matrix cannot.

This is what the Du/Interlat paper (arXiv:2511.09149) actually uses for
cross-model communication. They achieved cross-family transfer (Qwen→LLaMA)
that outperformed same-family with a trained MHA adapter.

### Training data

Paired hidden states from both models on the same text:
1. Take 500 GSM8K training questions
2. For each question, run it through BOTH models
3. Capture Qwen layer 13 states and Gemma layer 22 states (all token positions)
4. That gives ~500 × 50 tokens = 25,000 paired vectors
5. Train the MLP to minimize MSE: `loss = ||MLP(qwen_states) - gemma_states||²`


CRITICAL: The training pairs must come from the same text processed by both
models independently. Both models tokenize the same question, each produces
hidden states at their respective layers. We resample to aligned positions
(same approach as `calibrate_multitoken_incremental` in `contextual_procrustes.py`).

CRITICAL: Include GSM8K training questions specifically — domain-match the
calibration data to the test domain. Do NOT use Wikipedia or generic text.

CRITICAL: Also include latent-loop-style states in the training data. Run
the LatentMAS loop on Qwen for some prompts, capture the recurrent states.
These are the actual states that will be projected at runtime. If you only
train on normal forward-pass states, the adapter will fail on latent-loop
states for the same reason Procrustes fails — distribution mismatch.

### Implementation

New file: `mlxmas/mlp_adapter.py`

```python
class MLPAdapter(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=2560, output_dim=3584):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        return self.linear2(nn.silu(self.linear1(x)))
```

Training: standard MLX gradient descent.
- Optimizer: AdamW, lr=1e-3
- Batch size: 256 paired vectors per step
- Loss: MSE between MLP(qwen_states) and gemma_states
- Both models FROZEN — only the adapter trains
- Train/val split: 80/20 of the paired vectors
- Stop when val loss plateaus (~50-200 steps)


### Integration with existing pipeline

At runtime, replace `apply_cross_realignment()` with:
```python
projected = mlp_adapter(sender_hidden)  # [1, 33, 3584]
```

No L2 normalization. No norm matching. The adapter learns the correct
magnitudes as part of the mapping. Feed the output directly into
`gemma_forward_from_layer()` at start_layer=22.

### Diagnostic validation

After training, re-run `diagnose_latents.py` but with the MLP adapter
instead of Procrustes. Check:
- Logit decode: do the projected vectors now decode to question-related tokens?
- Mean vector cosine: should be >> 0.22
- Distribution comparison: norms should have natural dynamic range

### Scripts to create

1. `mlxmas/collect_paired_states.py` — Collects paired (Qwen L13, Gemma L22)
   states from GSM8K training prompts. Saves to `data/paired_states_S13_R22.npz`.
   Include both normal forward-pass states AND latent-loop states.

2. `mlxmas/train_mlp_adapter.py` — Loads paired states, trains the MLP,
   saves to `data/mlp_adapter_S13_R22.npz`. Reports train/val MSE and cosine.

3. Update `test_cross.py` to accept `--adapter mlp` flag that loads the
   trained adapter instead of Procrustes.


## Approach 2: Canonical Correlation Analysis (CCA)

### What it is

Instead of rotating the FULL 2560-dim Qwen space into the FULL 3584-dim
Gemma space, CCA finds the SUBSPACE where the two models' representations
are maximally correlated. Maybe only 200 of those 2560 dimensions carry
transferable information. CCA identifies exactly those dimensions and
ignores the rest.

### Why it should work

Procrustes treats all dimensions equally — it finds the best global rotation
across all 2560 dimensions. But many of those dimensions may be model-specific
noise that has no counterpart in the other model. CCA separates "shared
structure" from "model-specific noise" by finding the canonical directions
where the two models agree most.

The diagnostic showed 0.93 variance correlation but only 0.22 mean cosine.
This means the same dimensions are active, but they encode different things
in different directions. CCA finds the subset of directions where both
models encode the SAME thing.

### How it works

Given paired matrices X [N, D_a] and Y [N, D_b]:
1. Compute cross-covariance: C_xy = X^T @ Y
2. Compute within-covariances: C_xx = X^T @ X, C_yy = Y^T @ Y
3. Solve generalized eigenvalue problem
4. Output: projection matrices W_a [D_a, K] and W_b [D_b, K] that map
   both models into a shared K-dimensional space where correlations are maximal


At runtime: project Qwen states through W_a into shared space, then through
W_b^T back into Gemma's space. This is still a linear projection, but it
operates in the subspace where the models actually agree.

### Key parameter: K (shared dimensionality)

K determines how many canonical dimensions to keep. Start with K=256, 512,
1024 and measure diagnostic cosine for each. Higher K retains more information
but also more noise. Lower K is cleaner but may lose important features.

### Implementation

Closed-form, no training. Uses the same paired states as Approach 1.

New file: `mlxmas/cca_adapter.py`

```python
def fit_cca(X, Y, K=512, reg=1e-4):
    """Compute CCA projections.
    X: [N, D_a] sender states
    Y: [N, D_b] receiver states
    K: number of canonical dimensions

    Returns: W_a [D_a, K], W_b [D_b, K], correlations [K]
    """
    # Center
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    # Within-covariance (regularized)
    C_xx = Xc.T @ Xc / N + reg * I_a
    C_yy = Yc.T @ Yc / N + reg * I_b

    # Cross-covariance
    C_xy = Xc.T @ Yc / N


    # Whitened cross-covariance
    C_xx_inv_sqrt = matrix_power(C_xx, -0.5)  # via eigendecomposition
    C_yy_inv_sqrt = matrix_power(C_yy, -0.5)
    T = C_xx_inv_sqrt @ C_xy @ C_yy_inv_sqrt

    # SVD of whitened cross-covariance
    U, S, Vt = svd(T)

    # Take top K canonical directions
    W_a = C_xx_inv_sqrt @ U[:, :K]     # [D_a, K]
    W_b = C_yy_inv_sqrt @ Vt[:K, :].T  # [D_b, K]

    return W_a, W_b, S[:K], mu_x, mu_y
```

Runtime projection:
```python
def apply_cca(x, W_a, W_b, mu_x, mu_y, target_norm):
    shared = (x - mu_x) @ W_a       # [N, K] — shared space
    projected = shared @ W_b.T       # [N, D_b] — back to receiver space
    projected = projected + mu_y     # uncenter
    # Scale norms to match receiver's operating range
    ...
    return projected
```

### Memory note for CCA

The matrix_power (inverse square root) requires eigendecomposition of
[2560, 2560] and [3584, 3584] matrices. Use `mx.linalg.eigh()` on CPU
stream. This is a one-time offline cost, same as Procrustes calibration.

### Scripts to create

1. `mlxmas/cca_adapter.py` — Fit CCA, save projections, apply at runtime.
   Uses the same paired states as Approach 1.
2. Update `test_cross.py` to accept `--adapter cca` flag.
3. Sweep K values: 128, 256, 512, 1024.


## Approach 3: Optimal Transport (Sinkhorn)

### What it is

Procrustes assumes the two distributions are rotations of each other.
The diagnostic proved they're not — different norm ranges, different
variance profiles (before the fix), different covariance structure.

Optimal Transport finds the actual mapping that transforms one distribution
into the other, handling differences in shape, spread, and density.
No assumption of linearity or orthogonality. The Sinkhorn algorithm
gives an approximate solution in polynomial time.

### Why it might work

OT is fundamentally about matching distributions. Given N sender vectors
and N receiver vectors (paired by token position on the same text), OT
finds the transport plan that moves sender vectors to receiver vectors
with minimum total displacement. Unlike Procrustes (rotation only) or
CCA (linear subspace), OT can handle:
- Different covariance structures
- Nonlinear relationships
- Multimodal distributions

### The practical limitation

Full OT between N=25,000 paired vectors in D=2560/3584 dimensions is
expensive. Sinkhorn-Knopp approximation with entropy regularization makes
it tractable but it produces a transport PLAN (soft assignment matrix),
not a transport MAP (a function). To get a reusable function, you need
to either:

(a) Fit a linear map to the transported points (which gets you back to
    something Procrustes-like but with better point correspondences), or
(b) Use a neural network to approximate the transport map — which is
    essentially Approach 1 (MLP adapter) with a different loss function.


### Implementation

New file: `mlxmas/ot_adapter.py`

Use the POT (Python Optimal Transport) library for Sinkhorn.
Install: `pip install POT --break-system-packages`

```python
import ot
import numpy as np

def fit_ot_linear(X, Y, reg=0.05):
    """Compute optimal transport then fit linear map to transported points.
    X: [N, D_a] sender states (numpy)
    Y: [N, D_b] receiver states (numpy)
    """
    # Compute Sinkhorn transport plan
    a = np.ones(N) / N  # uniform source weights
    b = np.ones(N) / N  # uniform target weights
    M = ot.dist(X, Y)   # cost matrix [N, N]
    T = ot.sinkhorn(a, b, M, reg=reg)  # transport plan [N, N]

    # Transport X to Y-space via barycentric mapping
    X_transported = N * T @ Y  # [N, D_b]

    # Now fit a linear map from X to X_transported
    # This is like Procrustes but with OT-corrected correspondences
    W = np.linalg.lstsq(X, X_transported, rcond=None)[0]

    return W  # [D_a, D_b]
```

### Why this is ranked #3

Option (a) collapses back to a linear map — potentially better than
Procrustes because the correspondences are improved by OT, but still
limited by linearity. Option (b) is just Approach 1 with extra steps.
OT's theoretical appeal is strongest, but practically it either reduces
to something we already have or requires significant compute.

Still worth trying because the OT-corrected correspondences might give
a better linear map than Procrustes.


## Execution Order

All three approaches share the same paired-states data. Collect once, use
for all three.

```
Step 1: Collect paired states
    python -m mlxmas.collect_paired_states \
        --sender mlx-community/Qwen3-4B-8bit \
        --receiver mlx-community/gemma-2-9b-it-8bit \
        --sender-layer 13 --receiver-layer 22 \
        --n-prompts 500 --n-positions 50 \
        --include-latent-states \
        --output mlxmas/data/paired_states_S13_R22.npz
    ↓
Step 2: Train MLP adapter (Approach 1)
    python -m mlxmas.train_mlp_adapter \
        --paired-data mlxmas/data/paired_states_S13_R22.npz \
        --output mlxmas/data/mlp_adapter_S13_R22.npz
    ↓
Step 3: Run diagnostics with MLP adapter
    python -u -m mlxmas.diagnose_latents \
        --receiver mlx-community/gemma-2-9b-it-8bit \
        --adapter-type mlp \
        --adapter-path mlxmas/data/mlp_adapter_S13_R22.npz \
        --sender-layer 13 --receiver-layer 22
    ↓
Step 4: If MLP diagnostics show signal, run GSM8K eval
    python -u -m mlxmas.test_cross \
        --receiver mlx-community/gemma-2-9b-it-8bit \
        --adapter mlp --adapter-path mlxmas/data/mlp_adapter_S13_R22.npz \
        --sender-layer 13 --start-layer 22 \
        --max-samples 5
    ↓
Step 5: Fit CCA (Approach 2) on same paired states
    python -m mlxmas.cca_adapter \
        --paired-data mlxmas/data/paired_states_S13_R22.npz \
        --K 512 \
        --output mlxmas/data/cca_adapter_S13_R22_K512.npz
    ↓
Step 6: Run diagnostics with CCA, compare against MLP
    ↓
Step 7: If time/interest, try OT (Approach 3)
```


## Collecting Paired States (SHARED BY ALL APPROACHES)

### Script: `mlxmas/collect_paired_states.py`

This is the most important piece. All three approaches depend on quality
paired data.

For each GSM8K training question:
1. Tokenize with BOTH tokenizers
2. Run through Qwen → capture hidden states at layer 13 (all tokens)
3. Run through Gemma → capture hidden states at layer 22 (all tokens)
4. Resample both to N evenly-spaced positions (handle different tokenizer lengths)
5. Store as paired arrays

The existing `calibrate_multitoken_incremental()` in `contextual_procrustes.py`
already does steps 1-4 for Procrustes. Reuse that pattern.

### CRITICAL: Include latent-loop states

This is what killed Procrustes. The calibration used normal forward-pass
states, but runtime uses latent-loop states which live in a different
distribution.

For ~100 of the 500 prompts, also run the LatentMAS loop on Qwen:
1. Run planner/critic/refiner with 10 latent steps each
2. Capture hidden states at layer 13 during the loop
3. For these states, the "paired" Gemma state is Gemma's layer-22
   activation on the same question text (best available pairing)

This ensures the adapter sees latent-loop-style states during training.

### Output format

```
paired_states_S13_R22.npz:
  sender_states: [N, D_a]     # Qwen layer 13, float32
  receiver_states: [N, D_b]   # Gemma layer 22, float32
  n_forward_pass: int         # how many are from normal forward passes
  n_latent_loop: int          # how many are from latent loop states
```


## Diagnostic Protocol

After EACH approach, run `diagnose_latents.py` (update it to accept
`--adapter-type` and `--adapter-path` flags). Compare these metrics:

| Metric | Procrustes (baseline) | Target |
|---|---|---|
| Logit decode | `itſelf`, `setVerticalGroup` (garbage) | Question-related tokens |
| Variance profile correlation | 0.9274 | ≥ 0.93 (maintain) |
| Mean vector cosine | 0.2221 | > 0.50 at minimum |
| Norm range | [69, 1314] vs real [190, 4713] | Closer to real range |

If logit decode shows question-related tokens (eggs, ducks, market, sells),
there is real signal. Then and ONLY then, run the GSM8K eval.

## GSM8K Test Protocol

CRITICAL: The receiver prompt must NOT contain the question text.

The whole point of this work is to test whether the latent channel carries
information. If the receiver can read the question in text, the test is
worthless.

The prompt for the receiver should be:
```
Using the reasoning context provided, solve the problem step by step.
Put your final numerical answer inside \boxed{}.
```

NO question text. NO hints. If the latent channel works, the receiver
extracts the problem from the injected states. If it doesn't, 0%.

Run with `--max-samples 5` first. If there's signal, scale to 50.


## What NOT To Do

1. **Do NOT put the question text in the receiver prompt.** This was the bug
   that invalidated every prior result. The receiver must extract ALL
   information from the injected latent states.

2. **Do NOT flatten norms.** `apply_cross_realignment` previously forced every
   vector to the same magnitude, destroying the dynamic range. The current
   version preserves per-vector norms. The MLP adapter should learn norms
   naturally through MSE training — do not add any norm matching.

3. **Do NOT train on only forward-pass states.** Include latent-loop states
   in the training data. This is what killed Procrustes.

4. **Do NOT use `latent_comm.compute_alignment()` for the sender.** That
   function returns identity for tied-embedding models. It is superseded
   by the trained adapter.

5. **Do NOT use Wikipedia or generic text for calibration.** Use GSM8K
   training questions — domain-match the calibration to the test.

6. **Do NOT apply the adapter output through `apply_cross_realignment()`.**
   The adapter directly maps Qwen→Gemma. There is no intermediate cross-
   alignment step. The adapter output goes straight into
   `gemma_forward_from_layer()`.

7. **Do NOT read `model.model.embed_tokens.weight` directly.** Quantized
   models store packed weights. Always dequantize by passing token IDs
   through `model.model.embed_tokens()`.

8. **Do NOT assume the Gemma forward pass starts at layer 0.** It starts
   at `start_layer=22`. Gemma applies `sqrt(hidden_size)` scaling only
   at layer 0. When `start_layer > 0`, no scaling is applied — the
   adapter output is fed directly into the layer loop.


## What Success Looks Like

**Minimum viable signal:** Logit decode on projected vectors shows at least
SOME question-related tokens. Mean vector cosine > 0.40. Receiver generates
coherent (not necessarily correct) text about the problem WITHOUT seeing the
question in text.

**Real success:** Receiver answers correctly on 1+ of 5 questions WITHOUT
the question text in the prompt. This proves information transferred through
the latent channel.

**Full success:** Receiver accuracy significantly above chance (> 20%) on 50
questions without question text.

## Ranking

1. **MLP Adapter** — Highest confidence. This is what works in the literature.
   Nonlinear, can learn context-dependent mappings, trains in minutes.
2. **CCA** — Closed-form, no training. Finds the shared subspace. Good
   diagnostic even if it doesn't work as well as MLP — it tells you the
   dimensionality of the shared structure.
3. **Optimal Transport** — Theoretically interesting but practically reduces
   to either a linear map or an MLP. Try last.

## References

- Du et al., "Enabling Agents to Communicate Entirely in Latent Space,"
  arXiv:2511.09149 (trained MHA adapter for cross-model communication)
- Zou et al., "Latent Collaboration in Multi-Agent Systems,"
  arXiv:2511.20639 (LatentMAS same-model communication)
- `mlxmas/diagnose_latents.py` — existing diagnostic script
- `mlxmas/residual_adapter.py` — existing (partial) adapter infrastructure
- `docs/barycentric_projector.md` — prior failed approach (barycentric ridge)
