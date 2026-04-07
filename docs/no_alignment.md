---
title: Cross-Model Latent Communication — Architecture & Findings
author: Nick Fox
date: 2026-04-04
status: Active Research
---

# Cross-Model Latent Communication

## The Problem

Two different LLM families (Qwen3-4B and Gemma-2-2B) need to communicate through latent hidden states instead of text tokens. Same-model latent communication works perfectly — 5/5 GSM8K with zero inter-agent tokens. Cross-model communication degrades because the models have different embedding geometries, different hidden dimensions (2560 vs 2304), and different architectural choices.

## What Works: Same-Model Latent Communication

Based on the LatentMAS paper (Zou et al., arXiv:2511.20639), same-model communication passes KV cache directly between agents. The alignment matrix maps a model's output hidden space back to its own input embedding space. For tied-embedding models like Qwen3-4B, this is approximately identity — the dominant operation is norm-matching.

**Pipeline:**
```
Question → Qwen [planner] ──KV──▶ [critic] ──KV──▶ [refiner] ──KV──▶ [judger] → Answer
             (latent)         (latent)          (latent)          (text output)
```

**Verified results:**

| Configuration | Accuracy | Time/Sample | Memory |
|---|---|---|---|
| MLX 4-bit, sequential, latent_steps=10 | 5/5 (100%) | 13.4s | 5.6 GB |
| PyTorch fp16/MPS, sequential, latent_steps=10 | 5/5 (100%) | 64.1s | 13.2 GB |

This is solved. The same model shares an embedding manifold with itself. No alignment problem exists.

---

## What We Tried: Cross-Model Communication

The challenge: Qwen3-4B (D=2560, 36 layers, 32 attn heads, 8 KV heads) and Gemma-2-2B (D=2304, 26 layers, 8 attn heads, 4 KV heads) have incompatible KV caches. We cannot share KV cache across architectures. Instead, we collect the sender's hidden states, project them into the receiver's embedding space, and feed them as `input_embeddings` to the receiver's transformer.

### Approach 1: Vocabulary Least-Squares (First Attempt)

**Method:** Find 25,139 tokens shared between both vocabularies. Extract their embeddings from both models. Compute a least-squares projection $W_{cross}: \mathbb{R}^{2560} \rightarrow \mathbb{R}^{2304}$ that minimizes reconstruction error on these anchor pairs.

**Pipeline:**
```
Qwen hidden state (Layer-36)
    → self-align to Qwen embedding space (LatentMAS W_a, ~identity for tied embeddings)
    → cross-align to Gemma embedding space (W_cross, vocabulary LS)
    → norm-match to Gemma's embedding scale
    → feed as input_embeddings to Gemma (bypasses embed_tokens)
    → Gemma generates text
```

**Variations tested:**

| Method | A→B Cosine | B→A Cosine |
|---|---|---|
| Raw least-squares | 0.6833 | 0.6361 |
| Mean-centered | 0.6300 | 0.5998 |
| L2-normalized | 0.6843 | 0.6380 |

Centering made it worse — the global offset carries useful structural information in Layer-0 embeddings. L2-normalization was negligibly different from raw.

**Downstream result:** 1/5 GSM8K. Gemma produced coherent, well-structured text but made logical errors — forgetting a subtraction step, dropping a multiplier. The errors were reasoning failures, not gibberish. The 0.68 cosine projection preserved enough structure for coherent generation but not enough for reliable multi-step arithmetic.

**Implementation:** `mlxmas/cross_align.py`, `mlxmas/test_cross.py`

### Approach 2: Contextual Semi-Orthogonal Procrustes (Second Attempt)

**Motivation:** Both GPT-5.4-high and Gemini-3.1-pro independently diagnosed the same root cause (see "External Analysis" section below): the alignment matrix was computed on **Layer-0 vocabulary embeddings** (static, lexical, context-free) but applied to **Layer-N contextual reasoning states** (dynamic, attention-mixed, logic-encoding). This is a distributional mismatch — fitting a map on one type of object to apply to another.

**Method:** Run 500 GSM8K training prompts through both models. Extract the last-layer hidden state of the final token from each. These paired contextual states become the anchor points. Compute SVD-based semi-orthogonal Procrustes instead of unconstrained least-squares, guaranteeing isometric projection (distances and angles preserved).

$$C = H_{qwen,c}^T \cdot H_{gemma,c}$$
$$C = U \Sigma V^T$$
$$W_{ortho} = U V^T \in \mathbb{R}^{2560 \times 2304}$$

**Calibration cosine:** 0.7380 (with 500 prompts). A 50-prompt test showed 0.92 but was overfitting.

**Downstream result:** 0/5 GSM8K. Complete gibberish — repetitive fragments, no coherent reasoning.

**Why it failed:** The Procrustes matrix was computed to map **Qwen Layer-N → Gemma Layer-N**. But the pipeline feeds projected vectors as `input_embeddings` to Gemma at **Layer-0**. Gemma then processes them through all 26 transformer layers. The receiver needs vectors that look like valid **input embeddings**, not like valid **deep contextual states**.

The pipeline architecture makes this clear:

```
VOCABULARY LS (Approach 1) — worked (1/5, coherent text):
  Qwen Layer-36 hidden → self-align to Qwen Layer-0 → cross-align Layer-0→Layer-0 → Gemma Layer-0 input ✓

CONTEXTUAL PROCRUSTES (Approach 2) — failed (0/5, gibberish):
  Qwen Layer-36 hidden → self-align to Qwen Layer-0 → Procrustes Layer-N→Layer-N → Gemma Layer-0 input ✗
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^
                                                       Wrong! Input is Layer-0, not Layer-N.
                                                       Matrix receives Layer-0 data but was
                                                       fit on Layer-N data.
```

**Implementation:** `mlxmas/contextual_procrustes.py`

### The Critical Insight

The external analysis was correct that we had a Layer-0/Layer-N distributional mismatch — but the fix they proposed (align on Layer-N states) was wrong **for this specific pipeline architecture**. The vocabulary alignment was actually doing the right thing: it maps Layer-0 → Layer-0, which is exactly what the receiver needs. The self-alignment step (LatentMAS's W_a) already handles the Layer-N → Layer-0 projection within the sender's space.

The real problem isn't which layer we compute alignment on. It's that a linear projection between two independently trained model families can only capture ~68% of the directional structure, regardless of which layer we use. The remaining 32% is in nonlinear manifold geometry that no single matrix can recover.

---

## External Analysis

Two frontier models (GPT-5.4-high and Gemini-3.1-pro) were given the full problem context and asked for solutions.

### Shared Diagnosis (Both Models Agreed)

Both independently identified the same root cause: **fitting Layer-0 vocabulary embeddings to transfer Layer-N contextual states is a distributional mismatch.** Layer-0 embeddings are static and lexical. Layer-N hidden states are deeply contextualized, attention-mixed, and encode logical structure. A map fit on one distribution and applied to another will underperform.

This diagnosis is correct in principle. Where it went wrong is in the proposed fix — both suggested computing alignment on Layer-N contextual state pairs. This would be correct if we could inject into the receiver at Layer-N (like same-model KV cache sharing). But our cross-model pipeline injects at Layer-0 via `input_embeddings`, so the receiver expects Layer-0-like vectors, not Layer-N states.

### GPT-5.4-high Proposals

**Solution 1: Contextual whitening + CCA/Procrustes.** Pair contextual hidden states, whiten both, compute SVD-based alignment in the whitened space with an α-parameter interpolating between Procrustes (geometry-preserving) and CCA (correlation-maximizing). Mathematically elegant but over-engineered for a first attempt. The simpler SVD Procrustes achieves the same core benefit.

**Solution 2: Piecewise-linear mixture of local experts.** Cluster sender states into K regions, learn local residual maps per cluster with soft gating. Creative but risky — with ~500 calibration pairs split across K=8 clusters, each expert sees ~62 points in 2560 dimensions. Underdetermined. Also suggested stratifying by agent role (planner/critic/refiner), which adds complexity without clear benefit — the alignment should work regardless of what produced the hidden states.

**Solution 3: Tiny latent adapter distilled on contextual pairs.** A 2-layer residual MLP trained on precomputed contextual pairs with cosine + covariance + KL losses. This is essentially a lightweight version of Interlat's communication adapter. Highest-upside but requires training. Estimated ~30 minutes on M2 Pro.

**Evaluation insight (important):** Stop measuring cosine on vocabulary embeddings. The true metric should be held-out contextual cosine, or better yet, downstream task accuracy. A method can improve reasoning transfer substantially while barely moving the static-embedding metric. This point is well-taken.

### Gemini-3.1-pro Proposals

**Solution 1: Semi-orthogonal Procrustes.** Same core idea as GPT's solution 1 but presented more cleanly — 4-line SVD formulation. Correctly noted that Procrustes guarantees isometric projection ($W^T W = I$), preventing the variance-shrinkage problem of unconstrained least-squares.

**Solution 2: Contextual activation anchoring.** Collect Layer-N states from both models on identical prompts, use those as alignment anchors instead of vocabulary. This is the approach we tested. Correct in principle, wrong for our pipeline architecture (see above).

**Solution 3: Non-linear latent residual adapter.** Frozen Procrustes baseline + trainable 2-layer MLP residual. Very similar to GPT's solution 3 and to Interlat's architecture.

**Recommended combination:** Gemini suggested combining Solutions 1 and 2 — contextual Procrustes. We tested this. It failed for the architectural reason described above.

---

## What We Learned

### 1. Layer-space matching matters more than alignment algorithm

The choice of alignment algorithm (least-squares vs Procrustes vs CCA) is secondary to getting the layer-space mapping right. A simple least-squares on Layer-0 embeddings (0.68 cosine, 1/5 accuracy) dramatically outperformed a sophisticated Procrustes on Layer-N states (0.74 cosine on calibration data, 0/5 accuracy) because the pipeline architecture was injecting at Layer-0.

### 2. Cosine similarity on calibration data is not a reliable predictor of downstream performance

The contextual Procrustes had higher measured cosine (0.74) than the vocabulary LS (0.68) but produced complete gibberish downstream. The metric was measuring the right quantity on the wrong distribution. Downstream task accuracy is the only metric that matters.

### 3. The vocabulary alignment is the best current baseline for Layer-0 injection

The two-step process — self-align (Layer-N → Layer-0 in sender space) then cross-align (Layer-0 → Layer-0 across models) — correctly matches the pipeline's injection point. It is structurally sound but limited: a shared-vocabulary token table is a rough bridge between two model families' representations, not a final solution.

### 4. The 0.68 cosine is the ceiling of the current anchor/objective, not necessarily of linear transfer

What we've shown is that raw least-squares on shared token embeddings tops out around 0.68, and centering/L2 normalization don't help. We have **not** exhausted contextual anchors at the right target space, whitening/anisotropy correction, or layer-matched injection. The ceiling may be higher with better-matched anchor types and injection points.

### 5. Cross-model communication at 0.68 cosine is partially functional

At 1/5 accuracy, the linear projection is good enough for coherent text generation but not for reliable multi-step reasoning. The errors are logical (missing an arithmetic step), not syntactic (gibberish). This suggests the geometric structure for "language production" survives the projection better than the structure for "logical chaining."

### 6. Interlat's cross-family success uses a fundamentally different approach

The Interlat paper (Du et al., 2026, arXiv:2511.09149v3) achieved cross-family results where Qwen→LLaMA *outperformed* same-family communication. But they used a **trained** communication adapter (8-head MHA + layer norm + projection) with curriculum learning, separation loss, and plan-aligned regularization. The adapter learns a nonlinear, context-dependent, sequence-aware transformation — not a single global matrix. This required 8×A100-80GB GPUs to train.

### 7. We are not limited to Layer-0 injection

This is the most important insight from the second round of external analysis. Our `gemma_forward_with_embeddings()` function already manually iterates through Gemma's transformer layers — we built it because Gemma's MLX implementation lacks `input_embeddings`. There is nothing stopping us from starting at Layer-4, or Layer-8, or any layer. The "receiver accepts only Layer-0" constraint was self-imposed based on the stock API, not the manual forward pass we already control.

---

## Second Round External Analysis

After Approach 2 failed, the architecture document was sent to both GPT-5.4-high and Gemini-3.1-pro for review. Their responses corrected and extended the analysis.

### Gemini-3.1-pro (Second Round)

**Confirmed:** The Layer-N→Layer-N Procrustes failure diagnosis was correct.

**Warned against direct L-N→L-0 Procrustes.** Gemini identified a fatal flaw in the proposed path: pairing Qwen's Layer-36 reasoning state (which encodes the model's impending *solution*) with Gemma's Layer-0 mean-pooled prompt embeddings (which encode the *question*) is a semantic mismatch — fitting the matrix to map answers back to questions. When run at inference, Gemma would receive a compressed echo of the prompt, not Qwen's reasoning.

**Recommended:** Keep the structurally correct Approach 1 pipeline (self-align → cross-align → Layer-0 inject) but replace the static $W_{cross}$ with a trained residual adapter. Frozen linear baseline + 2-layer SiLU MLP, ~5 minutes training on M2 Pro. Applied strictly at the L-0→L-0 step.

### GPT-5.4-high (Second Round)

**Key insight: Intermediate layer injection.** Since we already have manual forward-pass control over Gemma, we are not limited to Layer-0 injection. We can inject at **any** residual-stream layer by starting the Gemma forward loop at block `m` and running layers `m` through 26.

**Argued:** Cross-model representations are often more alignable at intermediate layers than at Layer-0 (too lexical/tokenizer-dependent) or Layer-N (too task-specialized). The middle layers tend to converge on more universal semantic representations. Fitting a contextual map $h_{Qwen}^{(s)} \rightarrow h_{Gemma}^{(m)}$ and injecting at Gemma layer $m$ is more natural than forcing a deep reasoning state to masquerade as a lexical embedding.

**Proposed search grid:** Sender layer $s \in \{23, 29, 35\}$ (0-indexed), receiver layer $m \in \{0, 2, 4, 6, 8\}$.

**Also proposed:** A factorized bridge — contextual hidden-to-hidden Procrustes ($R_{cross}$) followed by Gemma's own self-alignment ($A_{Gemma}$: Layer-m → Layer-0). Both matrices are closed-form, still training-free.

**Cautioned:** 500 calibration prompts may be too thin for a stable 2560×2304 cross-space estimate. Recommended several thousand paired vectors. Also noted that calibration data should match the actual transmitted objects (latent thought steps, not just final-token states).

### Synthesis: Where Both Were Right and Wrong

**Both agreed:** Direct L-N→L-0 with mean-pooled targets is a weak pairing. Both independently flagged problems with it. This path is demoted from "correct path forward" to "cheap ablation, expected to underperform."

**GPT was stronger on architecture:** The intermediate layer injection insight is the most important new idea. It reframes the entire problem — instead of degrading contextual states down to lexical embeddings, inject contextual states where the receiver processes contextual representations.

**Gemini was stronger on the fallback:** If training-free methods plateau, the residual adapter architecture (frozen linear + trainable MLP residual) is well-specified and feasible on M2 Pro.

---

## The Path Forward (Revised)

### Priority 1: Intermediate Layer Injection (Training-Free)

**Core idea:** Instead of injecting projected states at Gemma Layer-0 (where they must look like lexical embeddings), inject at an intermediate Gemma layer where the representations are contextual and more likely to align across model families.

**Implementation:** Modify `gemma_forward_with_embeddings()` to accept a starting layer parameter:

```python
def gemma_forward_from_layer(model, hidden_states, cache, start_layer=0):
    """Run Gemma forward pass starting from an intermediate layer."""
    h = hidden_states
    if start_layer == 0:
        h = h * (model.model.args.hidden_size ** 0.5)  # embedding scaling
    
    layers = model.model.layers[start_layer:]
    cache_slice = cache[start_layer:]
    
    mask = create_attention_mask(h, cache_slice[0], return_array=True)
    for layer, c in zip(layers, cache_slice):
        h = layer(h, mask, c)
    
    h = model.model.norm(h)
    out = model.model.embed_tokens.as_linear(h)
    out = mx.tanh(out / model.final_logit_softcapping)
    out = out * model.final_logit_softcapping
    return out
```

**Calibration:** Run ~2000 GSM8K prompts through both models. At each layer of interest, extract the final-token hidden state. Compute Procrustes SVD for each (sender_layer, receiver_layer) pair.

**Calibration distribution — alternatives to investigate:**

The default calibration (final-token hidden states) may introduce a distributional mismatch analogous to the Layer-0/Layer-N error from Approach 2. In the LatentMAS pipeline, the actual transmitted objects are multi-token latent thought sequences across multiple steps — not single final-token vectors. If the Procrustes matrix is calibrated on final-token states but applied at inference to multi-token latent step outputs, the alignment could underperform on the real transmission distribution. Three alternatives worth testing:

1. **Mean-pooled sequence states.** Instead of extracting only the final token's hidden state per prompt, extract all token hidden states from the sequence and mean-pool them. This gives calibration vectors that better represent the average geometry of the full token sequence, though it washes out position-dependent structure.

2. **Latent step outputs from the pipeline itself.** Run the actual LatentMAS pipeline (same-model, Qwen→Qwen) and intercept the hidden states at each latent thought step as they would be transmitted. These are the real objects the cross-model alignment must handle. Calibrating on them eliminates the proxy mismatch entirely, at the cost of requiring a working same-model pipeline to generate calibration data.

3. **Stratified calibration combining both.** Compute Procrustes on a mixture of final-token states (cheap to collect, good coverage of the embedding space) and actual latent step outputs (expensive but distribution-matched). Weight the latent step pairs more heavily since they represent the true operating distribution.

The risk is the same pattern as Approach 2: optimizing a metric (calibration cosine) on a distribution that doesn't match the inference distribution. Given that this exact failure mode already cost us an entire experimental round, it's worth running at least alternatives 1 and 2 as ablations alongside the default final-token calibration.

**Layer grid:**
- Sender (Qwen): layers 23, 29, 35 (0-indexed; Qwen3-4B has 36 layers, 0-35)
- Receiver (Gemma): layers 0, 2, 4, 6, 8 (0-indexed; Gemma-2-2B has 26 layers, 0-25)

**Evaluation:** Measure held-out contextual cosine AND downstream GSM8K accuracy for each pair. The cosine metric is only useful when measured on the correct distribution — same layer types as the actual transmission.

**Complexity:** Low-medium. Modifying the forward function is ~10 lines. Calibration sweep is ~15 minutes for 2000 prompts × 2 models × multiple layers. Each Procrustes solve is <1 second.

### Priority 2: Factorized Bridge (Training-Free Fallback)

If no single layer pair produces strong alignment, use a two-step factorized approach:

$$h_{Qwen}^{(s)} \xrightarrow{R_{cross}} h_{Gemma}^{(m)} \xrightarrow{A_{Gemma}} e_{Gemma}^{(0)}$$

Where $R_{cross}$ is contextual hidden-to-hidden Procrustes (fit on paired contextual states), and $A_{Gemma}$ is Gemma's own self-alignment (Layer-m hidden states → Layer-0 compatible embeddings), fit purely from Gemma data:

$$A_{Gemma} = \arg\min_A \|H_{Gemma}^{(m)} A - E_{Gemma}^{(0)}\|_F^2$$

Both matrices are closed-form. Still training-free. Inject the result at Layer-0.

### Priority 3: Residual Adapter (Lightweight Training)

If all linear methods plateau, the cross-model relationship is genuinely nonlinear. Train a micro-adapter on the best linear baseline:

$$f_\theta(x) = x W_{best} + W_2 \cdot \text{SiLU}(W_1 \cdot \text{LayerNorm}(x))$$

Where $W_{best}$ is frozen (best Procrustes from Priority 1 or 2), $W_1 \in \mathbb{R}^{D_{sender} \times 512}$, $W_2 \in \mathbb{R}^{512 \times D_{receiver}}$, ~2.6M trainable parameters. Initialize $W_2$ near zero so the adapter starts at the linear baseline.

Train on precomputed contextual pairs with cosine + MSE loss. AdamW, weight decay 0.01, dropout 0.1, early stopping. ~5-10 minutes on M2 Pro.

### What We Would NOT Do

- **Direct L-N→L-0 Procrustes with mean-pooled targets.** Both GPT and Gemini flagged this as a weak pairing. Demoted to "cheap ablation" if we want a data point, not a mainline approach.
- **Role-stratified alignment.** Adds complexity without clear benefit. The alignment should work on whatever tensor it receives regardless of cognitive role.
- **Full Interlat-style SFT.** Overkill for this stage. 8×A100 training is not warranted when we haven't exhausted training-free options.

---

## Summary of All Results

| Approach | Cosine | GSM8K | Output Quality |
|---|---|---|---|
| Same-model LatentMAS (Qwen→Qwen) | ~1.0 | 5/5 | Perfect |
| Vocab LS, L-0→L-0 inject (Qwen→Gemma) | 0.68 | 1/5 | Coherent, logical errors |
| Contextual Procrustes, L-N→L-N inject at L-0 | 0.74* | 0/5 | Gibberish |
| **Next: Contextual Procrustes, inject at Gemma L-m** | TBD | TBD | TBD |

*\*Cosine measured on calibration data, not held-out. Misleading metric.*

---

## Implementation Files

```
mlxmas/
├── latent_comm.py              # Core: alignment, latent thoughts, KV transfer (same-model)
├── cross_align.py              # Vocabulary-based cross-model alignment (Approach 1)
├── contextual_procrustes.py    # Contextual Procrustes calibration + save/load
├── test_cross.py               # Cross-model test harness (supports both alignment methods)
├── run.py                      # Same-model LatentMAS entry point
├── prompts.py                  # Agent prompt builders
└── utils.py                    # Answer extraction, normalization
```

## References

**Core papers:**
- Zou et al., "Latent Collaboration in Multi-Agent Systems," arXiv:2511.20639, 2025 (LatentMAS — same-model latent communication, KV cache transfer, alignment matrix)
- Du et al., "Enabling Agents to Communicate Entirely in Latent Space," arXiv:2511.09149v3, 2026 (Interlat — trained adapter, cross-family communication, latent compression)

**Cross-space alignment:**
- Conneau et al., "Word Translation Without Parallel Data," arXiv:1710.04087, 2018 (Orthogonal Procrustes for cross-lingual embedding alignment — the industry standard)
- Smith et al., "Offline Bilingual Word Vectors, Orthogonal Transformations and the Inverted Softmax," 2017

**Representation analysis:**
- Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis," NeurIPS 2017 (comparing neural network representations across layers/models)
- Kornblith et al., "Similarity of Neural Network Representations Revisited," ICML 2019

**Adapter architectures:**
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," 2019
- Mohiuddin & Joty, "Revisiting Adversarial Autoencoder for Unsupervised Word Translation," 2019 (non-linear cross-lingual mappings)

**Mixture of experts:**
- Jacobs et al., "Adaptive Mixtures of Local Experts," 1991
- Jordan & Jacobs, "Hierarchical Mixtures of Experts," 1994


---

## Implementation Spec for Priority 1: Intermediate Layer Injection

This section is a task list for implementation. Each task specifies exactly what to change, where, and how to verify it works.

### Prerequisites (Cleanup)

**Task 0a: Add `__init__.py` to `mlxmas/`**

Create `mlxmas/__init__.py` (empty file). This allows package-style imports and lets the code be run from the project root.

**Task 0b: Move `gemma_forward_with_embeddings` out of `test_cross.py`**

Move `gemma_forward_with_embeddings()` from `test_cross.py` into a new file `mlxmas/cross_comm.py`. This function is the core of cross-model communication and will be modified heavily. It should not live in a test file.

`cross_comm.py` will also hold the new `gemma_forward_from_layer()` function (Task 1).

Update `test_cross.py` to import from `cross_comm`.

**Task 0c: Remove dead code in `contextual_procrustes.py`**

`extract_final_hidden_state_gemma()` (lines 50-62) is identical to `extract_final_hidden_state()` (lines 35-48). Delete the Gemma version. Update `collect_contextual_pairs()` to always use the single function.

**Task 0d: Normalize arg parsing in `test_cross.py`**

Replace manual `sys.argv` parsing for `--procrustes` (lines 212-215) with argparse, consistent with every other script.


### Task 1: Parameterized Gemma Forward Pass

**File:** `mlxmas/cross_comm.py` (new)

**Function signature:**

```python
def gemma_forward_from_layer(model, hidden_states, cache, start_layer=0):
    """Run Gemma forward pass starting from an intermediate layer.
    
    Args:
        model: Gemma-2 MLX model (the full Model, not model.model)
        hidden_states: [B, seq, D] tensor to inject
        cache: list of KVCache objects (one per layer, all 26)
        start_layer: which Gemma layer to begin processing at
        
    Returns:
        logits: [B, seq, vocab] output logits
        
    CRITICAL: sqrt(hidden_size) embedding scaling is ONLY applied when
    start_layer == 0. Intermediate representations are already at 
    operating scale — scaling them would distort magnitudes.
    """
```

**Implementation notes:**

- When `start_layer == 0`: apply `h = h * (hidden_size ** 0.5)` (Gemma embedding scaling)
- When `start_layer > 0`: skip the scaling entirely
- Iterate `model.model.layers[start_layer:]` with `cache[start_layer:]`
- Attention mask must be computed from the correct cache entry. Use `cache[start_layer]` (the first cache entry that will actually have data), NOT `cache[0]` (which will be empty for `start_layer > 0`)
- Apply `model.model.norm(h)` after the layer loop
- Apply logit soft-capping: `embed_tokens.as_linear(h)` → `tanh(out / softcap) * softcap`

**Verify:** Call with `start_layer=0` on same inputs as current `gemma_forward_with_embeddings()`. Outputs must be identical (bitwise).


### Task 2: Cache Handling for Intermediate Injection

**File:** `mlxmas/cross_comm.py`

**Problem:** `make_prompt_cache(model)` creates KV cache entries for all 26 Gemma layers. When injecting at layer `m`, layers 0 through `m-1` are skipped — their cache entries are never written. The attention mask in `create_attention_mask(h, cache[start_layer])` computes position offsets from the cache. If `cache[start_layer]` is empty (offset=0), the mask will be correct for the first injection. On subsequent injections into the same cache, `cache[start_layer].offset` will track the sequence position correctly for layers `m` through 25, while `cache[0]` through `cache[m-1]` remain at offset 0.

**Investigation needed:** Verify that MLX's `create_attention_mask` only depends on the cache entry passed to it, not on other cache entries. If so, passing `cache[start_layer]` is sufficient. If the mask function inspects all cache entries, we need to either:
1. Pad the skipped cache entries with zeros to maintain consistent offsets, or
2. Create a partial cache list containing only entries for layers `m` through 25

**Verify:** Feed 3 sequential embeddings at layer `m` into a fresh cache. Check that `cache[start_layer].offset == 3` and `cache[0].offset == 0`. Confirm the generated text is coherent (not broken by mask issues).

### Task 3: Multi-Layer Hidden State Extraction

**File:** `mlxmas/contextual_procrustes.py` (extend)

**New function:**

```python
def extract_hidden_at_layers(model, tokenizer, prompt_text, layers):
    """Run prompt through model, return hidden states at specified layers.
    
    Args:
        model: MLX model
        tokenizer: tokenizer
        prompt_text: input prompt
        layers: list of int, which layer outputs to capture
        
    Returns:
        dict mapping layer_index -> [1, D] final-token hidden state
        
    Implementation: Hook into the forward pass by manually iterating
    model.model.layers and capturing intermediate h values.
    """
```

**Implementation notes:**

- Create cache with `make_prompt_cache(model)`
- Tokenize and embed: `h = model.model.embed_tokens(input_ids)`
- For Gemma: apply `h = h * (hidden_size ** 0.5)`
- For Qwen: apply whatever scaling Qwen does (check `other/qwen3.py` or the mlx_lm source)
- Compute mask once
- Iterate layers, capturing `h` after each layer in `layers`
- Extract final-token `h[:, -1, :]` at each captured point
- Do NOT run through `model.model.norm` or lm_head — we want raw residual-stream states

**Add held-out split:** When collecting calibration pairs, reserve 20% of prompts. Report cosine on both train and held-out sets. Flag if held-out is >0.05 below train (overfitting signal).


### Task 4: Layer-Pair Sweep

**File:** New script `mlxmas/sweep_layers.py`

**Purpose:** Compute Procrustes alignment for each (sender_layer, receiver_layer) pair and evaluate downstream accuracy.

**Parameters:**
- Sender layers (Qwen): `[23, 29, 35]` (0-indexed)
- Receiver layers (Gemma): `[0, 2, 4, 6, 8]` (include 0 as baseline)
- Calibration prompts: 2000 GSM8K training prompts
- Held-out: 400 prompts (20%)
- Eval: the existing 5 GSM8K test questions

**Workflow per (s, m) pair:**

1. Load both models
2. Extract hidden states at layer `s` from Qwen, layer `m` from Gemma, for all 2400 prompts
3. Split 2000 train / 400 held-out
4. Compute Procrustes on train pairs: `H_qwen_s` → `H_gemma_m`
5. Report train cosine and held-out cosine
6. Run 5-question GSM8K eval using `gemma_forward_from_layer(start_layer=m)`
7. Log all results to JSON

**Output format:** `sweep_results.json`

```json
{
  "pairs": [
    {
      "sender_layer": 36,
      "receiver_layer": 0,
      "train_cosine": 0.74,
      "heldout_cosine": 0.71,
      "gsm8k_accuracy": 0.2,
      "gsm8k_correct": 1,
      "gsm8k_total": 5
    },
    ...
  ]
}
```


**Memory budget:**

- Qwen3-4B 4-bit: ~2.5 GB
- Gemma-2-2B 4-bit: ~1.5 GB
- KV caches (both models): ~2 GB worst case
- Calibration tensors (2400 prompts × 2560 × float32): ~23 MB per layer — negligible
- Total: ~6 GB active, well within 32 GB

Both models can stay loaded throughout the sweep. No need to load/unload between pairs.

**Calibration distribution alternatives (from review):**

For each (s, m) pair, run three calibration variants:

1. **Final-token states** (default): Extract final-token hidden at layer `s`/`m` per prompt. This is the cheapest and most straightforward.

2. **Mean-pooled sequence states**: Extract all token hidden states at layer `s`/`m`, mean-pool across the sequence dimension. Produces calibration vectors that represent the average geometry of the full sequence rather than a single position.

3. **Latent step outputs** (if time permits): Run the same-model Qwen LatentMAS pipeline, intercept the self-aligned hidden states at each latent thought step. These are the actual objects the cross-model alignment will handle at inference. Pair them with Gemma Layer-`m` states extracted from the same prompts.

Run variants 1 and 2 for every (s, m) pair. Run variant 3 for the top-2 pairs by held-out cosine from variant 1.

### Task 5: Update `test_cross.py` for Intermediate Injection

**File:** `mlxmas/test_cross.py`

Once the best (s, m) pair is identified from the sweep:

1. Update `collect_sender_hidden_states()` to extract at layer `s` instead of the final layer. This means calling `extract_hidden_at_layers(sender_model, ..., layers=[s])` during the latent loop instead of relying on `model.model()` which returns the last layer's output.

2. Update `generate_with_cross_latents()` to call `gemma_forward_from_layer(start_layer=m)` instead of `gemma_forward_with_embeddings()`.

3. Update the Procrustes loading to expect the layer-pair-specific alignment matrix.

4. Run the 5-question eval. Compare against the baseline (1/5 with vocab LS at Layer-0).


### Execution Order

```
0a. Create mlxmas/__init__.py
0b. Move gemma_forward_with_embeddings → cross_comm.py
0c. Delete extract_final_hidden_state_gemma
0d. Fix argparse in test_cross.py
 ↓
1.  Implement gemma_forward_from_layer(start_layer) in cross_comm.py
    Verify: start_layer=0 matches old function bitwise
 ↓
2.  Investigate cache behavior for start_layer > 0
    Verify: 3 sequential embeddings at layer m, check offsets
 ↓
3.  Implement extract_hidden_at_layers() in contextual_procrustes.py
    Add held-out split to compute_procrustes()
 ↓
4.  Run sweep_layers.py — all (s, m) pairs with final-token + mean-pooled calibration
    Pick best pair by held-out cosine + downstream accuracy
 ↓
5.  Update test_cross.py to use best (s, m) pair
    Run 5-question eval, compare against 1/5 baseline
```

### Current Function Signatures (Reference)

These are the existing functions that the implementation will call or modify:

```python
# latent_comm.py
def compute_alignment(model) -> tuple[mx.array, mx.array]
    # Returns (W_a, target_norm) — self-alignment for same-model
def apply_realignment(hidden, W_a, target_norm, use_realign=True) -> mx.array
    # Project hidden → embedding space (same model)
def latent_forward(model, tokenizer, prompt_text, cache, latent_steps, W_a, target_norm, use_realign=True) -> list
    # Run prompt + latent loop, returns updated cache
def generate_with_cache(model, tokenizer, prompt_text, cache, max_tokens=1024, temperature=0.6, top_p=0.95) -> str

# cross_align.py
def compute_cross_alignment(model_a, tok_a, model_b, tok_b) -> dict
    # Returns {W_ab, W_ba, mean_a, mean_b, target_norm_a, target_norm_b, ...}
def apply_cross_realignment(hidden, W_cross, mean_sender, mean_receiver, target_norm) -> mx.array

# contextual_procrustes.py
def extract_final_hidden_state(model, tokenizer, prompt_text) -> mx.array  # [1, D]
def collect_contextual_pairs(model_a, tok_a, model_b, tok_b, prompts, is_gemma_b=True) -> tuple[mx.array, mx.array]
def compute_procrustes(H_a, H_b) -> dict
    # Returns {W_ortho, mean_a, mean_b, target_norm_a, target_norm_b, cos_sim, ...}
def save_alignment(result, path)
def load_alignment(path) -> dict

# test_cross.py
def collect_sender_hidden_states(model, tokenizer, prompt_text, cache, latent_steps, W_a_self, target_norm_self) -> mx.array
    # Returns [1, latent_steps+1, D_sender] in sender embedding space
def gemma_forward_with_embeddings(model, input_embeddings, cache) -> mx.array
    # → Moving to cross_comm.py, replaced by gemma_forward_from_layer
def generate_with_cross_latents(receiver_model, receiver_tokenizer, projected_embeddings, question, ...) -> str
```


### Risks and Open Questions

1. **Cache mask at intermediate layers.** This is the biggest unknown. If `create_attention_mask` doesn't work correctly with a partially-populated cache, the entire intermediate injection approach needs a cache workaround. Task 2 investigates this before any sweep runs. If it breaks, the fix is to create a truncated cache list for layers `m:26` only.

2. **Qwen intermediate layer extraction changes the sender pipeline.** Currently `model.model(input_ids, cache=cache)` returns the final-layer output and populates the full cache. To extract at layer `s < 36`, we need to either (a) run the full forward pass and capture intermediate `h` values, or (b) stop early at layer `s`. Option (a) is safer — the sender's cache still gets fully populated for the latent thought loop. Option (b) saves compute but means the sender's latent loop operates on shallower representations, which changes the nature of the "thinking."

3. **Norm mismatch between sender layer `s` and receiver layer `m`.** Procrustes preserves angles but not scales. The norm-matching step after projection uses `target_norm_b` computed from receiver layer `m` states. If the variance of norms at layer `m` is high, a single scalar target may be insufficient. Monitor per-sample norm ratios during calibration.

4. **The 5-question eval is too small for statistical significance.** A 1/5 → 2/5 improvement could be noise. Once a promising (s, m) pair is found, expand to 50+ GSM8K problems for confirmation. But 5 questions is fine for rapid iteration during the sweep.

5. **Latent step calibration (variant 3) requires same-model pipeline running.** This means Qwen must be loaded and running LatentMAS during calibration data collection. Memory: Qwen 4-bit (~2.5 GB) + Gemma 4-bit (~1.5 GB) + Qwen KV cache (~1 GB) ≈ 5 GB. Fits comfortably.


---

## Empirical Layer Selection (April 4, 2026)

The implementation spec above proposed a layer-pair sweep with sender layers `{23, 29, 35}` and receiver layers `{0, 2, 4, 6, 8}`. Those values were adapted from GPT-5.4-high's initial suggestion of `{24, 30, 36}` (converted to 0-indexed). However, those numbers were a heuristic guess — "sample the final 25% of the sender" — with no empirical basis for these specific models.

Before implementing the cross-model pipeline, we ran three empirical tests to determine the correct layer pairs using actual data from Qwen3-4B and Gemma-2-2B on GSM8K tasks, rather than extrapolating from papers about different models.

### Literature Review

Three peer-reviewed papers informed the test design:

1. **"Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning"** (arXiv:2510.02091, Jan 2026). Systematic layer pruning on Qwen3-8B and LLaMA-3.1-8B. Found that reasoning performance on GSM8K is highly sensitive to shallow and middle layers. For Qwen3-8B, layers 6, 23, and 35 showed the largest accuracy drops when pruned. Key finding: "depth usage is inherently task-dependent, highly metric-sensitive, and strongly model-specific."

2. **"Semantic Convergence: Investigating Shared Representations Across Scaled LLMs"** (arXiv:2507.22918, Jul 2025). Compared Gemma-2-2B and Gemma-2-9B using SAE features. Found that middle layers yield the strongest cross-model overlap, while early and late layers show much less similarity.

3. **"Bridging Critical Gaps in Convergent Learning"** (arXiv:2502.18710, Nov 2025). Cross-architecture comparison using Procrustes alignment. Found that layers at similar proportional depths align most closely across architectures, and that Procrustes and linear alignment scores are nearly identical, validating Procrustes as the right tool.


### Test 1: Full CKA Heatmap (`test1_cka_heatmap.py`)

**Purpose:** Measure cross-model representational alignment at every possible layer pair, without assumptions about which layers matter.

**Method:** Ran 200 GSM8K training prompts through both Qwen3-4B (8-bit) and Gemma-2-2B (8-bit). At every layer of both models, extracted the final-token hidden state (raw residual stream, no RMSNorm or lm_head). Computed linear CKA (Kornblith et al., ICML 2019) for all 36 × 26 = 936 (sender, receiver) layer pairs. CKA measures representational similarity invariant to rotation and scaling — a value of 1 means identical structure, 0 means orthogonal.

**Results:** The top CKA pairs are all in the early-to-middle layers:

| Rank | Qwen Layer | Gemma Layer | CKA |
|------|-----------|-------------|-----|
| 1 | 6 | 4 | 0.831 |
| 2 | 7 | 4 | 0.831 |
| 3 | 7 | 6 | 0.831 |
| 4 | 12 | 9 | 0.824 |
| 5 | 12 | 10 | 0.820 |
| 6 | 3 | 2 | 0.816 |
| 7 | 14 | 10 | 0.813 |
| 8 | 13 | 10 | 0.811 |

Not a single late Qwen layer (28, 31, 34, 35) appeared in the top 50 pairs. Cross-model alignment peaks in the middle of both networks and drops off sharply in the final layers where model-specific specialization diverges.

**Data:** `mlxmas/cka_heatmap.json`


### Test 2: Layer Skip Test (`test2_layer_skip.py`)

**Purpose:** Determine which layers are critical for reasoning in each model, independently. This answers: where should the sender extract from (where reasoning lives), and where should the receiver accept injection (where it needs runway for its own reasoning).

**Method:** For each model, skip one layer at a time (replace its output with its input — identity residual) and measure the degradation in log-probability of the correct GSM8K answer. Larger negative delta = more critical layer. 10 GSM8K questions, 8-bit models.

**Qwen3-4B Results — Most critical layers (negative delta = hurts when skipped):**

| Rank | Layer | Depth | Delta |
|------|-------|-------|-------|
| 1 | 0 | 0% | -3.06 |
| 2 | 1 | 3% | -0.65 |
| 3 | 11 | 31% | -0.42 |
| 4 | 7 | 19% | -0.39 |
| 5 | 23 | 64% | -0.33 |
| 6 | 6 | 17% | -0.31 |
| 7 | 13 | 36% | -0.24 |
| 8 | 28 | 78% | -0.19 |

Qwen3-4B least critical layers (skipping *improves* log-prob): Layer 34 (+1.04), 35 (+0.75), 29 (+0.70), 19 (+0.67). The final two layers are the most dispensable — this is the logit lens collapse effect. The model's reasoning concentrates in two bands: early (6-13) and mid-deep (23, 28).


**Gemma-2-2B Results — Most critical layers:**

| Rank | Layer | Depth | Delta |
|------|-------|-------|-------|
| 1 | 25 | 96% | -0.69 |
| 2 | 22 | 85% | -0.33 |
| 3 | 23 | 88% | -0.28 |
| 4 | 20 | 77% | -0.19 |
| 5 | 21 | 81% | -0.17 |

Only layers 19-25 actually hurt when skipped. Layers 0-18 are all dispensable — skipping layer 4 *improves* log-prob by +1.15. Gemma's reasoning is concentrated entirely in the last 25% of the network (layers 19-25). The first 18 layers handle lexical and syntactic processing that can be safely bypassed.

**Key insight from Test 2:** Qwen's reasoning lives in layers 6-13 and 23-28. Gemma's reasoning lives in layers 19-25. These are different absolute positions but similar proportional depths for the reasoning-critical bands.

**Data:** `mlxmas/layer_skip_Qwen3-4B-8bit.json`, `mlxmas/layer_skip_gemma-2-2b-it-8bit.json`


### Test 3: Injection Probe (`test3_injection_probe.py`)

**Purpose:** The definitive test — actually inject Procrustes-projected Qwen states into Gemma at specific layers and measure whether Gemma can use them.

**Method:** For each candidate layer pair (s, m):
1. Collect 500 GSM8K calibration prompts
2. Extract hidden states at layer `s` from Qwen and layer `m` from Gemma for all 500 prompts
3. Compute semi-orthogonal Procrustes alignment on the paired states
4. For 5 GSM8K eval questions: extract Qwen's hidden state at layer `s`, project via Procrustes into Gemma's layer-`m` space, inject into Gemma at layer `m` (bypassing layers 0 through `m-1`), then feed the question and correct answer through Gemma with the injected state in its KV cache
5. Measure Gemma's log-probability of the correct answer — higher (less negative) means the injection was more useful

**Layer pairs tested:** Selected from the intersection of "Qwen has reasoning signal" (Test 2) and "high CKA alignment" (Test 1): `{7,4; 7,8; 11,6; 11,9; 11,10; 13,9; 13,10; 13,12}`

**Results (ranked by receiver log-probability, less negative = better):**

| Rank | Sender (Qwen) | Receiver (Gemma) | Cal. Cosine | Log-prob |
|------|--------------|-----------------|-------------|----------|
| 1 | 13 | 12 | 0.877 | -6.062 |
| 2 | 13 | 10 | 0.883 | -6.116 |
| 3 | 11 | 10 | 0.878 | -6.120 |
| 4 | 11 | 9 | 0.879 | -6.162 |
| 5 | 7 | 8 | 0.856 | -6.162 |
| 6 | 13 | 9 | 0.881 | -6.163 |
| 7 | 11 | 6 | 0.823 | -6.197 |
| 8 | 7 | 4 | 0.863 | -6.233 |

Gemma's standalone baseline (no injection, from Test 2): -5.997


**Winner: Qwen Layer 13 → Gemma Layer 12.**

The calibration cosine of 0.877 is a 29% relative improvement over the 0.68 from the vocabulary least-squares approach. The injection degrades Gemma's log-prob by only 0.065 (from -5.997 to -6.062), meaning a single projected hidden state from Qwen barely disrupts Gemma's reasoning — the foreign representation is nearly compatible.

**Data:** `mlxmas/injection_probe.json`

### Synthesis: What the Data Shows

The three tests converge on a single conclusion that contradicts both GPT-5.4-high's and Gemini-3.1-pro's recommendations from the earlier external analysis rounds.

**What GPT recommended:** Extract from late Qwen layers (28, 31, 34, 35), inject at early Gemma layers (2, 4, 6). Rationale: "the last 25% is where reasoning lives, give the receiver maximum runway."

**What Gemini recommended:** Stay at Layer-0 injection, replace the linear W_cross with a trained residual adapter. Rationale: "the Layer-0→Layer-0 pipeline is structurally correct, just hit the linear ceiling."

**What the data shows:** Both were wrong.


1. **Late Qwen layers are the worst senders, not the best.** The layer skip test shows layers 34 and 35 are the *least* critical for reasoning — skipping them improves log-prob. This is the logit lens collapse: the final layers over-specialize toward output vocabulary and lose abstract reasoning structure. The CKA heatmap confirms this: late Qwen layers have the lowest alignment with any Gemma layer. Extracting from layer 35 means extracting a degraded signal with poor cross-model compatibility.

2. **Layer-0 injection wastes Gemma's dispensable layers.** The layer skip test shows Gemma's layers 0-18 are all dispensable for reasoning. Injecting at Layer-0 means the projected state must traverse 18 layers that contribute nothing to reasoning before reaching the layers that matter (19-25). Layer-0 injection also forces the projected state to masquerade as a lexical embedding, which is a geometric mismatch the CKA data makes explicit: CKA between Qwen reasoning layers and Gemma Layer-0 is ~0.20.

3. **The sweet spot is the middle of both networks.** Qwen layers 7-13 carry reasoning signal (layer skip deltas of -0.24 to -0.42) AND have high cross-model alignment with Gemma layers 9-12 (CKA 0.80-0.83). Gemma layers 9-12 are in the "dispensable" zone (not needed for reasoning), which means they provide a geometrically compatible injection point that still leaves Gemma's full reasoning stack (layers 19-25) intact downstream.

4. **A trained adapter at Layer-0 is unnecessary.** Gemini argued a residual adapter was needed to break the 0.68 cosine ceiling. The ceiling was an artifact of the layer choice, not a fundamental limit of linear transfer. Moving to the right layers achieves 0.877 calibration cosine with a training-free Procrustes — 30% higher than the "ceiling" — without any training at all.


### Revised Pipeline Architecture

The cross-model latent communication pipeline changes from:

```
OLD (Approach 1, 1/5 GSM8K):
  Qwen Layer-36 hidden → self-align to Qwen Layer-0 → cross-align L0→L0 (0.68 cos) → Gemma Layer-0 input

NEW (data-driven):
  Qwen Layer-13 hidden → Procrustes L13→L12 (0.877 cos) → inject at Gemma Layer-12 → Gemma layers 12-25 → output
```

The self-alignment step (LatentMAS W_a) is no longer needed. We extract directly from Qwen's intermediate layer and project into Gemma's intermediate space. No Layer-0 bottleneck, no vocabulary bridge, no pretending a reasoning state is a lexical embedding.

### Revised Implementation Plan for Claude Code

**Prerequisites (unchanged from earlier spec):**
- Task 0a: Add `mlxmas/__init__.py`
- Task 0b: Move `gemma_forward_with_embeddings` to `cross_comm.py`
- Task 0c: Remove dead `extract_final_hidden_state_gemma`
- Task 0d: Fix argparse in `test_cross.py`


**Task 1: `gemma_forward_from_layer(start_layer=12)` in `cross_comm.py`**

Parameterize the Gemma forward pass to start at any layer. Critical: do NOT apply sqrt(hidden_size) embedding scaling when `start_layer > 0` — intermediate representations are already at operating scale. Use `cache[start_layer]` for attention mask computation, not `cache[0]`.

Verify: `start_layer=0` must produce identical output to the existing `gemma_forward_with_embeddings()`.

**Task 2: `extract_sender_at_layer(layer=13)` in `cross_comm.py`**

Extract Qwen's hidden state at a specific intermediate layer during the latent thought loop. Run the full forward pass (so the KV cache is fully populated for subsequent latent steps), but capture the residual stream at layer 13 instead of the final layer output. The state should be raw (no RMSNorm, no lm_head).

This replaces the current two-step process of: (1) get final hidden → (2) self-align to Layer-0. No self-alignment is needed because we are projecting directly from Qwen Layer-13 to Gemma Layer-12.

**Task 3: Procrustes calibration at the correct layers**

Reuse `contextual_procrustes.py` but parameterize it for arbitrary (sender_layer, receiver_layer) pairs. Default: sender=13, receiver=12. Use 500+ calibration prompts. Add a 20% held-out split and report both train and held-out cosine. Expected held-out cosine: ~0.87.


**Task 4: Update `test_cross.py` for intermediate injection**

Wire the new components together:
1. `collect_sender_hidden_states()` extracts at Qwen layer 13 (not final layer)
2. `apply_cross_realignment()` uses the layer-13→layer-12 Procrustes matrix (not vocab LS)
3. `generate_with_cross_latents()` calls `gemma_forward_from_layer(start_layer=12)` for the latent prefill, then continues with normal token generation

The sender still runs the full LatentMAS pipeline (planner → critic → refiner with latent thinking steps). The difference is that the hidden states collected for cross-model transfer come from layer 13 instead of the final layer, and they're injected at Gemma layer 12 instead of layer 0.

**Task 5: Run 5-question GSM8K eval**

Compare against the Approach 1 baseline (vocab LS, L0→L0, 1/5 accuracy). The injection probe data (Test 3) suggests the projected states are geometrically compatible. The question is whether the full multi-agent pipeline (33 projected embeddings from 3 agents × 11 latent steps) translates geometric compatibility into downstream reasoning accuracy.

**Cache handling note:** When injecting at Gemma layer 12, layers 0-11 have empty cache entries. The attention mask uses `cache[start_layer]` which correctly tracks the sequence offset for layers 12-25. Layers 0-11 will have offset=0 in their cache; when the question tokens are subsequently processed through the full model via `receiver_model(input_ids, cache=cache)`, those layers will process the question tokens without attending to the injected states (correct behavior — the injection is only visible to layers 12+).


### Execution Order

```
0a-0d. Cleanup prerequisites
  ↓
1. Implement gemma_forward_from_layer(start_layer) in cross_comm.py
   Verify: start_layer=0 matches old function bitwise
  ↓
2. Implement extract_sender_at_layer(layer=13) in cross_comm.py
  ↓
3. Calibrate Procrustes at (sender=13, receiver=12) with 500+ prompts
   Verify: held-out cosine ~0.87
  ↓
4. Update test_cross.py to use new pipeline
  ↓
5. Run 5-question GSM8K eval
   Compare against 1/5 baseline
```

### Empirical Test Scripts (Reference)

The three test scripts are in `mlxmas/` and can be re-run to validate any changes:

```
# Test 1: Full CKA heatmap (36×26 layer pairs)
python -m mlxmas.test1_cka_heatmap --n_prompts 200

# Test 2: Layer skip test (per model)
python -m mlxmas.test2_layer_skip --model mlx-community/Qwen3-4B-8bit
python -m mlxmas.test2_layer_skip --model mlx-community/gemma-2-2b-it-8bit

# Test 3: Injection probe (specific layer pairs)
python -m mlxmas.test3_injection_probe --pairs "13,12;13,10;11,10"
python -m mlxmas.test3_injection_probe --from_cka cka_heatmap.json --top_k 10
```

All scripts default to 8-bit models. Results are saved as JSON in `mlxmas/`.
