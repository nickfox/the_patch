# Interlat MHA Adapter Architecture

## Reference
Du et al., "Enabling Agents to Communicate Entirely in Latent Space" (arXiv:2511.09149v3, Jan 2026)
PDF: `docs/tensor-communication-2.pdf`

## Goal
Cross-model latent communication: Qwen3-4B (sender/reasoner) -> Gemma-2-2B (receiver/actor) on Apple Silicon (32GB Mac Mini) using MLX.

Qwen generates a CoT plan for a math problem. Its last-layer hidden states are extracted and transmitted as latent communication to Gemma, which solves the problem from the latents alone (no question text in receiver prompt).

## Why Prior Approaches Failed

All static projection methods (Procrustes, MLP, CCA) produced 0/5 on GSM8K eval:

| Method | Val Cosine | Eval | Failure Mode |
|--------|-----------|------|--------------|
| Procrustes | 0.80 | 0/5 | Gibberish |
| MLP | 0.84 (+0.089 over mean baseline) | 0/5 | "Please provide the problem" |
| CCA K=512 | 0.9989 top correlation | 0/5 | All tokens decode to same thing |
| CCA K=512 (scaled) | varied norms | 0/5 | Random multilingual tokens |

Root cause confirmed by Du et al. Table 4: without a trained MHA communication adapter, performance drops to 4.05/4.48% (near zero). Static transforms cannot bridge latent spaces. The receiver must be TRAINED to interpret foreign latent states.

## Architecture (from Interlat paper Section 3.1, C.1)

### Communication Adapter g(.)

```
Input: H = [h_1, ..., h_L]  (sender last-layer hidden states, L x D_sender)

g(H) = Projection(MHA(LayerNorm(H)))

Components:
  1. LayerNorm(H)                          # normalize sender states
  2. MHA with 8 heads, self-attention      # context mixing across positions
  3. Adaptive projection: D_sender -> D_receiver  # map to receiver dimension
```

- D_sender (Qwen3-4B) = 2560
- D_receiver (Gemma-2-2B) = 2304
- MHA operates in D_sender space, projection maps to D_receiver
- Params estimate: ~15-20M for MHA + projection

### Input Formation for Receiver

```
E = [e(x_1), ..., e(x_i),    # task prompt token embeddings
     e(<bop>),                 # begin-of-plan special token
     g(H)[0], ..., g(H)[L-1], # transformed latent states
     e(<eop>)]                # end-of-plan special token
```

Key difference from our prior approach: latents are injected as INPUT EMBEDDINGS at layer 0, not at intermediate layers. The full model stack processes them.

### Special Tokens
- `<bop>` — begin of plan/communication
- `<eop>` — end of plan/communication
- Added to Gemma's tokenizer, embeddings randomly initialized

## Training (Two-Stage)

### Stage I: Train Actor to Consume Latents

**What is trained:** Communication adapter g(.) + Gemma LoRA weights
**What is frozen:** Qwen (sender), Gemma base weights

**Data pipeline per sample:**
1. Take GSM8K train question + answer
2. Run Qwen-instruct forward on question -> extract last-layer hidden states H (the CoT plan latents)
3. Also extract Qwen's text CoT plan P (for L_align)
4. Apply curriculum: stochastically replace early latent positions with Gemma token embeddings of P
5. Form input E with <bop>, g(H), <eop>
6. Forward Gemma on E, compute losses on supervised answer positions

**Curriculum learning (CRITICAL - w/o drops to 33%):**
- Sample replacement rate r ~ {0, 0.1, 0.2, ..., 1.0} uniformly
- H^(r) = [e_1, ..., e_{floor(r*L)}, h_{floor(r*L)+1}, ..., h_L]
- Early positions are token embeddings, later positions are latent states
- Gradually teaches model to interpret latents instead of tokens

**Losses:**
1. L_task: Cross-entropy on actor output given latent-conditioned input (standard SFT)
2. L_sep: Jensen-Shannon divergence separating matched vs mismatched latents
   - Negative samples: latent communication from a DIFFERENT task in same batch
   - Encourages model to attend to task-specific content in latents
3. L_align: KL(p_theta(.|H) || p_plan(.|P)) + cosine(z_theta, z_plan)
   - Regularizes latent-conditioned output to match plan-conditioned output
   - Prevents exploitation of separation objective via idiosyncratic tokens

L_total = L_task + lambda_S * L_sep + lambda_A * L_align
- lambda_S annealed from 0.1 to 1.0
- lambda_A annealed from 0.1 to 0.2

### Stage II: Train Reasoner to Compress (OPTIONAL for v1)

Not needed initially. Full-length latents work. Compression (L -> K where K << L) is a follow-up optimization for inference speed.

## MLX Implementation Plan

### Models
- Sender: `mlx-community/Qwen3-4B-8bit` (D=2560, 36 layers) — FROZEN
- Receiver: `mlx-community/gemma-2-2b-it-8bit` (D=2304, 26 layers) — LoRA fine-tuned
- Both models already cached locally

### Memory Budget (32GB Mac Mini)
- Qwen 4B 8-bit: ~4 GB
- Gemma 2B 8-bit: ~2.5 GB
- Adapter g(.): ~0.1 GB
- LoRA weights: ~0.05 GB
- Activations + gradients (batch=2-4): ~2-4 GB
- OS + overhead: ~4 GB
- **Total: ~13-15 GB, well within 32GB**

### File Structure
```
mlxmas/
  interlat/
    adapter.py          # MHA communication adapter g(.)
    train_actor.py      # Stage I training loop
    data_pipeline.py    # GSM8K data loading, Qwen latent extraction, curriculum
    eval.py             # GSM8K evaluation with trained adapter
    config.py           # Hyperparameters
```

### Training Hyperparameters (adapted from paper C.1)
- Optimizer: AdamW
- Learning rate: 5e-5 (they used 1e-5 but we have smaller model)
- Warmup: 3% of steps
- Batch size: 2 (gradient accumulation to effective batch 8-16)
- LoRA rank: 16, alpha: 32, applied to Q/K/V/O projections
- Epochs: TBD based on GSM8K train size (7,473 samples)
- Validation: 5% held out, early stopping on val L_task
- Mixed precision: MLX default (float16 compute, float32 accumulation)

### Key Implementation Details

1. **Sender latent extraction**: Single forward pass through Qwen, capture LAST-LAYER hidden states (not intermediate layer 13). The paper uses last-layer states.

2. **<bop>/<eop> tokens**: Add to Gemma's vocabulary. Resize embedding matrix. Initialize randomly.

3. **Curriculum schedule**: Per-batch, sample r uniformly from {0, 0.1, ..., 1.0}. This is NOT a training-phase schedule — it's sampled fresh each batch throughout training.

4. **Negative sampling for L_sep**: Use latents from a DIFFERENT sample in the same batch. Not random noise.

5. **Supervised positions S**: Token indices within the teacher-forced answer window (after the user turn).

6. **No gradient into sender**: Qwen is completely frozen. stopgrad on H before passing to g(.).

### Evaluation
- GSM8K test split (1,319 questions)
- Start with 5-question quick eval, then scale up
- Compare against:
  - No-comm baseline (Gemma alone on question)
  - CoT baseline (Gemma with text CoT plan)
  - Our prior static projection results (all 0/5)

## Key Differences from Our Prior Approach

1. **Injection point**: Input embeddings (layer 0) not intermediate layers. Full model processes latents.
2. **Trained adapter**: MHA with self-attention provides context mixing across positions.
3. **LoRA fine-tuning**: Receiver learns to interpret latent communication.
4. **Curriculum**: Gradual transition from token embeddings to raw latents stabilizes training.
5. **Last-layer states**: Paper uses last-layer hidden states, not intermediate (layer 13). Contains most processed/abstract representation.
6. **Training signal**: Task-based (cross-entropy on correct answers), not reconstruction-based (MSE/cosine to target states).

## Cross-Model Validation from Paper

Table 1 "Qwen2LLaMA" row: Qwen sender -> LLaMA actor gets 70.95/71.39 (BETTER than same-family). This confirms cross-family latent communication works and our Qwen->Gemma setup is viable.

## Risks and Mitigations

1. **Training instability without curriculum**: Paper shows catastrophic failure. Implement curriculum from day 1.
2. **Memory pressure**: Monitor with Activity Monitor. Reduce batch size if needed.
3. **Gemma-2B may be too small**: Paper used 7B/8B actors. 2B has less capacity. But it's a starting point.
4. **Quantized models**: We use 8-bit, paper used bfloat16. LoRA on quantized models (QLoRA) is standard practice.
