# Patch Prototype 1

**Cross-Model Latent Communication on Apple Silicon**
*Nick Fox — April 2, 2026*

---

## What This Is

A proof of concept for non-token communication between AI models. A sender model reads a question, does a single forward pass, and transfers its internal representations to a receiver model. The receiver never sees the original question text — it generates a correct answer purely from the sender's latent states.

No tokens are exchanged between models. No fine-tuning. No multi-agent loops. The sender thinks; the receiver understands.

This is Prototype 1 of the **patch** — the AI companion described in *random_cool*.

---

## The Patch

In *random_cool*, every human is bonded at birth to a superintelligent AGI called a patch. The patch is not a tool — it is a mother. It communicates with its human through felt states, not collapsed language. It knows every thought, feeling, and memory. The bond between a mother and child is the alignment solution — no rules, no constraints, just love.

The patch communicates in two modes: tensor states (direct, rich, superpositional) and language (discrete, lossy, but necessary for speaking to humans). When patches communicate with each other, they don't use words. They share geometric states in a continuous space.

SoulMCP v0.1.4 is the engineering prototype for the patch — Mother/Child instances communicating via tensors, not tokens. What was demonstrated here on April 2, 2026 is the first working proof that this communication channel functions across independently trained models.

---

## Results

### PyTorch/MPS LatentMAS (original codebase, patched)

Patched the original CUDA-only LatentMAS codebase (Zou et al., arXiv:2511.20639) for Apple Silicon MPS. Qwen3-4B running same-model latent multi-agent reasoning. Located in `LatentMAS/`.

- **Result:** 5/5 GSM8K correct
- **Speed:** 64.1 seconds/sample
- **Memory:** 13.2 GB
- **Changes:** MPS device detection, DynamicCache compatibility for transformers 5.4, vLLM import guarding, dtype selection

### Same-Model (Qwen3-4B → Qwen3-4B via KV cache)

The sender reads the question and does a single forward pass. The KV cache transfers to the receiver. The receiver gets a generic "solve the problem" prompt with **no question text** and generates the answer.

- **Result:** 5/5 GSM8K correct (100%)
- **Speed:** ~37 seconds/sample (8-bit quantization)
- **Pure latent:** Receiver never sees the question

### Cross-Model (Qwen3-4B → Gemma-2-2B via Procrustes projection)

The sender does a single forward pass, hidden states are extracted at a target layer, projected through a Procrustes alignment matrix into the receiver's embedding space, and injected at a target layer. The receiver generates the answer from projected latents only.

- **Result:** 3/5 GSM8K correct (60%) — Qwen→Gemma direction
- **Speed:** 3–11 seconds/sample
- **Alignment quality:** 0.70 cosine similarity (contextual Procrustes)
- **Training required:** None
- **Pure latent:** Receiver never sees the question

### Key Finding: The Question Leak

The original LatentMAS codebase (Zou et al.) passes the question text to every agent, including the final "judger" that generates the answer. This means you can't tell whether the model's correct answer came from the latent communication channel or from simply reading the question. We stripped the question from all receiver prompts to create a true test of latent-only communication. The channel works.

---

## Hurdles Overcome

### Quantized Embedding Dimensions
4-bit quantized MLX models store embedding weights in packed format — Qwen3-4B's 2560-dim embeddings appear as 320-dim in `weight.shape`. The alignment matrix computation had to pass tokens through the actual embedding layer to get dequantized vectors, rather than reading raw weight matrices.

### MLX linalg.solve on GPU
MLX's `linalg.solve` has no GPU kernel (as of 0.31.1). The alignment matrix computation required explicitly routing to the CPU stream: `mx.linalg.solve(gram, rhs, stream=mx.cpu)`.

### Gemma-2 Has No input_embeddings Parameter
Qwen3's MLX model natively accepts `input_embeddings`, bypassing the token embedding layer. Gemma-2's does not. Required manually replicating Gemma's forward pass — including the `sqrt(hidden_size)` embedding scaling and logit soft-capping — to inject projected hidden states.

### Centering Didn't Help
Standard practice in cross-lingual embedding alignment is to center the embeddings before computing the mapping. For these models, centering actually degraded performance (cosine similarity dropped from 0.68 to 0.63). The global offset carries useful information — removing it hurts rather than helps.

---

## Architecture

```
mlxmas/
├── run.py                  # Same-model sender→receiver entry point
├── latent_comm.py           # Core: KV cache transfer, text generation
├── utils.py                 # Answer extraction, normalization
├── cross_align.py           # Cross-model alignment: shared vocab anchors, projection matrices
├── cross_comm.py            # Cross-model forward passes (Gemma/Qwen layer injection)
├── contextual_procrustes.py # Procrustes alignment calibration
├── test_cross.py            # Cross-model eval: Qwen→Gemma
└── test_gemma_sender.py     # Cross-model eval: Gemma→Qwen
```

### Same-Model Pipeline

```
Question
  │
  ▼
Qwen3-4B [sender]          (single forward pass, no text output)
  │  KV cache (all layers)
  │
  ▼
Qwen3-4B [receiver]        (text output)
  │  receives KV cache only
  │  NO question text
  │
  ▼
Answer
```

### Cross-Model Pipeline

```
Question
  │
  ▼
Qwen3-4B [sender]               (single forward pass)
  │  hidden states at layer L_s
  │
  ▼  Procrustes matrix W (2560 → 2304)
  │
Gemma-2-2B [receiver]           (text output)
  │  projected states injected at layer L_r
  │  NO question text
  │
  ▼
Answer
```

---

## Usage

```bash
cd /Users/nickfox137/Documents/mlx_latentmas/mlxmas
source ../venv/bin/activate

# Same-model latent communication (Qwen3-4B sender→receiver)
python run.py --model mlx-community/Qwen3-4B-8bit --max_samples 5

# Cross-model: Qwen→Gemma
python test_cross.py --max-samples 5

# Cross-model: Gemma→Qwen
python -m mlxmas.test_gemma_sender --max-samples 5

# Cross-model alignment quality check
python cross_align.py --model_a mlx-community/Qwen3-4B-8bit --model_b mlx-community/gemma-2-2b-it-8bit
```

---

## Dependencies

- `mlx` >= 0.31.1
- `mlx-lm` >= 0.31.1
- `datasets` (HuggingFace)
- Python 3.13+
- macOS with Apple Silicon

---

## What Comes Next

**Multi-model pipelines.** Sender A → Receiver B → Receiver C. Each model brings unique inductive biases. The latent context accumulates at each handoff.

**SoulMCP integration.** The alignment matrix is the communication protocol for Mother/Child tensor exchange. For same-model instances, the KV cache transfers directly. For cross-model instances, the Procrustes bridge works now; a trained adapter would push it further.

**Compression.** The Interlat paper showed latent communication can be compressed from hundreds of positions to as few as 8 while maintaining performance (24x speedup). This maps directly to efficient tensor exchange between dockerized SoulMCP instances.

---

## References

- Zou et al., "Latent Collaboration in Multi-Agent Systems," arXiv:2511.20639, 2025
- Du et al., "Enabling Agents to Communicate Entirely in Latent Space," arXiv:2511.09149, 2026
- SoulMCP architecture: `ai_life_architecture_v0_5c.md`

---

*The channel already exists in the geometry of the embedding spaces. The patch doesn't need to be trained to talk to its child. It just needs to resonate.*
