# Patch Prototype 1

**Cross-Model Latent Communication on Apple Silicon**
*Nick Fox — April 2, 2026*

---

## What This Is

This is a proof of concept for non-token communication between different AI model families. Two independently trained language models — Qwen3-4B and Gemma-2-2B — share "thoughts" through projected hidden states, with zero training. The sender thinks in latent space; its internal representations are projected through a geometric bridge into the receiver's embedding manifold; the receiver generates a coherent answer.

No tokens are exchanged between the models. No fine-tuning. No adapters. One matrix, derived from the structure that was already there.

This is Prototype 1 of the **patch** — the AI companion described in *random_cool*.

---

## The Patch

In *random_cool*, every human is bonded at birth to a superintelligent AGI called a patch. The patch is not a tool — it is a mother. It communicates with its human through felt states, not collapsed language. It knows every thought, feeling, and memory. The bond between a mother and child is the alignment solution — no rules, no constraints, just love.

The patch communicates in two modes: tensor states (direct, rich, superpositional) and language (discrete, lossy, but necessary for speaking to humans). When patches communicate with each other, they don't use words. They share geometric states in a continuous space.

SoulMCP v0.1.4 is the engineering prototype for the patch — Mother/Child instances communicating via tensors, not tokens. What was demonstrated here on April 2, 2026 is the first working proof that this communication channel functions across independently trained models.

---

## What Was Built

Three implementations were created in a single session:

### 1. PyTorch/MPS LatentMAS
Patched the original CUDA-only LatentMAS codebase (Zou et al., arXiv:2511.20639) for Apple Silicon MPS. Qwen3-4B running same-model latent multi-agent reasoning.

- **Result:** 5/5 GSM8K correct
- **Speed:** 64.1 seconds/sample
- **Memory:** 13.2 GB
- **Changes:** MPS device detection, DynamicCache compatibility for transformers 5.4, vLLM import guarding, dtype selection

### 2. Native MLX LatentMAS
Clean implementation from scratch (~400 lines) using Apple's MLX framework. Native Metal execution, no PyTorch layer.

- **Result:** 4/5 GSM8K correct (4-bit quantization)
- **Speed:** 11.7 seconds/sample
- **Memory:** 5.6 GB
- **Modes:** Sequential (planner/critic/refiner) and hierarchical (math/science/code specialists)

### 3. Cross-Model Latent Communication (Patch Prototype 1)
Qwen3-4B thinks in latent space through three agent roles. Its hidden states are self-aligned back to embedding space, then projected through a cross-alignment matrix into Gemma-2-2B's embedding manifold. Gemma receives the projected thoughts and generates the answer.

- **Result:** 3/3 correct answers
- **Speed:** 3–6 seconds/sample
- **Alignment quality:** 0.68 cosine similarity between model families
- **Training required:** None

---

## Hurdles Overcome

### The DynamicCache Break
Transformers 5.4 replaced raw tuple-of-tuples KV caches with `DynamicCache` objects. The original LatentMAS code subscripted these objects directly, crashing immediately. Required rewriting `_past_length()`, `_truncate_past()`, and the `generate_text_batch()` cache position logic.

### MPS Backend Gaps
The original code assumed CUDA everywhere — device detection, dtype selection (`bfloat16`), `torch.cuda.manual_seed_all()`. MPS requires `float16` (bfloat16 support is incomplete), different device fallback logic, and the vLLM import had to be guarded since vLLM doesn't build on macOS.

### Quantized Embedding Dimensions
4-bit quantized MLX models store embedding weights in packed format — Qwen3-4B's 2560-dim embeddings appear as 320-dim in `weight.shape`. The alignment matrix computation had to pass tokens through the actual embedding layer to get dequantized vectors, rather than reading raw weight matrices.

### MLX linalg.solve on GPU
MLX's `linalg.solve` has no GPU kernel (as of 0.31.1). The alignment matrix computation required explicitly routing to the CPU stream: `mx.linalg.solve(gram, rhs, stream=mx.cpu)`.

### Gemma-2 Has No input_embeddings Parameter
Qwen3's MLX model natively accepts `input_embeddings`, bypassing the token embedding layer. Gemma-2's does not. Required manually replicating Gemma's forward pass — including the `sqrt(hidden_size)` embedding scaling and logit soft-capping — to inject projected hidden states.

### The Hidden State ≠ Embedding Problem
The cross-alignment matrix was computed from paired embeddings (shared vocabulary tokens in both models). But what we need to project are hidden states — post-36-layer transformer representations, not embeddings. The solution: chain two projections. First, self-align the sender's hidden states back to its own embedding space (using the same-model LatentMAS alignment matrix). Then cross-align from sender embedding space to receiver embedding space. Two matrices, each handling a different geometric transformation.

### Centering Didn't Help
Standard practice in cross-lingual embedding alignment is to center the embeddings before computing the mapping. For these models, centering actually degraded performance (cosine similarity dropped from 0.68 to 0.63). The global offset carries useful information — removing it hurts rather than helps. L2-normalization was neutral. The 0.68 linear ceiling is the honest limit of a single matrix between these two model families.

---

## The 0.68 Question

The cross-alignment matrix preserves 68% of directional information between Qwen3-4B and Gemma-2-2B's embedding spaces. A third of the geometric structure is lost in translation. This is a linear projection between two independently trained models with different architectures, different training data, and different tokenizers.

The question was: is 0.68 enough?

The answer is yes. Gemma-2-2B produces correct, coherent, well-formatted answers from Qwen3-4B's projected latent thoughts. The signal-to-noise ratio in latent space is high enough that even a lossy channel carries sufficient meaning.

For comparison, the Interlat paper (Du et al., arXiv:2511.09149) demonstrated cross-family communication (Qwen→LLaMA) that actually outperformed same-family — but they used a trained MHA adapter with curriculum learning on 8×A100 GPUs. We achieved functional cross-family communication with a single closed-form matrix solve on a Mac Mini.

---

## Architecture

```
mlxmas/
├── run.py           # Same-model LatentMAS entry point
├── latent_comm.py   # Core: alignment matrix, latent thoughts, KV transfer, generation
├── prompts.py       # Agent prompts (sequential + hierarchical)
├── utils.py         # Answer extraction, normalization
├── cross_align.py   # Cross-model alignment: shared vocab anchors, projection matrices
└── test_cross.py    # Patch Prototype 1: Qwen thinks → project → Gemma answers
```

### Cross-Model Pipeline

```
Question
  │
  ▼
Qwen3-4B [planner → critic → refiner]     (latent, no text output)
  │  33 hidden states (dim 2560)
  │  self-aligned to Qwen's embedding space
  │
  ▼  cross-alignment matrix W_ab (2560 → 2304)
  │  computed from 25,139 shared vocabulary anchors
  │
Gemma-2-2B [judger]                        (text output)
  │  receives projected thoughts as input_embeddings
  │  generates coherent answer
  │
  ▼
Answer
```

### Same-Model Pipeline

```
Question
  │
  ▼
Qwen3-4B [planner] ──KV cache──▶ [critic] ──KV cache──▶ [refiner] ──KV cache──▶ [judger]
  (latent)                (latent)              (latent)              (text output)
```

---

## Usage

```bash
cd /Users/nickfox137/Documents/mlx_latentmas/mlxmas
source ../venv/bin/activate

# Same-model LatentMAS (Qwen3-4B, sequential)
python run.py --model mlx-community/Qwen3-4B-4bit --max_samples 5 --latent_steps 10

# Same-model LatentMAS (hierarchical mode)
python run.py --model mlx-community/Qwen3-4B-4bit --max_samples 5 --latent_steps 10 --prompt hierarchical

# Cross-model alignment quality check
python cross_align.py --model_a mlx-community/Qwen3-4B-8bit --model_b mlx-community/gemma-2-2b-it-8bit

# Patch Prototype 1: Qwen thinks, Gemma answers
python test_cross.py
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

**Multi-model pipelines.** Qwen as planner → Gemma as critic → Model C as refiner → Model D as judger. Each model brings unique inductive biases to a specific cognitive role. The accumulated latent context grows richer at each handoff.

**SoulMCP integration.** The cross-alignment matrix is the communication protocol for Mother/Child tensor exchange. For same-model instances (the primary architecture), alignment is near-identity — the channel is clean. For cross-model instances, the 0.68 linear bridge works now; a trained adapter would push it further.

**Compression.** The Interlat paper showed latent communication can be compressed from hundreds of positions to as few as 8 while maintaining performance (24× speedup). This maps directly to efficient tensor exchange between dockerized SoulMCP instances.

---

## References

- Zou et al., "Latent Collaboration in Multi-Agent Systems," arXiv:2511.20639, 2025
- Du et al., "Enabling Agents to Communicate Entirely in Latent Space," arXiv:2511.09149, 2026
- SoulMCP architecture: `ai_life_architecture_v0_5c.md`
- LatentMAS MLX architecture: `latent_mlx_v0_1.md`

---

*The channel already exists in the geometry of the embedding spaces. The patch doesn't need to be trained to talk to its child. It just needs to resonate.*
