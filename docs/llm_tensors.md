# Cross-Model Latent Communication via Logit Distribution Transfer

## Core Insight

LLMs do not think in tokens. Tokens are the lossy, human-readable interface. Under the hood, the model's actual output is a probability distribution over its entire vocabulary — a dense tensor of logits. Selecting a single token via argmax discards the vast majority of information the model computed.

Instead of converting Qwen's output to discrete tokens and feeding them to Gemma as text, we pass the **raw logit distribution** from Qwen's output head directly to Gemma's input, remapped through the shared vocabulary.

**This is a zero-shot approach. No training, no adapter, no LoRA.** It is a deterministic geometric operation using the shared vocabulary as anchor points.

## Why This Works

1. **No intermediate layers.** We only touch model boundaries — Qwen's output head and Gemma's embedding layer. Intermediate transformer layers are fundamentally incompatible across model families.

2. **Shared vocabulary as the bridge.** Both models tokenize text. Their vocabularies overlap substantially. The function `find_shared_tokens()` in `mlxmas/cross_align.py` already computes the mapping: `(token_str, qwen_id, gemma_id)`. This mapping lets us rearrange Qwen's logit vector into Gemma's vocabulary order.

3. **Full information preservation.** A logit vector encodes not just the top-1 prediction but the model's entire uncertainty landscape — what it thinks is likely, unlikely, and impossible. A single argmax token throws all of this away. The distribution is the thought.

## Why This Avoids Previous Failure Modes

Every previous cross-model approach failed because a trainable component learned to bypass the communication channel:

- **MHA adapter v1/v2/v3:** LoRA learned to ignore the adapter output and predict answers from fixed patterns. Mode collapse — adapter produced identical vectors regardless of input.
- **Procrustes alignment:** Linear rotation assumes isometry between embedding spaces. Wrong assumption for different model families.

This approach has **no trainable components**. There is no adapter to mode-collapse, no LoRA to learn to ignore the signal, no loss function to optimize away meaning. The remapping is a fixed mathematical operation. If Qwen's logits carry information (they do — it's a working language model), that information reaches Gemma.

## Architecture

```
Qwen (Sender) — Autoregressive generation
  1. Receive question, begin generating response
  2. At each step:
     a. Run forward pass → logits [vocab_size_qwen]
     b. Sample a discrete token for Qwen's OWN continuation (standard generation)
     c. ALSO capture the full logit distribution for Gemma (this is the "thought")
  3. Apply softmax to captured logits → probability distribution

Vocabulary Remap — Fixed index mapping, no training
  4. Using shared vocabulary map (qwen_id → gemma_id pairs):
     - Extract probabilities at shared Qwen token positions
     - Place them at corresponding Gemma token positions
     - Tokens in Qwen but not in Gemma: probability mass is dropped
     - Tokens in Gemma but not in Qwen: receive zero probability
  5. Renormalize so probabilities sum to 1.0

Gemma (Receiver) — Consumes soft embeddings
  6. Convert remapped probability distribution into a soft embedding:
     soft_embed = probs_gemma @ gemma_embedding_table → [hidden_dim]
     This is a weighted average of Gemma's embeddings, weighted by
     Qwen's output probabilities. It is the "soft token."
  7. Accumulate soft embeddings into a sequence (one per Qwen generation step)
  8. Feed full soft embedding sequence into Gemma as input context
  9. Gemma generates answer — never sees original question text
```

## Key Design Decisions

### Soft Embeddings vs Hard Tokens

- **Hard token (argmax):** Pick the single most likely token. Discrete, lossy. This is what normal text generation does.
- **Soft embedding (distribution @ embedding_table):** Weighted blend of all token embeddings by their probabilities. Continuous, preserves uncertainty. This is what we do.

The soft embedding for a high-confidence prediction (90% one token) will be very close to that token's embedding. For an uncertain prediction (30/25/20/15/10 across five tokens), the soft embedding is a blend that exists *between* the discrete token points in embedding space — a superposition of meanings that no single token can represent.

### What About Non-Shared Tokens?

The Qwen and Gemma vocabularies do not overlap 100%. Probability mass on Qwen-only tokens is lost during remapping. After remapping, we renormalize the Gemma-side distribution so it sums to 1.0. In practice, the shared vocabulary covers the vast majority of probability mass for natural language text because both models tokenize common words and subwords similarly.

### Qwen Generates Normally, We Eavesdrop

Qwen still samples discrete tokens to drive its own autoregressive loop — it needs a concrete token to feed back as input for the next step. But at each step, we *also* capture the full logit distribution before sampling. Qwen generates its chain-of-thought normally; we siphon off the continuous representation at every step.

### Multi-Step Transfer

Qwen generates a full response autoregressively (potentially hundreds of tokens of reasoning). At each step, we capture logits, remap, and convert to a soft embedding. Gemma receives the entire sequence of soft embeddings — Qwen's full chain of thought in continuous form, not a single prediction.

## Implementation Plan

### Step 1: Build Vocabulary Mapping

Reuse `find_shared_tokens()` from `mlxmas/cross_align.py`. Build index arrays for efficient remapping:

```python
shared = find_shared_tokens(qwen_tok, gemma_tok)
# shared: list of (token_str, qwen_id, gemma_id)

# Index arrays for fast gather/scatter
qwen_ids = mx.array([s[1] for s in shared])   # positions to read from Qwen logits
gemma_ids = mx.array([s[2] for s in shared])   # positions to write in Gemma probs
```

### Step 2: Sender Forward Pass with Logit Capture

Run Qwen autoregressively. At each step, sample for Qwen's continuation AND capture logits for Gemma:

```python
logits = qwen_forward(input_ids)[:, -1, :]     # [1, vocab_size_qwen]

# Sample token for Qwen's own continuation
token = sample(logits)
input_ids = mx.concatenate([input_ids, token], axis=-1)

# Capture full distribution for Gemma
probs_qwen = mx.softmax(logits, axis=-1)        # [1, vocab_size_qwen]

# Remap to Gemma vocabulary using index arrays
probs_shared = probs_qwen[:, qwen_ids]           # [1, n_shared]
probs_gemma = mx.zeros((1, vocab_size_gemma))
probs_gemma[:, gemma_ids] = probs_shared
probs_gemma = probs_gemma / mx.sum(probs_gemma, axis=-1, keepdims=True)

# Convert to soft embedding
soft_embed = probs_gemma @ gemma_embed_table     # [1, hidden_dim]
soft_embeds.append(soft_embed)
```

### Step 3: Receiver Generation from Soft Embeddings

Concatenate all soft embeddings from Step 2 into a sequence. Feed into Gemma's transformer layers (bypassing the embedding lookup), then let Gemma generate autoregressively from that context.

```python
# soft_embeds: [1, seq_len, hidden_dim] — Qwen's full response as soft tokens
# Feed directly into Gemma's transformer layers (bypass embedding lookup)
# Then generate answer autoregressively
```

### Step 4: Evaluation

Run on GSM8K (5 questions). The receiver (Gemma) never sees the original question text. It only receives the soft embedding sequence derived from Qwen's logit distributions. Compare accuracy against:

- Same-model KV cache transfer: 5/5 (current baseline)
- Cross-model intermediate layer transfer: 0/5 (failed approaches)

## Failure Modes

If this approach fails, the possible causes are:

1. **Insufficient vocabulary overlap** — too much probability mass lost during remapping. Diagnosable by measuring what fraction of Qwen's probability mass lands on shared tokens.
2. **Soft embeddings too far from any real token embedding** — Gemma's transformer layers may not handle inputs that are weighted blends of many embeddings. Diagnosable by measuring distance from nearest real embedding.
3. **Sequence length mismatch** — Qwen's chain-of-thought may be too long or too short for Gemma to interpret. Tunable.

None of these involve training failure or mode collapse.

## Files

| File | Purpose |
|------|---------|
| `mlxmas/cross_align.py` | `find_shared_tokens()` — vocabulary mapping (exists) |
| `mlxmas/logit_comm.py` | New: logit capture, remapping, soft embedding conversion |
| `mlxmas/eval_logit.py` | New: end-to-end eval on GSM8K |

## What We Are NOT Doing

- No intermediate layer access between models
- No trained adapter or projection matrix
- No LoRA fine-tuning
- No Procrustes or linear alignment of hidden states
- No discrete token transfer (argmax)
- No training of any kind
