---
title: "MHA Adapter: Interlat-Style Cross-Model Latent Communication on MLX"
author: Nick Fox
date: 2026-04-06
status: Implementation Spec for Claude Code
---

# MHA Adapter Architecture

## Read This Entire Document Before Writing Any Code

This document specifies the complete architecture for Interlat-style latent
communication between Qwen3-4B (sender) and Gemma-2-2B (receiver) on Apple
Silicon using MLX. It is based on Du et al. (arXiv:2511.09149) and their
open-source implementation at github.com/XiaoDu-flying/Interlat.

Every prior approach (Procrustes, MLP, CCA, OT, barycentric ridge) failed
because they all assume a **frozen receiver**. A frozen receiver has no way
to interpret alien activations. The Interlat approach fine-tunes the receiver
to USE injected hidden states via LoRA, trained on task loss.


## Why Every Prior Approach Failed

All six approaches applied position-independent transforms to map Qwen hidden
states into Gemma's space, then injected them at an intermediate layer into a
frozen Gemma. The diagnostics proved the pattern:

- Variance profile correlation: 0.93 (right dimensions active)
- Mean vector cosine: 0.22 (wrong directions within those dimensions)
- CCA top canonical correlation: 0.99 (shared structure EXISTS)
- Every method: 0/5 on GSM8K with question text removed

The problem is not the projection quality. The problem is that Gemma was
never trained to interpret projected Qwen states. Without fine-tuning, Gemma
treats injected states as noise.


## Models

- **Sender:** `mlx-community/Qwen3-4B-8bit` (D=2560, 36 layers, tied embeddings)
- **Receiver:** `mlx-community/gemma-2-2b-it-8bit` (D=2304, 26 layers, tied embeddings)
- Both are cached at `~/.cache/huggingface/hub/`
- Memory: Qwen ~4GB + Gemma ~2.5GB = ~6.5GB. Leaves ~25GB for training.

Do NOT use Gemma-9B. Do NOT use 4-bit models. Do NOT use Phi-4.


## The Interlat Architecture (What We Are Porting)

Interlat (Du et al.) does three things differently from everything we tried:

1. **Fine-tunes the receiver** with LoRA so it learns to read injected states
2. **Injects at layer 0** (embedding level), not intermediate layers
3. **Trains on task loss** (cross-entropy on correct answers), not MSE on paired states

The adapter processes sender hidden states through self-attention MHA before
injection. The receiver sees the processed states as part of its input
embedding sequence, wrapped in special `<bop>...<eop>` tokens.


## Architecture Overview

```
Sender (Qwen3-4B, frozen, 8-bit):
  - Generates solution for GSM8K question
  - output_hidden_states=True captures last-layer hidden state per token
  - Result: [num_gen_tokens, 2560]

MHA Adapter (trainable, ~25M params):
  - input_projector: nn.Linear(2560, 2304)     # cross-model dimension mapping
  - pre_ln: nn.LayerNorm(2304)
  - self_attn: nn.MultiHeadAttention(2304, 8)  # Q=K=V from same hidden states
  - post_ln: nn.LayerNorm(2304)                # with residual connection
  - adaptive_proj: AdaptiveProjection(2304)     # learnable scale + MLP + output_scale
  - Result: [num_gen_tokens, 2304]

Injection (layer 0):
  - Wrap adapter output in <bop>...<eop> special token embeddings
  - Concatenate: [user_prompt_embeds | <bop> | adapter_output | <eop> | assistant_prompt_embeds]
  - Feed through Gemma from layer 0

Receiver (Gemma-2-2B, LoRA fine-tuned):
  - LoRA rank 8 on attention layers (last 16 layers)
  - Generates answer from the combined embedding sequence
  - Trained on cross-entropy against gold answers
```


## MLX API Reference (Verified from venv site-packages)

These are the exact signatures from the installed MLX version (Python 3.14).

### nn.MultiHeadAttention
```python
nn.MultiHeadAttention(
    dims: int,                          # Model dimensions (also default for Q/K/V/output)
    num_heads: int,                     # Number of attention heads
    query_input_dims: Optional[int],    # Input dims for queries (default: dims)
    key_input_dims: Optional[int],      # Input dims for keys (default: dims)
    value_input_dims: Optional[int],    # Input dims for values (default: key_input_dims)
    value_dims: Optional[int],          # Dims of values after projection (default: dims)
    value_output_dims: Optional[int],   # Output projection dims (default: dims)
    bias: bool = False,                 # Whether to use bias in projections
)
# Call: mha(queries, keys, values, mask=None)
# For self-attention: mha(x, x, x)
# Input shapes: [batch, seq_len, dims]
# Output shape: [batch, seq_len, value_output_dims]
# Uses mx.fast.scaled_dot_product_attention internally
```

### nn.LayerNorm
```python
nn.LayerNorm(
    dims: int,
    eps: float = 1e-5,
    affine: bool = True,
    bias: bool = True,
)
# Call: ln(x) -> uses mx.fast.layer_norm internally
```

### nn.Linear
```python
nn.Linear(input_dims: int, output_dims: int, bias: bool = True)
# Call: linear(x) -> x @ weight.T + bias
```

### LoRA (from mlx_lm.tuner)
```python
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

# Apply LoRA to model layers:
linear_to_lora_layers(
    model,                  # nn.Module with .layers attribute
    num_layers=16,          # Number of layers from the END to convert
    config={
        "rank": 8,
        "dropout": 0.0,
        "scale": 20.0,
    },
)
# This freezes the model, then converts Linear/QuantizedLinear layers
# in the last num_layers to LoRALinear, which are unfrozen.
```

### Training Loop (from mlx_lm.tuner.trainer)
```python
# MLX training pattern:
loss_value_and_grad = nn.value_and_grad(model, loss_fn)
# loss_fn signature: loss_fn(model, batch_data...) -> (loss, num_tokens)

# Compiled step:
@partial(mx.compile, inputs=state, outputs=state)
def step(batch, prev_grad, do_update):
    (lvalue, toks), grad = loss_value_and_grad(model, *batch)
    ...
    optimizer.update(model, grad)
    return lvalue, toks, grad
```

### Loss Functions (from mlx_lm.tuner.losses)
```python
from mlx_lm.tuner.losses import kl_div_loss, js_div_loss
# kl_div_loss(logits_q, logits_p) -> per-token KL divergence (Metal kernel)
# js_div_loss(logits_q, logits_p) -> per-token JSD (Metal kernel)
# Both operate on raw logits (not probabilities), handle logsumexp internally
```

### Gemma-2-2B Forward Pass (from other/gemma2.py)
```python
# Gemma-2-2B does NOT accept input_embeddings parameter.
# Must manually replicate the forward pass:
h = model.model.embed_tokens(inputs)       # [B, seq, 2304]
h = h * (model.model.args.hidden_size ** 0.5)  # sqrt(2304) scaling
mask = create_attention_mask(h, cache[0], return_array=True)
for layer, c in zip(model.model.layers, cache):
    h = layer(h, mask, c)
h = model.model.norm(h)                    # RMSNorm
out = model.model.embed_tokens.as_linear(h)  # tied embeddings
out = mx.tanh(out / model.final_logit_softcapping)  # soft-cap 30.0
out = out * model.final_logit_softcapping
```

### Qwen3-4B Forward Pass (from other/qwen3.py)
```python
# Qwen3-4B accepts input_embeddings natively:
# model.model(inputs, cache, input_embeddings=emb)
# No sqrt(hidden_size) scaling applied to embeddings.
# Tied embeddings via model.model.embed_tokens.as_linear(h)
```


## Component 1: Data Collection — `mlxmas/collect_sender_hidden.py`

### Purpose
Run Qwen on GSM8K training questions, generate solutions, capture last-layer
hidden states per generated token. Store alongside generated text (the "plan")
and the gold answer.

### How Interlat Collects Hidden States
From `data_collection/base_data_collector.py`:
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    return_dict_in_generate=True,
    output_hidden_states=True
)
# Extract last-layer hidden state of each newly generated token:
for step_idx in range(len(generation_steps)):
    last_layer_hidden = generation_steps[step_idx][-1]
    new_token_hidden = last_layer_hidden[:, -1, :]  # [1, hidden_size]
    step_hidden_states.append(new_token_hidden)
# Stack: [num_gen_steps, hidden_size]
```

### MLX Implementation
`mlx_lm` does not expose `output_hidden_states` in its generate API. We must
capture hidden states manually during autoregressive generation.

Strategy: Run Qwen's forward pass one token at a time. After each step,
capture the last-layer hidden state (the output of `model.model.norm(h)` —
the post-norm representation before lm_head) for the newly generated token.

```python
def generate_with_hidden_capture(model, tokenizer, prompt, max_tokens=512, temp=0.6):
    """Generate text and capture last-layer hidden states per token.
    
    Returns: (generated_text, hidden_states [num_tokens, D])
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array([tokens])
    cache = make_prompt_cache(model)
    
    # Prefill: process input prompt
    h = model.model.embed_tokens(input_ids)
    # Qwen does NOT apply sqrt(hidden_size) scaling
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(model.model.layers, cache):
        h = layer(h, mask, c)
    normed = model.model.norm(h)
    # Capture the LAST token's normalized hidden state from prefill
    # (This represents Qwen's understanding of the question)
    
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(normed)
    else:
        logits = model.lm_head(normed)
    mx.eval(logits, *[c.state for c in cache])
    
    # Autoregressive generation with hidden state capture
    hidden_states = []
    generated_tokens = []
    sampler = make_sampler(temp=temp, top_p=0.95)
    next_logits = logits[:, -1, :]
    
    for _ in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated_tokens.append(token_id)
        
        next_input = mx.array([[token_id]])
        h = model.model.embed_tokens(next_input)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(model.model.layers, cache):
            h = layer(h, mask, c)
        normed = model.model.norm(h)
        
        # Capture this token's last-layer hidden state
        hidden_states.append(normed[0, -1, :])  # [D]
        
        if model.args.tie_word_embeddings:
            next_logits = model.model.embed_tokens.as_linear(normed)
        else:
            next_logits = model.lm_head(normed)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, hidden_states[-1], *[c.state for c in cache])
    
    all_hidden = mx.stack(hidden_states, axis=0)  # [num_tokens, D]
    mx.eval(all_hidden)
    text = tokenizer.decode(generated_tokens)
    return text, all_hidden
```

### Data Collection Script
```
python -m mlxmas.collect_sender_hidden \
    --sender mlx-community/Qwen3-4B-8bit \
    --n-prompts 500 \
    --max-tokens 512 \
    --output mlxmas/data/sender_hidden_states.npz
```

Output format:
```
sender_hidden_states.npz:
  hidden_states: list of arrays, each [num_tokens_i, 2560]
  plans: list of strings (generated solutions)
  questions: list of strings (GSM8K training questions)
  gold_answers: list of strings (gold numerical answers)
  n_samples: int
```

Memory: Only Qwen loaded (~4GB). No memory pressure.


## Component 2: MHA Adapter — `mlxmas/mha_adapter.py`

### Architecture (from Interlat's custom_model.py)

The adapter has these layers in order:

1. **input_projector**: `nn.Linear(2560, 2304, bias=True)` — maps Qwen dim to Gemma dim
2. **pre_ln**: `nn.LayerNorm(2304, eps=1e-6)` — normalize before attention
3. **hidden_mha**: `nn.MultiHeadAttention(2304, num_heads=8)` — self-attention (Q=K=V)
4. **post_ln**: `nn.LayerNorm(2304, eps=1e-6)` — normalize with residual: `post_ln(normed + attn_out)`
5. **adaptive_proj**: `AdaptiveProjection(2304)` — scale + MLP + output_scale

### AdaptiveProjection (from Interlat)
```python
class AdaptiveProjection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = mx.array(0.2)         # learnable
        self.output_scale = mx.array(0.1)  # learnable
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        # Init: linear1 weights N(0, 0.02), linear2 xavier gain=1e-2
        # All biases zero

    def __call__(self, x):
        residual = x * self.scale
        h = self.linear1(residual)
        h = nn.gelu(h)
        h = self.ln(h)
        h = self.linear2(h)
        return (residual + h) * self.output_scale
```

The output is deliberately small initially (~0.02 magnitude) so the adapter
starts near-zero and gradually learns to inject meaningful signal.

### Full MHA Adapter Class
```python
class MHAAdapter(nn.Module):
    """Interlat-style MHA adapter for cross-model latent communication.
    
    Processes sender hidden states through self-attention and projects
    them into receiver's embedding space. Output is deliberately small
    initially (scale=0.2, output_scale=0.1) so the adapter starts
    near-zero and the receiver can learn to use the signal gradually.
    
    Input:  [batch, seq_len, 2560]  (Qwen last-layer hidden states)
    Output: [batch, seq_len, 2304]  (Gemma embedding-space vectors)
    """
    def __init__(self, sender_dim=2560, receiver_dim=2304, num_heads=8):
        super().__init__()
        self.input_projector = nn.Linear(sender_dim, receiver_dim, bias=True)
        self.pre_ln = nn.LayerNorm(receiver_dim, eps=1e-6)
        self.hidden_mha = nn.MultiHeadAttention(
            dims=receiver_dim,
            num_heads=num_heads,
        )
        self.post_ln = nn.LayerNorm(receiver_dim, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(receiver_dim)

    def __call__(self, x):
        """
        Args:
            x: [batch, seq_len, sender_dim] sender hidden states
        Returns:
            [batch, seq_len, receiver_dim] processed states for injection
        """
        x = self.input_projector(x)          # [B, S, 2304]
        normed = self.pre_ln(x)              # [B, S, 2304]
        attn_out = self.hidden_mha(normed, normed, normed)  # self-attention
        out = self.post_ln(normed + attn_out)  # residual + norm
        out = self.adaptive_proj(out)        # scale down
        return mx.clip(out, -10.0, 10.0)    # clamp for stability
```

### Parameter Count
- input_projector: 2560 × 2304 + 2304 = ~5.9M
- pre_ln: 2304 + 2304 = ~4.6K
- hidden_mha: Q/K/V projections (2304×2304 × 3) + out_proj (2304×2304) = ~21.2M
- post_ln: ~4.6K
- adaptive_proj: linear1 (2304×2304) + linear2 (2304×2304) + ln + scales = ~10.6M
- **Total: ~37.7M trainable parameters**

### Weight Initialization (from Interlat)
```python
def init_adapter_weights(adapter):
    """Initialize adapter weights following Interlat conventions."""
    # input_projector: default Xavier (nn.Linear default in MLX)
    # MHA: Xavier uniform with gain=1/sqrt(3) for Q/K/V, gain=1.0 for out_proj
    # adaptive_proj.linear1: N(0, 0.02)
    # adaptive_proj.linear2: Xavier gain=1e-2
    # All biases: zero
    # scale: 0.2, output_scale: 0.1
```

### Save/Load
```python
def save_adapter(adapter, path):
    weights = dict(mx.utils.tree_flatten(adapter.parameters()))
    mx.save_safetensors(path, weights)

def load_adapter(path, sender_dim=2560, receiver_dim=2304, num_heads=8):
    adapter = MHAAdapter(sender_dim, receiver_dim, num_heads)
    weights = mx.load(path)
    adapter.load_weights(list(weights.items()))
    return adapter
```


## Component 3: LoRA on Gemma-2-2B

### Why LoRA
We cannot fine-tune the full Gemma-2-2B (2.6B params). LoRA adds ~2-4M
trainable parameters by decomposing weight updates into low-rank matrices.
The base model stays frozen in 8-bit quantization.

### Configuration
```python
lora_config = {
    "rank": 8,
    "dropout": 0.0,
    "scale": 20.0,
}
num_lora_layers = 16  # Last 16 of 26 layers

# Apply LoRA to Gemma:
from mlx_lm.tuner.utils import linear_to_lora_layers
receiver_model.freeze()
linear_to_lora_layers(receiver_model, num_lora_layers, lora_config)
```

`linear_to_lora_layers` automatically:
1. Freezes the entire model
2. Finds all `nn.Linear` and `nn.QuantizedLinear` in the last 16 layers
3. Replaces them with `LoRALinear` (which wraps the frozen base + trainable A/B)
4. The LoRA layers are unfrozen (trainable)

### What Gets LoRA
In each Gemma TransformerBlock:
- `self_attn.q_proj` (QuantizedLinear 2304→2304)
- `self_attn.k_proj` (QuantizedLinear 2304→1024)
- `self_attn.v_proj` (QuantizedLinear 2304→1024)
- `self_attn.o_proj` (QuantizedLinear 2304→2304)
- `mlp.gate_proj` (QuantizedLinear 2304→9216)
- `mlp.down_proj` (QuantizedLinear 9216→2304)
- `mlp.up_proj` (QuantizedLinear 2304→9216)

Each gets rank-8 LoRA adapters. Roughly ~2.5M additional params for 16 layers.


## Component 4: Layer-0 Injection — `mlxmas/interlat_forward.py`

### How Interlat Injects Hidden States
The processed adapter output is inserted into Gemma's embedding sequence at
the position after the user message, wrapped in special `<bop>` and `<eop>`
token embeddings. The entire sequence then goes through ALL Gemma layers
from layer 0 — standard transformer forward pass.

This is fundamentally different from our prior approach (injecting at layer 22).
Layer-0 injection means every Gemma layer gets to process the injected states.
The LoRA adapters in those layers learn how to interpret them.

### Special Tokens
Add `<bop>` (begin-of-plan) and `<eop>` (end-of-plan) to Gemma's tokenizer
and resize embeddings. These bracket the injected hidden states so Gemma knows
where the latent signal starts and ends.

```python
# During setup (ONCE):
special_tokens = {"additional_special_tokens": ["<bop>", "<eop>"]}
num_added = receiver_tok.add_special_tokens(special_tokens)
if num_added > 0:
    receiver_model.resize_token_embeddings(len(receiver_tok))
bop_id = receiver_tok.convert_tokens_to_ids("<bop>")
eop_id = receiver_tok.convert_tokens_to_ids("<eop>")
```

CRITICAL: `resize_token_embeddings` must happen BEFORE applying LoRA.
The LoRA layers need to see the correct vocab size.

### Embedding Assembly
```python
def assemble_with_latent(receiver_model, receiver_tok,
                         adapter_output, question, gold_answer,
                         bop_id, eop_id):
    """Build the full embedding sequence with injected latent states.
    
    Sequence layout:
    [user_prompt_embeds] [<bop>] [adapter_output] [<eop>] [assistant_prompt_embeds]
    
    Labels: IGNORE on everything except the assistant's answer tokens.
    
    Args:
        receiver_model: Gemma-2-2B model
        adapter_output: [1, num_latent_tokens, 2304] from MHA adapter
        question: str, the GSM8K question text
        gold_answer: str, the gold step-by-step solution
        bop_id, eop_id: int, special token IDs
    
    Returns:
        inputs_embeds: [1, total_seq_len, 2304]
        attention_mask: [1, total_seq_len]
        labels: [1, total_seq_len] with IGNORE_TOKEN_ID=-100 for non-target positions
    """
    emb_layer = receiver_model.model.embed_tokens
    
    # Build user prompt (question only — NO answer here)
    user_prompt = (
        f"<start_of_turn>user\n"
        f"Using the reasoning context provided, solve the problem step by step. "
        f"Put your final numerical answer inside \\boxed{{}}.\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    user_tokens = receiver_tok.encode(user_prompt, add_special_tokens=True)
    user_ids = mx.array([user_tokens])
    user_embeds = emb_layer(user_ids)  # [1, user_len, 2304]
    
    # Apply sqrt(hidden_size) scaling to text embeddings (Gemma convention)
    user_embeds = user_embeds * (receiver_model.model.args.hidden_size ** 0.5)
    
    # Build answer tokens (these are the training targets)
    answer_tokens = receiver_tok.encode(gold_answer, add_special_tokens=False)
    answer_ids = mx.array([answer_tokens])
    answer_embeds = emb_layer(answer_ids)
    answer_embeds = answer_embeds * (receiver_model.model.args.hidden_size ** 0.5)
    
    # Special token embeddings
    bop_embed = emb_layer(mx.array([[bop_id]]))  # [1, 1, 2304]
    eop_embed = emb_layer(mx.array([[eop_id]]))  # [1, 1, 2304]
    bop_embed = bop_embed * (receiver_model.model.args.hidden_size ** 0.5)
    eop_embed = eop_embed * (receiver_model.model.args.hidden_size ** 0.5)
    
    # Adapter output is NOT scaled by sqrt(hidden_size) — it's already
    # in Gemma's operating range via the AdaptiveProjection
    
    # Concatenate: [user | <bop> | latent | <eop> | answer]
    inputs_embeds = mx.concatenate([
        user_embeds,       # [1, user_len, 2304]
        bop_embed,         # [1, 1, 2304]
        adapter_output,    # [1, latent_len, 2304]
        eop_embed,         # [1, 1, 2304]
        answer_embeds,     # [1, answer_len, 2304]
    ], axis=1)
    
    total_len = inputs_embeds.shape[1]
    user_len = user_embeds.shape[1]
    latent_len = adapter_output.shape[1]
    prefix_len = user_len + 1 + latent_len + 1  # user + bop + latent + eop
    answer_len = answer_embeds.shape[1]
    
    # Attention mask: all ones (everything attends to everything)
    attention_mask = mx.ones((1, total_len))
    
    # Labels: IGNORE for prefix, real token IDs for answer
    ignore_labels = mx.full((1, prefix_len), -100, dtype=mx.int32)
    answer_labels = mx.array([answer_tokens], dtype=mx.int32)
    labels = mx.concatenate([ignore_labels, answer_labels], axis=1)
    
    return inputs_embeds, attention_mask, labels
```

### Gemma Forward Pass with Embeddings
```python
def gemma_forward_with_embeds(model, inputs_embeds, cache=None):
    """Run Gemma forward pass from pre-built embeddings.
    
    IMPORTANT: sqrt(hidden_size) scaling must already be applied to
    text embeddings before calling this. The adapter output should
    NOT have sqrt scaling (AdaptiveProjection handles its own range).
    
    This function does NOT apply sqrt scaling — caller is responsible.
    """
    h = inputs_embeds  # Already scaled where needed
    
    if cache is None:
        cache = [None] * len(model.model.layers)
    
    mask = create_attention_mask(h, cache[0], return_array=True)
    
    for layer, c in zip(model.model.layers, cache):
        h = layer(h, mask, c)
    
    h = model.model.norm(h)
    out = model.model.embed_tokens.as_linear(h)
    out = mx.tanh(out / model.final_logit_softcapping)
    out = out * model.final_logit_softcapping
    return out  # [1, total_seq_len, vocab_size]
```


## Component 5: Training — `mlxmas/train_interlat.py`

### Three Losses (from Interlat)

Interlat uses three forward passes per training step:

**Pass 1 — Normal (grad):** Inject real sender hidden states → CE loss on answer
**Pass 2 — Plan text (no grad):** Insert plan text tokens instead → reference logits  
**Pass 3 — Random (no grad):** Inject hidden states from a DIFFERENT sample → contrastive negative

#### Loss 1: Cross-Entropy (CE)
Standard next-token prediction loss on the gold answer tokens.
```python
# After forward pass with injected hidden states:
logits = gemma_forward_with_embeds(receiver_model, inputs_embeds)
# Shift logits/labels for next-token prediction
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
mask = (shift_labels != -100)
ce = nn.losses.cross_entropy(shift_logits, shift_labels) * mask
ce_loss = ce.sum() / mask.sum()
```

#### Loss 2: Plan Similarity
KL divergence between normal logits (with hidden states) and plan logits
(with plan text). This teaches the hidden states to carry the same
information as the text plan.
```python
from mlx_lm.tuner.losses import kl_div_loss
# Only compute on supervised positions (where labels != -100)
# plan_logits are detached (no gradient through plan path)
kl = kl_div_loss(normal_logits[mask], plan_logits[mask])
plan_sim_loss = kl.mean()
```

#### Loss 3: Random Contrast
Margin-based JSD between normal logits and random-hidden-state logits.
Pushes the model to distinguish real signal from noise.
```python
from mlx_lm.tuner.losses import js_div_loss
# random_logits from a different sample's hidden states (detached)
jsd = js_div_loss(normal_logits[mask], random_logits[mask])
margin = 0.69  # ln(2) — maximum possible JSD
contrast_loss = mx.maximum(margin - jsd.mean(), 0.0)
```

#### Combined Loss
```python
# Weights are dynamically adjusted per Interlat:
plan_w = 0.5    # initial, adjusted based on plan_sim_loss magnitude
random_w = 1.5  # initial, adjusted based on contrast_loss magnitude
total_loss = ce_loss + plan_w * plan_sim_loss + random_w * contrast_loss
```

### Training Loop Structure
```python
def train_step(adapter, receiver_model, sender_hidden, plan_text,
               random_hidden, question, gold_answer, bop_id, eop_id):
    """One training step. Returns total_loss.
    
    Trainable parameters: adapter (all) + receiver LoRA weights
    Frozen: receiver base weights, sender (not loaded during training)
    """
    # Pass 1: Normal (with gradient)
    adapter_out = adapter(sender_hidden)  # [1, S, 2304]
    embeds_n, mask_n, labels_n = assemble_with_latent(
        receiver_model, receiver_tok, adapter_out, question, gold_answer,
        bop_id, eop_id
    )
    logits_n = gemma_forward_with_embeds(receiver_model, embeds_n)
```
    
    # Pass 2: Plan text (no gradient)
    # Insert plan text tokens at the same position instead of hidden states
    plan_tokens = receiver_tok.encode(plan_text, add_special_tokens=False)
    plan_ids = mx.array([plan_tokens])
    plan_embeds = receiver_model.model.embed_tokens(plan_ids)
    plan_embeds = plan_embeds * (receiver_model.model.args.hidden_size ** 0.5)
    # Assemble with plan text instead of adapter output
    embeds_p, _, labels_p = assemble_with_plan_text(...)
    logits_p = mx.stop_gradient(gemma_forward_with_embeds(receiver_model, embeds_p))
    
    # Pass 3: Random (no gradient)
    random_adapter_out = mx.stop_gradient(adapter(random_hidden))
    embeds_r, _, labels_r = assemble_with_latent(
        receiver_model, receiver_tok, random_adapter_out, question, gold_answer,
        bop_id, eop_id
    )
    logits_r = mx.stop_gradient(gemma_forward_with_embeds(receiver_model, embeds_r))
    
    # Compute losses
    ce_loss = compute_ce_loss(logits_n, labels_n)
    plan_sim_loss = compute_plan_similarity(logits_n, logits_p, labels_n, labels_p)
    contrast_loss = compute_random_contrast(logits_n, logits_r, labels_n, labels_r)
    
    total_loss = ce_loss + plan_w * plan_sim_loss + random_w * contrast_loss
    return total_loss

### Training Script CLI
```
python -m mlxmas.train_interlat \
    --receiver mlx-community/gemma-2-2b-it-8bit \
    --hidden-data mlxmas/data/sender_hidden_states.npz \
    --lora-rank 8 \
    --lora-layers 16 \
    --lr 1e-4 \
    --batch-size 1 \
    --iters 500 \
    --output mlxmas/data/interlat_adapter/ \
    --grad-checkpoint
```

### Optimizer and Gradient Setup
```python
import mlx.optimizers as optim

# Trainable parameters: adapter + receiver LoRA weights
# Use nn.value_and_grad to compute loss and gradients jointly
optimizer = optim.AdamW(learning_rate=1e-4)

# The loss function must be structured for nn.value_and_grad:
def loss_fn(adapter, receiver_model, batch):
    """
    adapter and receiver_model are nn.Modules with trainable params.
    nn.value_and_grad will differentiate w.r.t. adapter's params.
    Receiver's LoRA params also need gradients — see note below.
    """
    ...
    return total_loss

loss_and_grad = nn.value_and_grad(adapter, loss_fn)
```

IMPORTANT: `nn.value_and_grad(model, fn)` differentiates w.r.t. `model`'s
trainable parameters. But we need gradients for BOTH the adapter AND
Gemma's LoRA weights. Two options:

**Option A:** Wrap both in a single nn.Module container:
```python
class InterlatModel(nn.Module):
    def __init__(self, adapter, receiver_model):
        super().__init__()
        self.adapter = adapter
        self.receiver = receiver_model
```
Then `nn.value_and_grad(interlat_model, loss_fn)` differentiates w.r.t. all
trainable params in both sub-modules. This is the clean approach.

**Option B:** Manual gradient accumulation. More complex, avoid if possible.

Use Option A.


### Memory Budget (32GB Mac Mini M2 Pro)

| Component | Memory |
|---|---|
| Gemma-2-2B 8-bit weights | ~2.5 GB |
| LoRA adapter weights (rank 8, 16 layers) | ~10 MB |
| MHA adapter weights (37.7M params, float32) | ~150 MB |
| Optimizer state (AdamW: 2× trainable params) | ~300 MB |
| Forward pass activations (seq ~800, batch 1) | ~2-4 GB |
| Gradient checkpointing overhead | ~1 GB |
| **Total training** | **~6-8 GB** |

Qwen is NOT loaded during training. Hidden states are pre-collected and
loaded from disk as numpy arrays. This is critical — loading both models
simultaneously during training would not leave enough room for gradients.

With gradient checkpointing enabled, batch size 1 should fit comfortably.
Batch size 2 might work. Test empirically.


## Component 6: Evaluation — Update `mlxmas/test_cross.py`

### Eval Pipeline
At eval time, both models are loaded (no gradients, inference only):
1. Load Qwen (sender, ~4GB)
2. Load Gemma with LoRA adapters (~2.5GB + adapters)
3. Load trained MHA adapter
4. For each GSM8K test question:
   a. Run Qwen, capture last-layer hidden states
   b. Process through MHA adapter
   c. Inject at layer 0 into Gemma
   d. Generate answer autoregressively
   e. Extract boxed answer, compare to gold

Total eval memory: ~4GB + ~2.5GB + ~150MB + inference cache ≈ 8-10GB. Fine.

### Eval Script Updates
```
python -m mlxmas.test_cross \
    --receiver mlx-community/gemma-2-2b-it-8bit \
    --adapter-type interlat \
    --adapter-path mlxmas/data/interlat_adapter/ \
    --max-samples 5
```

The adapter directory contains:
- `mha_adapter.safetensors` — MHA adapter weights
- `lora_adapters.safetensors` — Gemma LoRA weights
- `adapter_config.json` — LoRA config (rank, scale, num_layers, etc.)
- `training_log.jsonl` — per-step loss components


## Execution Order

```
Phase 1: Data Collection (Qwen only, ~4GB)
    python -m mlxmas.collect_sender_hidden \
        --sender mlx-community/Qwen3-4B-8bit \
        --n-prompts 500 --max-tokens 512 \
        --output mlxmas/data/sender_hidden_states.npz

Phase 2: Training (Gemma only, ~6-8GB)
    python -m mlxmas.train_interlat \
        --receiver mlx-community/gemma-2-2b-it-8bit \
        --hidden-data mlxmas/data/sender_hidden_states.npz \
        --lora-rank 8 --lora-layers 16 \
        --lr 1e-4 --batch-size 1 --iters 500 \
        --output mlxmas/data/interlat_adapter/ \
        --grad-checkpoint

Phase 3: Diagnostic (both models, ~8GB, no grad)
    python -m mlxmas.diagnose_latents \
        --receiver mlx-community/gemma-2-2b-it-8bit \
        --adapter-type interlat \
        --adapter-path mlxmas/data/interlat_adapter/ \
        --sender-layer last --receiver-layer 0

Phase 4: GSM8K Eval (both models, ~8GB, no grad)
    python -m mlxmas.test_cross \
        --receiver mlx-community/gemma-2-2b-it-8bit \
        --adapter-type interlat \
        --adapter-path mlxmas/data/interlat_adapter/ \
        --max-samples 5
```


## What NOT To Do

1. **Do NOT load both models during training.** Qwen is only needed for
   data collection. Training uses pre-collected hidden states from disk.
   Loading both models + gradients will OOM.

2. **Do NOT inject at an intermediate layer.** Interlat injects at layer 0
   (embedding level). The LoRA adapters in ALL layers learn to process the
   injected states. Intermediate injection skips the early layers.

3. **Do NOT put question text in the receiver prompt during eval.** This was
   the bug that invalidated every prior result. The receiver prompt says
   "Using the reasoning context provided, solve the problem step by step."
   No question text. No hints.

4. **Do NOT train on MSE between paired activations.** This is what the MLP
   adapter did and it failed. Train on TASK LOSS (cross-entropy on correct
   answers). The model learns what the receiver NEEDS, not what the sender
   PRODUCES.

5. **Do NOT skip the plan similarity loss.** Without it, the model may learn
   a degenerate solution where it ignores the hidden states entirely and
   memorizes answers from the training set. Plan similarity forces the
   hidden states to carry the same information as the text plan.

6. **Do NOT skip the random contrast loss.** Without it, the model has no
   incentive to actually USE the hidden states. It could learn to produce
   the same output regardless of what hidden states are injected.

7. **Do NOT apply sqrt(hidden_size) scaling to the adapter output.** Only
   text embeddings from `embed_tokens()` get sqrt scaling. The adapter's
   AdaptiveProjection handles its own magnitude.

8. **Do NOT use `apply_cross_realignment()` or any Procrustes code.** The
   MHA adapter completely replaces all prior projection methods.

9. **Do NOT use the LatentMAS loop.** Single forward pass through Qwen,
   capture hidden states during generation. No planner/critic/refiner.

10. **Do NOT resize Gemma's token embeddings AFTER applying LoRA.** Resize
    first (for <bop>/<eop>), then apply LoRA. Order matters.


## New Files to Create

```
mlxmas/
├── mha_adapter.py              # MHAAdapter + AdaptiveProjection classes
├── collect_sender_hidden.py    # Data collection: Qwen generate + capture hidden states
├── interlat_forward.py         # assemble_with_latent, gemma_forward_with_embeds
├── train_interlat.py           # Training loop: 3 losses, optimizer, save/load
└── data/
    ├── sender_hidden_states.npz    # Collected hidden states (Phase 1 output)
    └── interlat_adapter/           # Trained weights (Phase 2 output)
        ├── mha_adapter.safetensors
        ├── lora_adapters.safetensors
        ├── adapter_config.json
        └── training_log.jsonl
```

### Files to Modify
- `test_cross.py` — Add `--adapter-type interlat` support
- `diagnose_latents.py` — Add `--adapter-type interlat` support


## Success Criteria

**Minimum viable signal:** After training, `diagnose_latents.py` with the
interlat adapter shows question-related tokens in logit decode (eggs, ducks,
market, sells). Receiver generates coherent text about the problem WITHOUT
seeing the question.

**Real success:** 1+ of 5 correct on GSM8K without question text.

**Full success:** >20% accuracy on 50 GSM8K questions without question text.

**Comparison point:** Qwen standalone baseline is 96% (48/50). We are not
trying to match that. We are trying to prove ANY information transfers
through the latent channel with a trained adapter + fine-tuned receiver.


## Key Differences from Original Interlat

| Aspect | Interlat (Du et al.) | Our Implementation |
|---|---|---|
| Framework | PyTorch + CUDA + flash_attention_2 | MLX + Metal (Apple Silicon) |
| Sender | Qwen2.5-7B | Qwen3-4B-8bit |
| Receiver | Qwen2.5-7B / LLaMA-3.1-8B | Gemma-2-2B-8bit |
| Receiver fine-tuning | Full SFT | LoRA (rank 8, memory constraint) |
| Training hardware | 8× A100 GPUs | 1× Mac Mini M2 Pro 32GB |
| Batch size | 2 | 1 |
| Hidden states | Last-layer, ~800 tokens | Last-layer, ~200-500 tokens |
| Data collection | Distributed multi-GPU | Sequential, single process |
| Task domain | ALFWorld + MATH | GSM8K |
| Curriculum mixing | Random ratio hidden/plan | Same (random ratio 0.1-0.9) |
| Compression | Optional student distillation | Not implemented (future) |

The architecture of the adapter itself (MHA + AdaptiveProjection) is
identical. The losses (CE + plan similarity + random contrast) are identical.
The injection method (layer 0, <bop>/<eop> wrapping) is identical. Only the
framework, models, and training scale differ.


## References

- Du et al., "Enabling Agents to Communicate Entirely in Latent Space,"
  arXiv:2511.09149 (Interlat)
- github.com/XiaoDu-flying/Interlat (reference implementation)
- github.com/nickfox/the_patch (our codebase)
- Zou et al., "Latent Collaboration in Multi-Agent Systems,"
  arXiv:2511.20639 (LatentMAS)
- MLX source: venv/lib/python3.14/site-packages/mlx/nn/layers/transformer.py
- MLX LoRA: venv/lib/python3.14/site-packages/mlx_lm/tuner/lora.py
- MLX losses: venv/lib/python3.14/site-packages/mlx_lm/tuner/losses.py
- Gemma-2 model: other/gemma2.py
- Qwen3 model: other/qwen3.py
