#!/usr/bin/env python3
"""
Train Interlat MHA adapter + Gemma LoRA for cross-model latent communication.

Minimal version: CE loss only. No curriculum, no contrastive, no plan similarity.
One forward pass per step. Prove the concept works first.

Qwen is NOT loaded — hidden states are pre-collected from disk.
Only Gemma + adapter are in memory (~6-8GB).

Usage:
    python -m mlxmas.train_interlat \
        --receiver mlx-community/gemma-2-2b-it-8bit \
        --hidden-data mlxmas/data/sender_hidden_states.npz \
        --output mlxmas/data/interlat_adapter/
"""

import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.utils import linear_to_lora_layers

from mlxmas.mha_adapter import MHAAdapter, save_adapter


class InterlatModel(nn.Module):
    """Wrapper so nn.value_and_grad differentiates both adapter and LoRA."""

    def __init__(self, adapter, receiver):
        super().__init__()
        self.adapter = adapter
        self.receiver = receiver


def add_special_tokens(model, tokenizer):
    """Add <bop> and <eop> to tokenizer, resize embeddings.

    Must be called BEFORE applying LoRA.

    Returns: (bop_id, eop_id)
    """
    special = {"additional_special_tokens": ["<bop>", "<eop>"]}
    num_added = tokenizer.add_special_tokens(special)
    if num_added > 0:
        old_embed = model.model.embed_tokens
        dim = model.model.args.hidden_size  # actual dim, not quantized weight shape
        # Dequantize existing embeddings by running dummy tokens through
        dummy_ids = mx.arange(old_embed.num_embeddings).reshape(-1)
        old_embeds_full = old_embed(dummy_ids)  # [vocab, dim]
        mx.eval(old_embeds_full)
        old_vocab = old_embeds_full.shape[0]
        new_vocab = old_vocab + num_added

        new_weight = mx.concatenate([
            old_embeds_full,
            mx.random.normal((num_added, dim)) * 0.02,
        ], axis=0)
        model.model.embed_tokens = nn.Embedding(new_vocab, dim)
        model.model.embed_tokens.weight = new_weight
        model.model.embed_tokens.freeze()
        mx.eval(model.model.embed_tokens.weight)
        print(f"  Resized embeddings: {old_vocab} -> {new_vocab} (dim={dim})", flush=True)

    bop_id = tokenizer.convert_tokens_to_ids("<bop>")
    eop_id = tokenizer.convert_tokens_to_ids("<eop>")
    return bop_id, eop_id


def gemma_forward_with_embeds(model, inputs_embeds):
    """Run Gemma forward pass from pre-built embeddings.

    sqrt(hidden_size) scaling must already be applied to text embeddings.
    Adapter output should NOT have sqrt scaling.
    No cache — single forward pass for training.
    """
    h = inputs_embeds
    # Causal mask for training (no cache, offset=0)
    mask = create_attention_mask(h, return_array=True)

    for layer in model.model.layers:
        h = layer(h, mask, cache=None)

    h = model.model.norm(h)
    out = model.model.embed_tokens.as_linear(h)

    if hasattr(model, 'final_logit_softcapping') and model.final_logit_softcapping:
        out = mx.tanh(out / model.final_logit_softcapping)
        out = out * model.final_logit_softcapping

    return out


def assemble_input(receiver_model, receiver_tok, adapter_output,
                   gold_answer, bop_id, eop_id):
    """Build embedding sequence with injected latent states.

    Layout: [user_prompt | <bop> | adapter_output | <eop> | answer]
    Labels: IGNORE on prefix, real token IDs on answer.

    No question text in the user prompt.
    """
    emb_layer = receiver_model.model.embed_tokens
    scale = receiver_model.model.args.hidden_size ** 0.5

    # User prompt (no question)
    user_prompt = (
        "<start_of_turn>user\n"
        "Using the reasoning context provided, solve the problem step by step. "
        "Put your final numerical answer inside \\boxed{}.\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    )
    user_tokens = receiver_tok.encode(user_prompt, add_special_tokens=True)
    user_embeds = emb_layer(mx.array([user_tokens])) * scale

    # Answer tokens (training target)
    answer_tokens = receiver_tok.encode(gold_answer, add_special_tokens=False)
    answer_embeds = emb_layer(mx.array([answer_tokens])) * scale

    # Special token embeddings
    bop_embed = emb_layer(mx.array([[bop_id]])) * scale
    eop_embed = emb_layer(mx.array([[eop_id]])) * scale

    # Concatenate: [user | <bop> | latent | <eop> | answer]
    inputs_embeds = mx.concatenate([
        user_embeds,
        bop_embed,
        adapter_output,  # NOT scaled by sqrt — adapter handles its own range
        eop_embed,
        answer_embeds,
    ], axis=1)

    prefix_len = user_embeds.shape[1] + 1 + adapter_output.shape[1] + 1
    ignore_labels = mx.full((1, prefix_len), -100, dtype=mx.int32)
    answer_labels = mx.array([answer_tokens], dtype=mx.int32)
    labels = mx.concatenate([ignore_labels, answer_labels], axis=1)

    return inputs_embeds, labels


def compute_ce_loss(logits, labels):
    """Cross-entropy loss on supervised positions."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = (shift_labels != -100)

    if mask.sum().item() == 0:
        return mx.array(0.0)

    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="none")
    ce = ce * mask
    return ce.sum() / mask.sum()


def loss_fn(interlat_model, sender_hidden, gold_answer, receiver_tok,
            bop_id, eop_id):
    """Compute CE loss for one sample."""
    adapter_out = interlat_model.adapter(sender_hidden)
    embeds, labels = assemble_input(
        interlat_model.receiver, receiver_tok,
        adapter_out, gold_answer, bop_id, eop_id
    )
    logits = gemma_forward_with_embeds(interlat_model.receiver, embeds)
    loss = compute_ce_loss(logits, labels)
    return loss


def main():
    parser = argparse.ArgumentParser(description="Train Interlat adapter")
    parser.add_argument("--receiver", type=str,
                        default="mlx-community/gemma-2-2b-it-8bit")
    parser.add_argument("--hidden-data", type=str,
                        default="mlxmas/data/sender_hidden_states.npz")
    parser.add_argument("--output", type=str,
                        default="mlxmas/data/interlat_adapter/")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    # Load pre-collected sender hidden states
    print(f"Loading sender hidden states from {args.hidden_data}...", flush=True)
    data = np.load(args.hidden_data, allow_pickle=True)
    hidden_list = data["hidden_states"]
    questions = data["questions"]
    gold_answers = data["gold_answers"]
    n_samples = int(data["n_samples"])
    print(f"  {n_samples} samples loaded", flush=True)

    # Train/val split
    n_val = max(1, int(n_samples * args.val_split))
    n_train = n_samples - n_val
    indices = np.random.RandomState(42).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    print(f"  Train: {n_train}, Val: {n_val}", flush=True)

    # Load receiver
    print(f"\nLoading receiver: {args.receiver}", flush=True)
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)

    # Add special tokens BEFORE LoRA
    bop_id, eop_id = add_special_tokens(receiver_model, receiver_tok)
    print(f"  <bop>={bop_id}, <eop>={eop_id}", flush=True)

    # Apply LoRA
    lora_config = {"rank": args.lora_rank, "dropout": 0.0, "scale": 20.0}
    linear_to_lora_layers(receiver_model, args.lora_layers, lora_config)
    n_lora = sum(p.size for _, p in
                 tree_flatten(receiver_model.trainable_parameters()))
    print(f"  LoRA: rank={args.lora_rank}, layers={args.lora_layers}, "
          f"params={n_lora:,}", flush=True)

    # Create adapter
    adapter = MHAAdapter(sender_dim=2560, receiver_dim=2304, num_heads=1)
    mx.eval(adapter.parameters())
    n_adapter = sum(p.size for _, p in
                    tree_flatten(adapter.parameters()))
    print(f"  Adapter params: {n_adapter:,}", flush=True)

    # Wrap in single module for joint gradient computation
    interlat = InterlatModel(adapter, receiver_model)
    n_total = sum(p.size for _, p in
                  tree_flatten(interlat.trainable_parameters()))
    print(f"  Total trainable: {n_total:,}\n", flush=True)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # Gradient function
    loss_and_grad = nn.value_and_grad(
        interlat,
        lambda model, h, g: loss_fn(model, h, g, receiver_tok, bop_id, eop_id)
    )

    # Output directory
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, "training_log.jsonl")
    log_file = open(log_path, "w")

    # Training loop
    print(f"=== Training: {args.epochs} epochs, lr={args.lr} ===", flush=True)
    global_step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        perm = np.random.permutation(n_train)

        for batch_i in range(n_train):
            idx = train_idx[perm[batch_i]]
            h = mx.array(hidden_list[idx]).reshape(1, -1, 2560)
            gold = str(gold_answers[idx])

            loss, grads = loss_and_grad(interlat, h, gold)
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(interlat, grads)
            mx.eval(interlat.parameters(), optimizer.state, grad_norm)

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            if global_step % args.log_every == 0:
                from datetime import datetime
                now = datetime.now().strftime("%-I:%M%p %-m-%-d-%y").lower()
                elapsed = time.time() - t0
                el_m, el_s = divmod(int(elapsed), 60)
                total_steps = n_train * args.epochs
                steps_done = (epoch * n_train) + batch_i + 1
                rate = steps_done / elapsed if elapsed > 0 else 0
                eta_sec = (total_steps - steps_done) / rate if rate > 0 else 0
                eta_m, eta_s = divmod(int(eta_sec), 60)
                avg = epoch_loss / (batch_i + 1)
                gn = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                print(f"  {now} [{steps_done}/{total_steps}] loss={loss_val:.4f} "
                      f"avg={avg:.4f} gnorm={gn:.2f} elapsed={el_m}m{el_s:02d}s ETA={eta_m}m{eta_s:02d}s", flush=True)
                log_file.write(json.dumps({
                    "step": global_step, "epoch": epoch + 1,
                    "loss": loss_val, "avg_loss": avg,
                    "elapsed_sec": round(elapsed, 1),
                }) + "\n")
                log_file.flush()

        # End of epoch — validation
        val_loss_sum = 0.0
        for vi in range(n_val):
            idx = val_idx[vi]
            h = mx.array(hidden_list[idx]).reshape(1, -1, 2560)
            gold = str(gold_answers[idx])
            vl = loss_fn(interlat, h, gold, receiver_tok, bop_id, eop_id)
            val_loss_sum += vl.item()

        val_loss = val_loss_sum / n_val
        train_loss = epoch_loss / n_train
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({elapsed:.0f}s)", flush=True)

        log_file.write(json.dumps({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "elapsed_sec": round(elapsed, 1),
        }) + "\n")
        log_file.flush()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter(adapter, os.path.join(args.output, "mha_adapter.safetensors"))
            # Save LoRA weights
            lora_weights = dict(tree_flatten(
                receiver_model.trainable_parameters()
            ))
            mx.save_safetensors(
                os.path.join(args.output, "lora_adapters.safetensors"),
                lora_weights
            )
            print(f"  Saved best model (val_loss={val_loss:.4f})", flush=True)

    log_file.close()
    total_time = time.time() - t0

    # Save config
    config = {
        "sender_dim": 2560,
        "receiver_dim": 2304,
        "num_heads": 1,
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "lora_scale": 20.0,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_train": n_train,
        "n_val": n_val,
        "best_val_loss": best_val_loss,
        "total_time_sec": round(total_time, 1),
        "bop_id": bop_id,
        "eop_id": eop_id,
        "receiver_model": args.receiver,
    }
    with open(os.path.join(args.output, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Training complete in {total_time:.0f}s", flush=True)
    print(f"Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
