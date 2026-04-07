"""
Collect paired hidden states from Qwen (sender) and Gemma (receiver).

Both models run independently on the same text via forward pass.
Hidden states are captured at target layers, resampled to aligned positions.

Output: .npz with sender_states [N, D_a] and receiver_states [N, D_b].
Shared by all alignment approaches (MLP, CCA, OT).
"""

import argparse
import os
import time

import mlx.core as mx
import mlx_lm
import numpy as np

from mlxmas.contextual_procrustes import (
    extract_all_tokens_at_layer,
    load_calibration_prompts,
)


def resample(arr, n_pos):
    """Resample array to n_pos evenly-spaced positions (nearest-neighbor)."""
    seq_len = arr.shape[0]
    if seq_len <= n_pos:
        return arr
    indices = np.linspace(0, seq_len - 1, n_pos).astype(int)
    return arr[indices]


def collect_forward_pass_pairs(sender_model, sender_tok, receiver_model, receiver_tok,
                               prompts, sender_layer, receiver_layer, n_positions):
    """Collect paired states from normal forward passes on both models."""
    all_sender, all_receiver = [], []
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        h_sender = extract_all_tokens_at_layer(
            sender_model, sender_tok, prompt, sender_layer, is_gemma=False
        )
        h_receiver = extract_all_tokens_at_layer(
            receiver_model, receiver_tok, prompt, receiver_layer, is_gemma=True
        )

        h_sender = resample(h_sender, n_positions)
        h_receiver = resample(h_receiver, n_positions)
        n = min(h_sender.shape[0], h_receiver.shape[0])
        all_sender.append(h_sender[:n].astype(np.float32))
        all_receiver.append(h_receiver[:n].astype(np.float32))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            total_vecs = sum(a.shape[0] for a in all_sender)
            print(f"    [forward] {i+1}/{len(prompts)} ({rate:.1f}/s, "
                  f"ETA {eta:.0f}s) vectors={total_vecs}")

    return all_sender, all_receiver


def main():
    parser = argparse.ArgumentParser(
        description="Collect paired hidden states for adapter training"
    )
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--receiver", default="mlx-community/gemma-2-9b-it-8bit")
    parser.add_argument("--sender-layer", type=int, default=13)
    parser.add_argument("--receiver-layer", type=int, default=22)
    parser.add_argument("--n-prompts", type=int, default=100,
                        help="Total number of GSM8K train prompts")
    parser.add_argument("--n-positions", type=int, default=50,
                        help="Resampled positions per prompt (forward-pass mode)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "data",
            f"paired_states_S{args.sender_layer}_R{args.receiver_layer}.npz"
        )

    print(f"=== Collecting paired states: S{args.sender_layer} -> R{args.receiver_layer} ===")
    print(f"  Prompts: {args.n_prompts}")
    print(f"  Positions per prompt: {args.n_positions}")
    print(f"  Sender: {args.sender}")
    print(f"  Receiver: {args.receiver}")

    # Load prompts from GSM8K train split
    prompts = load_calibration_prompts(args.n_prompts)

    # Load both models
    print(f"\nLoading sender: {args.sender}")
    sender_model, sender_tok = mlx_lm.load(args.sender)

    print(f"Loading receiver: {args.receiver}")
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)

    # Collect forward-pass pairs
    print(f"\n--- Collecting {len(prompts)} prompts ---")
    t_start = time.time()
    all_sender, all_receiver = collect_forward_pass_pairs(
        sender_model, sender_tok, receiver_model, receiver_tok,
        prompts, args.sender_layer, args.receiver_layer, args.n_positions,
    )
    elapsed = time.time() - t_start

    H_sender = np.concatenate(all_sender, axis=0)
    H_receiver = np.concatenate(all_receiver, axis=0)

    print(f"\n=== Collection complete ===")
    print(f"  sender_states:   {H_sender.shape} ({H_sender.nbytes / 1e6:.1f} MB)")
    print(f"  receiver_states: {H_receiver.shape} ({H_receiver.nbytes / 1e6:.1f} MB)")
    print(f"  Total: {H_sender.shape[0]} vectors in {elapsed:.1f}s")

    # Save
    np.savez(args.output,
             sender_states=H_sender,
             receiver_states=H_receiver,
             sender_layer=args.sender_layer,
             receiver_layer=args.receiver_layer)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
