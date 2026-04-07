#!/usr/bin/env python3
"""
Cross-model latent communication test:
  Qwen3-4B (sender) → Gemma-2-9B (receiver)

Single forward pass through Qwen captures hidden states at the target layer.
These are projected into Gemma's space via the selected adapter (Procrustes,
MLP, CCA, or residual) and injected into Gemma at the target receiver layer.
Gemma generates an answer from the injected latents only (no question text).
"""

import argparse
import json
import mlx.core as mx
import mlx_lm
import time
import os
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from mlxmas.cross_align import apply_cross_realignment
from mlxmas.contextual_procrustes import load_alignment
from mlxmas.cross_comm import generate_with_cross_latents_from_layer
from mlxmas.utils import extract_boxed_answer, normalize_answer


def load_gsm8k_test(max_samples=-1):
    """Load GSM8K test set questions with gold answers."""
    import re
    import random
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = []
    for item in ds:
        gold_match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", item["answer"])
        gold = gold_match.group(1) if gold_match else None
        questions.append({
            "question": item["question"].strip(),
            "gold": gold,
        })
    random.shuffle(questions)
    if max_samples > 0:
        questions = questions[:max_samples]
    return questions

def create_parser():
    """Create argument parser for cross-model test."""
    parser = argparse.ArgumentParser(
        description="Cross-model latent communication test: Qwen3-4B → Gemma-2"
    )
    parser.add_argument("--receiver", default="mlx-community/gemma-2-2b-it-8bit",
                        help="Receiver model (default: gemma-2-2b-it-8bit)")
    parser.add_argument("--adapter", choices=["procrustes", "mlp", "cca", "residual"],
                        default="procrustes", help="Alignment method")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to adapter/alignment .npz (auto-discovered if None)")
    parser.add_argument("--start-layer", type=int, default=12,
                        help="Gemma receiver layer to inject at (default 12)")
    parser.add_argument("--sender-layer", type=int, default=13,
                        help="Qwen sender layer to extract from (default 13)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Number of GSM8K test questions to evaluate (default 50, -1 for all)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    sender_name = "mlx-community/Qwen3-4B-8bit"
    receiver_name = args.receiver

    # Step 1: Load alignment / adapter
    adapter_type = args.adapter
    method_name = f"{adapter_type}_S{args.sender_layer}_R{args.start_layer}"

    if adapter_type == "mlp":
        from mlxmas.mlp_adapter import load_mlp_adapter, apply_mlp_adapter
        adapter_path = args.adapter_path
        if adapter_path is None:
            adapter_path = os.path.join(
                os.path.dirname(__file__), "data",
                f"mlp_adapter_S{args.sender_layer}_R{args.start_layer}.npz"
            )
        if not os.path.exists(adapter_path):
            print(f"MLP adapter not found: {adapter_path}")
            print(f"Train one first with train_mlp_adapter.py")
            return
        print(f"=== Loading MLP adapter from {adapter_path} ===")
        mlp_adapter, mlp_meta = load_mlp_adapter(adapter_path)
        print(f"  Val cosine: {mlp_meta['val_cosine']:.4f}")
        print(f"  Val MSE: {mlp_meta['val_mse']:.6f}")

    elif adapter_type == "cca":
        from mlxmas.cca_adapter import load_cca, apply_cca
        adapter_path = args.adapter_path
        if adapter_path is None:
            adapter_path = os.path.join(
                os.path.dirname(__file__), "data",
                f"cca_adapter_S{args.sender_layer}_R{args.start_layer}_K512.npz"
            )
        if not os.path.exists(adapter_path):
            print(f"CCA adapter not found: {adapter_path}")
            print(f"Fit one first with cca_adapter.py")
            return
        print(f"=== Loading CCA adapter from {adapter_path} ===")
        cca_Wa, cca_Wb, cca_mu_x, cca_mu_y, cca_target_norm, cca_meta = load_cca(adapter_path)
        print(f"  K={cca_meta['K']}, top correlation: {cca_meta['top_correlation']:.4f}, "
              f"mean: {cca_meta['mean_correlation']:.4f}")

    elif adapter_type == "residual":
        from mlxmas.residual_adapter import load_adapter, apply_adapter
        adapter_path = args.adapter_path
        if adapter_path is None:
            adapter_path = os.path.join(
                os.path.dirname(__file__), "data",
                f"adapter_S{args.sender_layer}_R{args.start_layer}.npz"
            )
        if not os.path.exists(adapter_path):
            print(f"Residual adapter not found: {adapter_path}")
            return
        print(f"=== Loading residual adapter from {adapter_path} ===")
        res_adapter, W_base, mean_a, mean_b, target_norm_b, res_meta = load_adapter(adapter_path)
        print(f"  Baseline cosine: {res_meta['baseline_cosine']:.4f}")
        print(f"  Adapter val cosine: {res_meta['adapter_val_cosine']:.4f}")

    else:  # procrustes
        procrustes_path = args.adapter_path
        if procrustes_path is None:
            procrustes_path = os.path.join(
                os.path.dirname(__file__), "data",
                f"procrustes_S{args.sender_layer}_R{args.start_layer}_multitoken.npz"
            )
        if not os.path.exists(procrustes_path):
            print(f"Procrustes alignment file not found: {procrustes_path}")
            print(f"Run calibration first:")
            print(f"  python -m mlxmas.contextual_procrustes "
                  f"--sender_layer {args.sender_layer} "
                  f"--receiver_layer {args.start_layer}")
            return
        print(f"=== Loading Procrustes alignment from {procrustes_path} ===")
        alignment = load_alignment(procrustes_path)
        W_ab = alignment["W_ortho"]
        mean_a = alignment["mean_a"]
        mean_b = alignment["mean_b"]
        target_norm_b = alignment["target_norm_b"]
        if "heldout_cosine" in alignment:
            print(f"  Held-out cosine: {alignment['heldout_cosine']:.4f}, "
                  f"train cosine: {alignment['train_cosine']:.4f}")
        elif "cos_sim" in alignment:
            print(f"  Calibration cosine: {alignment['cos_sim']:.4f}")
        if "sender_layer" in alignment:
            print(f"  Layers: sender={alignment['sender_layer']}, "
                  f"receiver={alignment['receiver_layer']}")
        print(f"  W shape: {W_ab.shape}")

    # Step 2: Load models (8-bit)
    print(f"\n=== Loading inference models (8-bit) ===")
    print(f"Sender: {sender_name}")
    sender_model, sender_tok = mlx_lm.load(sender_name)

    print(f"Receiver: {receiver_name}")
    receiver_model, receiver_tok = mlx_lm.load(receiver_name)

    # Load test questions
    questions = load_gsm8k_test(max_samples=args.max_samples)
    print(f"\nLoaded {len(questions)} GSM8K test questions")
    print()

    # Step 3: Run test questions
    correct = 0
    total_start = time.time()
    for i, item in enumerate(questions):
        question = item["question"]
        gold = item["gold"]
        print(f"==================== Question #{i+1} ====================")
        print(f"Q: {question[:100]}...")
        print(f"Gold: {gold}")

        t0 = time.time()

        # Sender (Qwen): single forward pass, capture at sender_layer
        tokens = sender_tok.encode(question, add_special_tokens=True)
        input_ids = mx.array([tokens])
        sender_cache = make_prompt_cache(sender_model)
        h = sender_model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, sender_cache[0], return_array=True)
        for li, (layer, c) in enumerate(zip(sender_model.model.layers, sender_cache)):
            h = layer(h, mask, c)
            if li == args.sender_layer:
                sender_hidden = h.astype(mx.float32)
                break
        mx.eval(sender_hidden)
        print(f"  Sender hidden: {sender_hidden.shape}")

        # Project into Gemma's layer-{start_layer} space
        if adapter_type == "mlp":
            projected = apply_mlp_adapter(sender_hidden, mlp_adapter)
        elif adapter_type == "cca":
            projected = apply_cca(sender_hidden, cca_Wa, cca_Wb, cca_mu_x, cca_mu_y, cca_target_norm)
        elif adapter_type == "residual":
            projected = apply_adapter(
                sender_hidden, res_adapter, W_base, mean_a, mean_b, target_norm_b
            )
        else:
            projected = apply_cross_realignment(
                sender_hidden, W_ab, mean_a, mean_b, target_norm_b
            )
        mx.eval(projected)
        print(f"  Projected to receiver space: {projected.shape}")

        # Inject at start_layer + 1: the adapter/alignment targets the OUTPUT
        # of start_layer, so the projected vectors should enter at the NEXT layer.
        # Feeding them into start_layer would double-process through that layer.
        inject_layer = args.start_layer + 1
        print(f"  Generating with receiver (Gemma, inject at layer {inject_layer} "
              f"[adapter targets L{args.start_layer} output])...")
        output = generate_with_cross_latents_from_layer(
            receiver_model, receiver_tok,
            projected,
            start_layer=inject_layer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        elapsed = time.time() - t0
        pred = normalize_answer(extract_boxed_answer(output))
        ok = False
        if pred and gold:
            try:
                ok = float(pred) == float(gold)
            except (ValueError, TypeError):
                ok = (pred == gold)
        if ok:
            correct += 1

        print(f"\n[Gemma output]\n{output[:500]}\n")
        print(f"Result: Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
        print(f"Running: {correct}/{i+1} = {100*correct/(i+1):.0f}%")
        print()

    total_time = time.time() - total_start
    total = len(questions)
    acc = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"FINAL: {correct}/{total} = {acc*100:.0f}% accuracy")
    print(f"Method: {method_name}")
    print(f"Baseline (vocab LS, L0→L0): 1/5 = 20%")
    print(f"{'='*60}")

    print(json.dumps({
        "method": method_name,
        "sender": sender_name,
        "receiver": receiver_name,
        "sender_layer": args.sender_layer,
        "start_layer": args.start_layer,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "total_time_sec": round(total_time, 2),
        "time_per_sample_sec": round(total_time / total, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
