#!/usr/bin/env python3
"""
Layer-pair sweep for intermediate injection.

Computes Procrustes alignment for each (sender_layer, receiver_layer)
pair and evaluates downstream accuracy on 5 GSM8K questions.

Usage:
    python -m mlxmas.sweep_layers
    python -m mlxmas.sweep_layers --n_calibration 2000
    python -m mlxmas.sweep_layers --sender_layers 24,30,36 --receiver_layers 0,2,4,6,8
"""

import argparse
import json
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from mlxmas.contextual_procrustes import (
    extract_hidden_at_layers,
    compute_procrustes_with_heldout,
    load_calibration_prompts,
)
from mlxmas.cross_align import apply_cross_realignment
from mlxmas.cross_comm import generate_with_cross_latents_from_layer
from mlxmas.latent_comm import compute_alignment, apply_realignment
from mlxmas.prompts import build_prompt
from mlxmas.utils import extract_boxed_answer, normalize_answer


QUESTIONS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "gold": "18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "gold": "3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "gold": "70000"},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "gold": "540"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?", "gold": "20"},
]

AGENTS = ["planner", "critic", "refiner"]


def collect_all_hidden_states(model, tokenizer, prompts, layers, is_gemma=False):
    """Extract hidden states at multiple layers for all prompts.

    Returns: dict mapping layer_index -> [N, D] matrix of states
    """
    by_layer = {l: [] for l in layers}

    print(f"  Extracting hidden states at layers {layers} for {len(prompts)} prompts...")
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        captured = extract_hidden_at_layers(model, tokenizer, prompt, layers, is_gemma)
        for l in layers:
            by_layer[l].append(captured[l])
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = len(prompts) - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            pct = 100 * (i + 1) / len(prompts)
            print(f"    {i+1}/{len(prompts)} ({pct:.0f}%) | {rate:.1f} prompts/s | ETA {eta:.0f}s")

    # Stack into matrices
    result = {}
    for l in layers:
        stacked = mx.concatenate(by_layer[l], axis=0).astype(mx.float32)
        mx.eval(stacked)
        result[l] = stacked

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return result


def collect_sender_hidden_for_eval(model, tokenizer, question, latent_steps,
                                    W_a_self, target_norm_self):
    """Run sender latent loop and collect hidden states for one question."""
    all_hidden = []
    cache = make_prompt_cache(model)

    for role in AGENTS:
        prompt = build_prompt(tokenizer, role, question)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array([tokens])

        hidden = model.model(input_ids, cache=cache)
        mx.eval(hidden, *[c.state for c in cache])

        last_hidden = hidden[:, -1:, :]
        aligned = apply_realignment(last_hidden, W_a_self, target_norm_self)
        collected = [aligned]

        for _ in range(latent_steps):
            aligned = apply_realignment(last_hidden, W_a_self, target_norm_self)
            hidden = model.model(None, cache=cache, input_embeddings=aligned)
            mx.eval(hidden, *[c.state for c in cache])
            last_hidden = hidden[:, -1:, :]
            aligned_step = apply_realignment(last_hidden, W_a_self, target_norm_self)
            collected.append(aligned_step)

        agent_hidden = mx.concatenate(collected, axis=1)
        all_hidden.append(agent_hidden)

    sender_hidden = mx.concatenate(all_hidden, axis=1)
    mx.eval(sender_hidden)
    return sender_hidden


def eval_pair(sender_model, sender_tok, receiver_model, receiver_tok,
              W_cross, mean_a, mean_b, target_norm_b,
              W_a_self, target_norm_self, start_layer, latent_steps=10):
    """Evaluate one (sender_layer, receiver_layer) pair on 5 GSM8K questions."""
    correct = 0

    for i, item in enumerate(QUESTIONS):
        question = item["question"]
        gold = item["gold"]

        sender_hidden = collect_sender_hidden_for_eval(
            sender_model, sender_tok, question, latent_steps,
            W_a_self, target_norm_self,
        )

        projected = apply_cross_realignment(
            sender_hidden, W_cross, mean_a, mean_b, target_norm_b
        )
        mx.eval(projected)

        output = generate_with_cross_latents_from_layer(
            receiver_model, receiver_tok,
            projected,
            start_layer=start_layer,
            max_tokens=2048, temperature=0.6,
        )

        pred = normalize_answer(extract_boxed_answer(output))
        ok = False
        if pred and gold:
            try:
                ok = float(pred) == float(gold)
            except (ValueError, TypeError):
                ok = (pred == gold)
        if ok:
            correct += 1
        print(f"    Q{i+1}: pred={pred}, gold={gold}, ok={ok}")

    return correct, len(QUESTIONS)


def main():
    parser = argparse.ArgumentParser(description="Layer-pair sweep for intermediate injection")
    parser.add_argument("--sender_layers", type=str, default="23,29,35",
                        help="Comma-separated sender layer indices (Qwen3-4B has layers 0-35)")
    parser.add_argument("--receiver_layers", type=str, default="0,2,4,6,8",
                        help="Comma-separated receiver layer indices (Gemma-2-2B has layers 0-25)")
    parser.add_argument("--n_calibration", type=int, default=2000,
                        help="Number of calibration prompts")
    parser.add_argument("--latent_steps", type=int, default=10,
                        help="Latent thinking steps per agent")
    parser.add_argument("--output", type=str, default="sweep_results.json",
                        help="Output JSON file")
    parser.add_argument("--calibration_only", action="store_true",
                        help="Only compute calibration cosines, skip downstream eval")
    args = parser.parse_args()

    sender_layers = [int(x) for x in args.sender_layers.split(",")]
    receiver_layers = [int(x) for x in args.receiver_layers.split(",")]

    print(f"=== Layer-pair sweep ===")
    print(f"  Sender layers:   {sender_layers}")
    print(f"  Receiver layers: {receiver_layers}")
    print(f"  Calibration:     {args.n_calibration} prompts")

    # Load calibration prompts
    prompts = load_calibration_prompts(args.n_calibration)
    print(f"  Loaded {len(prompts)} prompts")

    # Load models
    sender_name = "mlx-community/Qwen3-4B-4bit"
    receiver_name = "mlx-community/gemma-2-2b-it-4bit"

    print(f"\nLoading sender: {sender_name}")
    sender_model, sender_tok = mlx_lm.load(sender_name)
    W_a_self, target_norm_self = compute_alignment(sender_model)

    print(f"Loading receiver: {receiver_name}")
    receiver_model, receiver_tok = mlx_lm.load(receiver_name)

    # Extract hidden states at all layers of interest in one pass per model
    print(f"\nSender extraction (Qwen):")
    sender_states = collect_all_hidden_states(
        sender_model, sender_tok, prompts, sender_layers, is_gemma=False
    )
    print(f"Receiver extraction (Gemma):")
    receiver_states = collect_all_hidden_states(
        receiver_model, receiver_tok, prompts, receiver_layers, is_gemma=True
    )

    # Sweep all (s, m) pairs
    results = []
    for s in sender_layers:
        for m in receiver_layers:
            print(f"\n--- Pair (sender={s}, receiver={m}) ---")
            H_a = sender_states[s]
            H_b = receiver_states[m]

            for cal_variant in ["final_token"]:
                for center in [True, False]:
                    label = f"s{s}_m{m}_{cal_variant}_{'centered' if center else 'uncentered'}"
                    print(f"  [{label}]")

                    proc = compute_procrustes_with_heldout(H_a, H_b, center=center)
                    print(f"    train_cos={proc['train_cosine']:.4f}, "
                          f"heldout_cos={proc['heldout_cosine']:.4f}")

                    entry = {
                        "sender_layer": s,
                        "receiver_layer": m,
                        "calibration": cal_variant,
                        "centered": center,
                        "train_cosine": proc["train_cosine"],
                        "heldout_cosine": proc["heldout_cosine"],
                        "n_train": proc["n_train"],
                        "n_heldout": proc["n_heldout"],
                    }

                    if not args.calibration_only:
                        print(f"    Running downstream eval...")
                        W_cross = proc["W_ortho"]
                        mean_a = proc["mean_a"]
                        mean_b = proc["mean_b"]
                        target_norm_b = proc["target_norm_b"]

                        n_correct, n_total = eval_pair(
                            sender_model, sender_tok,
                            receiver_model, receiver_tok,
                            W_cross, mean_a, mean_b, target_norm_b,
                            W_a_self, target_norm_self,
                            start_layer=m,
                            latent_steps=args.latent_steps,
                        )
                        entry["gsm8k_correct"] = n_correct
                        entry["gsm8k_total"] = n_total
                        entry["gsm8k_accuracy"] = n_correct / n_total
                        print(f"    GSM8K: {n_correct}/{n_total}")

                    results.append(entry)

    # Save results
    output = {"pairs": results, "sender_layers": sender_layers,
              "receiver_layers": receiver_layers, "n_calibration": len(prompts)}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Sender':>8} {'Receiver':>8} {'Center':>8} {'Train':>8} {'Heldout':>8}", end="")
    if not args.calibration_only:
        print(f" {'GSM8K':>8}", end="")
    print()
    print("-" * 60)
    for r in results:
        ctr = "yes" if r["centered"] else "no"
        print(f"{r['sender_layer']:>8} {r['receiver_layer']:>8} {ctr:>8} "
              f"{r['train_cosine']:>8.4f} {r['heldout_cosine']:>8.4f}", end="")
        if "gsm8k_accuracy" in r:
            print(f" {r['gsm8k_accuracy']:>8.1%}", end="")
        print()


if __name__ == "__main__":
    main()
