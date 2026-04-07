#!/usr/bin/env python3
"""
Test 2: Layer Skip Test — Where does reasoning live in each model?

For each model, skip one layer at a time (replace its output with its input,
i.e. identity residual) and measure GSM8K accuracy. Layers where skipping
causes the biggest accuracy drop are the ones critical for reasoning.

This tells you:
  - Sender: which layers to EXTRACT from (where reasoning is fully formed)
  - Receiver: which layers are first to do semantic processing (injection point)

Usage:
    python -m mlxmas.test2_layer_skip
    python -m mlxmas.test2_layer_skip --model mlx-community/Qwen3-4B-4bit --n_questions 20
"""

import argparse
import time
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask


QUESTIONS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "gold": "18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "gold": "3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "gold": "70000"},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "gold": "540"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?", "gold": "20"},
    {"question": "Kylar went to the store to get water and some apples. He spent $8 on water and spent three times as much on apples. How much money did Kylar spend in total?", "gold": "32"},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "gold": "260"},
    {"question": "Carla is downloading a 200 GB file. She can download 2 GB/minute. For the first 60 minutes, she downloads at 2 GB/minute. Then for the next 60 minutes, a weather event slows her connection to 1 GB/minute. After that, the connection goes back to normal. How many minutes does it take for her to download the entire file?", "gold": "160"},
    {"question": "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the rest of the time driving at 80 mph. How far is he from home when the 4 hours are up?", "gold": "20"},
    {"question": "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?", "gold": "460"},
]


def forward_skip_layer(model, input_ids, skip_layer, is_gemma=False):
    """Full forward pass, but skip one layer (identity residual).

    When layer == skip_layer, output = input (layer does nothing).
    Returns logits.
    """
    h = model.model.embed_tokens(input_ids)
    if is_gemma:
        h = h * (model.model.args.hidden_size ** 0.5)

    cache = make_prompt_cache(model)
    mask = create_attention_mask(h, cache[0], return_array=True)

    for i, (layer, c) in enumerate(zip(model.model.layers, cache)):
        if i == skip_layer:
            # Identity — skip this layer entirely
            continue
        h = layer(h, mask, c)

    h = model.model.norm(h)

    # Model-specific logit computation
    if is_gemma:
        out = model.model.embed_tokens.as_linear(h)
        out = mx.tanh(out / model.final_logit_softcapping)
        out = out * model.final_logit_softcapping
    elif hasattr(model, 'lm_head'):
        out = model.lm_head(h)
    else:
        # Tied embeddings (Qwen3 default)
        out = model.model.embed_tokens.as_linear(h)

    mx.eval(out)
    return out


def logprob_with_skip(model, tokenizer, question, gold_answer, skip_layer,
                      is_gemma=False):
    """Compute log-probability of the gold answer given the question,
    with one layer skipped.

    More principled than generating: directly measures how much
    skipping a layer degrades the model's ability to produce the
    correct answer.

    Returns: mean log-prob of gold answer tokens (more negative = worse).
    """
    if is_gemma:
        prompt = (f"<start_of_turn>user\nSolve: {question}<end_of_turn>\n"
                  f"<start_of_turn>model\nThe answer is {gold_answer}.")
    else:
        prompt = (f"<|im_start|>user\nSolve: {question}<|im_end|>\n"
                  f"<|im_start|>assistant\nThe answer is {gold_answer}.")

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    # Forward with skip
    logits = forward_skip_layer(model, input_ids, skip_layer, is_gemma)

    # Compute log-probs: shift logits and targets by 1
    log_probs = nn.log_softmax(logits[0, :-1, :], axis=-1)  # [seq-1, vocab]
    targets = mx.array(tokens[1:])  # [seq-1]

    # Gather log-probs of actual next tokens
    token_logprobs = log_probs[mx.arange(len(targets)), targets]
    mx.eval(token_logprobs)

    # Return mean log-prob (over last 20 tokens — the answer region)
    answer_logprobs = np.array(token_logprobs.astype(mx.float32))[-20:]
    return float(np.mean(answer_logprobs))


def run_baseline(model, tokenizer, questions, is_gemma=False):
    """Compute baseline log-prob with no layers skipped."""
    n_layers = len(model.model.layers)
    # Use skip_layer = -1 as "skip nothing" — need a small hack
    # Actually, just pass skip_layer = n_layers (out of range, nothing skipped)
    logprobs = []
    for q in questions:
        lp = logprob_with_skip(model, tokenizer, q["question"], q["gold"],
                               skip_layer=n_layers, is_gemma=is_gemma)
        logprobs.append(lp)
    return float(np.mean(logprobs))


def main():
    parser = argparse.ArgumentParser(description="Layer skip test — find critical layers")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--is_gemma", action="store_true",
                        help="Set if model is Gemma-2")
    parser.add_argument("--n_questions", type=int, default=10,
                        help="Number of GSM8K questions to test")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: layer_skip_{model}.json)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    n_layers = len(model.model.layers)
    is_gemma = args.is_gemma or "gemma" in args.model.lower()
    print(f"  {n_layers} layers, is_gemma={is_gemma}")

    questions = QUESTIONS[:args.n_questions]
    print(f"  Testing {len(questions)} questions\n")

    # Baseline (no skip)
    print("=== Baseline (no skip) ===")
    baseline = run_baseline(model, tokenizer, questions, is_gemma)
    print(f"  Baseline mean log-prob: {baseline:.4f}\n")

    # Skip each layer
    print(f"=== Skipping each of {n_layers} layers ===")
    results = []
    t0 = time.time()

    for layer_idx in range(n_layers):
        logprobs = []
        for q in questions:
            lp = logprob_with_skip(model, tokenizer, q["question"], q["gold"],
                                   skip_layer=layer_idx, is_gemma=is_gemma)
            logprobs.append(lp)

        mean_lp = float(np.mean(logprobs))
        delta = mean_lp - baseline
        results.append({
            "layer": layer_idx,
            "mean_logprob": mean_lp,
            "delta_from_baseline": delta,
        })

        elapsed = time.time() - t0
        rate = (layer_idx + 1) / elapsed
        eta = (n_layers - layer_idx - 1) / rate
        print(f"  Layer {layer_idx:2d}: logprob={mean_lp:.4f}  "
              f"delta={delta:+.4f}  "
              f"| {rate:.1f} layers/s | ETA {eta:.0f}s")

    # Sort by impact (most negative delta = most critical layer)
    sorted_results = sorted(results, key=lambda x: x["delta_from_baseline"])

    print(f"\n=== Most critical layers (biggest accuracy drop when skipped) ===")
    for i, entry in enumerate(sorted_results[:10]):
        pct = 100 * entry["layer"] / n_layers
        print(f"  {i+1:2d}. Layer {entry['layer']:2d} ({pct:.0f}% depth)  "
              f"delta={entry['delta_from_baseline']:+.4f}")

    print(f"\n=== Least critical layers (safest to skip) ===")
    for i, entry in enumerate(sorted_results[-5:]):
        pct = 100 * entry["layer"] / n_layers
        print(f"  Layer {entry['layer']:2d} ({pct:.0f}% depth)  "
              f"delta={entry['delta_from_baseline']:+.4f}")

    # Save
    model_short = args.model.split("/")[-1]
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "data", f"layer_skip_{model_short}.json"
    )
    output = {
        "model": args.model,
        "n_layers": n_layers,
        "is_gemma": is_gemma,
        "n_questions": len(questions),
        "baseline_logprob": baseline,
        "layer_results": results,
        "most_critical": [r["layer"] for r in sorted_results[:10]],
        "total_time_sec": round(time.time() - t0, 1),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
