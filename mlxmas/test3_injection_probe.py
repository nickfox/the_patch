#!/usr/bin/env python3
"""
Test 3: Injection Probe — Does the projected state actually help the receiver?

For the top N layer pairs (from CKA heatmap or manual selection):
1. Extract Qwen hidden state at sender layer s
2. Compute Procrustes alignment on calibration data
3. Project Qwen state into Gemma's space at layer m
4. Inject into Gemma at layer m
5. Measure Gemma's log-probability of the correct answer tokens

The winning pair is the one where Gemma gets the highest confidence
in the correct answer after receiving Qwen's projected reasoning state.

Usage:
    python -m mlxmas.test3_injection_probe
    python -m mlxmas.test3_injection_probe --pairs "31,4;31,8;28,6"
    python -m mlxmas.test3_injection_probe --from_cka cka_heatmap.json --top_k 10
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


def load_gsm8k_prompts(n=500):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    return [item["question"].strip() for item in ds][:n]


EVAL_QUESTIONS = [
    {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "gold": "18"},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "gold": "3"},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "gold": "70000"},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "gold": "540"},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?", "gold": "20"},
]


def extract_hidden_at_layer(model, tokenizer, prompt, layer_idx, is_gemma=False):
    """Extract final-token hidden state at a specific layer."""
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array([tokens])

    cache = make_prompt_cache(model)
    h = model.model.embed_tokens(input_ids)
    if is_gemma:
        h = h * (model.model.args.hidden_size ** 0.5)

    mask = create_attention_mask(h, cache[0], return_array=True)

    for i, (layer, c) in enumerate(zip(model.model.layers, cache)):
        h = layer(h, mask, c)
        if i == layer_idx:
            state = h[0, -1, :].astype(mx.float32)
            mx.eval(state)
            result = np.array(state)
            # Continue forward pass to completion (for memory cleanup)
            for j, (layer2, c2) in enumerate(
                zip(model.model.layers[i+1:], cache[i+1:])):
                h = layer2(h, mask, c2)
            mx.eval(h)
            return result

    raise ValueError(f"Layer {layer_idx} not reached")


def compute_procrustes_pair(H_a, H_b):
    """Compute semi-orthogonal Procrustes: H_a [N, D_a] -> H_b [N, D_b].
    Returns W_ortho, mean_a, mean_b, target_norm_b, train_cosine.
    """
    mean_a = H_a.mean(axis=0, keepdims=True)
    mean_b = H_b.mean(axis=0, keepdims=True)
    Ha_c = H_a - mean_a
    Hb_c = H_b - mean_b

    C = Ha_c.T @ Hb_c
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    D_b = H_b.shape[1]
    W_ortho = U[:, :D_b] @ Vt[:D_b, :]

    # Calibration cosine
    Hb_hat = Ha_c @ W_ortho
    cos = np.mean(
        np.sum(Hb_hat * Hb_c, axis=1) /
        (np.linalg.norm(Hb_hat, axis=1) * np.linalg.norm(Hb_c, axis=1) + 1e-8)
    )
    target_norm_b = np.mean(np.linalg.norm(H_b, axis=1))

    return W_ortho, mean_a, mean_b, target_norm_b, float(cos)


def project_state(h, W_ortho, mean_a, mean_b, target_norm_b):
    """Project a single sender state to receiver space via Procrustes."""
    h = h.reshape(1, -1) - mean_a
    projected = h @ W_ortho + mean_b
    norm = np.linalg.norm(projected)
    if norm > 1e-6:
        projected = projected * (target_norm_b / norm)
    return projected.reshape(-1)

def inject_and_measure_logprob(receiver_model, receiver_tok, projected_state,
                                inject_layer, question, gold_answer):
    """Inject projected state at a specific Gemma layer, then measure
    log-probability of the gold answer.

    1. Feed the projected state through Gemma layers [inject_layer..25]
    2. Then feed the question + answer tokens through the full model
       with the KV cache from the injection
    3. Measure log-prob of the answer tokens
    """
    D = receiver_model.model.args.hidden_size

    # Projected state as [1, 1, D] tensor
    h_inject = mx.array(projected_state.reshape(1, 1, D))

    # Create cache for all layers
    cache = make_prompt_cache(receiver_model)

    # Only apply embedding scaling if injecting at layer 0
    if inject_layer == 0:
        h_inject = h_inject * (D ** 0.5)

    # Run through layers [inject_layer .. end]
    layers = receiver_model.model.layers[inject_layer:]
    cache_slice = cache[inject_layer:]
    mask = create_attention_mask(h_inject, cache_slice[0], return_array=True)

    h = h_inject
    for layer, c in zip(layers, cache_slice):
        h = layer(h, mask, c)
    # Only eval cache entries that were actually written to
    mx.eval(h, *[c.state for c in cache_slice if c.keys is not None])

    # Now feed question + answer as normal tokens
    # We CANNOT use receiver_model() because cache offsets are inconsistent
    # (layers 0..inject_layer-1 have offset=0, layers inject_layer..25 have offset=1)
    # Instead, do the full forward pass manually with per-layer mask handling.
    prompt = (f"<start_of_turn>user\nSolve: {question}<end_of_turn>\n"
              f"<start_of_turn>model\nThe answer is {gold_answer}.")
    tokens = receiver_tok.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    # Embed the question tokens
    h = receiver_model.model.embed_tokens(input_ids)
    h = h * (D ** 0.5)

    # Process each layer with its own mask (based on its own cache state)
    for i, (layer, c) in enumerate(zip(receiver_model.model.layers, cache)):
        layer_mask = create_attention_mask(h, c, return_array=True)
        h = layer(h, layer_mask, c)

    h = receiver_model.model.norm(h)
    logits = receiver_model.model.embed_tokens.as_linear(h)
    logits = mx.tanh(logits / receiver_model.final_logit_softcapping)
    logits = logits * receiver_model.final_logit_softcapping
    mx.eval(logits)

    # Log-prob of answer tokens (last 20 tokens)
    log_probs = nn.log_softmax(logits[0, :-1, :], axis=-1)
    targets = mx.array(tokens[1:])
    token_logprobs = log_probs[mx.arange(len(targets)), targets]
    mx.eval(token_logprobs)

    answer_lps = np.array(token_logprobs.astype(mx.float32))[-20:]
    return float(np.mean(answer_lps))


def main():
    parser = argparse.ArgumentParser(description="Injection probe: test cross-model transfer")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated s,m pairs e.g. '31,4;31,8;28,6'")
    parser.add_argument("--from_cka", type=str, default=None,
                        help="Load top pairs from CKA heatmap JSON")
    parser.add_argument("--top_k", type=int, default=10,
                        help="How many top CKA pairs to test (with --from_cka)")
    parser.add_argument("--n_calibration", type=int, default=500,
                        help="Number of calibration prompts for Procrustes")
    parser.add_argument("--sender", default="mlx-community/Qwen3-4B-8bit")
    parser.add_argument("--receiver", default="mlx-community/gemma-2-2b-it-8bit")
    parser.add_argument("--output", default="injection_probe.json")
    args = parser.parse_args()

    # Determine which layer pairs to test
    if args.pairs:
        pairs = []
        for p in args.pairs.split(";"):
            s, m = p.split(",")
            pairs.append((int(s), int(m)))
    elif args.from_cka:
        with open(args.from_cka) as f:
            cka_data = json.load(f)
        pairs = [(p["sender_layer"], p["receiver_layer"])
                 for p in cka_data["top_pairs"][:args.top_k]]
    else:
        # Default grid based on literature
        pairs = [(28, 4), (28, 8), (28, 12),
                 (31, 4), (31, 8), (31, 12),
                 (34, 4), (34, 8), (34, 12),
                 (35, 4), (35, 8), (35, 12)]

    print(f"Testing {len(pairs)} layer pairs: {pairs}\n")

    total_start = time.time()

    # Load models
    print(f"Loading sender: {args.sender}")
    sender_model, sender_tok = mlx_lm.load(args.sender)
    print(f"Loading receiver: {args.receiver}")
    receiver_model, receiver_tok = mlx_lm.load(args.receiver)

    # Get unique layers needed
    sender_layers = sorted(set(s for s, m in pairs))
    receiver_layers = sorted(set(m for s, m in pairs))
    print(f"Sender layers needed: {sender_layers}")
    print(f"Receiver layers needed: {receiver_layers}")

    # Load calibration prompts
    print(f"\nLoading {args.n_calibration} calibration prompts...")
    cal_prompts = load_gsm8k_prompts(args.n_calibration)

    # Collect calibration states at each needed layer
    print(f"\n=== Collecting sender calibration states ===")
    sender_cal = {}
    for sl in sender_layers:
        print(f"  Sender layer {sl}...")
        states = []
        t0 = time.time()
        for i, p in enumerate(cal_prompts):
            h = extract_hidden_at_layer(sender_model, sender_tok, p, sl, is_gemma=False)
            states.append(h)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(cal_prompts)} ({rate:.1f}/s)")
        sender_cal[sl] = np.stack(states, axis=0)
        print(f"    Done: {sender_cal[sl].shape}")

    print(f"\n=== Collecting receiver calibration states ===")
    receiver_cal = {}
    for rl in receiver_layers:
        print(f"  Receiver layer {rl}...")
        states = []
        t0 = time.time()
        for i, p in enumerate(cal_prompts):
            h = extract_hidden_at_layer(receiver_model, receiver_tok, p, rl, is_gemma=True)
            states.append(h)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(cal_prompts)} ({rate:.1f}/s)")
        receiver_cal[rl] = np.stack(states, axis=0)
        print(f"    Done: {receiver_cal[rl].shape}")

    # For each pair: compute Procrustes, then injection probe
    print(f"\n=== Running injection probes ===")
    results = []

    for s, m in pairs:
        print(f"\n  Pair: Qwen L{s} → Gemma L{m}")

        # Procrustes alignment
        W, mean_a, mean_b, tnorm, cal_cos = compute_procrustes_pair(
            sender_cal[s], receiver_cal[m]
        )
        print(f"    Procrustes calibration cosine: {cal_cos:.4f}")

        # Test on each eval question
        logprobs = []
        for qi, q in enumerate(EVAL_QUESTIONS):
            # Extract sender state for this question
            h_sender = extract_hidden_at_layer(
                sender_model, sender_tok, q["question"], s, is_gemma=False
            )
            # Project to receiver space
            h_projected = project_state(h_sender, W, mean_a, mean_b, tnorm)
            # Inject and measure
            lp = inject_and_measure_logprob(
                receiver_model, receiver_tok, h_projected,
                m, q["question"], q["gold"]
            )
            logprobs.append(lp)
            print(f"      Q{qi+1}: logprob={lp:.4f}")

        mean_lp = float(np.mean(logprobs))
        entry = {
            "sender_layer": s,
            "receiver_layer": m,
            "calibration_cosine": cal_cos,
            "mean_logprob": mean_lp,
            "per_question": logprobs,
        }
        results.append(entry)
        print(f"    → Mean logprob: {mean_lp:.4f}")

    # Sort by mean logprob (highest = best)
    results.sort(key=lambda x: x["mean_logprob"], reverse=True)

    print(f"\n{'='*60}")
    print(f"=== RESULTS (ranked by receiver confidence) ===")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  {i+1:2d}. Qwen L{r['sender_layer']:2d} → Gemma L{r['receiver_layer']:2d}  "
              f"logprob={r['mean_logprob']:.4f}  "
              f"cal_cos={r['calibration_cosine']:.4f}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "data", args.output)
    output = {
        "sender": args.sender,
        "receiver": args.receiver,
        "n_calibration": len(cal_prompts),
        "n_eval_questions": len(EVAL_QUESTIONS),
        "pairs_tested": len(results),
        "results": results,
        "best_pair": {
            "sender_layer": results[0]["sender_layer"],
            "receiver_layer": results[0]["receiver_layer"],
            "mean_logprob": results[0]["mean_logprob"],
            "calibration_cosine": results[0]["calibration_cosine"],
        },
        "total_time_sec": round(time.time() - total_start, 1),
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    total = time.time() - total_start
    print(f"Total time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
