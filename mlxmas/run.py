#!/usr/bin/env python3
"""
MLX LatentMAS — Same-model latent communication on Apple Silicon.

Sender reads the question → single forward pass → KV cache
Receiver gets ONLY the latent context (no question text) → generates answer

Usage:
    python run.py --model mlx-community/Qwen3-4B-8bit --max_samples 5
"""

import argparse
import time
import json

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from utils import extract_boxed_answer, normalize_answer, extract_gold


def load_gsm8k(split="test", max_samples=-1):
    """Load GSM8K dataset, shuffled."""
    import random
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    items = []
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        items.append({"question": question, "solution": solution, "gold": gold})
    random.shuffle(items)
    if max_samples > 0:
        items = items[:max_samples]
    return items


def build_sender_prompt(tokenizer, question: str) -> str:
    """Sender prompt: reads the question and reasons about it."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f"Think carefully about the following question step by step.\n\n"
            f"Question: {question}\n\n"
            f"Reason through this problem thoroughly."
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_receiver_prompt(tokenizer, no_think=False) -> str:
    """Receiver prompt: NO question text, only instruction to use latent context."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            "Using the reasoning context provided, solve the problem step by step. "
            "Put your final numerical answer inside \\boxed{}."
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if no_think:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def run_latent_comm(model, tokenizer, question: str,
                    max_tokens: int = 2048, temperature: float = 0.6,
                    no_think: bool = False) -> str:
    """Sender reads question → KV cache → receiver generates from latents only."""
    cache = make_prompt_cache(model)

    # Sender: single forward pass, populates KV cache
    sender_prompt = build_sender_prompt(tokenizer, question)
    tokens = tokenizer.encode(sender_prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])
    model.model(input_ids, cache=cache)
    mx.eval(*[c.state for c in cache])
    print(f"  [sender] cache offset: {cache[0].offset}", flush=True)

    # Receiver: generates answer from latent context only (no question)
    receiver_prompt = build_receiver_prompt(tokenizer, no_think=no_think)
    tokens = tokenizer.encode(receiver_prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])
    logits = model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])

    # Autoregressive generation
    sampler = make_sampler(temp=temperature, top_p=0.95)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = tokenizer.eos_token_id

    for i in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1} tokens", flush=True)
        next_input = mx.array([[token_id]])
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return tokenizer.decode(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description="MLX LatentMAS — sender→receiver")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-8bit",
                        help="HuggingFace model name")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--no-think", action="store_true",
                        help="Skip Qwen's <think> block on the receiver")
    args = parser.parse_args()

    print(f"Loading model: {args.model}", flush=True)
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Model loaded.\n", flush=True)

    print(f"Loading {args.task} dataset...", flush=True)
    items = load_gsm8k(max_samples=args.max_samples)
    print(f"Loaded {len(items)} items. Starting evaluation.\n", flush=True)

    correct = 0
    total = len(items)
    start_time = time.time()

    for i, item in enumerate(items):
        q_start = time.time()
        print(f"==================== Problem #{i+1}/{total} ====================", flush=True)
        print(f"Q: {item['question'][:120]}...", flush=True)

        output = run_latent_comm(
            model, tokenizer, item["question"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            no_think=args.no_think,
        )

        pred = normalize_answer(extract_boxed_answer(output))
        gold = item["gold"]
        ok = (pred == gold) if (pred and gold) else False
        if ok:
            correct += 1

        elapsed = time.time() - q_start
        print(f"[Receiver output]\n{output}\n")
        print(f"Result: Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
        print(f"Running: {correct}/{i+1} = {100*correct/(i+1):.0f}%")
        print()

    total_time = time.time() - start_time
    acc = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"FINAL: {correct}/{total} = {acc*100:.0f}% accuracy")
    print(f"{'='*60}")

    print(json.dumps({
        "method": "latent_comm",
        "model": args.model,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "total_time_sec": round(total_time, 2),
        "time_per_sample_sec": round(total_time / total, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
