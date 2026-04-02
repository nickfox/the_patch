#!/usr/bin/env python3
"""
MLX LatentMAS — Latent multi-agent reasoning on Apple Silicon.

Usage:
    python run.py --model Qwen/Qwen3-4B --task gsm8k --max_samples 5 --latent_steps 10
"""

import argparse
import time
import json

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

from latent_comm import compute_alignment, latent_forward, generate_with_cache
from prompts import build_prompt, build_prompt_hierarchical
from utils import extract_boxed_answer, normalize_answer, extract_gold


SEQUENTIAL_AGENTS = ["planner", "critic", "refiner"]       # judger decodes text
HIERARCHICAL_AGENTS = ["math_agent", "science_agent", "code_agent"]  # task_summarizer decodes


def load_gsm8k(split="test", max_samples=-1):
    """Load GSM8K dataset."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    items = []
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        items.append({"question": question, "solution": solution, "gold": gold})
        if max_samples > 0 and len(items) >= max_samples:
            break
    return items


def run_latent_mas(model, tokenizer, question: str, latent_steps: int,
                   W_a, target_norm, use_realign: bool = True,
                   max_tokens: int = 1024, temperature: float = 0.6,
                   prompt_mode: str = "sequential") -> str:
    """Run full LatentMAS pipeline on a single question.

    Sequential: Planner → Critic → Refiner (latent) → Judger (text)
    Hierarchical: Math → Science → Code (latent) → Summarizer (text)
    """
    # Fresh KV cache for this question
    cache = make_prompt_cache(model)

    if prompt_mode == "hierarchical":
        agents = HIERARCHICAL_AGENTS
        judger_role = "task_summarizer"
        prompt_fn = build_prompt_hierarchical
    else:
        agents = SEQUENTIAL_AGENTS
        judger_role = "judger"
        prompt_fn = build_prompt

    # Latent agents: each builds on the accumulated KV cache
    for role in agents:
        prompt = prompt_fn(tokenizer, role, question)
        cache = latent_forward(
            model, tokenizer, prompt, cache,
            latent_steps=latent_steps,
            W_a=W_a, target_norm=target_norm,
            use_realign=use_realign,
        )
        print(f"  [{role}] cache offset: {cache[0].offset}")

    # Final agent: generates text using accumulated latent context
    judger_prompt = prompt_fn(tokenizer, judger_role, question)
    output = generate_with_cache(
        model, tokenizer, judger_prompt, cache,
        max_tokens=max_tokens, temperature=temperature,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="MLX LatentMAS")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace model name")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential", "hierarchical"],
                        help="Agent architecture: sequential (planner/critic/refiner) "
                             "or hierarchical (math/science/code)")
    parser.add_argument("--realign", action="store_true",
                        help="Use full realignment matrix (vs norm-match only)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    print(f"Model loaded. Computing alignment matrix...")

    W_a, target_norm = compute_alignment(model)
    print(f"Alignment matrix ready. target_norm={target_norm.item():.4f}")

    print(f"Loading {args.task} dataset...")
    items = load_gsm8k(max_samples=args.max_samples)
    print(f"Loaded {len(items)} items. Starting evaluation.\n")

    correct = 0
    total = len(items)
    start_time = time.time()

    for i, item in enumerate(items):
        q_start = time.time()
        print(f"==================== Problem #{i+1}/{total} ====================")
        print(f"Q: {item['question'][:120]}...")

        output = run_latent_mas(
            model, tokenizer, item["question"],
            latent_steps=args.latent_steps,
            W_a=W_a, target_norm=target_norm,
            use_realign=args.realign,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            prompt_mode=args.prompt,
        )

        pred = normalize_answer(extract_boxed_answer(output))
        gold = item["gold"]
        ok = (pred == gold) if (pred and gold) else False
        if ok:
            correct += 1

        elapsed = time.time() - q_start
        print(f"[Judger output]\n{output}\n")
        print(f"Result: Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
        print()

    total_time = time.time() - start_time
    acc = correct / total if total > 0 else 0.0

    print(json.dumps({
        "method": "mlx_latent_mas",
        "model": args.model,
        "prompt": args.prompt,
        "latent_steps": args.latent_steps,
        "realign": args.realign,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "total_time_sec": round(total_time, 2),
        "time_per_sample_sec": round(total_time / total, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
