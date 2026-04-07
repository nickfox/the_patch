#!/usr/bin/env python3
"""
Cross-model latent communication test (simplified):
  Qwen3-4B (sender) -> Phi-4-mini (receiver)

Single forward pass per model. No subagents, no latent loop.
  1. Qwen reads the question -> extract post-norm hidden states
  2. Project to embedding space via barycentric ridge self-projector
  3. Cross-align into receiver's embedding space
  4. Inject into receiver as latent context
  5. Receiver generates the answer
"""

import argparse
import mlx.core as mx
import mlx_lm
import time
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from cross_align import (compute_cross_alignment, apply_cross_realignment,
                         find_shared_tokens)
from self_projector import BarycentricRidgeSelfProjector, get_embedding_matrix
from utils import extract_boxed_answer, normalize_answer


def _is_gemma(model):
    """Detect Gemma-2 (needs sqrt scaling + logit softcapping)."""
    return hasattr(model, "final_logit_softcapping")


def receiver_forward_with_embeddings(model, input_embeddings, cache):
    """Run receiver forward pass with pre-computed embeddings.

    Model-agnostic: auto-detects Gemma sqrt scaling and logit softcapping.
    For non-Gemma models (Phi-4, Qwen3, etc.), does a standard forward.
    """
    h = input_embeddings
    if _is_gemma(model):
        h = h * (model.model.args.hidden_size ** 0.5)

    if cache is None:
        cache = [None] * len(model.model.layers)

    mask = create_attention_mask(h, cache[0], return_array=True)

    for layer, c in zip(model.model.layers, cache):
        h = layer(h, mask, c)

    h = model.model.norm(h)

    if _is_gemma(model):
        out = model.model.embed_tokens.as_linear(h)
        out = mx.tanh(out / model.final_logit_softcapping)
        out = out * model.final_logit_softcapping
    elif model.args.tie_word_embeddings:
        out = model.model.embed_tokens.as_linear(h)
    else:
        out = model.lm_head(h)

    return out


def generate_with_cross_latents(receiver_model, receiver_tokenizer,
                                 projected_embeddings, question,
                                 max_tokens=2048, temperature=0.6):
    """Inject projected latent context into receiver and generate answer."""
    from mlx_lm.sample_utils import make_sampler

    cache = make_prompt_cache(receiver_model)

    # Step 1: Feed projected latent context
    logits = receiver_forward_with_embeddings(
        receiver_model, projected_embeddings, cache
    )
    mx.eval(logits, *[c.state for c in cache])
    print(f"    Latent prefill done, cache offset: {cache[0].offset}", flush=True)

    # Step 2: Feed solve prompt using the receiver's chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            "You have been given latent context about the following question. "
            "Use it to solve the problem step by step.\n\n"
            f"Question: {question}\n\n"
            "Put your final numerical answer inside \\boxed{}."
        )},
    ]
    prompt = receiver_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = receiver_tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = mx.array([tokens])

    logits = receiver_model(input_ids, cache=cache)
    mx.eval(logits, *[c.state for c in cache])
    print(f"    Prompt prefilled, cache offset: {cache[0].offset}", flush=True)

    # Step 3: Autoregressive generation
    sampler = make_sampler(temp=temperature, top_p=0.95)
    generated_tokens = []
    next_logits = logits[:, -1, :]
    eos_token = receiver_tokenizer.eos_token_id

    for _ in range(max_tokens):
        token = sampler(next_logits)
        token_id = token.item()
        if token_id == eos_token:
            break
        generated_tokens.append(token_id)
        next_input = mx.array([[token_id]])
        next_logits = receiver_model(next_input, cache=cache)
        next_logits = next_logits[:, -1, :]
        mx.eval(next_logits, *[c.state for c in cache])

    return receiver_tokenizer.decode(generated_tokens)


def load_gsm8k_test(max_samples=-1):
    """Load GSM8K test set questions with gold answers."""
    import re
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
        if max_samples > 0 and len(questions) >= max_samples:
            break
    return questions


def run_diagnostics(sender_model, sender_tok, receiver_model, receiver_tok,
                    projector, E_raw, W_ab, mean_a, mean_b, target_norm_b):
    """Run all three diagnostic metrics. No GSM8K eval."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    # --- Diagnostic 1: Projector Fidelity ---
    print("\n--- Diagnostic 1: Projector Fidelity ---")
    print("cos(project_fast(z), project_exact(z)) on held-out prompts")
    ds = load_dataset("gsm8k", "main", split="test")
    val_prompts = [item["question"].strip() for item in ds][:5]

    all_cos = []
    for i, prompt in enumerate(val_prompts):
        tokens = sender_tok.encode(prompt, add_special_tokens=True)
        input_ids = mx.array([tokens])
        cache = make_prompt_cache(sender_model)
        z = sender_model.model(input_ids, cache=cache)
        mx.eval(z)

        z_2d = z[0]  # [seq, D]
        y_fast = projector.project_fast(z_2d)
        y_exact = projector.project_exact(z_2d, E_raw, k=projector.k, tau=projector.tau)
        mx.eval(y_fast, y_exact)

        cos = mx.mean(mx.sum(y_fast * y_exact, axis=-1)).item()
        all_cos.append(cos)
        print(f"  Prompt {i + 1}: cos={cos:.4f} ({len(tokens)} tokens)")

    mean_fidelity = sum(all_cos) / len(all_cos)
    print(f"  Mean fidelity: {mean_fidelity:.4f} (target > 0.85)")

    # --- Diagnostic 2: Static W_AB Ceiling ---
    print("\n--- Diagnostic 2: Static W_AB Ceiling ---")
    print("cos(normalize(E_A @ W_AB), normalize(E_B)) on shared tokens")
    shared = find_shared_tokens(sender_tok, receiver_tok)
    n_eval = min(1000, len(shared))
    eval_shared = shared[:n_eval]

    ids_a = mx.array([[s[1] for s in eval_shared]])
    ids_b = mx.array([[s[2] for s in eval_shared]])
    E_a = sender_model.model.embed_tokens(ids_a)[0].astype(mx.float32)
    E_b = receiver_model.model.embed_tokens(ids_b)[0].astype(mx.float32)
    mx.eval(E_a, E_b)

    # Normalize E_a, project through W_AB, normalize result
    E_a_n = E_a / (mx.linalg.norm(E_a, axis=-1, keepdims=True) + 1e-8)
    E_b_hat = E_a_n @ W_ab
    E_b_hat_n = E_b_hat / (mx.linalg.norm(E_b_hat, axis=-1, keepdims=True) + 1e-8)
    E_b_n = E_b / (mx.linalg.norm(E_b, axis=-1, keepdims=True) + 1e-8)

    cos_static = mx.mean(mx.sum(E_b_hat_n * E_b_n, axis=-1)).item()
    print(f"  Static W_AB cosine ({n_eval} shared tokens): {cos_static:.4f}")

    # --- Diagnostic 3: Projected W_AB Ceiling ---
    print("\n--- Diagnostic 3: Projected W_AB Ceiling ---")
    print("Refit W_AB on project_exact(z_A) vectors, measure cos vs E_B")

    # Collect projected sender states and corresponding receiver embeddings
    # for shared tokens that appear in the calibration prompts
    cal_prompts = [item["question"].strip() for item in ds][:20]

    u_A_all = []
    e_B_all = []

    for prompt in cal_prompts:
        tokens = sender_tok.encode(prompt, add_special_tokens=True)
        input_ids = mx.array([tokens])
        cache = make_prompt_cache(sender_model)
        z = sender_model.model(input_ids, cache=cache)
        mx.eval(z)

        # Project sender hidden states to embedding space
        u_A = projector.project_exact(z[0], E_raw, k=projector.k, tau=projector.tau)
        mx.eval(u_A)

        # For each token position, get the corresponding receiver embedding
        # by looking up the sender token in the receiver's vocabulary
        sender_vocab = sender_tok.get_vocab()
        receiver_vocab = receiver_tok.get_vocab()
        inv_sender = {v: k for k, v in sender_vocab.items()}

        for pos_idx in range(len(tokens)):
            tok_id = tokens[pos_idx]
            tok_str = inv_sender.get(tok_id)
            if tok_str and tok_str in receiver_vocab:
                recv_id = receiver_vocab[tok_str]
                u_A_all.append(u_A[pos_idx])
                recv_emb = receiver_model.model.embed_tokens(
                    mx.array([[recv_id]])
                )[0, 0].astype(mx.float32)
                mx.eval(recv_emb)
                e_B_all.append(recv_emb)

    if len(u_A_all) < 50:
        print(f"  Only {len(u_A_all)} matched tokens — too few for reliable refit")
    else:
        U_A = mx.stack(u_A_all, axis=0)  # [N, D_sender]
        E_B_matched = mx.stack(e_B_all, axis=0)  # [N, D_receiver]
        mx.eval(U_A, E_B_matched)

        # Split into train (80%) and test (20%)
        N_total = U_A.shape[0]
        N_train = int(N_total * 0.8)
        U_train, U_test = U_A[:N_train], U_A[N_train:]
        E_train, E_test = E_B_matched[:N_train], E_B_matched[N_train:]

        # L2-normalize
        U_train_n = U_train / (mx.linalg.norm(U_train, axis=-1, keepdims=True) + 1e-8)
        U_test_n = U_test / (mx.linalg.norm(U_test, axis=-1, keepdims=True) + 1e-8)
        E_train_n = E_train / (mx.linalg.norm(E_train, axis=-1, keepdims=True) + 1e-8)
        E_test_n = E_test / (mx.linalg.norm(E_test, axis=-1, keepdims=True) + 1e-8)

        # Refit W_AB on projected vectors
        D_s = U_train_n.shape[-1]
        gram = U_train_n.T @ U_train_n + 1e-5 * mx.eye(D_s)
        rhs = U_train_n.T @ E_train_n
        mx.eval(gram, rhs)
        W_ab_new = mx.linalg.solve(gram, rhs, stream=mx.cpu)
        mx.eval(W_ab_new)

        # Evaluate on test set
        E_pred = U_test_n @ W_ab_new
        E_pred_n = E_pred / (mx.linalg.norm(E_pred, axis=-1, keepdims=True) + 1e-8)
        cos_refit = mx.mean(mx.sum(E_pred_n * E_test_n, axis=-1)).item()
        print(f"  Matched token pairs: {N_total} (train={N_train}, test={N_total - N_train})")
        print(f"  Refitted W_AB cosine on projected vectors: {cos_refit:.4f}")
        print(f"  (Compare to static W_AB ceiling: {cos_static:.4f})")
        if cos_refit > cos_static:
            print(f"  Improvement: +{cos_refit - cos_static:.4f} — sender mismatch was a bottleneck")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model latent communication: Qwen sender -> receiver"
    )
    parser.add_argument("--receiver", default="mlx-community/Phi-4-mini-instruct-8bit",
                        help="Receiver model")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--mode", choices=["fast", "exact"], default="fast",
                        help="Projection mode: fast (ridge) or exact (barycentric)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run diagnostics only (skip GSM8K eval)")
    parser.add_argument("--projector-path", default="data/qwen_self_projector.npz",
                        help="Path to saved projector")
    args = parser.parse_args()

    sender_name = "mlx-community/Qwen3-4B-8bit"
    receiver_name = args.receiver

    # Step 1: Load sender model
    print(f"=== Loading models (8-bit) ===", flush=True)
    print(f"Sender: {sender_name}", flush=True)
    sender_model, sender_tok = mlx_lm.load(sender_name)

    # Step 2: Load self-projector
    print(f"Loading self-projector from {args.projector_path}", flush=True)
    projector = BarycentricRidgeSelfProjector.load(args.projector_path)
    print(f"  Projector loaded (k={projector.k}, tau={projector.tau}, lam={projector.lam})")

    # Step 3: Compute E_raw if needed (exact mode or diagnostics)
    E_raw = None
    if args.mode == "exact" or args.diagnostics:
        print("Computing dequantized embedding matrix (needed for exact mode)...",
              flush=True)
        E_raw = get_embedding_matrix(sender_model)
        print(f"  E_raw shape: {E_raw.shape}")

    # Step 4: Load receiver model
    print(f"Receiver: {receiver_name}", flush=True)
    receiver_model, receiver_tok = mlx_lm.load(receiver_name)
    print()

    # Step 5: Compute cross-alignment
    print("=== Computing cross-alignment matrix ===", flush=True)
    t0 = time.time()
    alignment = compute_cross_alignment(sender_model, sender_tok,
                                         receiver_model, receiver_tok)
    W_ab = alignment["W_ab"]
    mean_a = alignment["mean_a"]
    mean_b = alignment["mean_b"]
    target_norm_b = alignment["target_norm_b"]
    print(f"Alignment computed in {time.time() - t0:.1f}s\n")

    # Diagnostics mode: print metrics and exit
    if args.diagnostics:
        run_diagnostics(sender_model, sender_tok, receiver_model, receiver_tok,
                        projector, E_raw, W_ab, mean_a, mean_b, target_norm_b)
        return

    # Step 6: Load test questions
    questions = load_gsm8k_test(max_samples=args.max_samples)
    print(f"Loaded {len(questions)} GSM8K test questions\n")

    # Step 7: Run
    correct = 0
    total_start = time.time()
    for i, item in enumerate(questions):
        question = item["question"]
        gold = item["gold"]
        print(f"===== Question #{i + 1} =====", flush=True)
        print(f"Q: {question[:100]}...", flush=True)
        print(f"Gold: {gold}", flush=True)

        t0 = time.time()

        # 1. Forward through sender — post-norm hidden states
        tokens = sender_tok.encode(question, add_special_tokens=True)
        input_ids = mx.array([tokens])
        cache = make_prompt_cache(sender_model)
        z = sender_model.model(input_ids, cache=cache)  # [1, seq, D]
        mx.eval(z)

        # 2. Project to embedding space via self-projector
        if args.mode == "exact":
            sender_projected = projector.project_exact(
                z[0], E_raw, k=projector.k, tau=projector.tau
            )
            sender_projected = sender_projected[None, :]  # [1, seq, D]
        else:
            sender_projected = projector.project_fast(z)  # [1, seq, D]
        mx.eval(sender_projected)
        print(f"  Sender projected ({args.mode}): {sender_projected.shape}")

        # 3. Cross-align into receiver's embedding space
        projected = apply_cross_realignment(
            sender_projected, W_ab, mean_a, mean_b, target_norm_b
        )
        mx.eval(projected)
        print(f"  Cross-projected: {projected.shape}")

        # 4. Inject into receiver and generate
        output = generate_with_cross_latents(
            receiver_model, receiver_tok,
            projected, question,
            max_tokens=args.max_tokens, temperature=args.temperature,
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

        print(f"\n[Output]\n{output[:500]}\n")
        print(f"Pred={pred} | Gold={gold} | OK={ok} | Time={elapsed:.1f}s")
        print(f"Running: {correct}/{i + 1} = {100 * correct / (i + 1):.0f}%\n")

    total_time = time.time() - total_start
    total = len(questions)
    acc = correct / total if total > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"FINAL: {correct}/{total} = {acc * 100:.0f}% accuracy")
    print(f"Sender: {sender_name}")
    print(f"Receiver: {receiver_name}")
    print(f"Mode: {args.mode}")
    print(f"Time: {total_time:.1f}s ({total_time / total:.1f}s/question)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
