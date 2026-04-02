"""Prompt builders for LatentMAS agents — Qwen3 format."""


def build_prompt(tokenizer, role: str, question: str, task: str = "gsm8k") -> str:
    """Build a chat prompt for the given agent role.

    Returns the full rendered prompt string (not tokenized).
    """
    system_msg = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    if role == "planner":
        user_msg = (
            f"You are a Planner Agent. Given an input question, design a clear, "
            f"step-by-step plan for how to solve the question.\n\n"
            f"Question: {question}\n\n"
            f"Your outlined plan should be concise with a few bulletpoints for each step. "
            f"Do not produce the final answer.\n"
            f"Now output your plan to solve the question below:"
        )
    elif role == "critic":
        user_msg = (
            f"Question: {question}\n\n"
            f"You are a Critic Agent to evaluate the correctness of the input plan "
            f"for the given question and provide helpful feedback for improving the plan.\n"
            f"The plan information is provided in latent KV representation format. "
            f"Review the plan and question and output:\n"
            f"(1) original plan contents\n"
            f"(2) constructive feedback on the original plan.\n\n"
            f"Format your response as follows:\n"
            f"Original Plan: [Copy the provided Planner Agent's plan here]\n"
            f"Feedback: [Your detailed feedback to improve the plan here]\n\n"
            f"Now, output your response below:"
        )
    elif role == "refiner":
        user_msg = (
            f"Question: {question}\n\n"
            f"You are a Refiner Agent to provide a refined step-by-step plan for solving "
            f"the given question.\nYou are provided with:\n"
            f"(1) latent-format information: a previous plan with feedback\n"
            f"(2) text-format information: the input question you need to solve.\n\n"
            f"Based on the input, write a refined and improved plan to solve the question. "
            f"Make sure your output plan is correct and concise.\n\n"
            f"Now, output your refined plan below:"
        )
    elif role == "judger":
        user_msg = (
            f"Target Question: {question}\n\n"
            f"You are a helpful assistant. You are provided with latent information "
            f"for reference and a target question to solve.\n\n"
            f"The latent information might contain irrelevant contents. Ignore it if "
            f"it is not helpful for solving the target question.\n\n"
            f"You must reason step-by-step to solve the provided Target Question "
            f"without outputting other irrelevant information.\n\n"
            f"Now, reason step by step and output the final answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompt_hierarchical(tokenizer, role: str, question: str,
                               task: str = "gsm8k") -> str:
    """Build hierarchical MAS prompt — specialist agents working independently.

    Roles: math_agent, science_agent, code_agent, task_summarizer
    """
    system_msg = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    if role == "math_agent":
        user_msg = (
            f"You are a math agent. Given the input question, reason step-by-step "
            f"and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n\n"
            f"Input Question: {question}\n\nYour response:"
        )
    elif role == "science_agent":
        user_msg = (
            f"You are a science agent. Given the input question, reason step-by-step "
            f"and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n\n"
            f"Input Question: {question}\n\nYour response:"
        )
    elif role == "code_agent":
        user_msg = (
            f"You are a code agent. Given the input question, reason step-by-step "
            f"and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n\n"
            f"Input Question: {question}\n\nYour response:"
        )
    elif role == "task_summarizer":
        user_msg = (
            f"You are a task summarizer. Given the input question and responses from "
            f"previous agents as reference, reason step-by-step and put the final "
            f"answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n\n"
            f"Input Question: {question}\n\nYour response:"
        )
    else:
        raise ValueError(f"Unknown hierarchical role: {role}")

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
