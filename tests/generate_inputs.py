import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin


def to_chat_text(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return text


def main():
    model_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: GenerationMixin = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="bfloat16"
    ).to("cuda")

    prompts = [
        "Give me a short introduction to large language model.",
        "What is the capital of France?",
    ]
    texts = [to_chat_text(tokenizer, prompt) for prompt in prompts]
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        outputs = model.model.forward(**model_inputs)

    os.makedirs("qwen3-0.6b", exist_ok=True)
    hidden_states = outputs.last_hidden_state[:, -1, :]  # [n_hidden_states, D]
    torch.save(hidden_states, "qwen3-0.6b/hidden_states.pt")
    weights = model.lm_head.weight.data  # [V, D]
    torch.save(weights, "qwen3-0.6b/weights.pt")


if __name__ == "__main__":
    main()
