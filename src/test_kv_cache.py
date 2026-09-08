from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT_PATH = Path("data/context.txt")


def clone_cache(cache):
    cache_data = []

    for key, value, sliding_window in cache:
        layer_data = [key.clone(), value.clone()]

        if sliding_window is not None:
            layer_data.append(sliding_window.clone())

        cache_data.append(tuple(layer_data))

    return DynamicCache(cache_data)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )

    model.eval()

    context = CONTEXT_PATH.read_text(encoding="utf-8")

    prefix = (
        "Utiliza exclusivamente la siguiente información para responder "
        "a las preguntas.\n\n"
        f"{context}\n\n"
    )

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
    )

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
        )

    base_cache = outputs.past_key_values

    formal_cache = clone_cache(base_cache)
    colloquial_cache = clone_cache(base_cache)

    print("Cache base:", base_cache.get_seq_length())
    print("Cache formal:", formal_cache.get_seq_length())
    print("Cache coloquial:", colloquial_cache.get_seq_length())

    print(
        "Formal comparte tensor con base:",
        formal_cache.layers[0].keys.data_ptr()
        == base_cache.layers[0].keys.data_ptr()
    )

    print(
        "Coloquial comparte tensor con base:",
        colloquial_cache.layers[0].keys.data_ptr()
        == base_cache.layers[0].keys.data_ptr()
    )


if __name__ == "__main__":
    main()