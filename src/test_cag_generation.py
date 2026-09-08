from linecache import cache
from multiprocessing import context
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXT_PATH = Path("data/context.txt")

FORMAL_QUESTION = (
    "De acuerdo con la información proporcionada, "
    "¿qué se entiende por ilusión?"
)

COLLOQUIAL_QUESTION = (
    "Según lo que pone ahí, ¿qué es eso de tener ilusión?"
)


def clone_cache(cache):
    cache_data = []

    for key, value, sliding_window in cache:
        layer_data = [key.clone(), value.clone()]

        if sliding_window is not None:
            layer_data.append(sliding_window.clone())

        cache_data.append(tuple(layer_data))

    return DynamicCache(cache_data)


def generate_from_cache(model, tokenizer, base_cache, question):
    # Cada pregunta parte de una copia limpia del mismo conocimiento.
    cache = clone_cache(base_cache)

    # Solo tokenizamos la pregunta.
    # El contexto NO vuelve a introducirse porque ya está en el KV-cache.
    question_inputs = tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=False,
    )

    question_ids = question_inputs["input_ids"]

    past_length = cache.get_seq_length()
    question_length = question_ids.shape[1]

    attention_mask = torch.ones(
        (1, past_length + question_length),
        dtype=torch.long,
    )

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=question_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # generate() devuelve también los input_ids de la pregunta.
        # Nos quedamos únicamente con los tokens generados.
        generated_ids = outputs[0, question_length:]

        return tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )
    model.eval()

    context = CONTEXT_PATH.read_text(encoding="utf-8")

    prefix = (
        "Responde de forma breve utilizando únicamente el contexto.\n\n"
        "### CONTEXTO\n"
        f"{context}\n\n"
        "### FIN DEL CONTEXTO\n\n"
        "### CONSULTA\n"
    )

    prefix_inputs = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # PREFILL: procesamos el conocimiento una única vez.
    with torch.inference_mode():
        prefix_outputs = model(
            **prefix_inputs,
            use_cache=True,
        )

    base_cache = prefix_outputs.past_key_values

    print("KV base:", base_cache.get_seq_length(), "tokens")

    formal_response = generate_from_cache(
        model,
        tokenizer,
        base_cache,
        FORMAL_QUESTION,
    )

    colloquial_response = generate_from_cache(
        model,
        tokenizer,
        base_cache,
        COLLOQUIAL_QUESTION,
    )

    print("\n--- FORMAL ---")
    print("Pregunta:", FORMAL_QUESTION)
    print("Respuesta:", formal_response)

    print("\n--- COLOQUIAL ---")
    print("Pregunta:", COLLOQUIAL_QUESTION)
    print("Respuesta:", colloquial_response)

    print("\nKV base al terminar:", base_cache.get_seq_length(), "tokens")


if __name__ == "__main__":
    main()
