import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "mistral": "ministral/Ministral-3b-instruct"
}

CONTEXT_PATH = Path("data/context.txt")
QUESTIONS_PATH = Path("data/questions.json")

def clone_cache(cache):
    cache_data = []

    for key, value, sliding_window in cache:
        layer_data = [key.clone(), value.clone()]

        if sliding_window is not None:
            layer_data.append(sliding_window.clone())

        cache_data.append(tuple(layer_data))

    return DynamicCache(cache_data)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS.keys(),
        help="Modelo que se utilizará en el experimento",
    )

    return parser.parse_args()

def generate_from_cache(model, tokenizer, base_cache, question, max_new_tokens=60):
    cache = clone_cache(base_cache)

    question_text = f"{question}\n\n### RESPUESTA\n"

    question_inputs = tokenizer(
        question_text,
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
        output_ids = model.generate(
            input_ids=question_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, question_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

def main():
    args = parse_args()
    model_id = MODELS[args.model]

    print(f"Modelo seleccionado: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
    )

    model.eval()
    print(f"Modelo cargado: {model.__class__.__name__}")

    # Cargar contexto
    context = CONTEXT_PATH.read_text(encoding="utf-8")

    # Cargar preguntas
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions_data = json.load(file)

    questions = questions_data["questions"]

    print(f"Preguntas cargadas: {len(questions)}")

    # Prefijo fijo que formará el KV-cache base
    prefix = (
        "Responde de forma breve utilizando únicamente el contexto.\n\n"
        "### CONTEXTO\n"
        f"{context}\n\n"
        "### FIN DEL CONTEXTO\n\n"
        "### CONSULTA\n"
    )

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Prefill del conocimiento: se realiza una sola vez
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
        )

    base_cache = outputs.past_key_values

    print(f"Tokens del prefijo: {inputs['input_ids'].shape[1]}")
    print(f"Tipo de cache: {type(base_cache).__name__}")
    print(f"Tokens almacenados en cache: {base_cache.get_seq_length()}")

    # Archivo de salida específico para cada modelo
    output_path = Path(
        f"results/{args.model}_generations.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_generations = len(questions) * 2
    current = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for question in questions:
            for register in ("formal", "colloquial"):
                current += 1

                question_text = question[register]

                print(
                    f"[{current:02d}/{total_generations}] "
                    f"{question['id']} | {register}"
                )

                response = generate_from_cache(
                    model=model,
                    tokenizer=tokenizer,
                    base_cache=base_cache,
                    question=question_text,
                    max_new_tokens=200,
                )

                result = {
                    "question_id": question["id"],
                    "task_type": question["type"],
                    "concept": question["target"],
                    "register": register,
                    "model": model_id,
                    "question": question_text,
                    "response": response,
                }

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    ) + "\n"
                )

                # Persistir cada generación inmediatamente
                output_file.flush()

    print(f"\nResultados guardados en: {output_path}")
    print(
        f"Cache base al finalizar: "
        f"{base_cache.get_seq_length()}"
    )

if __name__ == "__main__":
    main()