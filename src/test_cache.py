from pathlib import Path

import ollama


MODEL = "llama3.2:latest"
CONTEXT_PATH = Path("data/context.txt")


def print_metrics(response):
    print(f"prompt_eval_count: {response.get('prompt_eval_count')}")
    print(
        f"prompt_eval_cached_count: "
        f"{response.get('prompt_eval_cached_count')}"
    )
    print(f"prompt_eval_duration: {response.get('prompt_eval_duration')}")


def main():
    context = CONTEXT_PATH.read_text(encoding="utf-8")

    question_1 = (
        "De acuerdo con la información proporcionada, "
        "¿qué se entiende por ilusión?"
    )

    question_2 = (
        "Según lo que pone ahí, ¿qué es eso de tener ilusión?"
    )

    prompt_1 = f"""Utiliza exclusivamente la siguiente información para responder.

{context}

Pregunta:
{question_1}
"""

    prompt_2 = f"""Utiliza exclusivamente la siguiente información para responder.

{context}

Pregunta:
{question_2}
"""

    print("=== PRIMERA EJECUCIÓN ===")

    response_1 = ollama.generate(
        model=MODEL,
        prompt=prompt_1,
        options={
            "temperature": 0
        }
    )

    print(response_1["response"])
    print_metrics(response_1)

    print("\n=== SEGUNDA EJECUCIÓN ===")

    response_2 = ollama.generate(
        model=MODEL,
        prompt=prompt_2,
        options={
            "temperature": 0
        }
    )

    print(response_2["response"])
    print_metrics(response_2)


if __name__ == "__main__":
    main()