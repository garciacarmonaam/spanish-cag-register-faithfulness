from pathlib import Path

import ollama


MODEL = "llama3.2:latest"
CONTEXT_PATH = Path("data/context.txt")


def main():
    knowledge = CONTEXT_PATH.read_text(encoding="utf-8")

    # 1. Procesamos únicamente el conocimiento.
    print("=== PREFILL DEL CONOCIMIENTO ===")

    prefill = ollama.generate(
        model=MODEL,
        prompt=(
            "Utiliza exclusivamente la siguiente información "
            "para responder a las preguntas posteriores.\n\n"
            f"{knowledge}"
        ),
        options={
            "temperature": 0,
            "num_predict": 1
        }
    )

    cached_context = prefill.get("context")

    print(f"Tokens procesados: {prefill.get('prompt_eval_count')}")
    print(
        "Contexto reutilizable recibido:",
        cached_context is not None
    )

    if cached_context is None:
        print("Ollama no ha devuelto el campo 'context'.")
        return

    print(f"Longitud del context: {len(cached_context)}")

    # 2. Pregunta formal partiendo del estado común.
    print("\n=== FORMAL ===")

    formal = ollama.generate(
        model=MODEL,
        prompt=(
            "De acuerdo con la información proporcionada, "
            "¿qué se entiende por ilusión?"
        ),
        context=cached_context,
        options={
            "temperature": 0
        }
    )

    print(formal["response"])
    print(
        "prompt_eval_count:",
        formal.get("prompt_eval_count")
    )

    # 3. Pregunta coloquial partiendo DEL MISMO estado.
    print("\n=== COLOQUIAL ===")

    colloquial = ollama.generate(
        model=MODEL,
        prompt=(
            "Según lo que pone ahí, "
            "¿qué es eso de tener ilusión?"
        ),
        context=cached_context,
        options={
            "temperature": 0
        }
    )

    print(colloquial["response"])
    print(
        "prompt_eval_count:",
        colloquial.get("prompt_eval_count")
    )


if __name__ == "__main__":
    main()
