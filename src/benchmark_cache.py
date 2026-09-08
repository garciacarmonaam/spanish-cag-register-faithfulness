from pathlib import Path
import statistics
import ollama


MODEL = "llama3.2:latest"
CONTEXT_PATH = Path("data/context.txt")
RUNS = 5

INSTRUCTION = (
    "Utiliza exclusivamente la siguiente información "
    "para responder a las preguntas posteriores.\n\n"
)

QUESTION = (
    "De acuerdo con la información proporcionada, "
    "¿qué se entiende por ilusión?"
)


def ms(ns):
    return ns / 1_000_000


def main():
    knowledge = CONTEXT_PATH.read_text(encoding="utf-8")
    knowledge_prompt = INSTRUCTION + knowledge

    print("=== CREANDO CONTEXTO BASE ===")

    prefill = ollama.generate(
        model=MODEL,
        prompt=knowledge_prompt,
        options={
            "temperature": 0,
            "num_predict": 1
        }
    )

    base_context = prefill["context"]

    print("Tokens del contexto:", len(base_context))

    without_context = []
    with_context = []

    for i in range(RUNS):

        # A: contexto completo
        response_a = ollama.generate(
            model=MODEL,
            prompt=knowledge_prompt + "\n\nPregunta:\n" + QUESTION,
            options={
                "temperature": 0,
                "num_predict": 1
            }
        )

        duration_a = ms(response_a["prompt_eval_duration"])
        without_context.append(duration_a)

        # B: contexto previamente obtenido
        response_b = ollama.generate(
            model=MODEL,
            prompt=QUESTION,
            context=base_context,
            options={
                "temperature": 0,
                "num_predict": 1
            }
        )

        duration_b = ms(response_b["prompt_eval_duration"])
        with_context.append(duration_b)

        print(
            f"Run {i + 1}: "
            f"sin context={duration_a:.2f} ms | "
            f"con context={duration_b:.2f} ms"
        )

    median_a = statistics.median(without_context)
    median_b = statistics.median(with_context)

    print("\n=== RESULTADOS ===")
    print(f"Mediana sin context: {median_a:.2f} ms")
    print(f"Mediana con context: {median_b:.2f} ms")

    if median_b > 0:
        print(f"Ratio: {median_a / median_b:.2f}x")


if __name__ == "__main__":
    main()
