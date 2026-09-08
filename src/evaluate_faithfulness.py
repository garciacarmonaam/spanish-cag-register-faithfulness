import argparse
import asyncio
import json
from pathlib import Path

from langchain_ollama import ChatOllama
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness


CONTEXT_FILE = Path("data/context.txt")
JUDGE_MODEL = "granite3.2:latest"

MODELS = ["qwen", "smollm", "gemma", "mistral", "phi"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Modelo generador cuyas respuestas se evaluarán",
    )
    args = parser.parse_args()

    model = args.model

    input_file = Path(f"results/{model}_generations.jsonl")
    output_file = Path(f"results/{model}_faithfulness.jsonl")

    context = CONTEXT_FILE.read_text(encoding="utf-8")

    with input_file.open("r", encoding="utf-8") as f:
        generations = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    judge = ChatOllama(
        model=JUDGE_MODEL,
        temperature=0,
    )

    metric = Faithfulness(
        llm=LangchainLLMWrapper(judge)
    )

    print(f"Modelo evaluado: {model}")
    print(f"Generaciones cargadas: {len(generations)}")
    print(f"Juez: {JUDGE_MODEL}")
    print()

    with output_file.open("w", encoding="utf-8") as output_file_handle:
        for i, item in enumerate(generations, start=1):
            sample = SingleTurnSample(
                user_input=item["question"],
                response=item["response"],
                retrieved_contexts=[context],
            )

            try:
                score = await metric.single_turn_ascore(sample)
            except Exception as exc:
                print(
                    f"[{i}/{len(generations)}] "
                    f"{item['question_id']} / {item['register']} -> ERROR: {exc}"
                )
                score = None

            result = {
                "question_id": item["question_id"],
                "task_type": item["task_type"],
                "concept": item["concept"],
                "register": item["register"],
                "model": item["model"],
                "question": item["question"],
                "response": item["response"],
                "faithfulness": score,
                "judge_model": JUDGE_MODEL,
            }

            output_file_handle.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file_handle.flush()

            print(
                f"[{i}/{len(generations)}] "
                f"{item['question_id']} / {item['register']} "
                f"-> Faithfulness: {score}"
            )

    print()
    print(f"Resultados guardados en: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())