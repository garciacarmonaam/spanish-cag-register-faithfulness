import json
from pathlib import Path


KNOWLEDGE_PATH = Path("data/knowledge.json")
OUTPUT_PATH = Path("data/context.txt")


def build_context() -> str:
    with KNOWLEDGE_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    blocks = []

    for item in data["knowledge"]:
        block = [
            f"Término: {item['term']}",
            f"Definición (acepción {item['sense']}): {item['definition']}"
        ]

        if "reference" in item:
            reference = item["reference"]
            block.append(
                f"Referencia: {reference['term']} "
                f"(acepción {reference['sense']}): "
                f"{reference['definition']}"
            )

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def main():
    context = build_context()
    OUTPUT_PATH.write_text(context, encoding="utf-8")
    print(f"Contexto generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()