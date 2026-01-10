import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean and normalize Uzbek text.
    - Lowercase
    - Remove URLs
    - Keep Uzbek Cyrillic + Latin characters
    - Remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁўқғҳʼ’ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_file(input_path: str, output_path: str):
    """
    Read raw text file, clean each line, and save processed corpus.
    Each line is treated as a single sentence.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = clean_text(line)
        if len(cleaned.split()) >= 3:   # juda qisqa gaplarni tashlaymiz
            cleaned_lines.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")

    print(f"Processed {len(cleaned_lines)} lines")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    # 🔴 ASOSIY KORPUS (katta)
    preprocess_file(
        input_path="data/raw/texts_big.txt",
        output_path="data/processed/corpus_big.txt"
    )

    # 🟡 AGAR KICHIK TEST KORPUS KERAK BO‘LSA (ixtiyoriy)
    # preprocess_file(
    #     input_path="data/raw/texts.txt",
    #     output_path="data/processed/corpus.txt"
    # )
