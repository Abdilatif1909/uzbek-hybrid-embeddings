import re
from pathlib import Path
from typing import List


def clean_text(text: str) -> str:
    """
    Clean and normalize Uzbek text.
    - Lowercase
    - Remove URLs
    - Keep Uzbek Cyrillic + Latin characters and common punctuation used for sentence splitting
    - Remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    # allow Latin, Cyrillic, Uzbek specific letters and basic punctuation for splitting
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁўқғҳғӣӣʼ’\-\.,!\?\:\;\(\)\\n ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation heuristics.
    This is a lightweight splitter suitable for Uzbek corpora (no external dependencies).
    """
    # Normalize ellipsis
    text = text.replace('…', '...')
    # Split on sentence enders followed by space or linebreak
    parts = re.split(r'(?<=[\.!?])\s+', text)
    sentences = [p.strip() for p in parts if p and len(p.split()) >= 3]
    return sentences


def preprocess_all_raw(raw_dir: str = "data/raw", out_path: str = "data/processed/corpus.txt", out_big: str = "data/processed/corpus_big.txt", sample_limit: int = None):
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    out_big = Path(out_big)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    texts = []
    for p in sorted(raw_dir.glob("*.txt")):
        with p.open(encoding="utf-8") as f:
            content = f.read()
            if not content:
                continue
            content = clean_text(content)
            sents = split_sentences(content)
            texts.extend(sents)

    if sample_limit:
        sampled = texts[:sample_limit]
    else:
        sampled = texts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in sampled:
            f.write(s + "\n")

    # save full corpus as well
    with out_big.open("w", encoding="utf-8") as f:
        for s in texts:
            f.write(s + "\n")

    print(f"Processed sentences (saved sample): {len(sampled)} -> {out_path}")
    print(f"Processed sentences (full): {len(texts)} -> {out_big}")


if __name__ == "__main__":
    # default behavior: process all raw files and create a full and a sample corpus
    preprocess_all_raw(
        raw_dir="data/raw",
        out_path="data/processed/corpus.txt",
        out_big="data/processed/corpus_big.txt",
        sample_limit=None
    )
