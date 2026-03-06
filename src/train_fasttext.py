import yaml
from gensim.models import FastText
from pathlib import Path
from typing import List

CONFIG_PATH = Path("config.yaml")
CORPUS_PATH = Path("data/processed/corpus.txt")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "fasttext.model"


def load_config(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_corpus(path: Path) -> List[List[str]]:
    with path.open(encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f if line.strip()]
    return sentences


def train_fasttext(sentences: List[List[str]], config: dict):
    model = FastText(
        vector_size=config.get("vector_size", 300),
        window=config.get("window", 5),
        min_count=config.get("min_count", 3),
        sg=config.get("sg", 1),
        min_n=config.get("min_n", 3),
        max_n=config.get("max_n", 6),
        workers=config.get("workers", 4),
        epochs=config.get("epochs", 10)
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
    return model


if __name__ == "__main__":
    config = load_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Processed corpus not found: {CORPUS_PATH}. Run src/preprocessing.py first.")

    sentences = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(sentences)} sentences from {CORPUS_PATH}")

    model = train_fasttext(sentences, config)

    model.save(str(MODEL_PATH))

    vocab_size = len(model.wv)
    (MODEL_DIR / "fasttext_vocab_size.txt").write_text(str(vocab_size), encoding="utf-8")

    print(f"FastText model saved to {MODEL_PATH}")
    print(f"Vocabulary size: {vocab_size}")
