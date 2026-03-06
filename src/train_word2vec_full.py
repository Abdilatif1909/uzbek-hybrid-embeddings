import yaml
from gensim.models import Word2Vec
from pathlib import Path
from typing import List

CONFIG_PATH = Path("config.yaml")
CORPUS_PATH = Path("data/processed/corpus_big.txt")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "word2vec_full.model"


def load_config(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_corpus(path: Path) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]


def train_and_save(sentences, config: dict):
    model = Word2Vec(
        sentences=sentences,
        vector_size=config.get("vector_size", 300),
        window=config.get("window", 5),
        min_count=1,
        workers=config.get("workers", 4),
        sg=config.get("sg", 1),
        epochs=config.get("epochs", 10),
        seed=config.get("seed", 42)
    )
    model.save(str(MODEL_PATH))
    # save vocab size
    vocab_size = len(model.wv)
    (MODEL_DIR / "word2vec_full_vocab_size.txt").write_text(str(vocab_size), encoding="utf-8")
    print(f"Saved Word2Vec full model to {MODEL_PATH} (vocab_size={vocab_size})")


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    sentences = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(sentences)} sentences from {CORPUS_PATH}")
    train_and_save(sentences, config)
