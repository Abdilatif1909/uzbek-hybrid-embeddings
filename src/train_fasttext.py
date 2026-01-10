from gensim.models import FastText
from pathlib import Path

CORPUS_PATH = "data/processed/corpus.txt"
MODEL_DIR = Path("models/fasttext")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f if line.strip()]
    return sentences

if __name__ == "__main__":
    sentences = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(sentences)} sentences")

    model = FastText(
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,      # skip-gram
        epochs=20
    )

    # MUHIM QADAM
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=model.epochs
    )

    model_path = MODEL_DIR / "fasttext.model"
    model.save(str(model_path))

    print(f"FastText model saved to {model_path}")

    word = "suniy"
    if word in model.wv:
        print(f"\nMost similar to '{word}':")
        for w, s in model.wv.most_similar(word):
            print(w, s)
