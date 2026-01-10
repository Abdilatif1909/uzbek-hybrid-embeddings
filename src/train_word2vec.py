from gensim.models import Word2Vec
from pathlib import Path

CORPUS_PATH = "data/processed/corpus.txt"
MODEL_DIR = Path("models/word2vec")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f if line.strip()]
    return sentences

def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,          # 1 = Skip-gram, 0 = CBOW
        epochs=20
    )
    return model

if __name__ == "__main__":
    sentences = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(sentences)} sentences")

    model = train_word2vec(sentences)

    model_path = MODEL_DIR / "word2vec.model"
    model.save(str(model_path))

    print(f"Word2Vec model saved to {model_path}")

    # test
    if "suniy" in model.wv:
        print("Most similar to 'suniy':")
        for word, score in model.wv.most_similar("suniy"):
            print(word, score)
