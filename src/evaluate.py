from gensim.models import Word2Vec, FastText
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# === PATHS ===
W2V_PATH = "models/word2vec/word2vec.model"
FT_PATH = "models/fasttext/fasttext.model"

TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TABLE_PATH = TABLE_DIR / "embedding_comparison.csv"
FIG_PATH = FIG_DIR / "embedding_comparison.png"

# === LOAD MODELS ===
w2v = Word2Vec.load(W2V_PATH)
ft = FastText.load(FT_PATH)

# === TEST WORDS (morphology check) ===
test_words = [
    "suniy", "suniylik",
    "intellekt", "intellektual",
    "texnologiya", "texnologiyalar"
]

word_pairs = [
    ("suniy", "intellekt"),
    ("suniy", "texnologiya"),
    ("texnologiya", "intellekt")
]

def oov_coverage(model, words):
    known = sum(1 for w in words if w in model.wv)
    return known / len(words)

def avg_cosine_similarity(model, pairs):
    scores = []
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            scores.append(model.wv.similarity(w1, w2))
    return float(np.mean(scores)) if scores else 0.0

# === RESULTS ===
results = {
    "Model": ["Word2Vec", "FastText"],
    "Vocabulary Size": [
        len(w2v.wv),
        len(ft.wv)
    ],
    "OOV Coverage": [
        oov_coverage(w2v, test_words),
        oov_coverage(ft, test_words)
    ],
    "Avg Cosine Similarity": [
        avg_cosine_similarity(w2v, word_pairs),
        avg_cosine_similarity(ft, word_pairs)
    ]
}

df = pd.DataFrame(results)
df.to_csv(TABLE_PATH, index=False)

print("\n=== EMBEDDING COMPARISON TABLE ===")
print(df)

# === PLOT ===
metrics_df = df.set_index("Model")[["OOV Coverage", "Avg Cosine Similarity"]]

metrics_df.plot(kind="bar")
plt.title("Word2Vec vs FastText Embedding Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.show()
MBERT_EMB_PATH = "models/mbert/sentence_embeddings.npy"

mbert_embeddings = np.load(MBERT_EMB_PATH)

# tasodifiy 3 ta jumla o‘xshashligi
sim_matrix = cosine_similarity(mbert_embeddings[:3])

print("\n=== mBERT Sentence Similarity (sample) ===")
print(sim_matrix)

print(f"\nTable saved to: {TABLE_PATH}")
print(f"Figure saved to: {FIG_PATH}")
