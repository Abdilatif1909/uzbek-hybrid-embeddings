import yaml
from pathlib import Path
import numpy as np
import torch
# Compatibility shim: some transformers versions expect torch.utils._pytree.register_pytree_node
# while torch may expose _register_pytree_node. Create a safe wrapper if needed before importing transformers.
try:
    pytree = getattr(torch.utils, "_pytree", None)
    if pytree is not None and not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
        # create a wrapper that accepts extra kwargs (e.g., serialized_type_name) and forwards to the underlying function
        def _register_wrapper(*args, **kwargs):
            # transformers may pass serialized_type_name kwarg which older torch impl doesn't accept
            if "serialized_type_name" in kwargs:
                kwargs.pop("serialized_type_name")
            return getattr(pytree, "_register_pytree_node")(*args, **kwargs)
        setattr(pytree, "register_pytree_node", _register_wrapper)
except Exception:
    # best-effort shim; proceed even if it fails
    pass
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import csv

CONFIG_PATH = Path("config.yaml")
CORPUS_PATH = Path("data/processed/corpus.txt")
MBERT_DIR = Path("models/mbert")
MBERT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CSV = TABLE_DIR / "mbert_results.csv"
EMB_NPY = MBERT_DIR / "sentence_embeddings.npy"

MODEL_NAME = "bert-base-multilingual-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_sentences(sentences: List[str], batch_size: int = 16) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            output = model(**encoded)
            sent_emb = mean_pooling(output, encoded["attention_mask"])  # shape (batch, dim)
            embeddings.append(sent_emb.cpu().numpy())
    return np.vstack(embeddings)


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    sample_limit = config.get("sample_limit")
    batch_size = config.get("mbert_batch_size", 32)

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Processed corpus not found: {CORPUS_PATH}. Run src/preprocessing.py first.")

    with CORPUS_PATH.open(encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if sample_limit:
        sentences = sentences[:sample_limit]

    print(f"Embedding {len(sentences)} sentences with mBERT on {DEVICE}...")

    embeddings = embed_sentences(sentences, batch_size=batch_size)

    np.save(EMB_NPY, embeddings)
    print(f"Saved embeddings to {EMB_NPY} (shape={embeddings.shape})")

    # Compute cosine similarity for a sample of random pairs
    rng = np.random.RandomState(config.get("seed", 42))
    n_pairs = min(500, max(100, len(sentences)//10))
    idx = rng.randint(0, len(sentences), size=(n_pairs, 2))

    with RESULT_CSV.open("w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx1", "idx2", "sentence1", "sentence2", "cosine"])
        for i1, i2 in idx:
            v1 = embeddings[i1].reshape(1, -1)
            v2 = embeddings[i2].reshape(1, -1)
            cos = float(cosine_similarity(v1, v2)[0, 0])
            writer.writerow([i1, i2, sentences[i1], sentences[i2], cos])

    print(f"Saved mBERT pairwise similarities to: {RESULT_CSV}")
