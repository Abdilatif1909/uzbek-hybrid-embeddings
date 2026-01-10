import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import numpy as np

MODEL_NAME = "bert-base-multilingual-cased"
CORPUS_PATH = "data/processed/corpus_big.txt"
SAVE_DIR = Path("models/mbert")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_sentences(sentences, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            output = model(**encoded)
            sent_emb = mean_pooling(output, encoded["attention_mask"])
            embeddings.append(sent_emb.cpu().numpy())
    return np.vstack(embeddings)

if __name__ == "__main__":
    with open(CORPUS_PATH, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # faqat bir qismini olamiz (tezlik uchun)
    sentences = sentences[:1000]

    print(f"Embedding {len(sentences)} sentences with mBERT...")

    sentence_embeddings = embed_sentences(sentences)

    np.save(SAVE_DIR / "sentence_embeddings.npy", sentence_embeddings)

    print("mBERT sentence embeddings saved.")
    print(f"Shape: {sentence_embeddings.shape}")
