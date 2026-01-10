# Uzbek Hybrid Embeddings

This repository presents a reproducible experimental framework for building and evaluating
hybrid embedding models for the Uzbek language. The study focuses on comparing classical
static embeddings with modern contextual representations in a low-resource language setting.

The repository is designed to support academic research and Scopus-indexed publications
in the fields of Natural Language Processing (NLP) and Artificial Intelligence.

---

## 📌 Objectives

- To investigate the effectiveness of different embedding approaches for the Uzbek language
- To compare static, subword-based, and contextual embeddings
- To analyze the impact of corpus size and morphology on embedding quality
- To provide a fully reproducible experimental pipeline

---

## 🧠 Models Included

- **Word2Vec** – static word embeddings
- **FastText** – subword-aware embeddings suitable for morphologically rich languages
- **mBERT** – multilingual contextual sentence embeddings

---

## 🏗️ System Architecture

The overall architecture of the proposed system is illustrated as a modular NLP pipeline:

Raw Uzbek Text Corpus
│
▼
Preprocessing Module
(text cleaning, normalization)
│
▼
Processed Corpus
│
├──► Word2Vec Training
│ │
│ ▼
│ Static Word Embeddings
│
├──► FastText Training
│ │
│ ▼
│ Subword-aware Embeddings
│
└──► mBERT Encoding
│
▼
Contextual Sentence Embeddings
│
▼
Evaluation Module
(OOV, similarity, comparison)

This modular design ensures extensibility and reproducibility across different experiments.

---

## 📂 Project Structure

uzbek-hybrid-embeddings/
│
├── src/
│ ├── preprocessing.py # Text cleaning and normalization
│ ├── train_word2vec.py # Word2Vec training script
│ ├── train_fasttext.py # FastText training script
│ ├── train_mbert.py # mBERT sentence embedding script
│ └── evaluate.py # Evaluation and visualization
│
├── data/
│ ├── raw/ # Raw text corpus
│ └── processed/ # Preprocessed corpus
│
├── results/
│ ├── tables/ # CSV result tables
│ └── figures/ # Plots and figures
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Evaluation Metrics

The models are evaluated using intrinsic evaluation metrics:

- **Vocabulary Size**
- **Out-of-Vocabulary (OOV) Coverage**
- **Average Cosine Similarity**

These metrics allow for a fair comparison between static, subword-based, and contextual embeddings.

---

## 📈 Experimental Results (Initial)

| Model     | Vocabulary Size | OOV Coverage | Avg Cosine Similarity |
|-----------|-----------------|--------------|-----------------------|
| Word2Vec  | 6               | 0.00         | 0.000                 |
| FastText  | 6               | 1.00         | -0.005                |

> Note: These are initial experimental results based on a small corpus.
> The corpus will be expanded in subsequent experiments to obtain more robust results.

---

## 🔬 Reproducibility

Due to GitHub file size limitations, trained model files are excluded from the repository.
However, all experiments are fully reproducible using the provided scripts.

To reproduce the experiments:

```bash
python src/preprocessing.py
python src/train_word2vec.py
python src/train_fasttext.py
python src/train_mbert.py
python src/evaluate.py

🧪 Dataset

The Uzbek text corpus consists of:

Educational texts

Scientific and technical descriptions

Synthetic data generated using large language models

This approach mitigates data sparsity in low-resource language scenarios.

📚 Research Context

This repository supports research on:

Low-resource NLP

Uzbek language processing

Hybrid embedding models

Comparative analysis of embedding techniques

The implementation is intended to accompany a Scopus-indexed research article.

📄 License

This project is released under the MIT License.

👤 Author

Abdilatif Meyliyev
PhD Researcher in Natural Language Processing
Uzbekistan