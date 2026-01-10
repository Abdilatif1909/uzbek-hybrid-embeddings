<<<<<<< HEAD
# Uzbek Hybrid Embeddings

This repository contains experimental implementations of hybrid embedding models
(Word2Vec, FastText, and mBERT) for the Uzbek language.

## Models
- Word2Vec (static embeddings)
- FastText (subword-aware embeddings)
- mBERT (contextual sentence embeddings)

## Corpus
The models are trained on a preprocessed Uzbek text corpus.
The corpus was synthetically expanded using large language models to mitigate data sparsity.

## Evaluation
The models are evaluated using intrinsic metrics:
- Vocabulary size
- Out-of-vocabulary (OOV) coverage
- Average cosine similarity

## Results (Initial)
| Model     | Vocabulary Size | OOV Coverage | Avg Cosine Similarity |
|-----------|-----------------|--------------|-----------------------|
| Word2Vec  | 6               | 0.00         | 0.000                 |
| FastText  | 6               | 1.00         | -0.005                |

> Note: These are initial experimental results. The corpus size will be expanded in subsequent experiments.

## Reproducibility
All experiments are fully reproducible using the provided scripts.

## License
MIT
=======
# uzbek-hybrid-embeddings
>>>>>>> 71f3f157dd518791d7d611007a49cca7846bcc49
