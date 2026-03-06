import yaml
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, KFold

CONFIG_PATH = Path("config.yaml")
CORPUS_PATH = Path("data/processed/corpus.txt")
W2V_PATH = Path("models/word2vec.model")
FT_PATH = Path("models/fasttext.model")
MBERT_EMB = Path("models/mbert/sentence_embeddings.npy")
RESULT_PATH = Path("results/tables/downstream_results.csv")
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_sentences(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Processed corpus not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sentence_embedding_w2v(sent: str, model: Word2Vec, dim: int):
    words = [w for w in sent.split() if w in model.wv]
    if not words:
        return np.zeros(dim, dtype=float)
    vecs = [model.wv[w] for w in words]
    return np.mean(vecs, axis=0)


def sentence_embedding_ft(sent: str, model: FastText, dim: int):
    words = sent.split()
    vecs = [model.wv[w] for w in words if w]
    if not vecs:
        return np.zeros(dim, dtype=float)
    return np.mean(vecs, axis=0)


if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    seed = config.get("seed", 42)
    sample_limit = config.get("sample_limit")

    sentences = load_sentences(CORPUS_PATH)
    if sample_limit:
        sentences = sentences[:sample_limit]

    # Define topic keyword sets for 4 classes: education, technology, science, society
    topics = {
        'education': {"ta'lim", "maktab", "talaba", "o'quv", "o'qituvchi", "o'quvchi", "ta'limiy", "dars", "universitet", "o'quv"},
        'technology': {"dastur", "dasturchi", "kompyuter", "texnologiya", "internet", "server", "kod", "algoritm", "sun'iy", "intellekt"},
        'science': {"ilm", "ilmiy", "laboratoriya", "tadqiqot", "tajriba", "ilm-fan", "akademik", "ma'lumot", "biologiya", "kimyo"},
        'society': {"jamiyat", "iqtisod", "siyosat", "madaniyat", "inson", "yozuv", "hayot", "shahar", "qishloq", "aholi"}
    }

    # collect sentences for each topic using keyword matching
    selected = {k: [] for k in topics}
    for s in sentences:
        tokens = set(s.split())
        for label, kws in topics.items():
            if tokens & kws:
                selected[label].append(s)

    # remove overlaps: keep only sentences that map to exactly one label to reduce noise
    final_sents = {k: [] for k in topics}
    for s in sentences:
        labels = [label for label, kws in topics.items() if set(s.split()) & kws]
        if len(labels) == 1:
            final_sents[labels[0]].append(s)

    # Ensure enough samples per class; sample up to N per class for balance
    min_per_class = 40
    max_per_class = 200
    for k in list(final_sents.keys()):
        n = len(final_sents[k])
        if n < min_per_class:
            print(f"Warning: only {n} sentences found for class '{k}' (min {min_per_class}). Consider adding more data.")
        # sample deterministically
        rng = np.random.RandomState(seed)
        final_sents[k] = list(rng.choice(final_sents[k], size=min(n, max_per_class), replace=False)) if n > 0 else []

    # Build dataset
    texts = []
    labels = []
    label_map = {lab: idx for idx, lab in enumerate(sorted(final_sents.keys()))}
    for lab, sents in final_sents.items():
        for s in sents:
            texts.append(s)
            labels.append(label_map[lab])

    if len(texts) < 50:
        raise RuntimeError("Not enough labelled sentences found across topics to run benchmark. Add more data or lower min_per_class.")

    print(f"Prepared dataset with {len(texts)} sentences across {len(final_sents)} classes.")

    # Load embedding models
    # Prefer full Word2Vec model if available
    full_w2v_path = Path("models/word2vec_full.model")
    if full_w2v_path.exists():
        w2v = Word2Vec.load(str(full_w2v_path))
    else:
        w2v = Word2Vec.load(str(W2V_PATH)) if W2V_PATH.exists() else None
    ft = FastText.load(str(FT_PATH)) if FT_PATH.exists() else None
    mbert_emb = np.load(MBERT_EMB) if MBERT_EMB.exists() else None

    # Map sentences to mBERT embeddings where possible
    sent_to_idx = {}
    with CORPUS_PATH.open(encoding='utf-8') as f:
        all_sents = [line.strip() for line in f if line.strip()]
    sent_to_idx = {s: i for i, s in enumerate(all_sents)}

    results = []

    def evaluate_embedding(X, y, name: str):
        seed_local = seed
        # Try stratified 80/20 split first
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_local, stratify=y)
            scores = []
            # Logistic Regression
            clf1 = LogisticRegression(max_iter=2000, random_state=seed_local)
            clf1.fit(X_train, y_train)
            p1 = clf1.predict(X_test)
            acc1 = accuracy_score(y_test, p1)
            f1_1 = f1_score(y_test, p1, average='macro')
            scores.append({'clf': 'LogisticRegression', 'acc': acc1, 'f1': f1_1})
            # LinearSVC
            clf2 = LinearSVC(max_iter=20000, random_state=seed_local)
            clf2.fit(X_train, y_train)
            p2 = clf2.predict(X_test)
            acc2 = accuracy_score(y_test, p2)
            f1_2 = f1_score(y_test, p2, average='macro')
            scores.append({'clf': 'LinearSVC', 'acc': acc2, 'f1': f1_2})
            # average
            avg_acc = float(np.mean([s['acc'] for s in scores]))
            avg_f1 = float(np.mean([s['f1'] for s in scores]))
            print(f"{name}: LR Acc={acc1:.3f} F1={f1_1:.3f}; SVC Acc={acc2:.3f} F1={f1_2:.3f}; AVG Acc={avg_acc:.3f} F1={avg_f1:.3f}")
            results.append({'model': name, 'accuracy': round(avg_acc, 3), 'f1_macro': round(avg_f1, 3)})
            return
        except ValueError as e:
            print(f"Stratified split failed: {e}. Falling back to cross-validation based evaluation.")

        # Fallback: use StratifiedKFold when possible
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        n_splits = 5
        if min_count >= n_splits:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_local)
            folds = skf.split(X, y)
        elif min_count >= 2:
            n_splits = min(5, min_count)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_local)
            folds = skf.split(X, y)
        else:
            # use regular KFold if stratification is impossible
            n_splits = min(5, len(y))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_local)
            folds = kf.split(X)

        accs_lr = []
        f1s_lr = []
        accs_svc = []
        f1s_svc = []
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf1 = LogisticRegression(max_iter=2000, random_state=seed_local)
            clf1.fit(X_train, y_train)
            p1 = clf1.predict(X_test)
            accs_lr.append(accuracy_score(y_test, p1))
            f1s_lr.append(f1_score(y_test, p1, average='macro'))

            clf2 = LinearSVC(max_iter=20000, random_state=seed_local)
            clf2.fit(X_train, y_train)
            p2 = clf2.predict(X_test)
            accs_svc.append(accuracy_score(y_test, p2))
            f1s_svc.append(f1_score(y_test, p2, average='macro'))

        # Average across folds and classifiers
        avg_acc = float(np.mean(accs_lr + accs_svc))
        avg_f1 = float(np.mean(f1s_lr + f1s_svc))
        print(f"{name}: Cross-val ({n_splits}) AVG Acc={avg_acc:.3f} F1={avg_f1:.3f}")
        results.append({'model': name, 'accuracy': round(avg_acc, 3), 'f1_macro': round(avg_f1, 3)})

    y = np.array(labels)

    # Word2Vec
    if w2v is not None:
        dim = w2v.wv.vector_size
        X = np.vstack([sentence_embedding_w2v(s, w2v, dim) for s in texts])
        evaluate_embedding(X, y, 'Word2Vec')

    # FastText
    if ft is not None:
        dim = ft.wv.vector_size
        X = np.vstack([sentence_embedding_ft(s, ft, dim) for s in texts])
        evaluate_embedding(X, y, 'FastText')

    # mBERT
    if mbert_emb is not None:
        emb_list = []
        missing = 0
        for s in texts:
            idx = sent_to_idx.get(s)
            if idx is None or idx >= len(mbert_emb):
                emb_list.append(np.zeros(mbert_emb.shape[1], dtype=float))
                missing += 1
            else:
                emb_list.append(mbert_emb[idx])
        if missing > 0:
            print(f"Warning: {missing} sentences missing mBERT embeddings; filled with zeros.")
        X = np.vstack(emb_list)
        evaluate_embedding(X, y, 'mBERT')

    df = pd.DataFrame(results)
    df.to_csv(RESULT_PATH, index=False)
    print(f"Downstream benchmark results saved to: {RESULT_PATH}")
