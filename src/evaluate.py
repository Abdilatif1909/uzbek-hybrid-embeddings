import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Word2Vec, FastText
import re
import csv

# Paths
CONFIG_PATH = Path("config.yaml")
W2V_PATH = Path("models/word2vec.model")
FT_PATH = Path("models/fasttext.model")
CORPUS_PATH = Path("data/processed/corpus.txt")
TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
EMBED_TABLE = TABLE_DIR / "embedding_results.csv"
COMPARE_TABLE = TABLE_DIR / "comparison_for_paper.csv"
FIG_PATH = FIG_DIR / "embedding_comparison.png"
MBERT_RESULTS = TABLE_DIR / "mbert_results.csv"
DOWNSTREAM_RESULTS = TABLE_DIR / "downstream_results.csv"
PAIRWISE_AUDIT = TABLE_DIR / "pairwise_audit.csv"


def load_config(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_models():
    # Prefer full Word2Vec model if available (trained on corpus_big)
    full_w2v = Path("models/word2vec_full.model")
    base_w2v = Path("models/word2vec.model")
    if full_w2v.exists():
        w2v_path = full_w2v
    else:
        w2v_path = base_w2v
    # Prefer full FastText if available
    full_ft = Path("models/fasttext_full.model")
    base_ft = Path("models/fasttext.model")
    if full_ft.exists():
        ft_path = full_ft
    else:
        ft_path = base_ft
    if not w2v_path.exists() or not ft_path.exists():
        raise FileNotFoundError("Trained models not found. Run training scripts first.")
    w2v = Word2Vec.load(str(w2v_path))
    ft = FastText.load(str(ft_path))
    return w2v, ft


def load_corpus(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Processed corpus not found: {path}")
    with path.open(encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


# Normalization utilities for Uzbek tokens
APOSTROPHE_VARIANTS = ["'", "’", "`", ""]


def normalize_token(tok: str) -> str:
    t = tok.lower()
    # unify apostrophes with ASCII '
    for a in APOSTROPHE_VARIANTS:
        t = t.replace(a, "'")
    t = t.replace("o'", "o'")  # keep as is; example: o'quvchi
    t = t.replace("g'", "g'")
    t = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁўқғҳ'_-]", "", t)
    t = t.strip("-_")
    return t


def generate_variants(word: str):
    # produce morphological / transliteration variants
    variants = set()
    base = normalize_token(word)
    variants.add(base)
    # common Uzbek suffixes
    suffixes = ["lar", "larni", "ning", "da", "dan", "ga", "larimiz", "imiz"]
    for s in suffixes:
        variants.add(base + s)
    # simplified transliteration: remove apostrophe
    variants.add(base.replace("'", ""))
    # replace o'->o, g'->g
    variants.add(base.replace("o'", "o").replace("g'", "g"))
    return list(variants)


# Initial curated seed pairs (short list) - will be expanded
SEED_RELATED = [
    ("kitob", "asar"), ("talaba", "o'quvchi"), ("maktab", "ta'lim"), ("kompyuter", "texnologiya"),
    ("yaxshi", "zo'r"), ("yomon", "salbiy"), ("daraxt", "barg"), ("barg", "yaproq"),
    ("shahar", "qishloq"), ("ilm", "fan"), ("ta'lim", "o'qitish"), ("o'qituvchi", "ustoz"),
    ("muallif", "yozuvchi"), ("dastur", "ilova"), ("internet", "tarmoq"), ("ma'lumot", "axborot"),
    ("inson", "odam"), ("avtomobil", "mashina"), ("telefon", "mobil"), ("laboratoriya", "tajriba"),
]

SEED_UNRELATED = [
    ("kitob", "kompyuter"), ("talaba", "mashina"), ("maktab", "o'simlik"), ("yaxshi", "telefon"),
    ("daraxt", "server"), ("shahar", "dengiz"), ("ilm", "muzika"), ("ta'lim", "ovqat"),
    ("ma'lumot", "meva"), ("inson", "kompyuter"), ("kasb", "rang"), ("taqdim", "yomg'ir"),
]

SEED_MORPH = [
    ("kitob", "kitoblar"), ("talaba", "talabalar"), ("ilm", "ilmiy"), ("maktab", "maktabda"),
    ("dastur", "dasturchi"), ("ma'lumot", "ma'lumotlar"), ("o'qituvchi", "o'qituvchilar"),
    ("ta'lim", "ta'limiy"), ("sun'iy", "suniy"), ("intellekt", "intellektual"),
]

# Expand and clean the pair lists to reach target sizes
TARGET_RELATED = 40
TARGET_UNRELATED = 40
TARGET_MORPH = 20


def expand_pairs(seed_pairs, target, existing_set=None):
    pairs = [] if existing_set is None else list(existing_set)
    skipped = []
    norm_set = set()
    for a, b in seed_pairs:
        na, nb = normalize_token(a), normalize_token(b)
        if na == nb:
            skipped.append(((a, b), 'identical_after_norm'))
            continue
        pairs.append((na, nb))
        norm_set.add((na, nb))
    # expand by generating variants of each token in seeds until target
    i = 0
    seeds = seed_pairs[:]
    while len(pairs) < target and i < len(seeds) * 50:
        a, b = seeds[i % len(seeds)]
        variants_a = generate_variants(a)
        variants_b = generate_variants(b)
        for va in variants_a:
            for vb in variants_b:
                na, nb = normalize_token(va), normalize_token(vb)
                if na == nb:
                    continue
                if (na, nb) in norm_set:
                    continue
                pairs.append((na, nb))
                norm_set.add((na, nb))
                if len(pairs) >= target:
                    break
            if len(pairs) >= target:
                break
        i += 1
    return pairs, skipped


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    w2v, ft = load_models()
    sentences = load_corpus(CORPUS_PATH)

    # Build expanded pair lists
    related_pairs, skipped_rel = expand_pairs(SEED_RELATED, TARGET_RELATED)
    unrelated_pairs, skipped_unrel = expand_pairs(SEED_UNRELATED, TARGET_UNRELATED)
    morph_pairs, skipped_morph = expand_pairs(SEED_MORPH, TARGET_MORPH)

    # combine morphological into related for some stats
    related_all = related_pairs + morph_pairs

    # Ensure category sizes
    if len(related_pairs) < TARGET_RELATED or len(unrelated_pairs) < TARGET_UNRELATED or len(morph_pairs) < TARGET_MORPH:
        raise RuntimeError("Failed to expand pair lists to required sizes. Inspect seed pairs.")

    # Prepare audit rows
    audit_rows = []

    def inspect_pair(a_orig, b_orig, category):
        na = normalize_token(a_orig)
        nb = normalize_token(b_orig)
        present_w2v = na in w2v.wv
        present_ft = nb in ft.wv if False else True  # placeholder to ensure variable exists
        present_ft = nb in ft.wv
        cosine_w2v = ''
        cosine_ft = ''
        skip_reason = 'used'
        if na == nb:
            skip_reason = 'identical_after_normalization'
        elif not present_w2v and not present_ft:
            skip_reason = 'oov_both'
        elif not present_w2v:
            skip_reason = 'oov_word2vec'
        elif not present_ft:
            skip_reason = 'oov_fasttext'
        else:
            try:
                cosine_w2v = float(w2v.wv.similarity(na, nb))
            except Exception:
                cosine_w2v = ''
            try:
                cosine_ft = float(ft.wv.similarity(na, nb))
            except Exception:
                cosine_ft = ''
        return {
            'original_word1': a_orig,
            'original_word2': b_orig,
            'normalized_word1': na,
            'normalized_word2': nb,
            'category': category,
            'present_in_word2vec': int(present_w2v),
            'present_in_fasttext': int(present_ft),
            'cosine_word2vec': round(cosine_w2v, 6) if cosine_w2v != '' else '',
            'cosine_fasttext': round(cosine_ft, 6) if cosine_ft != '' else '',
            'skip_reason': skip_reason
        }

    # audit related
    for a, b in related_pairs:
        audit_rows.append(inspect_pair(a, b, 'related'))
    # audit morph
    for a, b in morph_pairs:
        audit_rows.append(inspect_pair(a, b, 'morphological'))
    # audit unrelated
    for a, b in unrelated_pairs:
        audit_rows.append(inspect_pair(a, b, 'unrelated'))

    # save audit CSV
    with PAIRWISE_AUDIT.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'original_word1','original_word2','normalized_word1','normalized_word2','category',
            'present_in_word2vec','present_in_fasttext','cosine_word2vec','cosine_fasttext','skip_reason'
        ])
        writer.writeheader()
        for r in audit_rows:
            writer.writerow(r)

    # Count valid used pairs per category and per model
    audit_df = pd.read_csv(PAIRWISE_AUDIT)
    def count_used(df, category, model_flag):
        sub = df[df['category'] == category]
        used = sub[sub['skip_reason'] == 'used']
        return len(used)

    used_related_w2v = len(audit_df[(audit_df['category']=='related') & (audit_df['present_in_word2vec']==1) & (audit_df['skip_reason']=='used')])
    used_unrelated_w2v = len(audit_df[(audit_df['category']=='unrelated') & (audit_df['present_in_word2vec']==1) & (audit_df['skip_reason']=='used')])
    used_morph_w2v = len(audit_df[(audit_df['category']=='morphological') & (audit_df['present_in_word2vec']==1) & (audit_df['skip_reason']=='used')])

    used_related_ft = len(audit_df[(audit_df['category']=='related') & (audit_df['present_in_fasttext']==1) & (audit_df['skip_reason']=='used')])
    used_unrelated_ft = len(audit_df[(audit_df['category']=='unrelated') & (audit_df['present_in_fasttext']==1) & (audit_df['skip_reason']=='used')])
    used_morph_ft = len(audit_df[(audit_df['category']=='morphological') & (audit_df['present_in_fasttext']==1) & (audit_df['skip_reason']=='used')])

    print("Valid pairs used (Word2Vec): related=", used_related_w2v, ", unrelated=", used_unrelated_w2v, ", morphological=", used_morph_w2v)
    print("Valid pairs used (FastText): related=", used_related_ft, ", unrelated=", used_unrelated_ft, ", morphological=", used_morph_ft)

    # Prepare random 100 words and synthetic OOVs
    def get_random_words(sentences, k=100, seed=42):
        cnt = Counter()
        for s in sentences:
            cnt.update(s.split())
        words = list(cnt.keys())
        rng = np.random.RandomState(seed)
        if len(words) <= k:
            return words
        return list(rng.choice(words, size=k, replace=False))

    random100 = get_random_words(sentences, k=100, seed=cfg.get("seed", 42))
    # synthetic OOVs
    rng = np.random.RandomState(cfg.get("seed", 42))
    base_words = random100
    synthetic_oovs = []
    suffixes = ["zzz", "_x", "123", "-test", "qayta", "xx", "q"]
    i = 0
    while len(synthetic_oovs) < 50 and i < len(base_words) * 10:
        base = base_words[i % len(base_words)]
        cand = normalize_token(base) + suffixes[i % len(suffixes)]
        if cand not in w2v.wv and cand not in ft.wv and cand not in synthetic_oovs:
            synthetic_oovs.append(cand)
        i += 1

    # function to compute stats per category and log skipped pairs (OOV)
    def compute_stats(model, pairs, category_name):
        used_scores = []
        skipped = []
        for a, b in pairs:
            na, nb = normalize_token(a), normalize_token(b)
            if na == nb:
                skipped.append(((a, b), 'identical_after_norm'))
                continue
            if na not in model.wv or nb not in model.wv:
                skipped.append(((a, b), 'oov'))
                continue
            try:
                s = float(model.wv.similarity(na, nb))
                used_scores.append(s)
            except Exception:
                skipped.append(((a, b), 'error'))
        if used_scores:
            mean = float(np.mean(used_scores))
            median = float(np.median(used_scores))
            std = float(np.std(used_scores, ddof=1)) if len(used_scores) > 1 else 0.0
        else:
            mean = median = std = 0.0
        return {
            'n_pairs': len(used_scores),
            'mean': mean,
            'median': median,
            'std': std,
            'skipped': skipped
        }

    # compute OOV coverage
    def oov_coverage(model, words):
        if not words:
            return 0.0, 0
        known = sum(1 for w in words if w in model.wv)
        return known / len(words), known

    oov_random_w2v, known_rand_w2v = oov_coverage(w2v, random100)
    oov_random_ft, known_rand_ft = oov_coverage(ft, random100)
    oov_synth_w2v, known_synth_w2v = oov_coverage(w2v, synthetic_oovs)
    oov_synth_ft, known_synth_ft = oov_coverage(ft, synthetic_oovs)

    # Compute similarity stats for related and unrelated
    stats = {}
    stats['Word2Vec'] = {}
    stats['FastText'] = {}

    stats['Word2Vec']['related'] = compute_stats(w2v, related_all, 'related')
    stats['Word2Vec']['unrelated'] = compute_stats(w2v, unrelated_pairs, 'unrelated')
    stats['FastText']['related'] = compute_stats(ft, related_all, 'related')
    stats['FastText']['unrelated'] = compute_stats(ft, unrelated_pairs, 'unrelated')

    # overall aggregate cosine stats (combine related+unrelated used scores)
    def aggregate_scores(model, pairs_list):
        vals = []
        for a, b in pairs_list:
            na, nb = normalize_token(a), normalize_token(b)
            if na in model.wv and nb in model.wv:
                try:
                    vals.append(float(model.wv.similarity(na, nb)))
                except Exception:
                    continue
        return vals

    vals_w2v = aggregate_scores(w2v, related_all + unrelated_pairs)
    vals_ft = aggregate_scores(ft, related_all + unrelated_pairs)

    overall_w2v_mean = float(np.mean(vals_w2v)) if vals_w2v else 0.0
    overall_w2v_median = float(np.median(vals_w2v)) if vals_w2v else 0.0
    overall_w2v_std = float(np.std(vals_w2v, ddof=1)) if len(vals_w2v) > 1 else 0.0

    overall_ft_mean = float(np.mean(vals_ft)) if vals_ft else 0.0
    overall_ft_median = float(np.median(vals_ft)) if vals_ft else 0.0
    overall_ft_std = float(np.std(vals_ft, ddof=1)) if len(vals_ft) > 1 else 0.0

    # Vocabulary sizes
    vocab_w2v = len(w2v.wv)
    vocab_ft = len(ft.wv)

    # mBERT mean cosine if available (use full sample mean)
    mbert_mean = None
    mbert_median = None
    mbert_std = None
    if MBERT_RESULTS.exists():
        try:
            dfm = pd.read_csv(MBERT_RESULTS)
            if 'cosine' in dfm.columns:
                col = dfm['cosine'].dropna().astype(float)
                mbert_mean = float(col.mean())
                mbert_median = float(col.median())
                mbert_std = float(col.std(ddof=1)) if len(col) > 1 else 0.0
        except Exception:
            mbert_mean = mbert_median = mbert_std = None

    # Downstream results (load if present)
    downstream = pd.read_csv(DOWNSTREAM_RESULTS) if DOWNSTREAM_RESULTS.exists() else pd.DataFrame()

    # Build final comparison table with requested columns
    def build_row(model_name, vocab_size, oov_cov, avg_rel, avg_unrel, median_all, std_all, acc, f1):
        return {
            'Model': model_name,
            'Vocabulary Size': int(vocab_size) if not pd.isna(vocab_size) else '',
            'OOV Coverage': round(oov_cov, 3) if oov_cov is not None and not pd.isna(oov_cov) else '',
            'Avg Cosine (related)': round(avg_rel, 3) if avg_rel is not None else '',
            'Avg Cosine (unrelated)': round(avg_unrel, 3) if avg_unrel is not None else '',
            'Median Cosine': round(median_all, 3) if median_all is not None else '',
            'Std Cosine': round(std_all, 3) if std_all is not None else '',
            'Downstream Accuracy': round(acc, 3) if not pd.isna(acc) else '',
            'Downstream F1': round(f1, 3) if not pd.isna(f1) else ''
        }

    rows = []
    # Word2Vec row
    acc_w2v = np.nan
    f1_w2v = np.nan
    if not downstream.empty and 'Word2Vec' in downstream['model'].values:
        r = downstream[downstream['model'] == 'Word2Vec'].iloc[0]
        acc_w2v = r['accuracy']
        f1_w2v = r['f1_macro']

    rows.append(build_row('Word2Vec', vocab_w2v, oov_random_w2v,
                           stats['Word2Vec']['related']['mean'], stats['Word2Vec']['unrelated']['mean'],
                           overall_w2v_median, overall_w2v_std, acc_w2v, f1_w2v))

    # FastText row
    acc_ft = np.nan
    f1_ft = np.nan
    if not downstream.empty and 'FastText' in downstream['model'].values:
        r = downstream[downstream['model'] == 'FastText'].iloc[0]
        acc_ft = r['accuracy']
        f1_ft = r['f1_macro']

    rows.append(build_row('FastText', vocab_ft, oov_random_ft,
                           stats['FastText']['related']['mean'], stats['FastText']['unrelated']['mean'],
                           overall_ft_median, overall_ft_std, acc_ft, f1_ft))

    # mBERT row
    acc_mb = np.nan
    f1_mb = np.nan
    if not downstream.empty and 'mBERT' in downstream['model'].values:
        r = downstream[downstream['model'] == 'mBERT'].iloc[0]
        acc_mb = r['accuracy']
        f1_mb = r['f1_macro']

    rows.append(build_row('mBERT', np.nan, np.nan,
                           mbert_mean, None, mbert_median, mbert_std, acc_mb, f1_mb))

    cmp = pd.DataFrame(rows)
    cmp.to_csv(COMPARE_TABLE, index=False)
    print(f"Saved final comparison table to {COMPARE_TABLE}")

    # Also save detailed embedding_results.csv with per-model stats
    embed_rows = [
        {
            'Model': 'Word2Vec',
            'Vocabulary Size': vocab_w2v,
            'OOV random100_known': int(known_rand_w2v),
            'OOV random100_total': len(random100),
            'OOV synthetic50_known': int(known_synth_w2v),
            'OOV synthetic50_total': len(synthetic_oovs),
            'n_pairs_related': stats['Word2Vec']['related']['n_pairs'],
            'mean_related': round(stats['Word2Vec']['related']['mean'], 3),
            'median_related': round(stats['Word2Vec']['related']['median'], 3),
            'std_related': round(stats['Word2Vec']['related']['std'], 3),
            'n_pairs_unrelated': stats['Word2Vec']['unrelated']['n_pairs'],
            'mean_unrelated': round(stats['Word2Vec']['unrelated']['mean'], 3),
            'median_unrelated': round(stats['Word2Vec']['unrelated']['median'], 3),
            'std_unrelated': round(stats['Word2Vec']['unrelated']['std'], 3),
            'overall_mean': round(overall_w2v_mean, 3),
            'overall_median': round(overall_w2v_median, 3),
            'overall_std': round(overall_w2v_std, 3)
        },
        {
            'Model': 'FastText',
            'Vocabulary Size': vocab_ft,
            'OOV random100_known': int(known_rand_ft),
            'OOV random100_total': len(random100),
            'OOV synthetic50_known': int(known_synth_ft),
            'OOV synthetic50_total': len(synthetic_oovs),
            'n_pairs_related': stats['FastText']['related']['n_pairs'],
            'mean_related': round(stats['FastText']['related']['mean'], 3),
            'median_related': round(stats['FastText']['related']['median'], 3),
            'std_related': round(stats['FastText']['related']['std'], 3),
            'n_pairs_unrelated': stats['FastText']['unrelated']['n_pairs'],
            'mean_unrelated': round(stats['FastText']['unrelated']['mean'], 3),
            'median_unrelated': round(stats['FastText']['unrelated']['median'], 3),
            'std_unrelated': round(stats['FastText']['unrelated']['std'], 3),
            'overall_mean': round(overall_ft_mean, 3),
            'overall_median': round(overall_ft_median, 3),
            'overall_std': round(overall_ft_std, 3)
        }
    ]
    if mbert_mean is not None:
        embed_rows.append({
            'Model': 'mBERT',
            'Vocabulary Size': '',
            'OOV random100_known': '',
            'OOV random100_total': len(random100),
            'OOV synthetic50_known': '',
            'OOV synthetic50_total': len(synthetic_oovs),
            'n_pairs_related': '',
            'mean_related': round(mbert_mean, 3),
            'median_related': round(mbert_median, 3) if mbert_median is not None else '',
            'std_related': round(mbert_std, 3) if mbert_std is not None else '',
            'n_pairs_unrelated': '',
            'mean_unrelated': '',
            'median_unrelated': '',
            'std_unrelated': '',
            'overall_mean': round(mbert_mean, 3),
            'overall_median': round(mbert_median, 3) if mbert_median is not None else '',
            'overall_std': round(mbert_std, 3) if mbert_std is not None else ''
        })

    pd.DataFrame(embed_rows).to_csv(EMBED_TABLE, index=False)
    print(f"Saved detailed embedding results to {EMBED_TABLE}")

    # After previous code builds cmp DataFrame, add pair counts to compare table
    cmp = pd.read_csv(COMPARE_TABLE)
    # append pairs used columns (take Word2Vec counts as reference)
    cmp['pairs_used_related'] = [used_related_w2v if m=='Word2Vec' else used_related_ft if m=='FastText' else used_related_w2v for m in cmp['Model']]
    cmp['pairs_used_unrelated'] = [used_unrelated_w2v if m=='Word2Vec' else used_unrelated_ft if m=='FastText' else used_unrelated_w2v for m in cmp['Model']]
    cmp['pairs_used_morphological'] = [used_morph_w2v if m=='Word2Vec' else used_morph_ft if m=='FastText' else used_morph_w2v for m in cmp['Model']]
    # save updated compare table
    cmp.to_csv(COMPARE_TABLE, index=False)
    print(f"Updated comparison table with pair counts saved to {COMPARE_TABLE}")
