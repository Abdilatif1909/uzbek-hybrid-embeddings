import pandas as pd
import numpy as np
from pathlib import Path

AUDIT = Path("results/tables/pairwise_audit.csv")
COMPARE = Path("results/tables/comparison_for_paper.csv")

if not AUDIT.exists():
    raise SystemExit(f"Audit file not found: {AUDIT}")

df = pd.read_csv(AUDIT)

categories = ["related", "morphological", "unrelated"]
models = ["word2vec", "fasttext"]

summary = {}
for cat in categories:
    sub = df[df["category"] == cat]
    summary[cat] = {}
    total = len(sub)
    summary[cat]["total_pairs"] = int(total)
    # Word2Vec stats
    used_w = sub[(sub["present_in_word2vec"] == 1) & (sub["skip_reason"] == 'used')]
    vals_w = pd.to_numeric(used_w["cosine_word2vec"], errors='coerce').dropna().astype(float)
    summary[cat]["used_word2vec"] = int(len(vals_w))
    summary[cat]["w_mean"] = round(float(vals_w.mean()), 3) if len(vals_w) else None
    summary[cat]["w_median"] = round(float(vals_w.median()), 3) if len(vals_w) else None
    summary[cat]["w_std"] = round(float(vals_w.std(ddof=1)), 3) if len(vals_w) > 1 else None
    # FastText stats
    used_f = sub[(sub["present_in_fasttext"] == 1) & (sub["skip_reason"] == 'used')]
    vals_f = pd.to_numeric(used_f["cosine_fasttext"], errors='coerce').dropna().astype(float)
    summary[cat]["used_fasttext"] = int(len(vals_f))
    summary[cat]["f_mean"] = round(float(vals_f.mean()), 3) if len(vals_f) else None
    summary[cat]["f_median"] = round(float(vals_f.median()), 3) if len(vals_f) else None
    summary[cat]["f_std"] = round(float(vals_f.std(ddof=1)), 3) if len(vals_f) > 1 else None

# overall used counts
used_total_w = int(len(df[(df['present_in_word2vec']==1) & (df['skip_reason']=='used')]))
used_total_f = int(len(df[(df['present_in_fasttext']==1) & (df['skip_reason']=='used')]))

# skipped reasons
skipped_counts = df['skip_reason'].value_counts().to_dict()

# Print concise summary
print('\nPairwise audit summary:')
for cat in categories:
    s = summary[cat]
    print(f"\nCategory: {cat} — total pairs={s['total_pairs']}")
    print(f"  Word2Vec: used={s['used_word2vec']}, mean={s['w_mean']}, median={s['w_median']}, std={s['w_std']}")
    print(f"  FastText: used={s['used_fasttext']}, mean={s['f_mean']}, median={s['f_median']}, std={s['f_std']}")

print(f"\nOverall used pairs: Word2Vec={used_total_w}, FastText={used_total_f}")
print('\nTop skipped reasons:')
for k, v in skipped_counts.items():
    print(f"  {k}: {v}")

# Show where results are saved
print('\nGenerated files:')
print('  - Pairwise audit: ', AUDIT)
print('  - Comparison table for paper: ', COMPARE)

# Update comparison_for_paper.csv if present: ensure pairs_used_* columns exist
if COMPARE.exists():
    cmp = pd.read_csv(COMPARE)
    # fill pairs_used columns from audit summary (prefer Word2Vec counts)
    mapping = {
        'related': summary['related']['used_word2vec'],
        'unrelated': summary['unrelated']['used_word2vec'],
        'morphological': summary['morphological']['used_word2vec']
    }
    cmp['pairs_used_related'] = cmp['Model'].map(lambda m: mapping['related'])
    cmp['pairs_used_unrelated'] = cmp['Model'].map(lambda m: mapping['unrelated'])
    cmp['pairs_used_morphological'] = cmp['Model'].map(lambda m: mapping['morphological'])
    cmp.to_csv(COMPARE, index=False)
    print('\nUpdated comparison_for_paper.csv with pair counts.')
else:
    print('\ncomparison_for_paper.csv not found; skipping update.')
