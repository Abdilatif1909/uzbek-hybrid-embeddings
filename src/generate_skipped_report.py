import pandas as pd
from pathlib import Path

AUDIT = Path("results/tables/pairwise_audit.csv")
OUT = Path("results/tables/top_skipped_pairs.csv")

if not AUDIT.exists():
    raise SystemExit(f"Audit file not found: {AUDIT}")

df = pd.read_csv(AUDIT)
# consider skipped rows where skip_reason is not 'used'
skipped = df[df['skip_reason'] != 'used'].copy()
if skipped.empty:
    print("No skipped pairs found.")
    skipped.to_csv(OUT, index=False)
    raise SystemExit(0)

# Count reasons and order reasons by descending frequency
reason_counts = skipped['skip_reason'].value_counts()
ordered_reasons = list(reason_counts.index)

# Collect up to top_n pairs, prioritizing reasons with higher counts
top_n = 50
collected = []
remaining = top_n
for reason in ordered_reasons:
    group = skipped[skipped['skip_reason'] == reason]
    if group.empty:
        continue
    take = min(len(group), remaining)
    # take the first `take` rows preserving original order
    collected.append(group.head(take))
    remaining -= take
    if remaining <= 0:
        break

if collected:
    out_df = pd.concat(collected, axis=0)
else:
    out_df = skipped.head(top_n)

# Keep only requested columns and reorder
cols = ['original_word1','original_word2','normalized_word1','normalized_word2','skip_reason']
out_df = out_df[cols]
OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False, encoding='utf-8')

# Print summary
total_skipped = len(skipped)
oov_mask = skipped['skip_reason'].str.contains('oov', case=False, na=False)
ident_mask = skipped['skip_reason'].str.contains('ident', case=False, na=False)
num_oov = int(oov_mask.sum())
num_ident = int(ident_mask.sum())

print(f"Top skipped pairs written to: {OUT}")
print(f"Total skipped pairs: {total_skipped}")
print(f"Skipped due to OOV (any): {num_oov}")
print(f"Skipped due to identical tokens after normalization: {num_ident}")

# Also print counts by reason
print('\nSkip reason counts:')
for reason, cnt in reason_counts.items():
    print(f"  {reason}: {cnt}")
