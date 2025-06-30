import pandas as pd
import os
from glob import glob

results_dir = './results'
result_files = glob(os.path.join(results_dir, '*.csv'))

summary_rows = []
all_splits = set()

# First pass to collect all split names
for file in result_files:
    df = pd.read_csv(file)
    if 'split' in df.columns:
        all_splits.update(df['split'].dropna().unique())

all_splits = sorted(all_splits)

# Build summary table
for file in result_files:
    df = pd.read_csv(file)
    basename = os.path.basename(file)
    name = basename.split('eval_')[-1].removesuffix('.csv')
    model = df['model'].iloc[0] if 'model' in df.columns else 'unknown'
    overall_acc = df['score'].mean()

    row = {
        'name': name,
        'model': model,
        'overall': f"{overall_acc:.3f}"
    }

    for split in all_splits:
        if 'split' in df.columns:
            split_df = df[df['split'] == split]
            row[split] = f"{split_df['score'].mean():.3f}" if not split_df.empty else '-'
        else:
            row[split] = '-'

    summary_rows.append(row)

# Output as markdown
columns = ['name', 'model', 'overall'] + all_splits
df_summary = pd.DataFrame(summary_rows)[columns]
print(df_summary.to_markdown(index=False))
