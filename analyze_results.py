import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze accuracy statistics from one or more results CSV files.")
    parser.add_argument('csv_files', type=str, nargs='+', help='Path(s) to the CSV file(s) to analyze')
    args = parser.parse_args()

    dfs = []
    for csv_file in args.csv_files:
        df = pd.read_csv(csv_file)
        if 'score' not in df.columns:
            print(f"No 'score' column found in {csv_file}.")
            continue
        df['__source__'] = csv_file
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {csv_file}")
        print(f"  Average Accuracy: {df['score'].mean():.4f}")
        print(f"  Median Accuracy: {df['score'].median():.4f}")
        print(f"  Stddev Accuracy: {df['score'].std():.4f}")
        if 'split' in df.columns:
            print("  Breakdown by split:")
            for split, group in df.groupby('split'):
                print(f"    {split}: count={len(group)}, avg={group['score'].mean():.4f}")
        print()

    if len(dfs) < 2:
        return

    # Statistical significance test (t-test)
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        print("scipy is required for significance testing. Please install it with 'pip install scipy'.")
        return

    print("Significance testing (t-test, p<0.05):")
    for i in range(len(dfs)):
        for j in range(i+1, len(dfs)):
            scores1 = dfs[i]['score'].dropna()
            scores2 = dfs[j]['score'].dropna()
            tstat, pval = ttest_ind(scores1, scores2, equal_var=False)
            sig = pval < 0.05
            print(f"  {args.csv_files[i]} vs {args.csv_files[j]}: p-value={pval:.4g} {'*' if sig else ''}")
            if sig:
                print(f"    ==> There IS a significant difference between {args.csv_files[i]} and {args.csv_files[j]} (p={pval:.4g})")
            else:
                print(f"    ==> There is NO significant difference between {args.csv_files[i]} and {args.csv_files[j]} (p={pval:.4g})")

if __name__ == "__main__":
    main()
