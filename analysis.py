import pandas as pd
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_accuracy(csv_files):
    dfs = []
    for csv_file in csv_files:
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
        return dfs

    try:
        from scipy.stats import ttest_ind
        print("Significance testing (t-test, p<0.05):")
        for i in range(len(dfs)):
            for j in range(i+1, len(dfs)):
                scores1 = dfs[i]['score'].dropna()
                scores2 = dfs[j]['score'].dropna()
                tstat, pval = ttest_ind(scores1, scores2, equal_var=False)
                sig = pval < 0.05
                print(f"  {csv_files[i]} vs {csv_files[j]}: p-value={pval:.4g} {'*' if sig else ''}")
                if sig:
                    print(f"    ==> There IS a significant difference between {csv_files[i]} and {csv_files[j]} (p={pval:.4g})")
                else:
                    print(f"    ==> There is NO significant difference between {csv_files[i]} and {csv_files[j]} (p={pval:.4g})")
    except ImportError:
        print("scipy is required for significance testing. Please install it with 'pip install scipy'.")
    
    return dfs

def generate_table(results_dir):
    result_files = glob(os.path.join(results_dir, '*.csv'))
    summary_rows = []
    all_splits = set()

    for file in result_files:
        df = pd.read_csv(file)
        if 'split' in df.columns:
            all_splits.update(df['split'].dropna().unique())

    all_splits = sorted(all_splits)

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

    columns = ['name', 'model', 'overall'] + all_splits
    df_summary = pd.DataFrame(summary_rows)[columns]
    print(df_summary.to_markdown(index=False))
    return df_summary

def visualize_results(csv_files, output_dir="results/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            basename = os.path.basename(file_path)
            run_name = basename.split('_ewok_eval_')[-1].replace('.csv', '').replace('_', ' ').title()
            df['run_name'] = run_name
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_data:
        print("No valid data files found.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    plt.figure(figsize=(12, 8))
    
    if 'split' in combined_df.columns:
        splits = combined_df['split'].unique()
        run_names = combined_df['run_name'].unique()
        
        x = np.arange(len(run_names))
        width = 0.25
        
        for i, split in enumerate(splits):
            split_data = []
            for run in run_names:
                subset = combined_df[(combined_df['run_name'] == run) & (combined_df['split'] == split)]
                split_data.append(subset['score'].mean() if not subset.empty else 0)
            
            plt.bar(x + i * width, split_data, width, label=split)
        
        plt.xlabel('Experimental Runs')
        plt.ylabel('Average Accuracy')
        plt.title('Accuracy Comparison Across Runs and Splits')
        plt.xticks(x + width, run_names, rotation=45)
        plt.legend()
    else:
        run_names = combined_df['run_name'].unique()
        accuracies = [combined_df[combined_df['run_name'] == run]['score'].mean() for run in run_names]
        
        plt.bar(run_names, accuracies)
        plt.xlabel('Experimental Runs')
        plt.ylabel('Average Accuracy')
        plt.title('Accuracy Comparison Across Runs')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze results from CSV files.")
    parser.add_argument('csv_files', type=str, nargs='*', help='Path(s) to the CSV file(s) to analyze')
    parser.add_argument('--format', type=str, choices=['analyze', 'table', 'visualize', 'all'], 
                       default='analyze', help='Output format')
    parser.add_argument('--results_dir', type=str, default='./results', 
                       help='Directory containing result files (for table format)')
    args = parser.parse_args()

    if not args.csv_files and args.format not in ['table', 'visualize', 'all']:
        print("No CSV files provided. Use --format table, --format visualize, or --format all to analyze all files in results directory.")
        return

    if args.format == 'analyze' or args.format == 'all':
        if args.csv_files:
            analyze_accuracy(args.csv_files)
        else:
            result_files = glob(os.path.join(args.results_dir, '*.csv'))
            if result_files:
                analyze_accuracy(result_files[:5])
            else:
                print("No CSV files found for analysis.")
    
    if args.format == 'table' or args.format == 'all':
        generate_table(args.results_dir)
    
    if args.format == 'visualize' or args.format == 'all':
        if args.csv_files:
            visualize_results(args.csv_files)
        else:
            result_files = glob(os.path.join(args.results_dir, '*.csv'))
            if result_files:
                print(f"Found {len(result_files)} CSV files for visualization")
                visualize_results(result_files)
            else:
                print("No CSV files found for visualization.")

if __name__ == "__main__":
    main()