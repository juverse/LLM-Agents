import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def select_files(results_folder):
    """
    Open a file selection dialog to let user choose CSV files.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file selection dialog
    file_paths = filedialog.askopenfilenames(
        title="Select CSV files to visualize",
        initialdir=results_folder,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        multiple=True
    )
    
    root.destroy()
    return list(file_paths) if file_paths else []

def format_run_name(filename):
    """
    Format run name by removing underscores and capitalizing first letter of each part.
    """
    # Extract run name from filename
    run_name = filename.split('_ewok_eval_')[-1]
    
    # Split by underscores, capitalize each part, and join with spaces
    parts = run_name.split('_')
    formatted_parts = [part.capitalize() for part in parts if part]
    
    return ' '.join(formatted_parts)

def read_and_process_results(selected_files):
    """
    Read selected CSV files and process them for visualization.
    """
    all_data = []
    
    for file_path in selected_files:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            
            # Read only the columns we need to save memory
            df = pd.read_csv(file_path, usecols=['split', 'score'])
            
            # Extract and format run name from filename
            filename = Path(file_path).stem
            run_name = format_run_name(filename)
            
            # Group by split and calculate mean score
            split_scores = df.groupby('split')['score'].mean()
            
            # Create a record for this run
            run_data = {'run': run_name}
            for split, score in split_scores.items():
                run_data[split] = score
            
            all_data.append(run_data)
            
            # Clear memory
            del df
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(all_data)

def create_bar_plot(data_df, output_file='benchmark_results.png'):
    """
    Create a bar plot with 3 bars for each run (one for each split condition).
    """
    # Get all unique splits (excluding 'run' column)
    splits = [col for col in data_df.columns if col != 'run']
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Number of runs and splits
    n_runs = len(data_df)
    n_splits = len(splits)
    
    # Set the width of each bar and positions
    bar_width = 0.25
    r = np.arange(n_runs)
    
    # Colors optimized for digital screens (high contrast, colorblind friendly)
    colors = ['#4e79a7', '#f28e2b', '#59a14f']  # Blue, Orange, Green - excellent for digital displays
    
    # Create bars for each split
    for i, split in enumerate(splits):
        values = data_df[split].fillna(0)  # Fill NaN with 0 for missing splits
        bars = ax.bar(r + i * bar_width, values, bar_width, 
                     label=split, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not pd.isna(value) and value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Benchmark Runs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Benchmark Results by Split Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(r + bar_width * (n_splits - 1) / 2)
    ax.set_xticklabels(data_df['run'], rotation=45, ha='right')
    ax.legend(title='Split Conditions', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)  # Assuming scores are between 0 and 1
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for split in splits:
        if split in data_df.columns:
            mean_score = data_df[split].mean()
            std_score = data_df[split].std()
            print(f"{split}: Mean = {mean_score:.3f}, Std = {std_score:.3f}")
    
    return fig

def main():
    # Set the results folder path
    results_folder = "results"
    
    # Check if folder exists
    if not os.path.exists(results_folder):
        print(f"Error: Results folder '{results_folder}' not found!")
        return
    
    # Let user select files
    print("Opening file selection dialog...")
    selected_files = select_files(results_folder)
    
    if not selected_files:
        print("No files selected. Exiting.")
        return
    
    print(f"Selected {len(selected_files)} files")
    
    # Read and process selected CSV files
    print("Reading and processing CSV files...")
    data_df = read_and_process_results(selected_files)
    
    if data_df.empty:
        print("No data found in selected CSV files!")
        return
    
    print(f"Processed {len(data_df)} runs")
    print(f"Available splits: {[col for col in data_df.columns if col != 'run']}")
    
    # Create the bar plot
    fig = create_bar_plot(data_df)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()