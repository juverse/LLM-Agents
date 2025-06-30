#!/usr/bin/env python3
"""
Parallel Prompt Evaluation Runner

This script runs a comprehensive, parallelized evaluation of all prompt templates
in the 'prompts/' directory against a balanced sample of the EWoK dataset.

It calculates per-prompt accuracy, saves a detailed report in JSON and CSV formats,
and leverages ThreadPoolExecutor for faster processing.
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from prompt_tester import PromptTester, PromptResult

def run_full_evaluation(total_samples: int = 450, max_workers: int = 10):
    """
    Run a full, parallelized evaluation of all prompts against a dataset of a specified size.

    Args:
        total_samples (int): The total number of samples to evaluate against.
        max_workers (int): The number of parallel threads to use for API calls.
    """
    print("🚀 Starting Comprehensive Prompt Evaluation (Parallelized)")
    
    # The PromptTester will load up to `total_samples`
    tester = PromptTester(max_samples=total_samples) 
    
    # 1. Use the entire loaded dataset for evaluation
    evaluation_dataset = tester.dataset
    print(f"✅ Using all {len(evaluation_dataset)} loaded samples for the evaluation.")

    prompt_names = tester.prompt_loader.list_prompts()
    print(f"🔍 Found {len(prompt_names)} prompts to test: {', '.join(prompt_names)}")

    # 2. Prepare tasks for parallel execution
    tasks = []
    for prompt_name in prompt_names:
        for idx, row in evaluation_dataset.iterrows():
            # Use the DataFrame index as the unique sample_id
            tasks.append((prompt_name, row, idx))

    total_evaluations = len(tasks)
    print(f"\n🔬 Submitting {total_evaluations} tasks to {max_workers} parallel workers...")

    # 3. Run evaluation in parallel
    all_results: List[PromptResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(tester.evaluate_prompt_on_sample, p, r, i): (p, i) for p, r, i in tasks}
        
        for i, future in enumerate(as_completed(future_to_task)):
            try:
                result = future.result()
                all_results.append(result)
                print(f"  ({i + 1}/{total_evaluations}) Evaluation complete for prompt '{result.prompt_name}'...", end='\r')
            except Exception as e:
                task_info = future_to_task[future]
                print(f"❌ Task {task_info} generated an exception: {e}")

    print(f"\n\n✅ All evaluations complete!")

    # 4. Calculate and print summary
    summary = calculate_summary(all_results, prompt_names)
    print_summary(summary)

    # 5. Save results to JSON and CSV
    save_results(summary)

def calculate_summary(results: List[PromptResult], prompt_names: List[str]) -> Dict[str, Any]:
    """Calculate a detailed summary of the evaluation results."""
    prompt_summaries = {}
    for prompt_name in prompt_names:
        prompt_results = [r for r in results if r.prompt_name == prompt_name]
        if not prompt_results:
            continue
            
        correct = sum(1 for r in prompt_results if r.is_correct)
        total = len(prompt_results)
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = sum(r.response_time for r in prompt_results) / total if total > 0 else 0
        
        prompt_summaries[prompt_name] = {
            "accuracy_percent": accuracy,
            "correct_predictions": correct,
            "total_samples": total,
            "avg_response_time_s": avg_time
        }
        
    return {
        "evaluation_summary": prompt_summaries,
        "detailed_results": [res.to_dict() for res in results]
    }

def print_summary(summary: Dict[str, Any]):
    """Print a formatted summary of the evaluation."""
    print("\n\n📊 PROMPT EVALUATION SUMMARY 📊")
    print("===================================")
    
    sorted_prompts = sorted(
        summary['evaluation_summary'].items(),
        key=lambda item: item[1]['accuracy_percent'],
        reverse=True
    )
    
    for prompt_name, stats in sorted_prompts:
        print(f"\nPrompt: '{prompt_name}'")
        print(f"  - Accuracy: {stats['accuracy_percent']:.2f}% ({stats['correct_predictions']}/{stats['total_samples']})")
        print(f"  - Avg. Time: {stats['avg_response_time_s']:.2f}s")
        
    print("\n===================================")

def save_results(summary: Dict[str, Any]):
    """Save the summary to timestamped JSON and CSV files."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON file
    json_filename = os.path.join(results_dir, f"prompt_evaluation_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 JSON results saved to: {json_filename}")

    # Save CSV file
    try:
        csv_filename = os.path.join(results_dir, f"prompt_evaluation_{timestamp}.csv")
        df = pd.DataFrame(summary['detailed_results'])
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"💾 CSV results saved to: {csv_filename}")
    except Exception as e:
        print(f"⚠️ Could not save CSV file. Error: {e}")

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERROR: OPENROUTER_API_KEY is not set. Please create a .env file.")
    else:
        try:
            # Run with 450 total samples
            run_full_evaluation(total_samples=450, max_workers=10)
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()