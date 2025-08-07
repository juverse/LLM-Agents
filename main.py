import argparse
import sys
from scripts.evaluation import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM choices using modular prompts.")
    parser.add_argument('--backend', type=str, choices=['online', 'local', 'logprob', 'memory', 'gemini'], 
                       default='online', help='Backend to use for evaluation')
    parser.add_argument('--max_items', type=int, default=100, help='Maximum number of items per split (default: 100)')
    parser.add_argument('--model', type=str, default='mistralai/mistral-7b-instruct', help='Model name to use')
    parser.add_argument('--main_prompt', type=str, default='study', help='Main prompt template name (without .txt)')
    parser.add_argument('--sub_prompts', type=str, default='belief_desire_intention',
                        help='Comma separated sub-prompt template names (without .txt)')
    args = parser.parse_args()

    try:
        results, output_file = run_evaluation(args)
        print("Evaluation complete. Saved to", output_file)
        print("Average Accuracy:", results["score"].mean())
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

