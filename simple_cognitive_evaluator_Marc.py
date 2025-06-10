#!/usr/bin/env python3
"""Simple Cognitive Evaluator - Test LLM reasoning on EWoK dataset."""

import os, json, time, argparse, requests, pandas as pd
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SimpleResult:
    predicted: str
    correct: str
    is_correct: bool
    response_time: float
    domain: str = ""

class SimpleLLMClient:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-7b-instruct-v0.1')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        print(f"LLM Client initialized with model: {self.model}")

    def chat(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()

class SimpleDatasetLoader:
    def load_dataset(self):
        try:
            from datasets import load_dataset
            dataset = load_dataset("allenai/ewok")
            df = pd.DataFrame(dataset['test'])
            agentic_domains = ['agent-properties', 'social-interactions', 'social-properties']
            filtered_df = df[df['domain'].isin(agentic_domains)].copy()
            print(f"EWoK dataset loaded: {len(filtered_df)} samples from agentic domains")
            return filtered_df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self._create_fallback_dataset()

    def _create_fallback_dataset(self):
        fallback_data = [
            {
                'domain': 'agent-properties',
                'target_ent': 'person',
                'context': 'A person walks into a room.',
                'continuation_query': ['person goes to bed', 'person leaves room'],
                'correct_completion': '1-1,2-2'
            },
            {
                'domain': 'social-interactions',
                'target_ent': 'friend',
                'context': 'Two friends meet at a cafe.',
                'continuation_query': ['friends talk', 'friends order coffee'],
                'correct_completion': '1-1,2-2'
            },
            {
                'domain': 'social-properties',
                'target_ent': 'person',
                'context': 'A person helps someone in need.',
                'continuation_query': ['person is kind', 'person is selfish'],
                'correct_completion': '1-1,2-2'
            }
        ]
        print("Using fallback dataset with 3 samples")
        return pd.DataFrame(fallback_data)

class SimpleCognitiveEvaluator:
    def __init__(self):
        self.llm = SimpleLLMClient()
        self.loader = SimpleDatasetLoader()

    def evaluate_sample(self, sample):
        prompt = f"""Context: {sample['context']}
Target entity: {sample['target_ent']}
Options: {sample['continuation_query']}

Match each option to the target entity. Answer format: 1-1,2-2"""

        start_time = time.time()
        prediction = self.llm.chat(prompt)
        response_time = time.time() - start_time
        
        correct = sample['correct_completion']
        is_correct = prediction.strip() == correct.strip()
        
        return SimpleResult(
            predicted=prediction,
            correct=correct,
            is_correct=is_correct,
            response_time=response_time,
            domain=sample['domain']
        )

    def evaluate_dataset(self, max_samples=10, iterations=1):
        dataset = self.loader.load_dataset()
        all_results = []
        
        print(f"\nStarting evaluation: {iterations} iteration(s), up to {max_samples} samples")
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            samples = dataset.head(max_samples)
            
            for idx, (_, sample) in enumerate(samples.iterrows()):
                result = self.evaluate_sample(sample)
                all_results.append(result)
                status = "✓" if result.is_correct else "✗"
                print(f"  {idx+1}. {sample['domain']}: {status} ({result.response_time:.2f}s)")
        
        return all_results

    def calculate_summary(self, results):
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_time = sum(r.response_time for r in results) / total if total > 0 else 0
        
        # Domain accuracies
        domain_stats = {}
        for domain in set(r.domain for r in results):
            domain_results = [r for r in results if r.domain == domain]
            domain_correct = sum(1 for r in domain_results if r.is_correct)
            domain_accuracy = (domain_correct / len(domain_results)) * 100
            domain_stats[domain] = domain_accuracy
        
        return {
            'total_samples': total,
            'correct_predictions': correct,
            'overall_accuracy_percent': accuracy,
            'avg_response_time': avg_time,
            'domain_accuracies': domain_stats,
            'detailed_results': [
                {
                    'domain': r.domain,
                    'predicted': r.predicted,
                    'correct': r.correct,
                    'is_correct': r.is_correct,
                    'response_time': r.response_time
                } for r in results
            ]
        }

    def print_results(self, summary):
        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {summary['overall_accuracy_percent']:.1f}% ({summary['correct_predictions']}/{summary['total_samples']})")
        print(f"Average Response Time: {summary['avg_response_time']:.2f} seconds")
        print(f"\nDomain Accuracies:")
        for domain, acc in summary['domain_accuracies'].items():
            print(f"  {domain}: {acc:.1f}%")

    def save_results(self, summary, iterations):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_eval_results_{timestamp}.json"
        
        output_data = {
            'total_iterations': iterations,
            **summary
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Simple Cognitive Evaluator')
    parser.add_argument('--max_samples', type=int, default=10, help='Max samples per iteration')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    args = parser.parse_args()

    evaluator = SimpleCognitiveEvaluator()
    results = evaluator.evaluate_dataset(args.max_samples, args.iterations)
    summary = evaluator.calculate_summary(results)
    evaluator.print_results(summary)
    evaluator.save_results(summary, args.iterations)

if __name__ == "__main__":
    main() 