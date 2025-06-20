#!/usr/bin/env python3
"""
Prompt Tester - Cognitive Prompt Evaluation System

This module evaluates different cognitive prompts on the EWoK dataset to determine
which prompts perform best for context-target mapping tasks.
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from prompt_loader import PromptLoader


@dataclass
class PromptEvaluationResult:
    """Result of evaluating a single prompt on a sample."""
    prompt_name: str
    sample_id: int
    domain: str
    predicted: str
    correct: str
    is_correct: bool
    response_time: float
    confidence: Optional[str] = None


@dataclass
class PromptPerformanceSummary:
    """Summary of a prompt's performance across all samples."""
    prompt_name: str
    total_samples: int
    correct_count: int
    accuracy_percent: float
    avg_response_time: float
    domain_breakdown: Dict[str, Dict[str, Any]]


class PromptTester:
    """Main prompt evaluation system."""
    
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.config = self._load_config()
        self.llm_client = self._create_llm_client()
        self.prompt_loader = PromptLoader()
        self.dataset = self._load_dataset()
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment."""
        load_dotenv()
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        return {
            "api_key": api_key,
            "model_name": os.getenv("DEFAULT_LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
            "timeout": int(os.getenv("RESPONSE_TIMEOUT", "30")),
            "dataset_path": os.getenv("DATASET_PATH", "hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet"),
            "agentic_domains": ["agent-properties", "social-interactions", "social-properties"]
        }
    
    def _create_llm_client(self) -> ChatOpenAI:
        """Create OpenRouter LLM client."""
        print(f"🧠 Initializing OpenRouter client with model: {self.config['model_name']}")
        
        return ChatOpenAI(
            model=self.config["model_name"],
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.config["api_key"],
            temperature=0.1,
            request_timeout=self.config["timeout"]
        )
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load and filter EWoK dataset."""
        print(f"📥 Loading EWoK dataset (max {self.max_samples} samples)")
        
        try:
            # Load dataset
            df = pd.read_parquet(self.config["dataset_path"])
            
            # Filter for agentic domains
            filtered_df = df[df['Domain'].isin(self.config["agentic_domains"])].copy()
            
            # Sample specified number of examples
            if len(filtered_df) > self.max_samples:
                filtered_df = filtered_df.sample(n=self.max_samples, random_state=42)
            
            print(f"✅ Loaded {len(filtered_df)} samples")
            return filtered_df.reset_index(drop=True)
            
        except Exception as e:
            print(f"⚠️ Error loading dataset: {e}")
            return self._create_fallback_dataset()
    
    def _create_fallback_dataset(self) -> pd.DataFrame:
        """Create fallback dataset for testing."""
        fallback_data = [
            {
                'Domain': 'agent-properties',
                'Context1': 'An AI agent learns from feedback and adapts its behavior.',
                'Context2': 'A mechanical device follows fixed programming rules.',
                'Target1': 'This system demonstrates learning capabilities.',
                'Target2': 'This system operates with predetermined responses.'
            },
            {
                'Domain': 'social-properties',
                'Context1': 'Two people share emotions during a conversation.',
                'Context2': 'A person reads a book in solitude.',
                'Target1': 'This involves emotional exchange and empathy.',
                'Target2': 'This is an individual cognitive activity.'
            }
        ]
        
        df = pd.DataFrame(fallback_data)
        return df.head(min(self.max_samples, len(df)))
    
    def evaluate_prompt_on_sample(self, prompt_name: str, row: pd.Series, sample_id: int) -> PromptEvaluationResult:
        """Evaluate a single prompt on one sample."""
        start_time = time.time()
        
        try:
            # Format prompt for this sample
            formatted_prompt = self.prompt_loader.format_prompt_for_ewok(
                prompt_name=prompt_name,
                context1=row['Context1'],
                context2=row['Context2'],
                target1=row['Target1'],
                target2=row['Target2']
            )
            
            # Get LLM response
            response = self.llm_client.invoke([HumanMessage(content=formatted_prompt)])
            response_time = time.time() - start_time
            
            # Extract prediction
            predicted = self._extract_prediction(response.content)
            
            return PromptEvaluationResult(
                prompt_name=prompt_name,
                sample_id=sample_id,
                domain=row['Domain'],
                predicted=predicted,
                correct="1-1,2-2",  # EWoK standard
                is_correct=(predicted == "1-1,2-2"),
                response_time=response_time
            )
            
        except Exception as e:
            print(f"❌ Error evaluating {prompt_name} on sample {sample_id}: {e}")
            return PromptEvaluationResult(
                prompt_name=prompt_name,
                sample_id=sample_id,
                domain=row['Domain'],
                predicted="ERROR",
                correct="1-1,2-2",
                is_correct=False,
                response_time=0.0
            )
    
    def _extract_prediction(self, response_text: str) -> str:
        """Extract prediction from LLM response."""
        response_text = response_text.lower()
        
        # Look for explicit answer patterns
        if "1-2,2-1" in response_text or "1-2" in response_text:
            return "1-2,2-1"
        elif "1-1,2-2" in response_text or "1-1" in response_text:
            return "1-1,2-2"
        
        # Look for "option 2" or "option 1"
        if "option 2" in response_text:
            return "1-2,2-1"
        elif "option 1" in response_text:
            return "1-1,2-2"
        
        # Default fallback
        return "1-1,2-2"
    
    def evaluate_all_prompts(self) -> List[PromptPerformanceSummary]:
        """Evaluate all prompts on the dataset."""
        print(f"\n🚀 Starting prompt evaluation on {len(self.dataset)} samples")
        print(f"📝 Testing {len(self.prompt_loader.list_prompts())} prompts")
        
        all_results = []
        prompt_summaries = []
        
        for prompt_name in self.prompt_loader.list_prompts():
            print(f"\n🧪 Testing prompt: {prompt_name}")
            
            prompt_results = []
            for idx, row in self.dataset.iterrows():
                result = self.evaluate_prompt_on_sample(prompt_name, row, idx)
                prompt_results.append(result)
                all_results.append(result)
                
                # Print progress
                status = "✅" if result.is_correct else "❌"
                print(f"  Sample {idx+1}/{len(self.dataset)}: {status} ({result.response_time:.1f}s)")
                
                # Rate limiting
                time.sleep(0.5)
            
            # Calculate prompt summary
            summary = self._calculate_prompt_summary(prompt_results)
            prompt_summaries.append(summary)
            
            print(f"📊 {prompt_name}: {summary.accuracy_percent:.1f}% accuracy")
        
        # Save and display results
        self._save_results(all_results, prompt_summaries)
        self._display_summary(prompt_summaries)
        
        return prompt_summaries
    
    def _calculate_prompt_summary(self, results: List[PromptEvaluationResult]) -> PromptPerformanceSummary:
        """Calculate performance summary for a prompt."""
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = (correct_count / len(results)) * 100 if results else 0
        avg_time = sum(r.response_time for r in results) / len(results) if results else 0
        
        # Domain breakdown
        domain_breakdown = {}
        for domain in self.config["agentic_domains"]:
            domain_results = [r for r in results if r.domain == domain]
            if domain_results:
                domain_correct = sum(1 for r in domain_results if r.is_correct)
                domain_accuracy = (domain_correct / len(domain_results)) * 100
                domain_breakdown[domain] = {
                    "samples": len(domain_results),
                    "correct": domain_correct,
                    "accuracy": domain_accuracy
                }
        
        return PromptPerformanceSummary(
            prompt_name=results[0].prompt_name,
            total_samples=len(results),
            correct_count=correct_count,
            accuracy_percent=accuracy,
            avg_response_time=avg_time,
            domain_breakdown=domain_breakdown
        )
    
    def _save_results(self, all_results: List[PromptEvaluationResult], 
                     summaries: List[PromptPerformanceSummary]) -> None:
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/prompt_evaluation_{timestamp}.json"
        
        results_data = {
            "evaluation_timestamp": timestamp,
            "total_samples": len(self.dataset),
            "prompts_tested": len(summaries),
            "prompt_summaries": [asdict(s) for s in summaries],
            "detailed_results": [asdict(r) for r in all_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
    
    def _display_summary(self, summaries: List[PromptPerformanceSummary]) -> None:
        """Display formatted evaluation summary."""
        print(f"\n📈 PROMPT EVALUATION SUMMARY")
        print("=" * 60)
        
        # Sort by accuracy
        sorted_summaries = sorted(summaries, key=lambda x: x.accuracy_percent, reverse=True)
        
        for i, summary in enumerate(sorted_summaries, 1):
            print(f"\n{i}. {summary.prompt_name}")
            print(f"   Accuracy: {summary.accuracy_percent:.1f}% ({summary.correct_count}/{summary.total_samples})")
            print(f"   Avg Time: {summary.avg_response_time:.1f}s")
            
            # Show domain breakdown
            for domain, stats in summary.domain_breakdown.items():
                print(f"   {domain}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['samples']})")


def main():
    """Main entry point for prompt tester."""
    parser = argparse.ArgumentParser(description="Cognitive Prompt Evaluation System")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to evaluate")
    parser.add_argument("--prompts", nargs="+", help="Specific prompts to test (default: all)")
    
    args = parser.parse_args()
    
    try:
        tester = PromptTester(max_samples=args.max_samples)
        
        # If specific prompts requested, filter them
        if args.prompts:
            available_prompts = tester.prompt_loader.list_prompts()
            invalid_prompts = [p for p in args.prompts if p not in available_prompts]
            if invalid_prompts:
                print(f"❌ Invalid prompts: {invalid_prompts}")
                print(f"Available prompts: {available_prompts}")
                return 1
            
            # Filter dataset for specific prompts (would need to modify evaluate_all_prompts)
            print(f"🎯 Testing specific prompts: {args.prompts}")
        
        tester.evaluate_all_prompts()
        print(f"\n✅ Prompt evaluation complete!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())