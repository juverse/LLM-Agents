#!/usr/bin/env python3
"""
Cognitive System Evaluator - Clean Architecture Implementation

This module provides a clean, maintainable implementation of a cognitive evaluation system
that uses LangChain agents to evaluate EWoK dataset samples through chained cognitive tools.

Key Design Principles:
- Single Responsibility: Each class has one clear purpose
- Dependency Injection: LLM client injected for testability
- Configuration Management: Environment-based configuration
- Error Handling: Comprehensive exception handling
- Type Safety: Full type annotations
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


@dataclass
class EvaluationConfig:
    """Configuration settings for the cognitive evaluation system."""
    api_key: str
    model_name: str
    max_samples: int
    default_iterations: int
    timeout: int
    dataset_path: str
    agentic_domains: List[str]
    results_dir: str
    save_detailed: bool


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single sample."""
    iteration: int
    domain: str
    predicted: str
    correct: str
    is_correct: bool
    response_time: float
    agent_output: str
    sample_id: Optional[str] = None


class ConfigurationManager:
    """Manages configuration loading from environment variables and .env file."""
    
    @staticmethod
    def load_config() -> EvaluationConfig:
        """Load configuration from environment variables."""
        # Load .env file if it exists
        load_dotenv()
        
        # Validate required environment variables
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Please set it in your environment or .env file."
            )
        
        # Parse agentic domains from comma-separated string
        domains_str = os.getenv("AGENTIC_DOMAINS", "agent-properties,social-interactions,social-properties")
        agentic_domains = [domain.strip() for domain in domains_str.split(",")]
        
        return EvaluationConfig(
            api_key=api_key,
            model_name=os.getenv("DEFAULT_LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
            max_samples=int(os.getenv("MAX_SAMPLES", "3")),
            default_iterations=int(os.getenv("DEFAULT_ITERATIONS", "1")),
            timeout=int(os.getenv("RESPONSE_TIMEOUT", "30")),
            dataset_path=os.getenv("DATASET_PATH", "hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet"),
            agentic_domains=agentic_domains,
            results_dir=os.getenv("RESULTS_DIR", "results"),
            save_detailed=os.getenv("SAVE_DETAILED_RESULTS", "true").lower() == "true"
        )


class DatasetLoader:
    """Handles loading and filtering of the EWoK dataset."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def load_and_filter_dataset(self) -> pd.DataFrame:
        """Load EWoK dataset and filter for agentic variables."""
        print(f"📥 Loading EWoK dataset from: {self.config.dataset_path}")
        
        try:
            if self.config.dataset_path == "fallback":
                return self._create_fallback_dataset()
            
            # Load full dataset
            df = pd.read_parquet(self.config.dataset_path)
            
            # Filter for agentic domains
            filtered_df = df[df['Domain'].isin(self.config.agentic_domains)].copy()
            
            # Sample data if max_samples is specified
            if self.config.max_samples > 0:
                filtered_df = filtered_df.sample(
                    n=min(self.config.max_samples, len(filtered_df)), 
                    random_state=42
                )
            
            print(f"✅ Loaded {len(filtered_df)} samples from agentic domains")
            self._print_domain_distribution(filtered_df)
            
            return filtered_df.reset_index(drop=True)
            
        except Exception as e:
            print(f"⚠️ Error loading dataset: {e}")
            print("🔄 Using fallback dataset instead...")
            return self._create_fallback_dataset()
    
    def _create_fallback_dataset(self) -> pd.DataFrame:
        """Create a fallback dataset for testing purposes."""
        fallback_data = [
            {
                'Domain': 'agent-properties',
                'Context1': 'An AI agent processes information and makes decisions.',
                'Context2': 'A simple calculator only follows predefined rules.',
                'Target1': 'This system can adapt to new situations.',
                'Target2': 'This system operates with fixed programming.'
            },
            {
                'Domain': 'social-properties', 
                'Context1': 'Two people are having a conversation about their feelings.',
                'Context2': 'A person is reading a book alone in their room.',
                'Target1': 'This involves emotional exchange and understanding.',
                'Target2': 'This is a solitary intellectual activity.'
            },
            {
                'Domain': 'agent-properties',
                'Context1': 'A robot learns from its mistakes and improves performance.',
                'Context2': 'A traffic light changes colors on a fixed schedule.',
                'Target1': 'This demonstrates learning and adaptation capabilities.',
                'Target2': 'This follows a predetermined pattern without learning.'
            }
        ]
        
        df = pd.DataFrame(fallback_data)
        
        # Sample data if max_samples is specified
        if self.config.max_samples > 0:
            df = df.head(self.config.max_samples)
        
        print(f"✅ Created fallback dataset with {len(df)} samples")
        self._print_domain_distribution(df)
        
        return df.reset_index(drop=True)
    
    def _print_domain_distribution(self, df: pd.DataFrame) -> None:
        """Print the distribution of samples across domains."""
        for domain in self.config.agentic_domains:
            count = len(df[df['Domain'] == domain])
            print(f"   • {domain}: {count} samples")
    



class CognitiveToolsFactory:
    """Factory for creating cognitive processing tools."""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.llm_client = llm_client
    
    def create_working_memory_tool(self):
        """Create the working memory extraction tool."""
        @tool
        def working_memory_tool(context: str) -> str:
            """
            Extract key entities, relationships, and facts from context.
            
            This tool simulates working memory by identifying and structuring
            the important information from the given context.
            """
            prompt = self._build_working_memory_prompt(context)
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            return response.content
        
        return working_memory_tool
    
    def create_reasoning_tool(self):
        """Create the reasoning and decision-making tool."""
        @tool  
        def reasoning_tool(context1: str, context2: str, target1: str, target2: str, memory_state: str) -> str:
            """
            Reason about context-target mappings using working memory state.
            
            This tool takes the working memory output and uses it to determine
            the best mapping between contexts and targets.
            """
            prompt = self._build_reasoning_prompt(context1, context2, target1, target2, memory_state)
            response = self.llm_client.invoke([HumanMessage(content=prompt)])
            return response.content
        
        return reasoning_tool
    
    def _build_working_memory_prompt(self, context: str) -> str:
        """Build the prompt for working memory extraction."""
        return f"""Extract key information from this context:

Context: {context}

Please identify and return:
- Main entities (people, objects, concepts)
- Key relationships between entities
- Important facts and properties
- Context summary in one sentence

Keep the response brief and well-structured."""
    
    def _build_reasoning_prompt(self, context1: str, context2: str, target1: str, target2: str, memory_state: str) -> str:
        """Build the prompt for reasoning and decision-making."""
        return f"""Given the working memory state and contexts/targets, determine the best mapping.

Working Memory Analysis: {memory_state}

Context 1: {context1}
Context 2: {context2}
Target 1: {target1}
Target 2: {target2}

Evaluate both possible mappings:
1. Context1→Target1, Context2→Target2 (mapping: 1-1,2-2)
2. Context1→Target2, Context2→Target1 (mapping: 1-2,2-1)

Which context matches which target based on logical consistency and world knowledge?
Answer with either "1-1,2-2" or "1-2,2-1" and provide brief reasoning."""


class LLMClientFactory:
    """Factory for creating LLM clients with proper configuration."""
    
    @staticmethod
    def create_openrouter_client(config: EvaluationConfig) -> ChatOpenAI:
        """Create and configure OpenRouter LLM client."""
        print(f"🧠 Initializing OpenRouter client with model: {config.model_name}")
        
        return ChatOpenAI(
            model=config.model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=config.api_key,
            temperature=0.1,
            request_timeout=config.timeout
        )


class CognitiveAgent:
    """Manages the cognitive agent with chained tools."""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.tools_factory = CognitiveToolsFactory(llm_client)
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with cognitive tools."""
        # Create tools
        working_memory_tool = self.tools_factory.create_working_memory_tool()
        reasoning_tool = self.tools_factory.create_reasoning_tool()
        tools = [working_memory_tool, reasoning_tool]
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create and return agent executor
        agent = create_openai_tools_agent(self.llm_client, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the cognitive agent."""
        return """You are a cognitive reasoning system with two main capabilities:

1. Working Memory: Extract and structure information from contexts
2. Reasoning: Make logical decisions based on working memory state

Use the working_memory_tool first to analyze the contexts, then use the reasoning_tool 
to determine the best context-target mapping. Always follow this two-step process."""
    
    def evaluate_sample(self, contexts: Tuple[str, str], targets: Tuple[str, str]) -> str:
        """Evaluate a sample using the cognitive agent."""
        agent_input = f"""
Context 1: {contexts[0]}
Context 2: {contexts[1]}
Target 1: {targets[0]}
Target 2: {targets[1]}

Use working memory to extract information, then reasoning to determine the best context-target mapping.
"""
        
        result = self.agent_executor.invoke({"input": agent_input})
        return result["output"]


class ResultsManager:
    """Manages saving and formatting of evaluation results."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._ensure_results_directory()
    
    def _ensure_results_directory(self) -> None:
        """Ensure the results directory exists."""
        Path(self.config.results_dir).mkdir(exist_ok=True)
    
    def save_results(self, summary: Dict[str, Any]) -> str:
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(self.config.results_dir) / f"cognitive_eval_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return str(filename)
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted evaluation summary."""
        print(f"\n📈 COGNITIVE EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Iterations: {summary['total_iterations']}")
        print(f"Overall Accuracy: {summary['overall_accuracy_percent']:.1f}%")
        print(f"Total Correct: {summary['overall_correct']}/{summary['total_evaluations']}")
        print(f"Average Response Time: {summary['overall_avg_time']:.2f}s")
        
        print(f"\n📊 Per-Iteration Breakdown:")
        for iter_summary in summary['iteration_summaries']:
            print(f"  Iteration {iter_summary['iteration']}: {iter_summary['accuracy_percent']:.1f}% accuracy")


class CognitiveEvaluator:
    """Main cognitive evaluation system orchestrator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm_client = LLMClientFactory.create_openrouter_client(config)
        self.dataset_loader = DatasetLoader(config)
        self.cognitive_agent = CognitiveAgent(self.llm_client)
        self.results_manager = ResultsManager(config)
        
        # Load dataset
        self.dataset = self.dataset_loader.load_and_filter_dataset()
    
    def evaluate_single_sample(self, row: pd.Series, iteration: int) -> EvaluationResult:
        """Evaluate a single sample and return structured result."""
        start_time = time.time()
        
        try:
            # Use cognitive agent to evaluate
            agent_output = self.cognitive_agent.evaluate_sample(
                contexts=(row['Context1'], row['Context2']),
                targets=(row['Target1'], row['Target2'])
            )
            
            # Extract prediction from agent output
            predicted_answer = self._extract_prediction(agent_output)
            response_time = time.time() - start_time
            
            return EvaluationResult(
                iteration=iteration,
                domain=row['Domain'],
                predicted=predicted_answer,
                correct="1-1,2-2",  # EWoK standard
                is_correct=(predicted_answer == "1-1,2-2"),
                response_time=response_time,
                agent_output=agent_output
            )
            
        except Exception as e:
            print(f"❌ Error evaluating sample: {e}")
            return EvaluationResult(
                iteration=iteration,
                domain=row['Domain'],
                predicted="ERROR",
                correct="1-1,2-2",
                is_correct=False,
                response_time=0.0,
                agent_output=f"Error: {str(e)}"
            )
    
    def _extract_prediction(self, agent_output: str) -> str:
        """Extract the prediction from agent output."""
        if "1-2,2-1" in agent_output:
            return "1-2,2-1"
        return "1-1,2-2"  # Default
    
    def run_evaluation(self, iterations: int) -> Dict[str, Any]:
        """Run complete evaluation across all iterations."""
        print(f"\n🚀 Starting cognitive evaluation with {iterations} iteration(s)")
        
        all_results = []
        iteration_summaries = []
        
        for iteration in range(1, iterations + 1):
            print(f"\n🔄 Iteration {iteration}/{iterations}")
            iteration_results = self._run_single_iteration(iteration)
            
            all_results.extend(iteration_results)
            iteration_summaries.append(self._calculate_iteration_summary(iteration_results, iteration))
        
        # Calculate overall summary
        summary = self._calculate_overall_summary(all_results, iteration_summaries, iterations)
        
        # Save and display results
        if self.config.save_detailed:
            self.results_manager.save_results(summary)
        
        self.results_manager.print_summary(summary)
        
        return summary
    
    def _run_single_iteration(self, iteration: int) -> List[EvaluationResult]:
        """Run evaluation for a single iteration."""
        iteration_results = []
        
        for idx, row in self.dataset.iterrows():
            print(f"📝 Sample {idx+1}/{len(self.dataset)} - {row['Domain']}")
            
            result = self.evaluate_single_sample(row, iteration)
            iteration_results.append(result)
            
            # Print result
            status = "✅" if result.is_correct else "❌"
            print(f"   {status} Predicted: {result.predicted}, Time: {result.response_time:.2f}s")
            
            # Rate limiting
            time.sleep(1)
        
        return iteration_results
    
    def _calculate_iteration_summary(self, results: List[EvaluationResult], iteration: int) -> Dict[str, Any]:
        """Calculate summary for a single iteration."""
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = (correct_count / len(results)) * 100 if results else 0
        avg_time = sum(r.response_time for r in results) / len(results) if results else 0
        
        summary = {
            "iteration": iteration,
            "accuracy_percent": accuracy,
            "correct_count": correct_count,
            "total_samples": len(results),
            "avg_response_time": avg_time
        }
        
        print(f"📊 Iteration {iteration} Summary: {accuracy:.1f}% accuracy ({correct_count}/{len(results)})")
        return summary
    
    def _calculate_overall_summary(self, all_results: List[EvaluationResult], 
                                 iteration_summaries: List[Dict[str, Any]], 
                                 iterations: int) -> Dict[str, Any]:
        """Calculate overall evaluation summary."""
        overall_correct = sum(1 for r in all_results if r.is_correct)
        overall_accuracy = (overall_correct / len(all_results)) * 100 if all_results else 0
        overall_avg_time = sum(r.response_time for r in all_results) / len(all_results) if all_results else 0
        
        return {
            "total_iterations": iterations,
            "overall_accuracy_percent": overall_accuracy,
            "overall_correct": overall_correct,
            "total_evaluations": len(all_results),
            "overall_avg_time": overall_avg_time,
            "iteration_summaries": iteration_summaries,
            "detailed_results": [asdict(r) for r in all_results] if self.config.save_detailed else []
        }


def main():
    """Main entry point for the cognitive evaluator."""
    parser = argparse.ArgumentParser(description="Cognitive System Evaluator with Clean Architecture")
    parser.add_argument("--max_samples", type=int, help="Max samples to evaluate (overrides env)")
    parser.add_argument("--iterations", type=int, help="Number of evaluation iterations (overrides env)")
    parser.add_argument("--model", type=str, help="LLM model to use (overrides env)")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ConfigurationManager.load_config()
        
        # Override with command line arguments if provided
        if args.max_samples is not None:
            config.max_samples = args.max_samples
        if args.model is not None:
            config.model_name = args.model
        
        iterations = args.iterations if args.iterations is not None else config.default_iterations
        
        # Create and run evaluator
        evaluator = CognitiveEvaluator(config)
        results = evaluator.run_evaluation(iterations)
        
        print(f"\n✅ Cognitive evaluation complete!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())