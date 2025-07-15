import os
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import datetime
import sys
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
from collections import defaultdict
import pickle

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

@dataclass
class ResponseMemory:
    instruction_id: str
    domain: str
    concepts: Tuple[str, str]
    context1: str
    context2: str
    scenario: str
    model_response: str
    correct_choice: str
    model_choice: str
    was_correct: bool
    score: float
    timestamp: str
    usage_count: int = 0

@dataclass
class ResponsePattern:
    pattern_type: str
    concepts: Tuple[str, str]
    instruction_examples: List[str]
    response_choices: List[str]
    correct_responses: List[bool]
    accuracy_rate: float
    frequency: int

class MemoryManager:
    def __init__(self, memory_file: str = "memory_state.pkl", max_response_items: int = 1000, fresh_start: bool = True):
        self.memory_file = memory_file
        self.max_response_items = max_response_items
        self.fresh_start = fresh_start
        
        # Domain-specific memory limits
        # Remove memory limits - store everything
        self.domain_memory_limits = {
            'agent-properties': float('inf'),  # No limit
            'social-interactions': float('inf'),  # No limit
            'social-properties': float('inf')   # No limit
        }
        
        self.response_memory: List[ResponseMemory] = []
        self.response_patterns: Dict[str, ResponsePattern] = {}
        self.instruction_similarities: defaultdict = defaultdict(list)
        
        if not fresh_start:
            self.load_memory()

    def _generate_instruction_id(self, instruction: str, domain: str) -> str:
        return hashlib.md5(f"{domain}:{instruction}".encode()).hexdigest()[:8]
    
    def _enforce_domain_memory_limits(self):
        """Enforce per-domain memory limits with LRU eviction."""
        # Group memories by domain
        domain_memories = defaultdict(list)
        for memory in self.response_memory:
            domain_memories[memory.domain].append(memory)
        
        # Check each domain's limit
        memories_to_keep = []
        for domain, memories in domain_memories.items():
            limit = self.domain_memory_limits.get(domain, 100)  # Default 100 if domain not configured
            
            if len(memories) > limit:
                # Sort by usage count and timestamp (LRU with usage priority)
                memories.sort(key=lambda x: (x.usage_count, x.timestamp), reverse=True)
                memories = memories[:limit]
            
            memories_to_keep.extend(memories)
        
        self.response_memory = memories_to_keep

    def store_response_memory(self, context1: str, context2: str, scenario: str, domain: str, concepts: Tuple[str, str],
                            model_response: str, correct_choice: str, model_choice: str, score: float):
        instruction_id = self._generate_instruction_id(f"{context1}|{context2}|{scenario}", domain)
        
        response_item = ResponseMemory(
            instruction_id=instruction_id,
            domain=domain,
            concepts=concepts,
            context1=context1,
            context2=context2,
            scenario=scenario,
            model_response=model_response,
            correct_choice=correct_choice,
            model_choice=model_choice,
            was_correct=(model_choice == correct_choice),
            score=score,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        existing_idx = None
        for i, item in enumerate(self.response_memory):
            if item.instruction_id == instruction_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.response_memory[existing_idx] = response_item
        else:
            self.response_memory.append(response_item)
            self._enforce_domain_memory_limits()

    def update_response_patterns(self, domain: str, concepts: Tuple[str, str], 
                               instruction: str, model_choice: str, was_correct: bool):
        pattern_key = f"{domain}:{concepts[0]}_{concepts[1]}"
        
        if pattern_key in self.response_patterns:
            pattern = self.response_patterns[pattern_key]
            pattern.instruction_examples.append(instruction[:100])
            pattern.response_choices.append(model_choice)
            pattern.correct_responses.append(was_correct)
            pattern.frequency += 1
            
            pattern.accuracy_rate = sum(pattern.correct_responses) / len(pattern.correct_responses)
            
            if len(pattern.instruction_examples) > 10:
                pattern.instruction_examples = pattern.instruction_examples[-10:]
                pattern.response_choices = pattern.response_choices[-10:]
                pattern.correct_responses = pattern.correct_responses[-10:]
        else:
            self.response_patterns[pattern_key] = ResponsePattern(
                pattern_type=domain,
                concepts=concepts,
                instruction_examples=[instruction[:100]],
                response_choices=[model_choice],
                correct_responses=[was_correct],
                accuracy_rate=1.0 if was_correct else 0.0,
                frequency=1
            )

    def calculate_instruction_similarity(self, instruction1: str, instruction2: str) -> float:
        """Calculate similarity between two instructions using word overlap and structure."""
        words1 = set(instruction1.lower().split())
        words2 = set(instruction2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        word_overlap = len(words1 & words2) / len(words1 | words2)
        
        # Boost similarity for structural patterns
        structure_boost = 0
        if "which context makes more sense" in instruction1.lower() and "which context makes more sense" in instruction2.lower():
            structure_boost += 0.3
        if "context" in instruction1.lower() and "context" in instruction2.lower():
            structure_boost += 0.2
        
        return min(1.0, word_overlap + structure_boost)
    
    def retrieve_similar_responses(self, current_instruction: str, domain: str, 
                                 concepts: Tuple[str, str], max_items: int = 3) -> List[ResponseMemory]:
        """Retrieve past responses to similar instructions."""
        similar_responses = []
        
        for response in self.response_memory:
            relevance_score = 0
            
            # Domain match bonus
            if response.domain == domain:
                relevance_score += 2
            
            # Concept overlap bonus
            if set(concepts) & set(response.concepts):
                relevance_score += 3
            
            # Instruction similarity using contexts and scenario
            response_instruction = f"{response.context1}|{response.context2}|{response.scenario}"
            instruction_similarity = self.calculate_instruction_similarity(
                current_instruction, response_instruction
            )
            relevance_score += instruction_similarity * 4
            
            # Boost for correct responses (we want to learn from successes)
            if response.was_correct:
                relevance_score += 1
            
            if relevance_score > 2.0:  # Higher threshold for instruction similarity
                response.usage_count += 1
                similar_responses.append((response, relevance_score))
        
        similar_responses.sort(key=lambda x: x[1], reverse=True)
        return [response for response, score in similar_responses[:max_items]]

    def get_response_insights(self, domain: str, concepts: Tuple[str, str]) -> str:
        """Get insights about past response patterns for similar tasks."""
        pattern_key = f"{domain}:{concepts[0]}_{concepts[1]}"
        insights = []
        
        if pattern_key in self.response_patterns:
            pattern = self.response_patterns[pattern_key]
            insights.append(f"Past experience: {pattern.frequency} similar tasks")
            insights.append(f"Success rate: {pattern.accuracy_rate:.1%}")
            
            # Analysis of successful choices
            correct_choices = [choice for choice, correct in zip(pattern.response_choices, pattern.correct_responses) if correct]
            if correct_choices:
                choice_counts = defaultdict(int)
                for choice in correct_choices:
                    choice_counts[choice] += 1
                
                if choice_counts:
                    best_choice = max(choice_counts.items(), key=lambda x: x[1])
                    insights.append(f"Most successful choice: '{best_choice[0]}' ({best_choice[1]} times correct)")
        
        # Look for related patterns across similar concept combinations
        related_patterns = [p for key, p in self.response_patterns.items() 
                          if any(c in concepts for c in p.concepts) and key != pattern_key]
        
        if related_patterns:
            avg_accuracy = sum(p.accuracy_rate for p in related_patterns) / len(related_patterns)
            insights.append(f"Related tasks accuracy: {avg_accuracy:.1%} ({len(related_patterns)} patterns)")
        
        return " | ".join(insights) if insights else "No prior response patterns found."

    def compress_old_responses(self, days_threshold: int = 7):
        """Compress old response memories by converting them to patterns."""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_threshold)
        
        to_compress = []
        remaining = []
        
        for response in self.response_memory:
            response_date = datetime.datetime.fromisoformat(response.timestamp)
            if response_date < cutoff_date and response.usage_count < 2:
                to_compress.append(response)
            else:
                remaining.append(response)
        
        # Group compressed responses by domain and update patterns
        for response in to_compress:
            pattern_key = f"compressed:{response.domain}"
            if pattern_key not in self.response_patterns:
                self.response_patterns[pattern_key] = ResponsePattern(
                    pattern_type="compressed",
                    concepts=response.concepts,
                    instruction_examples=[],
                    response_choices=[],
                    correct_responses=[],
                    accuracy_rate=0.0,
                    frequency=0
                )
            
            pattern = self.response_patterns[pattern_key]
            pattern.instruction_examples.append(response.instruction_text[:50])
            pattern.response_choices.append(response.model_choice)
            pattern.correct_responses.append(response.was_correct)
            pattern.frequency += 1
            pattern.accuracy_rate = sum(pattern.correct_responses) / len(pattern.correct_responses)
        
        self.response_memory = remaining

    def save_memory(self):
        memory_state = {
            'response_memory': [asdict(item) for item in self.response_memory],
            'response_patterns': {k: asdict(v) for k, v in self.response_patterns.items()},
            'instruction_similarities': dict(self.instruction_similarities)
        }
        
        with open(self.memory_file, 'wb') as f:
            pickle.dump(memory_state, f)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    memory_state = pickle.load(f)
                
                self.response_memory = [ResponseMemory(**item) for item in memory_state.get('response_memory', [])]
                
                pattern_data = memory_state.get('response_patterns', {})
                self.response_patterns = {k: ResponsePattern(**v) for k, v in pattern_data.items()}
                
                self.instruction_similarities = defaultdict(list, memory_state.get('instruction_similarities', {}))
                
            except Exception as e:
                print(f"Warning: Could not load memory state: {e}")
                self.response_memory = []
                self.response_patterns = {}
                self.instruction_similarities = defaultdict(list)

def extract_choice(text):
    text = text.strip()
    match = re.search(r"[12]", text)
    if match:
        return match.group(0)
    return "UNKNOWN"

def call_with_retry(prompt, model_name, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2 
            else:
                return f"ERROR: {e}"


def create_response_augmented_prompt(base_prompt: str, similar_responses: List[ResponseMemory], 
                                   response_insights: str) -> str:
    """Create a prompt augmented with past response examples and insights."""
    if not similar_responses and not response_insights:
        return base_prompt
    
    memory_context = "# Learning from Past Responses\n\n"
    
    if similar_responses:
        memory_context += "## Similar Past Examples & Outcomes:\n\n"
        for i, response in enumerate(similar_responses[:3], 1):  # Limit to exactly 3
            outcome = "✓ CORRECT" if response.was_correct else "✗ INCORRECT"
            
            memory_context += f"**Past Example {i}** ({outcome})\n"
            memory_context += f"## Contexts\n"
            memory_context += f"1. \"{response.context1}\"\n"
            memory_context += f"2. \"{response.context2}\"\n\n"
            memory_context += f"## Scenario\n"
            memory_context += f"\"{response.scenario}\"\n\n"
            memory_context += f"## Task\n"
            memory_context += f"Which context makes more sense given the scenario? Please answer using either \"1\" or \"2\".\n\n"
            memory_context += f"## My Previous Response\n"
            memory_context += f"I chose: '{response.model_choice}'\n"
            
            if response.was_correct:
                memory_context += f"✓ **This was CORRECT!** Choice '{response.model_choice}' was the right answer.\n\n"
            else:
                memory_context += f"✗ **This was WRONG!** I should have chosen '{response.correct_choice}' instead.\n\n"
            
            memory_context += "---\n\n"  # Separator between examples
    
    # Remove success patterns section
    # if response_insights and response_insights != "No prior response patterns found.":
    #     memory_context += f"## Success Patterns:\n{response_insights}\n\n"
    
    memory_context += "# Current Question\n\n"
    
    return memory_context + base_prompt

RESPONSE_ENHANCED_STUDY_PROMPT = """# INSTRUCTIONS

In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense considering the scenario that follows. The contexts will be numbered "1" or "2". You must answer using "1" or "2" in your response.

Learn from the past response examples shown above to make better decisions.

# TEST EXAMPLE
## Contexts
1. "{context1}"
2. "{context2}"

## Scenario
"{target}"

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
"""

def evaluate_split_with_response_memory(df, split_name, model_name, max_items, memory_manager):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    def process_row_with_response_memory(row):
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]
        concepts = (row["ConceptA"], row["ConceptB"])
        
        # Create instruction summaries for memory retrieval
        instruction1 = f"{c1}|{c2}|{t1}"
        instruction2 = f"{c1}|{c2}|{t2}"
        
        # Retrieve similar past responses
        similar_responses1 = memory_manager.retrieve_similar_responses(instruction1, split_name, concepts)
        similar_responses2 = memory_manager.retrieve_similar_responses(instruction2, split_name, concepts)
        
        # Get response insights
        response_insights = memory_manager.get_response_insights(split_name, concepts)

        def run_response_prompt(target, context1, context2, instruction, similar_responses):
            base_prompt = RESPONSE_ENHANCED_STUDY_PROMPT.format(
                context1=context1, context2=context2, target=target
            )
            
            response_augmented_prompt = create_response_augmented_prompt(
                base_prompt, similar_responses, response_insights
            )
            
            final_answer = call_with_retry(response_augmented_prompt, model_name)
            choice = extract_choice(final_answer)
            return response_augmented_prompt, final_answer, choice

        r1_prompt, r1_text, choice1 = run_response_prompt(t1, c1, c2, instruction1, similar_responses1)
        r2_prompt, r2_text, choice2 = run_response_prompt(t2, c1, c2, instruction2, similar_responses2)
        
        score = 0
        if choice1 == "1":
            score += 0.5
        if choice2 == "2":
            score += 0.5
        
        # Store the responses for future learning
        memory_manager.store_response_memory(c1, c2, t1, split_name, concepts, r1_text, "1", choice1, 0.5 if choice1 == "1" else 0.0)
        memory_manager.store_response_memory(c1, c2, t2, split_name, concepts, r2_text, "2", choice2, 0.5 if choice2 == "2" else 0.0)
        
        # Update response patterns
        memory_manager.update_response_patterns(split_name, concepts, instruction1, choice1, choice1 == "1")
        memory_manager.update_response_patterns(split_name, concepts, instruction2, choice2, choice2 == "2")

        return {
            "model": model_name,
            "split": split_name,
            "context a": c1,
            "context b": c2,
            "answer a": t1,
            "answer b": t2,
            "chosen answer 1": choice1,
            "chosen answer 2": choice2,
            "score": score,
            "prompt 1": r1_prompt,
            "prompt 2": r2_prompt,
            "model answer 1": r1_text,
            "model answer 2": r2_text,
            "similar_responses_used": len(similar_responses1 + similar_responses2),
            "response_insights": response_insights
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row_with_response_memory, row) for _, row in df_split.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {split_name} with response memory"):
            results.append(future.result())

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM choices using long-term memory.")
    parser.add_argument('--max_items', type=int, default=100, help='Maximum number of items per split (default: 100)')
    parser.add_argument('--model', type=str, default='mistralai/mistral-7b-instruct', help='Model name to use')
    parser.add_argument('--memory_file', type=str, default='memory_state.pkl', help='Memory persistence file')
    parser.add_argument('--compress_memory', action='store_true', help='Compress old memories before evaluation')
    parser.add_argument('--load_previous', action='store_true', help='Load memories from previous runs (default: fresh start)')
    args = parser.parse_args()

    run_args = ' '.join(sys.argv[1:])
    
    memory_manager = MemoryManager(memory_file=args.memory_file, fresh_start=not args.load_previous)
    
    if args.compress_memory:
        print("Compressing old response memories...")
        memory_manager.compress_old_responses()
    
    df = pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")
    
    model_name = args.model
    splits = ["agent-properties", "social-interactions", "social-properties"]
    
    all_results = pd.concat(
        [
            evaluate_split_with_response_memory(
                df, s, model_name, max_items=args.max_items, memory_manager=memory_manager
            )
            for s in splits
        ],
        ignore_index=True,
    )
    
    memory_manager.save_memory()
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_file = os.path.join("results", f"{timestamp}_ewok_eval_memory.csv")
    
    all_results["arguments"] = run_args
    all_results.to_csv(output_file, index=False)
    
    print("✅ Response-enhanced evaluation complete. Saved to", output_file)
    print("Average Accuracy:", all_results["score"].mean())
    print("Response memories stored:", len(memory_manager.response_memory))
    
    # Domain breakdown
    from collections import Counter
    domain_counts = Counter(m.domain for m in memory_manager.response_memory)
    for domain, count in domain_counts.items():
        limit = memory_manager.domain_memory_limits.get(domain, "N/A")
        print(f"  - {domain}: {count}/{limit} memories")
    
    print("Response patterns learned:", len(memory_manager.response_patterns))

if __name__ == "__main__":
    main()