import os
import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
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
    def __init__(self, memory_file: str = "response_memory.pkl"):
        self.memory_file = memory_file
        self.response_memory: List[ResponseMemory] = []
        self.response_patterns: Dict[str, ResponsePattern] = {}
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.response_memory = data.get('response_memory', [])
                    self.response_patterns = data.get('response_patterns', {})
            except:
                pass

    def save_memory(self):
        with open(self.memory_file, 'wb') as f:
            pickle.dump({
                'response_memory': self.response_memory,
                'response_patterns': self.response_patterns
            }, f)

    def calculate_instruction_similarity(self, inst1: str, inst2: str) -> float:
        words1 = set(inst1.lower().split())
        words2 = set(inst2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def get_relevant_contexts(self, domain: str, concepts: Tuple[str, str], 
                            context1: str, context2: str, scenario: str, max_items: int = 3) -> List[ResponseMemory]:
        current_instruction = f"{context1}|{context2}|{scenario}"
        similar_responses = []
        
        for response in self.response_memory:
            relevance_score = 0
            
            if response.domain == domain:
                relevance_score += 2
            
            if set(concepts) & set(response.concepts):
                relevance_score += 3
            
            response_instruction = f"{response.context1}|{response.context2}|{response.scenario}"
            instruction_similarity = self.calculate_instruction_similarity(
                current_instruction, response_instruction
            )
            relevance_score += instruction_similarity * 4
            
            if response.was_correct:
                relevance_score += 1
            
            if relevance_score > 2.0:
                response.usage_count += 1
                similar_responses.append((response, relevance_score))
        
        similar_responses.sort(key=lambda x: x[1], reverse=True)
        return [response for response, score in similar_responses[:max_items]]

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

def extract_choice(text):
    text = text.strip()
    match = re.search(r"[12]", text)
    if match:
        return match.group(0)
    return "UNKNOWN"

def evaluate_memory(df, args, main_template, sub_templates):
    memory_manager = MemoryManager()
    splits = ["agent-properties", "social-interactions", "social-properties"]
    all_results = []
    
    for split in splits:
        df_split = df[df["Domain"] == split].head(args.max_items)
        results = []
        
        def process_row(row):
            c1, c2 = row["Context1"], row["Context2"]
            t1, t2 = row["Target1"], row["Target2"]
            concepts = (row.get("Concept1", ""), row.get("Concept2", ""))
            
            relevant_contexts = memory_manager.get_relevant_contexts(
                split, concepts, c1, c2, t1
            )
            
            memory_context = ""
            if relevant_contexts:
                memory_context = "\n\nPrevious similar examples:\n"
                for i, mem in enumerate(relevant_contexts[:3], 1):
                    memory_context += f"{i}. Context: {mem.context1[:100]}... -> Choice: {mem.model_choice} (Correct: {mem.was_correct})\n"
            
            sub_results = {}
            for name, template in sub_templates.items():
                sub_results[f"{name}_context1"] = call_with_retry(template.format(context=c1), args.model)
                sub_results[f"{name}_context2"] = call_with_retry(template.format(context=c2), args.model)

            def run_prompt(target):
                vars_dict = {
                    "context1": c1,
                    "context2": c2,
                    "target": target,
                }
                vars_dict.update(sub_results)
                final_prompt = main_template.format(**vars_dict) + memory_context
                final_answer = call_with_retry(final_prompt, args.model)
                choice = extract_choice(final_answer)
                return final_prompt, final_answer, choice

            r1_prompt, r1_text, choice1 = run_prompt(t1)
            r2_prompt, r2_text, choice2 = run_prompt(t2)

            score = 0
            if choice1 == "1":
                score += 0.5
            if choice2 == "2":
                score += 0.5
            
            memory_manager.response_memory.append(ResponseMemory(
                instruction_id=f"{split}_{len(results)}",
                domain=split,
                concepts=concepts,
                context1=c1,
                context2=c2,
                scenario=t1,
                model_response=r1_text,
                correct_choice="1",
                model_choice=choice1,
                was_correct=(choice1 == "1"),
                score=score,
                timestamp=datetime.datetime.now().isoformat()
            ))

            return {
                "model": f"{args.model}-memory",
                "split": split,
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
                "model answer 2": r2_text
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_row, row) for _, row in df_split.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {split}"):
                results.append(future.result())
        
        all_results.extend(results)
    
    memory_manager.save_memory()
    return pd.DataFrame(all_results)