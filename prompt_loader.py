#!/usr/bin/env python3
"""
Prompt Loader - Simple Prompt Management System

This module handles loading and managing cognitive prompts from markdown files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Represents a cognitive prompt template."""
    name: str
    content: str
    source_file: str


class PromptLoader:
    """Loads and manages cognitive prompts from markdown files."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self) -> None:
        """Load all markdown prompt files from the prompts directory."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        # Get all .md files in prompts directory
        md_files = list(self.prompts_dir.glob("*.md"))
        
        for md_file in md_files:
            try:
                prompt = self._load_prompt_file(md_file)
                self.prompts[prompt.name] = prompt
                print(f"✅ Loaded prompt: {prompt.name}")
            except Exception as e:
                print(f"⚠️ Failed to load {md_file.name}: {e}")
    
    def _load_prompt_file(self, file_path: Path) -> PromptTemplate:
        """Load a single prompt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Extract name from filename (remove .md extension)
        name = file_path.stem
        
        return PromptTemplate(
            name=name,
            content=content,
            source_file=str(file_path)
        )
    
    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get a specific prompt by name."""
        return self.prompts.get(name)
    
    def list_prompts(self) -> List[str]:
        """Get list of all available prompt names."""
        return list(self.prompts.keys())
    
    def format_prompt_for_ewok(self, prompt_name: str, context1: str, context2: str, 
                             target1: str, target2: str) -> str:
        """Format a prompt for EWoK dataset evaluation."""
        prompt_template = self.get_prompt(prompt_name)
        if not prompt_template:
            raise ValueError(f"Prompt not found: {prompt_name}")
        
        # Build the full prompt with EWoK context
        formatted_prompt = f"""Du bist ein Experte für Sprachverständnis und logisches Denken.

{prompt_template.content}

AUFGABE:
Du erhältst zwei Kontexte und zwei Ziele. Bestimme die korrekte Zuordnung.

Kontext A: {context1}
Kontext B: {context2}

Ziel 1: {target1}
Ziel 2: {target2}

Welche Zuordnung ist korrekt?
- Option 1: Kontext A → Ziel 1, Kontext B → Ziel 2 (Antwort: "1-1,2-2")
- Option 2: Kontext A → Ziel 2, Kontext B → Ziel 1 (Antwort: "1-2,2-1")

Folge der oben beschriebenen Methodik und gib am Ende deine Entscheidung an:
Antwort: [1-1,2-2 oder 1-2,2-1]"""
        
        return formatted_prompt
    
    def get_prompt_stats(self) -> Dict[str, int]:
        """Get statistics about loaded prompts."""
        stats = {
            "total_prompts": len(self.prompts),
            "avg_length": sum(len(p.content) for p in self.prompts.values()) // len(self.prompts) if self.prompts else 0
        }
        return stats


# Utility function for quick access
def load_prompts(prompts_dir: str = "prompts") -> PromptLoader:
    """Quick function to load all prompts."""
    return PromptLoader(prompts_dir)


if __name__ == "__main__":
    # Test the prompt loader
    loader = PromptLoader()
    print(f"\n📊 Loaded {len(loader.prompts)} prompts:")
    for name in loader.list_prompts():
        print(f"  • {name}")
    
    print(f"\n📈 Stats: {loader.get_prompt_stats()}")