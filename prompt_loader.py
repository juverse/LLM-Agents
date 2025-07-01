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
    """Loads and formats prompts from the 'prompts' directory."""

    def __init__(self, directory: str = "prompts"):
        self.directory = Path(directory)
        self.prompts = self._load_prompts_from_directory()

    def _load_prompts_from_directory(self) -> Dict[str, str]:
        """Load all .md files from the prompts directory."""
        prompts = {}
        for file_path in self.directory.glob("*.md"):
            prompt_name = file_path.stem  # Use filename without extension as name
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read()
            print(f"✅ Loaded prompt: {prompt_name}")
        return prompts

    def list_prompts(self) -> List[str]:
        """Return a list of available prompt names."""
        return list(self.prompts.keys())

    def get_prompt_template(self, prompt_name: str) -> Optional[str]:
        """Get the template content for a given prompt name."""
        return self.prompts.get(prompt_name)

    def format_prompt_for_ewok(self, prompt_name: str, context1: str, context2: str, target1: str, target2: str) -> str:
        """
        Formats a prompt template by replacing placeholders with EWoK data.
        """
        template = self.get_prompt_template(prompt_name)
        if not template:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        # Simple string replacement for placeholders
        formatted_prompt = template.replace("{{context1}}", context1)
        formatted_prompt = formatted_prompt.replace("{{context2}}", context2)
        formatted_prompt = formatted_prompt.replace("{{target1}}", target1)
        formatted_prompt = formatted_prompt.replace("{{target2}}", target2)

        return formatted_prompt

    def get_prompt_stats(self) -> Dict[str, int]:
        """Get statistics about loaded prompts."""
        stats = {
            "total_prompts": len(self.prompts),
            "avg_length": sum(len(p) for p in self.prompts.values()) // len(self.prompts) if self.prompts else 0
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