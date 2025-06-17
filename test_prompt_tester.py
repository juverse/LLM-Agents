#!/usr/bin/env python3
"""
Test Script for Prompt Tester

Quick validation of the prompt testing system without using many API calls.
"""

import os
from dotenv import load_dotenv
from prompt_tester import PromptTester


def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("🧪 Testing Prompt Tester Basic Functionality")
    
    # Test with minimal samples
    tester = PromptTester(max_samples=2)
    
    print(f"✅ Loaded {len(tester.dataset)} samples")
    print(f"✅ Loaded {len(tester.prompt_loader.list_prompts())} prompts")
    print(f"✅ LLM client initialized: {tester.config['model_name']}")
    
    # Test prompt formatting
    if len(tester.dataset) > 0:
        sample = tester.dataset.iloc[0]
        prompt_name = tester.prompt_loader.list_prompts()[0]
        
        formatted_prompt = tester.prompt_loader.format_prompt_for_ewok(
            prompt_name=prompt_name,
            context1=sample['Context1'],
            context2=sample['Context2'],
            target1=sample['Target1'],
            target2=sample['Target2']
        )
        
        print(f"✅ Prompt formatting successful for '{prompt_name}'")
        print(f"📝 Sample prompt length: {len(formatted_prompt)} characters")
    
    print("\n🎯 System validation complete!")
    return True


def test_single_evaluation():
    """Test single prompt evaluation (uses 1 API call)."""
    print("\n🔬 Testing Single Prompt Evaluation")
    
    # Check if API key is available
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️ No API key found - skipping API test")
        return True
    
    tester = PromptTester(max_samples=1)
    
    if len(tester.dataset) > 0:
        sample = tester.dataset.iloc[0]
        prompt_name = "BDI_small"  # Use a short prompt
        
        if prompt_name in tester.prompt_loader.list_prompts():
            print(f"🚀 Testing '{prompt_name}' on 1 sample...")
            
            result = tester.evaluate_prompt_on_sample(prompt_name, sample, 0)
            
            print(f"✅ Evaluation complete!")
            print(f"   Predicted: {result.predicted}")
            print(f"   Correct: {result.correct}")
            print(f"   Accuracy: {'✅' if result.is_correct else '❌'}")
            print(f"   Time: {result.response_time:.2f}s")
        else:
            print(f"⚠️ Prompt '{prompt_name}' not found")
    
    return True


if __name__ == "__main__":
    try:
        # Test basic functionality (no API calls)
        test_basic_functionality()
        
        # Test single evaluation (1 API call)
        test_single_evaluation()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)