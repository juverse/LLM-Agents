# 🧠 Simple Cognitive Evaluator Template

## Overview
Basic MVP template under 200 lines that:
- ✅ Loads and filters EWoK dataset for agentic variables
- ✅ Connects to OpenRouter via LangChain
- ✅ Chains 2 tools: Working Memory + Reasoning
- ✅ Evaluates accuracy in percentage
- ✅ Parser for multiple iterations

## Quick Start

### 1. Setup Environment
```bash
cd template
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-api-key"
```

### 2. Run Simple Evaluation
```bash
# Basic run (3 samples, 1 iteration)
python simple_cognitive_evaluator.py

# Custom parameters
python simple_cognitive_evaluator.py --max_samples 5 --iterations 3
```

## Features

### 🧠 LangChain Tools Architecture
- **Working Memory Tool**: Extracts entities and relationships from context
- **Reasoning Tool**: Makes context-target mapping decisions using memory state
- **Agent Executor**: Chains tools together automatically

### 📊 Evaluation & Parsing
- **Accuracy Calculation**: Percentage of correct predictions
- **Multi-Iteration Support**: Run multiple evaluation rounds
- **Performance Tracking**: Response times and iteration summaries
- **Result Saving**: JSON output with detailed results

### 🎯 Dataset Processing
- **Auto-Loading**: EWoK dataset from HuggingFace
- **Agentic Filtering**: Focus on agent-properties, social-interactions, social-properties
- **Sample Control**: Configurable number of samples per run

## Example Output
```
🚀 Starting evaluation with 2 iteration(s)

🔄 Iteration 1/2
📝 Sample 1/3 - agent-properties
✅ Predicted: 1-1,2-2, Time: 3.45s
📊 Iteration 1 Summary: 100.0% accuracy (3/3)

📈 EVALUATION PARSER SUMMARY
========================================
Total Iterations: 2
Overall Accuracy: 100.0%
Total Correct: 6/6
Average Response Time: 3.24s
```

## Architecture
- **Under 200 lines**: Simple, readable MVP code
- **LangChain Integration**: Modern AI agent framework
- **OpenRouter Connection**: Easy LLM model access
- **Tool Chaining**: Working Memory → Reasoning pipeline
- **Flexible Evaluation**: Configurable iterations and samples