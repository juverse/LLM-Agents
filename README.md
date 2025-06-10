# 🧠 Simple Cognitive Evaluator (< 200 Lines)

A lightweight cognitive evaluator for testing AI reasoning using the EWoK dataset. Optimized for simplicity and efficiency.

## 🎯 What This Does

Tests how well Large Language Models perform **cognitive reasoning tasks** by:
1. Loading EWoK dataset samples (or fallback test data)
2. Asking the LLM to map contexts to targets logically  
3. Measuring accuracy and response times

## 🚀 Quick Start

### 1. Setup
```bash
cd LLM-Agents

# Activate virtual environment if you have one
source venv_simple/bin/activate  # or your venv

# Install dependencies (if not already installed)
pip install requests python-dotenv pandas pyarrow fsspec
```

### 2. Configuration
The `.env` file is already configured with:
- **API Key**: Provided Mistral API key
- **Model**: `mistralai/mistral-7b-instruct-v0.1` 
- **Settings**: 3 samples per domain, 1 iteration

### 3. Run Evaluation
```bash
# Basic run with defaults (3 samples, 1 iteration)
python simple_cognitive_evaluator_Marc.py

# Custom parameters  
python simple_cognitive_evaluator_Marc.py --max_samples 2 --iterations 1

# Different model (optional)
python simple_cognitive_evaluator_Marc.py --model "mistralai/mistral-7b-instruct-v0.1" --max_samples 5
```

## 📊 Sample Output

```
🧠 Initialized LLM client with model: mistralai/mistral-7b-instruct-v0.1
📥 Loading EWoK dataset...
✅ Loaded 6 samples from agentic domains
🔬 Simple Cognitive Evaluator initialized

🚀 Starting evaluation with 1 iteration(s)

🔄 Iteration 1/1
📝 Sample 1/6 - agent-properties
   ✅ Predicted: 1-1,2-2, Time: 2.34s
📝 Sample 2/6 - social-interactions
   ❌ Predicted: 1-2,2-1, Time: 1.89s

📈 SIMPLE COGNITIVE EVALUATION SUMMARY
==================================================
Total Iterations: 1
Overall Accuracy: 83.3%
Correct Predictions: 5/6
Average Response Time: 2.1s

📊 Domain Accuracies:
  agent-properties: 100.0%
  social-interactions: 66.7%
  social-properties: 100.0%
💾 Results saved to simple_eval_results_20241210_143052.json
```

## 🏗️ Code Structure (Under 200 Lines)

### Core Classes
- **`SimpleLLMClient`**: Handles OpenRouter API calls
- **`SimpleDatasetLoader`**: Loads EWoK dataset with fallback
- **`SimpleCognitiveEvaluator`**: Main evaluation orchestrator
- **`SimpleResult`**: Dataclass for results

### Key Features
- ✅ Environment-based configuration
- ✅ Automatic dataset loading with fallback
- ✅ Clean error handling
- ✅ JSON results export
- ✅ Domain-specific accuracy tracking
- ✅ Rate limiting for API calls

## ⚙️ Configuration

### Environment Variables (`.env`)
```env
OPENROUTER_API_KEY=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
OPENROUTER_MODEL=mistralai/mistral-7b-instruct-v0.1
MAX_SAMPLES=3
ITERATIONS=1
```

### Command Line Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--max_samples` | Max samples per domain | 3 |
| `--iterations` | Number of evaluation runs | 1 |
| `--model` | Override LLM model | mistralai/mistral-7b-instruct-v0.1 |

## 🔍 How It Works

### 1. Cognitive Task Format
```
Context 1: Mohammed's foot is touching the volleyball.
Context 2: Mohammed's chair is touching the volleyball.
Target 1: Mohammed feels the volleyball.
Target 2: Mohammed does not feel the volleyball.

Question: Which context matches which target?
Answer: 1-1,2-2 or 1-2,2-1
```

### 2. Evaluation Logic
- **Correct Answer**: Always `1-1,2-2` (EWoK standard)
- **LLM Task**: Determine logical mapping
- **Scoring**: Binary correct/incorrect

### 3. Domains Tested
- **agent-properties**: Agent capabilities and limitations
- **social-interactions**: Social reasoning scenarios  
- **social-properties**: Social dynamics and relationships

## 📁 Output Files

Results automatically saved as timestamped JSON:
```json
{
  "total_iterations": 1,
  "total_samples": 6,
  "correct_predictions": 5,
  "overall_accuracy_percent": 83.3,
  "avg_response_time": 2.1,
  "domain_accuracies": {
    "agent-properties": 100.0,
    "social-interactions": 66.7,
    "social-properties": 100.0
  },
  "detailed_results": [...]
}
```

## 🛠️ Troubleshooting

### Common Issues
1. **Dataset Loading Failed**: Will use fallback test data automatically
2. **API Rate Limits**: Includes 0.5s delay between requests  
3. **Connection Errors**: Check internet and API key

### Debug Tips
```bash
# Test with minimal samples
python simple_cognitive_evaluator_Marc.py --max_samples 1 --iterations 1

# Check if dataset loads
python -c "import pandas as pd; print(pd.read_parquet('hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet').shape)"
```

## 📋 Dependencies

Minimal requirements:
- `requests` - API calls
- `python-dotenv` - Environment variables  
- `pandas` - Data handling
- `pyarrow` - Parquet support
- `fsspec` - File system support

## 🎯 Why This Design?

- **Under 200 Lines**: Easy to understand and modify
- **Single File**: All logic in one place
- **Minimal Dependencies**: Only essential packages
- **Error Resilient**: Fallback data when dataset fails
- **Production Ready**: Proper error handling and logging

---

**Ready to run!** The evaluator is configured with your API key and Mistral model. Just run:
```bash
python simple_cognitive_evaluator_Marc.py
``` 