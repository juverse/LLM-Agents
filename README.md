# LLM-Agents (Unified Architecture)

This project implements cognitively inspired LLM systems for assessing agentic skills using the EWOK benchmark. The codebase has been restructured with a unified architecture that supports multiple evaluation backends through a single interface.

## Quick Start

```bash
# Online evaluation (OpenRouter API)
python main.py --backend online --model mistralai/mistral-7b-instruct --max_items 20

# Local evaluation (vLLM server)
python main.py --backend local --max_items 20

# Log probability evaluation
python main.py --backend logprob --model mistralai/Mistral-7B-Instruct-v0.3

# Memory-assisted evaluation
python main.py --backend memory --model mistralai/mistral-7b-instruct
```

## Architecture

The unified structure consolidates functionality into clean, modular components:

```
llm_agents/
├── main.py                    # Unified CLI entry point
├── analysis.py               # Combined analysis tools
├── models/                   # Evaluation backends
│   ├── online.py             # OpenRouter API calls
│   ├── local.py              # vLLM local inference
│   ├── logprob.py            # Log probability evaluation
│   └── memory.py             # Memory-assisted evaluation
├── scripts/                     # Shared utilities
│   ├── evaluation.py         # Core evaluation logic
│   ├── visualize_results.py  # Visualization
│   └── utils.py              # Prompt loading, choice extraction
├── prompts/                  # Prompt templates (unchanged)
└── results/                  # Evaluation outputs (unchanged)
```

## Installation

1. Clone and install dependencies:
```bash
git clone https://github.com/juverse/LLM-Agents.git
cd LLM-Agents
pip install -r requirements.txt
```

2. Set up authentication:
```bash
# For online evaluation
export OPENROUTER_API_KEY="your_key_here"

# For HuggingFace datasets
huggingface-cli login
```

## Usage

### Evaluation Backends

**1. Online Evaluation (`--backend online`)**
- Uses OpenRouter API for cloud-based inference
- Supports any OpenRouter model
- Requires `OPENROUTER_API_KEY` environment variable

```bash
python main.py --backend online \
    --model mistralai/mistral-7b-instruct \
    --max_items 100 \
    --main_prompt study
```

**2. Local Evaluation (`--backend local`)**
- Uses local vLLM server (requires running server on localhost:8000)
- Faster inference, no API costs
- Requires vLLM server setup

```bash
# Start vLLM server first
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000

# Run evaluation
python main.py --backend local --max_items 100
```

**3. Log Probability Evaluation (`--backend logprob`)**
- Direct model inference using log probabilities
- No prompts, just compares context likelihood
- Supports both vLLM and HuggingFace backends

```bash
python main.py --backend logprob \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --max_items 100
```

**4. Memory-Assisted Evaluation (`--backend memory`)**
- Uses past evaluation context to improve performance
- Maintains memory of previous similar tasks
- Combines online evaluation with memory retrieval

```bash
python main.py --backend memory \
    --model mistralai/mistral-7b-instruct \
    --max_items 100
```

### Analysis Tools

The unified `analysis.py` combines all analysis functionality:

```bash
# Statistical analysis with significance testing
python analysis.py results/*.csv --format analyze

# Generate markdown summary table
python analysis.py --format table

# Create visualization plots
python analysis.py --format visualize

# Run all analysis types
python analysis.py --format all
```

### Command Line Arguments

**Main evaluation arguments:**
- `--backend` - Evaluation method: `online`, `local`, `logprob`, `memory`
- `--model` - Model identifier (varies by backend)
- `--max_items` - Number of items per split (default: 100)
- `--main_prompt` - Main prompt template name (default: 'main')
- `--sub_prompts` - Comma-separated sub-prompt names (default: 'belief_desire_intention')

**Analysis arguments:**
- `--format` - Output format: `analyze`, `table`, `visualize`, `all`
- `--results_dir` - Directory containing result files (default: './results')

## Performance Results

Based on comprehensive evaluations across multiple setups:

| Backend | Model | Overall Accuracy | Performance Notes |
|---------|-------|------------------|-------------------|
| memory | mistralai/mistral-7b-instruct | 77.2% | Best performing approach |
| online | deepseek/deepseek-chat-v3-0324 | 85.0% | Highest single model |
| local | vllm-quantized | 72.2% | ~1-2% below cloud model |
| logprob | quantized-mistral | 70.1% | No prompts, pure likelihood |

Memory-assisted evaluation achieves the best performance by leveraging historical context, while local quantized models provide near-identical performance to cloud APIs.

## Prompt System

The modular prompt system supports:
- **Main prompts** (`prompts/main_prompt/`): Final decision templates
- **Sub-prompts** (`prompts/sub_prompts/`): Cognitive module templates
- **Dynamic composition**: Sub-prompt results feed into main prompts

Example prompt combinations:
- `study.txt` - Exact prompt from research paper
- `combination_all.txt` - All cognitive modules combined
- `tom_main.txt` - Theory of Mind focused evaluation

## Authors & Contact

* Moritz Lönker
* Julia Lansche  
* Marc Baumholz

If you have any questions or comments, just let us know!
