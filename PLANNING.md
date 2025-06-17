# Prompt Tester - Planning Document

## 🎯 Project Overview
A cognitive prompt evaluation system that tests different prompting strategies on the EWoK dataset to determine which cognitive approaches perform best for context-target mapping tasks.

## 🏗️ Architecture

### Core Components
1. **PromptLoader** (`prompt_loader.py`)
   - Loads cognitive prompts from markdown files
   - Formats prompts for EWoK dataset evaluation
   - Manages prompt templates and metadata

2. **PromptTester** (`prompt_tester.py`)
   - Main evaluation orchestrator
   - Runs prompts against dataset samples
   - Calculates performance metrics and comparisons

### Design Principles
- **Single Responsibility**: Each class has one clear purpose
- **Modularity**: Split into focused, small files (<200 lines each)
- **Reusability**: Leverage existing cognitive evaluator patterns
- **Testability**: Clean separation of concerns

## 🧠 Cognitive Prompts Tested
- **BDI (Belief-Desire-Intention)**: Structured agent reasoning
- **Dual Process (S1/S2)**: Intuitive vs analytical thinking
- **Schema Theory**: Pattern matching and scripts
- **Theory of Mind**: Mental state reasoning
- **Common Ground**: Pragmatic understanding
- **Metacognition**: Self-reflective reasoning
- **ACT-R**: Cognitive architecture approach
- **Predictive Coding**: Bayesian inference
- **Grounded Cognition**: Embodied understanding
- **Analogy-Based Reasoning**: Comparative thinking

## 🔧 Technology Stack
- **Language**: Python 3.8+
- **LLM Integration**: LangChain + OpenRouter
- **Data Processing**: Pandas
- **Dataset**: EWoK (Evaluating World Knowledge)
- **Configuration**: Environment variables + .env

## 📊 Evaluation Metrics
- **Accuracy**: Correct context-target mappings
- **Response Time**: Processing speed per sample
- **Domain Breakdown**: Performance across cognitive domains
- **Comparative Analysis**: Prompt-to-prompt performance ranking

## 🎯 Success Criteria
- Evaluate 100+ samples from EWoK dataset
- Test 8+ different cognitive prompting strategies
- Generate clear performance comparisons
- Identify best-performing prompts for each domain
- Maintain <2s average response time per evaluation

## 🔒 Constraints
- Files must be <200 lines each
- Use existing OpenRouter configuration
- Focus on agentic cognitive domains
- Maintain clean, readable code
- No external dependencies beyond current requirements