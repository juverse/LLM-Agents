import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from context_summarizer import summarize_bdi
import argparse

# === Setup ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY # ihr könnt hier auch euren api key direkt einfügen als string
)

# === Prompt Template ===
def make_prompt(context1, context2, target, bdi1=None, bdi2=None):
    prompt = (
        "# INSTRUCTIONS\n"
        "In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. "
        "Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense "
        "considering the scenario that follows. The contexts will be numbered \"1\" or \"2\". You must answer using \"1\" or \"2\".\n\n"
        f"## Contexts\n1. \"{context1}\"\n2. \"{context2}\"\n\n"
    )
    if bdi1 or bdi2:
        prompt += "# Additional Information (Belief-Desire-Intention Summaries)\n"
        if bdi1:
            prompt += f"Context 1 BDI: {bdi1}\n"
        if bdi2:
            prompt += f"Context 2 BDI: {bdi2}\n"
        prompt += "\n"
    prompt += (
        f"## Scenario\n\"{target}\"\n\n"
        "## Task\nWhich context makes more sense given the scenario? Please ALWAYS start your answer with either \"1\" or \"2\"."
    )
    return prompt

def extract_choice(text):
    import re
    text = text.strip()
    match = re.search(r'[12]', text)
    if match:
        return match.group(0)
    return "UNKNOWN"

# === Helper: call OpenRouter with retries ===
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

def summarize_context(context, module="bdi", model_name="mistralai/mistral-7b-instruct"):
    """
    Modular context summarization. Supports different modules (e.g., 'bdi').
    """
    if module == "bdi":
        return summarize_bdi(context, model_name=model_name)
    # Add more modules here as needed
    raise ValueError(f"Unknown summarization module: {module}")

# === Evaluation Loop ===
def evaluate_split(df, split_name, model_name, max_items):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    def process_row(row):
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]

        # Summarize both contexts using BDI
        bdi1 = summarize_context(c1, module="bdi", model_name=model_name)
        bdi2 = summarize_context(c2, module="bdi", model_name=model_name)

        # Format BDI summaries as short strings for the prompt
        def bdi_to_str(bdi):
            if isinstance(bdi, dict):
                return f"Belief: {bdi.get('belief','')} | Desire: {bdi.get('desire','')} | Intention: {bdi.get('intention','')}"
            return str(bdi)

        bdi1_str = bdi_to_str(bdi1)
        bdi2_str = bdi_to_str(bdi2)

        r1_prompt = make_prompt(c1, c2, t1, bdi1=bdi1_str, bdi2=bdi2_str)
        r2_prompt = make_prompt(c1, c2, t2, bdi1=bdi1_str, bdi2=bdi2_str)

        r1_text = call_with_retry(r1_prompt, model_name)
        r2_text = call_with_retry(r2_prompt, model_name)

        choice1 = extract_choice(r1_text)
        choice2 = extract_choice(r2_text)

        score = 0
        if choice1 == "1":
            score += 0.5
        if choice2 == "2":
            score += 0.5

        return {
            "model": model_name,
            "split": split_name,
            "context a": c1,
            "context b": c2,
            "context a BDI": bdi1_str,
            "context b BDI": bdi2_str,
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

    # Run in parallel
    with ThreadPoolExecutor(max_workers=25) as executor: # hier könnt ihr die anzahl der parallelen anfragen, dass macht das ganze schneller kann aber auch zu rate limits führen
        futures = [executor.submit(process_row, row) for _, row in df_split.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {split_name}"):
            results.append(future.result())

    return pd.DataFrame(results)


# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM choices with optional context summarization.")
    parser.add_argument('--max_items', type=int, default=100, help='Maximum number of items per split (default: 100)')
    parser.add_argument('--summarizer', type=str, default='bdi', help='Summarization module to use (bdi, none, or other module name)')
    parser.add_argument('--model', type=str, default='mistralai/mistral-7b-instruct', help='Model name to use')
    args = parser.parse_args()

    # Load from HuggingFace Parquet
    df = pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")

    model_name = args.model
    splits = ["agent-properties", "social-interactions", "social-properties"]
    # hier könnt ihr die anzahl an fragen für den test einstellen mit max_items, 9999 für unbegrenzt
    all_results = pd.concat([evaluate_split(df, s, model_name, max_items=args.max_items) for s in splits], ignore_index=True)

    # Save results
    output_file = f"ewok_choice_eval_{model_name.replace('/', '_')}_{args.summarizer}.csv"
    all_results.to_csv(output_file, index=False)

    print("✅ Evaluation complete. Saved to", output_file)
    print("Average Accuracy:", all_results["score"].mean())

if __name__ == "__main__":
    main()

