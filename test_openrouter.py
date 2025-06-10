import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Setup ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY # ihr könnt hier auch euren api key direkt einfügen als string
)

# === Prompt Template ===
def make_prompt(context1, context2, target):
    return (
        "# INSTRUCTIONS\n"
        "In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. "
        "Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense "
        "considering the scenario that follows. The contexts will be numbered \"1\" or \"2\". You must answer using \"1\" or \"2\".\n\n"
        f"## Contexts\n1. \"{context1}\"\n2. \"{context2}\"\n\n"
        f"## Scenario\n\"{target}\"\n\n"
        "## Task\nWhich context makes more sense given the scenario? Please answer using ONLY either \"1\" or \"2\"."
    )

def extract_choice(text):
    text = text.strip()
    if "1" in text and "2" not in text:
        return "1"
    elif "2" in text and "1" not in text:
        return "2"
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
                max_tokens=10
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2 
            else:
                return f"ERROR: {e}"

# === Evaluation Loop ===
def evaluate_split(df, split_name, model_name, max_items):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    def process_row(row):
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]

        r1_text = call_with_retry(make_prompt(c1, c2, t1), model_name)
        r2_text = call_with_retry(make_prompt(c1, c2, t2), model_name)

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
            "answer a": t1,
            "answer b": t2,
            "chosen answer 1": choice1,
            "chosen answer 2": choice2,
            "score": score
        }

    # Run in parallel
    with ThreadPoolExecutor(max_workers=25) as executor: # hier könnt ihr die anzahl der parallelen anfragen, dass macht das ganze schneller kann aber auch zu rate limits führen
        futures = [executor.submit(process_row, row) for _, row in df_split.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {split_name}"):
            results.append(future.result())

    return pd.DataFrame(results)


# === Main ===
def main():
    # Load from HuggingFace Parquet
    df = pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")

    model_name = "mistralai/mistral-7b-instruct"
    splits = ["agent-properties", "social-interactions", "social-properties"]
    # hier könnt ihr die anzahl an fragen für den test einstellen mit max_items, 9999 für unbegrenzt
    all_results = pd.concat([evaluate_split(df, s, model_name, max_items=100) for s in splits], ignore_index=True)

    # Save results
    output_file = f"ewok_choice_eval_{model_name.replace('/', '_')}.csv"
    all_results.to_csv(output_file, index=False)

    print("✅ Evaluation complete. Saved to", output_file)
    print("Average Accuracy:", all_results["score"].mean())

if __name__ == "__main__":
    main()

