import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import datetime
import sys
import re

# === Setup ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY  # insert your API key here or set OPENROUTER_API_KEY env var
)

# === Prompt loading utilities ===
def load_prompt(path: str) -> str:
    """Read a prompt template from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



def load_subprompt(path: str) -> str:
    """Read a single subprompt template from ``path``."""
    return load_prompt(path)


def load_subprompts(directory: str, names=None) -> dict:
    """Load selected ``.txt`` subprompts from ``directory`` into a dict.

    If ``names`` is ``None``, all ``.txt`` files are loaded. Otherwise ``names``
    should be an iterable of base filenames (without ``.txt``).
    """
    subprompts = {}
    if not os.path.isdir(directory):
        return subprompts

    available = [fname for fname in os.listdir(directory) if fname.endswith(".txt")]
    if names is None:
        to_load = available
    else:
        to_load = [f"{name}.txt" for name in names if f"{name}.txt" in available]

    for fname in to_load:
        key = os.path.splitext(fname)[0]
        subprompts[key] = load_subprompt(os.path.join(directory, fname))

    return subprompts


# === Choice extraction ===
def extract_choice(text):
    import re
    text = text.strip()
    match = re.search(r"[12]", text)
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

# === Evaluation Loop ===
def evaluate_split(df, split_name, model_name, max_items, main_template, sub_templates):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    def process_row(row):
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]

        # Run all subprompts for each context
        sub_results = {}
        for name, template in sub_templates.items():
            sub_results[f"{name}_context1"] = call_with_retry(template.format(context=c1), model_name)
            sub_results[f"{name}_context2"] = call_with_retry(template.format(context=c2), model_name)

        def run_prompt(target):
            vars_dict = {
                "context1": c1,
                "context2": c2,
                "target": target,
            }
            vars_dict.update(sub_results)
            final_prompt = main_template.format(**vars_dict)
            final_answer = call_with_retry(final_prompt, model_name)
            choice = extract_choice(final_answer)
            return final_prompt, final_answer, choice

        r1_prompt, r1_text, choice1 = run_prompt(t1)
        r2_prompt, r2_text, choice2 = run_prompt(t2)

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
    parser = argparse.ArgumentParser(description="Evaluate LLM choices using modular prompts.")
    parser.add_argument('--max_items', type=int, default=100, help='Maximum number of items per split (default: 100)')
    parser.add_argument('--model', type=str, default='mistralai/mistral-7b-instruct', help='Model name to use')
    parser.add_argument('--main_prompt', type=str, default='main', help='Main prompt template name (without .txt)')
    parser.add_argument('--sub_prompts', type=str, default='belief_desire_intention',
                        help='Comma separated sub-prompt template names (without .txt)')
    args = parser.parse_args()

    run_args = ' '.join(sys.argv[1:])

    # Load from HuggingFace Parquet
    df = pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")

    model_name = args.model

    main_template_path = os.path.join("prompts", "main_prompt", f"{args.main_prompt}.txt")
    main_template = load_prompt(main_template_path)

    matches = re.findall(r"\{(\w+)_context[12]\}", main_template)
    sub_names = sorted(set(matches))  # deduplicate
    sub_templates = load_subprompts(os.path.join("prompts", "sub_prompts"), names=sub_names)

    splits = ["agent-properties", "social-interactions", "social-properties"]
    # hier könnt ihr die anzahl an fragen für den test einstellen mit max_items, 9999 für unbegrenzt
    all_results = pd.concat(
        [
            evaluate_split(
                df,
                s,
                model_name,
                max_items=args.max_items,
                main_template=main_template,
                sub_templates=sub_templates,
            )
            for s in splits
        ],
        ignore_index=True,
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_file = os.path.join(
        "results", f"{timestamp}_ewok_eval.csv"
    )

    all_results["arguments"] = run_args
    all_results.to_csv(output_file, index=False)

    print("✅ Evaluation complete. Saved to", output_file)
    print("Average Accuracy:", all_results["score"].mean())

if __name__ == "__main__":
    main()

