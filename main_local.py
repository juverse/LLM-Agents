import os
import time
import pandas as pd
from tqdm import tqdm
import argparse
import datetime
import sys
import logging
from openai import OpenAI

# === Reduce logging noise ===
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

# === Setup OpenAI-compatible local vLLM ===
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# === Prompt loading utilities ===
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_subprompt(path: str) -> str:
    return load_prompt(path)

def load_subprompts(directory: str, names=None) -> dict:
    subprompts = {}
    if not os.path.isdir(directory):
        return subprompts

    available = [fname for fname in os.listdir(directory) if fname.endswith(".txt")]
    to_load = [f"{name}.txt" for name in names if f"{name}.txt" in available] if names else available

    for fname in to_load:
        key = os.path.splitext(fname)[0]
        subprompts[key] = load_subprompt(os.path.join(directory, fname))

    return subprompts

# === Choice extraction ===
def extract_choice(text):
    import re
    text = text.strip()
    match = re.search(r"[12]", text)
    return match.group(0) if match else "UNKNOWN"

# === Local model call ===
def call_with_retry(prompt, max_retries=3):
    delay = 1
    for attempt in range(max_retries):
        try:
            if not prompt or not prompt.strip():
                return "ERROR: Empty prompt"
            response = client.chat.completions.create(
                model="neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.9,
                max_tokens=40
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return f"ERROR: {e}"

# === Evaluation Loop ===
def evaluate_split(df, split_name, max_items, main_template, sub_templates):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Evaluating {split_name}"):
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]

        sub_results = {}
        for name, template in sub_templates.items():
            sub_results[f"{name}_context1"] = call_with_retry(template.format(context=c1))
            sub_results[f"{name}_context2"] = call_with_retry(template.format(context=c2))

        def run_prompt(target):
            vars_dict = {
                "context1": c1,
                "context2": c2,
                "target": target,
                **sub_results
            }
            final_prompt = main_template.format(**vars_dict)
            final_answer = call_with_retry(final_prompt)
            choice = extract_choice(final_answer)
            return final_prompt, final_answer, choice

        r1_prompt, r1_text, choice1 = run_prompt(t1)
        r2_prompt, r2_text, choice2 = run_prompt(t2)

        score = 0.5 * (choice1 == "1") + 0.5 * (choice2 == "2")

        results.append({
            "model": "local-vllm",
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
        })

    return pd.DataFrame(results)

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_items', type=int, default=100)
    parser.add_argument('--main_prompt', type=str, default='main')
    parser.add_argument('--sub_prompts', type=str, default='belief_desire_intention')
    args = parser.parse_args()

    run_args = ' '.join(sys.argv[1:])

    df = pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")
    main_template = load_prompt(os.path.join("prompts", "main_prompt", f"{args.main_prompt}.txt"))
    sub_names = [n.strip() for n in args.sub_prompts.split(',') if n.strip()]
    sub_templates = load_subprompts(os.path.join("prompts", "sub_prompts"), names=sub_names)

    splits = ["agent-properties", "social-interactions", "social-properties"]
    all_results = []
    for split in tqdm(splits, desc="Evaluating splits"):
        df_result = evaluate_split(df, split, args.max_items, main_template, sub_templates)
        all_results.append(df_result)

    all_results = pd.concat(all_results, ignore_index=True)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_file = os.path.join("results", f"{timestamp}_ewok_eval.csv")

    all_results["arguments"] = run_args
    all_results.to_csv(output_file, index=False)

    print("✅ Evaluation complete. Saved to", output_file)
    print("Average Accuracy:", all_results["score"].mean())

if __name__ == "__main__":
    main()