import os
import time
import pandas as pd
from tqdm import tqdm
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def call_with_retry(prompt, max_retries=3):
    delay = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mistral-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return f"ERROR: {e}"

def extract_choice(text):
    import re
    text = text.strip()
    match = re.search(r"[12]", text)
    return match.group(0) if match else "UNKNOWN"

def evaluate_split(df, split_name, max_items, main_template, sub_templates):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    def process_row(row):
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
            }
            vars_dict.update(sub_results)
            final_prompt = main_template.format(**vars_dict)
            final_answer = call_with_retry(final_prompt)
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
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_row, row) for _, row in df_split.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {split_name}"):
            results.append(future.result())

    return pd.DataFrame(results)

def evaluate_local(df, args, main_template, sub_templates):
    splits = ["agent-properties", "social-interactions", "social-properties"]
    all_results = pd.concat(
        [
            evaluate_split(
                df,
                s,
                max_items=args.max_items,
                main_template=main_template,
                sub_templates=sub_templates,
            )
            for s in splits
        ],
        ignore_index=True,
    )
    return all_results