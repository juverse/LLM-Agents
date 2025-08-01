import os
import time
import pandas as pd
from tqdm import tqdm
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

def call_with_retry(prompt, model_name, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                # Check if it's a rate limiting error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # For rate limiting, use much longer delays
                    rate_limit_delay = 60 + (attempt * 30)  # 60s, 90s, 120s, 150s
                    print(f"Rate limit hit, waiting {rate_limit_delay}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(rate_limit_delay)
                else:
                    # For other errors, use exponential backoff
                    time.sleep(delay)
                    delay *= 2
            else:
                print(f"FAILED after {max_retries} attempts: {error_str}")
                return f"ERROR: {e}"

def extract_choice(text):
    import re
    text = text.strip()
    match = re.search(r"[12]", text)
    if match:
        return match.group(0)
    return "UNKNOWN"

def evaluate_split(df, split_name, model_name, max_items, main_template, sub_templates):
    df_split = df[df["Domain"] == split_name].head(max_items)
    results = []

    def process_row(row):
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]

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

    # Reduce concurrency for Gemini Pro to avoid rate limits
    max_workers = 5 if "pro" in model_name.lower() else 15
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in df_split.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {split_name}"):
            results.append(future.result())

    return pd.DataFrame(results)

def evaluate_gemini(df, args, main_template, sub_templates):
    splits = ["agent-properties", "social-interactions", "social-properties"]
    all_results = pd.concat(
        [
            evaluate_split(
                df,
                s,
                args.model,
                max_items=args.max_items,
                main_template=main_template,
                sub_templates=sub_templates,
            )
            for s in splits
        ],
        ignore_index=True,
    )
    return all_results