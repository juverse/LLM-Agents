from __future__ import annotations
import os
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import torch, math
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None

def make_vllm_client() -> OpenAI:
    return OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def lp_vllm(client: OpenAI, model: str, context: str, target: str) -> float:
    prompt = context.rstrip() + " " + target.lstrip()
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=0,
        temperature=0.0,
        logprobs=True,
        echo=True,
    )
    
    ctx_toks = len(resp.choices[0].logprobs.tokens) - 1
    tgt_logprobs = resp.choices[0].logprobs.token_logprobs[ctx_toks:]
    total_logprob = sum(tgt_logprobs)
    return total_logprob

def lp_hf(model, tokenizer, context: str, target: str) -> float:
    prompt = context.rstrip() + " " + target.lstrip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ctx_tokens = tokenizer(context.rstrip(), return_tensors="pt")["input_ids"]
    ctx_len = ctx_tokens.shape[-1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[ctx_len-1:-1].gather(1, inputs["input_ids"][0, ctx_len:].unsqueeze(1))
        total_log_prob = target_log_probs.sum().item()
    
    return total_log_prob

def score_row(row, lp_func) -> Dict[str, float]:
    c1, c2 = row["Context1"], row["Context2"]
    t1, t2 = row["Target1"], row["Target2"]
    lp11 = lp_func(c1, t1)
    lp12 = lp_func(c2, t1)
    gap1 = lp11 - lp12
    ans1 = "1" if gap1 > 0 else "2"

    lp21 = lp_func(c1, t2)
    lp22 = lp_func(c2, t2)
    gap2 = lp21 - lp22
    ans2 = "1" if gap2 > 0 else "2"

    score = 0.5 * (ans1 == "1") + 0.5 * (ans2 == "2")
    return {
        "context a": c1, "context b": c2,
        "answer a": t1,  "answer b": t2,
        "lp_ctx1_a": lp11, "lp_ctx2_a": lp12, "gap_a": gap1, "chosen answer 1": ans1,
        "lp_ctx1_b": lp21, "lp_ctx2_b": lp22, "gap_b": gap2, "chosen answer 2": ans2,
        "score": score,
    }

def evaluate_logprob(df, args, main_template=None, sub_templates=None):
    if args.backend == "vllm":
        if OpenAI is None:
            raise RuntimeError("openai package required for vllm backend")
        client = make_vllm_client()
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        def lp_func(ctx, tgt):
            return lp_vllm(client, args.model, ctx, tgt)
    else:
        if torch is None:
            raise RuntimeError("torch and transformers required for hf backend")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        def lp_func(ctx, tgt):
            return lp_hf(model, tokenizer, ctx, tgt)
    
    all_results = []
    splits = ["agent-properties", "social-interactions", "social-properties"]
    
    for split in splits:
        df_split = df[df["Domain"] == split]
        if args.max_items:
            df_split = df_split.head(args.max_items)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(score_row, row, lp_func) for _, row in df_split.iterrows()]
            split_results = []
            for future in tqdm(as_completed(futures), desc=f"Processing {split}", total=len(futures)):
                result = future.result()
                result["split"] = split
                result["model"] = args.model
                split_results.append(result)
        
        all_results.extend(split_results)
    
    return pd.DataFrame(all_results)