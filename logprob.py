#!/usr/bin/env python3
"""ewok_logprob_local.py

Evaluate EWOK‑core *CHOICE* with a **local Mistral‑7B‑Instruct** run under
vLLM or plain Transformers.  No prompts, no sub‑prompts – we pick the context
that gives a higher token‑level log‑probability for each target.

*If you already have a vLLM server running on http://localhost:8000* you can
set `--backend vllm` and we will query its `/v1/completions` endpoint with
`echo=True, logprobs=True` to avoid loading the model twice.

Otherwise the script falls back to **Transformers** and runs the model in‑proc
(GPU recommended).
"""

from __future__ import annotations
import os, argparse, datetime as dt
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# optional deps
try:
    from openai import OpenAI  # for vLLM backend
except ImportError:
    OpenAI = None
try:
    import torch, math
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None  # will error later if backend==hf

# ---------------------------------------------------------------------------
# 0.  Conditional log‑prob helpers                                           |
# ---------------------------------------------------------------------------

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
    tok_lp = resp.choices[0].logprobs.token_logprobs
    # count context tokens using same tokenizer vLLM uses internally (cheap hack: len of token_logprobs) – safer option is HF tokenizer, but vLLM leaks offsets? we approximate by len(context_token_logprobs) using text_offset None so we fallback to hf tokenizer
    # We'll re‑tokenise with HF tokenizer to slice correctly
    return tok_lp[-len(target.split()):]  # placeholder (we will override later)

# We'll actually compute ctx_len using HF tokenizer to ensure slice.

def build_hf(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    mdl.eval()
    if torch.cuda.is_available():
        mdl.to("cuda")
    return tok, mdl

def lp_hf(tk, mdl, context: str, target: str) -> float:
    prompt = context.rstrip() + " " + target.lstrip()
    ids = tk(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        ids = ids.to("cuda")
    with torch.no_grad():
        logits = mdl(ids).logits[:, :-1]
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    tgt_ids = ids[:, 1:]
    token_lp = logp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
    ctx_len = len(tk(context.rstrip()))
    return token_lp[0, ctx_len:].sum().item()

# ---------------------------------------------------------------------------
# 1.  Per‑row evaluation                                                     |
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 2.  Main                                                                   |
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")

    if args.backend == "vllm":
        if OpenAI is None:
            raise RuntimeError("openai package required for vllm backend")
        client = make_vllm_client()
        # need HF tokenizer for ctx_len
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        def lp_func(ctx, tgt):
            prompt = ctx.rstrip() + " " + tgt.lstrip()
            resp = client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=0,
                temperature=0.0,
                logprobs=True,
                echo=True,
            )
            lp_list = resp.choices[0].logprobs.token_logprobs
            ctx_len = len(tok(ctx.rstrip()))
            return sum(lp for lp in lp_list[ctx_len:] if lp is not None)
    else:
        if torch is None:
            raise RuntimeError("Install torch + transformers for hf backend")
        tok, mdl = build_hf(args.model)
        def lp_func(ctx, tgt):
            return lp_hf(tok, mdl, ctx, tgt)

    splits = ["agent-properties", "social-interactions", "social-properties"]
    rows: List[Dict] = []
    for sp in splits:
        part = df[df["Domain"] == sp]
        if args.max_items:
            part = part.head(args.max_items)
        for _, row in tqdm(part.iterrows(), total=len(part), desc=sp):
            res = score_row(row, lp_func)
            res.update({"model": args.model, "split": sp})
            rows.append(res)

    out_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    ts = dt.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    out_df.to_csv(f"results/{ts}_ewok_eval_logprob.csv", index=False)
    print("Average accuracy:", out_df["score"].mean())

if __name__ == "__main__":
    main()
