import os
import pandas as pd
import datetime
import re

def load_dataset():
    return pd.read_parquet("hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet")

def load_templates(args):
    from scripts.utils import load_prompt, load_subprompts
    
    main_template_path = os.path.join("prompts", "main_prompt", f"{args.main_prompt}.txt")
    main_template = load_prompt(main_template_path)

    matches = re.findall(r"\{(\w+)_context[12]\}", main_template)
    sub_names = sorted(set(matches))
    sub_templates = load_subprompts(os.path.join("prompts", "sub_prompts"), names=sub_names)
    
    return main_template, sub_templates

def save_results(results, args):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_file = os.path.join(
        "results", f"{timestamp}_ewok_eval_{args.backend}.csv"
    )
    
    run_args = ' '.join([f'--{k} {v}' for k, v in vars(args).items()])
    results["arguments"] = run_args
    results.to_csv(output_file, index=False)
    
    return output_file

def run_evaluation(args):
    df = load_dataset()
    main_template, sub_templates = load_templates(args)
    
    if args.backend == "online":
        from models.online import evaluate_online
        results = evaluate_online(df, args, main_template, sub_templates)
    elif args.backend == "local":
        from models.local import evaluate_local
        results = evaluate_local(df, args, main_template, sub_templates)
    elif args.backend == "logprob":
        from models.logprob import evaluate_logprob
        results = evaluate_logprob(df, args, main_template, sub_templates)
    elif args.backend == "memory":
        from models.memory import evaluate_memory
        results = evaluate_memory(df, args, main_template, sub_templates)
    elif args.backend == "gemini":
        from models.gemini import evaluate_gemini
        results = evaluate_gemini(df, args, main_template, sub_templates)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    output_file = save_results(results, args)
    return results, output_file