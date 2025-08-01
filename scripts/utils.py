import os

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
    if names is None:
        to_load = available
    else:
        to_load = [f"{name}.txt" for name in names if f"{name}.txt" in available]

    for fname in to_load:
        key = os.path.splitext(fname)[0]
        subprompts[key] = load_subprompt(os.path.join(directory, fname))

    return subprompts

def extract_choice(text):
    import re
    text = text.strip()
    match = re.search(r"[12]", text)
    if match:
        return match.group(0)
    return "UNKNOWN"