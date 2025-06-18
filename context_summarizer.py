import os
from openai import OpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def summarize_bdi(context, model_name="mistralai/mistral-7b-instruct"):
    """
    Summarizes the given context into Belief, Desire, and Intention using an LLM.
    Returns a dict with keys: 'belief', 'desire', 'intention'.
    """
    prompt = (
        "# INSTRUCTIONS\n"
        "Given the following context, extract and summarize the agent's Belief, Desire, and Intention (BDI) as short bullet points. "
        "Format your answer as JSON with keys 'belief', 'desire', and 'intention'.\n\n"
        f"## Context\n{context}\n\n"
        "## Task\nSummarize as JSON:"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    import json
    import re
    # Try to extract JSON from the response
    text = response.choices[0].message.content
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"belief": "", "desire": "", "intention": ""}

# You can add more summarization modules here, e.g.:
# def summarize_xyz(context, ...): ...
