'''Preparing data from Hugging Face AssetOpsBench dataset to use for training the classifier

    Downloads the "scenarios" subset from AssetOpsBench, and converts it to the jsonl format
    that lmtrain.py expects: {"text": "...", "label": "true" | "false"}

    The dataset is labeled using Llama 3.3 70B via WatsonX, which is prompted to return whether each query
    needs thinking mode enabled.
'''

import json
import os
from pathlib import Path
from datasets import load_dataset
from .litellm import LiteLLMBackend

# llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
llm = LiteLLMBackend("watsonx/ibm/granite-13b-chat-v2")

# load dataset
print(f"")
dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")
split = "train"

output_path = Path("data/lmtrain_data.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    for idx, sample in enumerate(dataset[split]):
        
        text = sample.get("text")
        if not text:
            raise ValueError("text field does not exist")
        
        text = text.strip()

        # prompting llm to label this sample as needing thinking mode or not
        prompt = f"Does this question need extended thinking/multi-step reasoning? Only answer with 'true' or 'false':\n{text}"
        response = llm.generate(prompt, temperature=0.0).strip().lower()
        label = "true" if "true" in response else "false"

        f.write(json.dumps({"text": text, "label": label}) + "\n")
        print(f"[{idx+1}] {text[:60]}... -> {label}")

print(f"\nSaved to {output_path}")
