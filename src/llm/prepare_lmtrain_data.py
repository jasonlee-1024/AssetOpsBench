#!/usr/bin/env python3
"""Download AssetOpsBench scenarios and use LLM to label for thinking mode."""

import json
import os
from pathlib import Path
from datasets import load_dataset

from .litellm import LiteLLMBackend

model_id = os.environ.get("LM_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")
llm = LiteLLMBackend(model_id)

# Load dataset
print(f"Loading AssetOpsBench scenarios with {model_id}...")
dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")
split = "train" if "train" in dataset else list(dataset.keys())[0]

# Save with LLM labels
output_path = Path("llm/lmtrain_data.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    for idx, sample in enumerate(dataset[split]):
        # AssetOpsBench uses "text" field for questions
        text = sample.get("text")
        if not text:
            continue
        
        text = text.strip()
        
        # Ask LLM if thinking mode is needed
        prompt = f"Does this question need extended thinking/multi-step reasoning? Answer only 'true' or 'false':\n{text}"
        response = llm.generate(prompt, temperature=0.0).strip().lower()
        label = "true" if "true" in response else "false"
        
        f.write(json.dumps({"text": text, "label": label}) + "\n")
        print(f"[{idx+1}] {text[:60]}... → {label}")

print(f"\nSaved to {output_path}")
