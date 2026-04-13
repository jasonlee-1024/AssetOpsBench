"""Thinking-mode routing via a minimal binary text classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import LLMBackend

# Training config: edit these values directly in this script.
DATA_FILE = Path("data/lmtrain_data.jsonl")
EVAL_RATIO = 0.25
OUTPUT_DIR = Path("models/lmtrain")
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
THRESHOLD = 0.5
EVAL_OUTPUT_PATH: Path | None = None
SEED = 42


def _load_and_split_examples(path: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load examples and split them into train/eval sets.

    Input schema per example: {"text": "...", "label": "true" | "false"}
    """
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "data" in data:
            rows = data["data"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON input must be a list or {'data': [...]} object")

    items: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if "text" not in row or "label" not in row:
            raise ValueError(f"Row {idx} missing required keys 'text' and/or 'label'")

        label_str = str(row["label"]).strip().lower()
        if label_str == "true":
            label = 1
        elif label_str == "false":
            label = 0
        else:
            raise ValueError(
                f"Row {idx} label must be 'true' or 'false', got: {row['label']!r}"
            )

        items.append({"text": str(row["text"]), "label": label})

    if len(items) < 2:
        return items, []

    import random

    random.Random(seed).shuffle(items)
    eval_size = max(1, int(len(items) * EVAL_RATIO))
    eval_examples = items[:eval_size]
    train_examples = items[eval_size:]
    if not train_examples:
        train_examples = eval_examples[:1]
        eval_examples = eval_examples[1:]
    return train_examples, eval_examples


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Compute binary classification metrics without extra dependencies."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = max(len(y_true), 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / total

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "support": total,
    }


def _print_eval_report(threshold: float, metrics: dict[str, Any]) -> None:
    """Print compact evaluation output suitable for terminal logs."""
    print(
        "Eval @ threshold "
        f"{threshold:.2f}: "
        f"acc={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


@dataclass
class ThinkModeClassifier:
    """Inference wrapper around a HF sequence classification model."""

    model: Any
    tokenizer: Any

    @classmethod
    def load(cls, model_path: str | Path) -> "ThinkModeClassifier":
        """Load a classifier model from local path or HF model id."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = str(model_path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        return cls(model=model, tokenizer=tokenizer)

    def predict_proba(self, text: str) -> float:
        """Return P(label=true) for a single text."""
        import torch

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        return float(probs[0][1].item())

    def should_use_thinking(self, text: str, threshold: float = 0.5) -> bool:
        """Return True if the model predicts thinking mode should be used."""
        return self.predict_proba(text) >= threshold


class ReasonRoutingLLMBackend(LLMBackend):
    """LLM backend wrapper that injects a think trigger based on classification."""

    def __init__(
        self,
        base_llm: LLMBackend,
        classifier: ThinkModeClassifier,
        threshold: float = 0.5,
        think_trigger: str = "</think>",
    ) -> None:
        self._base_llm = base_llm
        self._classifier = classifier
        self._threshold = threshold
        self._think_trigger = think_trigger

    def _route_prompt(self, prompt: str) -> str:
        if self._think_trigger in prompt:
            return prompt
        if self._classifier.should_use_thinking(prompt, self._threshold):
            return f"{prompt}\n{self._think_trigger}"
        return prompt

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        routed = self._route_prompt(prompt)
        return self._base_llm.generate(routed, temperature=temperature)


def train(
    data_file: Path,
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    threshold: float = 0.5,
    eval_output_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Train and save a binary text classifier using Hugging Face Trainer."""
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    train_examples, eval_examples = _load_and_split_examples(data_file, seed)
    if eval_examples:
        print(
            f"Auto split from {data_file}: "
            f"train={len(train_examples)} eval={len(eval_examples)}"
        )
    else:
        print(f"No eval split used from {data_file} (dataset too small).")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    train_ds = Dataset.from_list(train_examples).map(_tokenize, batched=True)
    eval_ds = Dataset.from_list(eval_examples).map(_tokenize, batched=True) if eval_examples else None

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=seed,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if eval_ds is not None else "no",
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if eval_ds is None:
        return None

    prediction_output = trainer.predict(eval_ds)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids.astype(int).tolist()

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    prob_true = probs[:, 1].tolist()

    pred_default = [1 if p >= threshold else 0 for p in prob_true]
    metrics = _binary_metrics(labels, pred_default)
    _print_eval_report(threshold, metrics)

    target = eval_output_path or (output_dir / "eval_metrics.json")
    report = {
        "threshold": threshold,
        "metrics": metrics,
    }
    target.write_text(
        json.dumps(report, indent=2)
    )
    print(f"Saved eval metrics to {target}")
    return report


def main() -> None:
    """Train using script-level constants defined at the top of this file."""
    train(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        threshold=THRESHOLD,
        eval_output_path=EVAL_OUTPUT_PATH,
        seed=SEED,
    )


if __name__ == "__main__":
    main()