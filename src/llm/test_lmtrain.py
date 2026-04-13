"""Tests for lmtrain routing helpers and backend wrapper."""

from __future__ import annotations

from llm import LLMBackend
from llm.lmtrain import (
    ReasonRoutingLLMBackend,
    _binary_metrics,
)


class _EchoLLM(LLMBackend):
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        return "ok"


class _StaticClassifier:
    def __init__(self, decision: bool) -> None:
        self._decision = decision

    def should_use_thinking(self, text: str, threshold: float = 0.5) -> bool:
        return self._decision


def test_lmtrain_appends_trigger_on_positive_prediction():
    base = _EchoLLM()
    router = ReasonRoutingLLMBackend(
        base_llm=base,
        classifier=_StaticClassifier(True),
        threshold=0.5,
        think_trigger="</think>",
    )

    router.generate("classify this prompt")
    assert base.last_prompt.endswith("</think>")


def test_lmtrain_does_not_append_on_negative_prediction():
    base = _EchoLLM()
    router = ReasonRoutingLLMBackend(
        base_llm=base,
        classifier=_StaticClassifier(False),
        threshold=0.5,
        think_trigger="</think>",
    )

    router.generate("simple prompt")
    assert base.last_prompt == "simple prompt"


def test_lmtrain_does_not_double_append_existing_trigger():
    base = _EchoLLM()
    router = ReasonRoutingLLMBackend(
        base_llm=base,
        classifier=_StaticClassifier(True),
        threshold=0.5,
        think_trigger="</think>",
    )

    router.generate("prompt already has token </think>")
    assert base.last_prompt == "prompt already has token </think>"


def test_binary_metrics_values():
    metrics = _binary_metrics(
        y_true=[1, 1, 0, 0],
        y_pred=[1, 0, 1, 0],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}