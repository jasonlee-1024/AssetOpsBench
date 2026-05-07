"""Tests for combined router demo formatting."""

from __future__ import annotations

from llm.lmtrain import ModelBasedRouter
from llm.router_demo import (
    format_combined_demo,
    format_model_router_demo,
    format_rule_router_demo,
)


class _ProbabilityClassifier:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities

    def predict_proba(self, text: str) -> float:
        return self._probabilities[text]

    def should_use_thinking(self, text: str, threshold: float = 0.5) -> bool:
        return self.predict_proba(text) >= threshold


def _router() -> ModelBasedRouter:
    return ModelBasedRouter(
        classifier=_ProbabilityClassifier(
            {
                "List sites": 0.55,
                "Detect bearing faults in WT-105": 0.66,
            }
        ),
        threshold=0.62,
    )


def test_rule_router_demo_prints_each_rule_status():
    output = format_rule_router_demo()

    assert "Running rule-based router" in output
    assert "Query: List sites" in output
    assert "  - multi_date      False" in output
    assert "Signals fired: -" in output
    assert "Query: Detect bearing faults in WT-105" in output
    assert "  - anomaly         True" in output
    assert "Decision: THINKING" in output


def test_model_router_demo_prints_formatted_scores():
    output = format_model_router_demo(_router())

    assert "Running model-based router" in output
    assert "Threshold: 0.62" in output
    assert "List sites" in output
    assert "0.5500" in output
    assert "Detect bearing faults in WT-105" in output
    assert "0.6600" in output


def test_combined_demo_prints_both_sections():
    output = format_combined_demo(_router())

    assert "Running rule-based router" in output
    assert "Running model-based router" in output
