"""Rule-based query router for adaptive thinking-mode selection.

Extracts five binary signals from a query string and returns True if any
signal is present, indicating the query is complex enough to warrant
reasoning mode.  Runs in sub-millisecond time with no LLM call.

Keyword lists are stored in rule_based_router_keywords.yaml — edit that file to add or
remove routing signals without touching Python code.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Load keyword config
# ---------------------------------------------------------------------------

_CONFIG = yaml.safe_load(
    (Path(__file__).parent / "rule_based_router_keywords.yaml").read_text()
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_word_pattern(keywords: list[str]) -> re.Pattern:
    """Build a regex that matches any keyword in the list as a complete word.

    For example, given ["fault", "anomaly"], the pattern matches "fault" or
    "anomaly" in a string but not "default" or "anomalies_list".
    Matching is case-insensitive.
    """
    escaped_keywords = [re.escape(keyword) for keyword in keywords]
    alternation = "|".join(escaped_keywords)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def _build_phrase_pattern(phrases: list[str]) -> re.Pattern:
    """Build a regex that matches any phrase in the list as a substring.

    Unlike _build_word_pattern, no word boundaries are added — multi-word
    phrases like "out of range" or "sensor fault" define their own boundaries.
    Matching is case-insensitive.
    """
    escaped_phrases = [re.escape(phrase) for phrase in phrases]
    alternation = "|".join(escaped_phrases)
    return re.compile(alternation, re.IGNORECASE)

# ---------------------------------------------------------------------------
# Patterns built from YAML — edit rule_based_router_keywords.yaml to change these
# ---------------------------------------------------------------------------

# Multi-date
_COMPARISON_WORD    = _build_word_pattern(_CONFIG["multi_date"]["comparison_words"])
_MONTH_NAME         = _build_word_pattern(_CONFIG["multi_date"]["month_names"])

# Derived metric
_DERIVED_METRIC     = _build_word_pattern(_CONFIG["derived_metric"]["keywords"])

# Anomaly / diagnosis
_ANOMALY_KEYWORD    = _build_word_pattern(_CONFIG["anomaly"]["keywords"])
_ANOMALY_PHRASE     = _build_phrase_pattern(_CONFIG["anomaly"]["phrases"])

# Conditional filter
_CONDITIONAL_PHRASE = _build_phrase_pattern(_CONFIG["conditional"]["phrases"])

# ---------------------------------------------------------------------------
# Structural patterns — edit this file to change these
# ---------------------------------------------------------------------------

# Multi-date: matches "between X and Y" where X is up to 60 characters
_BETWEEN_AND        = re.compile(r"\bbetween\b.{1,60}?\band\b", re.IGNORECASE)

# Multi-date: matches quarter labels Q1, Q2, Q3, Q4
_QUARTER            = re.compile(r"\bQ[1-4]\b",                  re.IGNORECASE)

# Multi-date: matches ISO-format dates like 2024-01-15
_ISO_DATE           = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Multi-asset: matches asset IDs like WT-105, CH-06, SITE-01
_ASSET_ID           = re.compile(r"\b[A-Z]{1,6}-\d+\b")

# Conditional filter: matches operator conditions like "when Tonnage > 0"
_OPERATOR_CONDITION = re.compile(r"\b(?:when|where)\s+\w+\s*[><=!]+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

def has_multi_date(query: str) -> bool:
    """Return True if the query contains two or more distinct date expressions
    or an explicit temporal comparison phrase (e.g. 'January versus March')."""
    if _COMPARISON_WORD.search(query):
        return True
    if _BETWEEN_AND.search(query):
        return True
    n_date_tokens = (
        len(_MONTH_NAME.findall(query))
        + len(_QUARTER.findall(query))
        + len(_ISO_DATE.findall(query))
    )
    return n_date_tokens >= 2


def has_derived_metric(query: str) -> bool:
    """Return True if the query requests a computed value rather than a raw
    retrieval (e.g. 'total energy output', 'average efficiency')."""
    return bool(_DERIVED_METRIC.search(query))


def has_anomaly_keywords(query: str) -> bool:
    """Return True if the query asks about abnormal values, root cause, or
    physical plausibility (e.g. 'sensor fault', 'why is RPM out of range')."""
    return bool(_ANOMALY_KEYWORD.search(query) or _ANOMALY_PHRASE.search(query))


def has_multi_asset(query: str) -> bool:
    """Return True if the query references more than one distinct asset or site
    (e.g. 'compare WT-101 and WT-105')."""
    matches = _ASSET_ID.findall(query)
    return len(set(matches)) >= 2


def has_conditional_filter(query: str) -> bool:
    """Return True if the query filters on a secondary signal condition
    (e.g. 'during operating hours', 'when Tonnage > 0', 'non-zero only')."""
    return bool(_CONDITIONAL_PHRASE.search(query) or _OPERATOR_CONDITION.search(query))

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route(query: str) -> bool:
    """Return True if the query is complex enough to warrant reasoning mode.

    Checks all five signals and returns True if any one fires.  The threshold
    is intentionally conservative: a borderline query defaults to the reasoning
    planner (higher accuracy) rather than the standard planner (lower latency).
    """
    signals = [
        has_multi_date(query),
        has_derived_metric(query),
        has_anomaly_keywords(query),
        has_multi_asset(query),
        has_conditional_filter(query),
    ]
    return any(signals)

# ---------------------------------------------------------------------------
# Classifier adapter
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """Drop-in replacement for ThinkModeClassifier using deterministic rules.

    Implements the same should_use_thinking() interface so it can be passed
    directly to ReasonRoutingLLMBackend without any other changes.

    No model loading, no GPU, no external dependencies beyond pyyaml.
    """

    def should_use_thinking(self, text: str, threshold: float = 0.5) -> bool:
        """Return True if the query should be routed to reasoning mode.

        The threshold parameter is accepted for interface compatibility with
        ThinkModeClassifier but is ignored — the rule-based decision is
        deterministic.
        """
        return route(text)
