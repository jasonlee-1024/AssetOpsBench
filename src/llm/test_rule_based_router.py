"""Tests for the rule-based query router."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm import LLMBackend
from llm.lmtrain import ReasonRoutingLLMBackend
from llm.rule_based_router import (
    RuleBasedClassifier,
    has_anomaly_keywords,
    has_conditional_filter,
    has_derived_metric,
    has_multi_asset,
    has_multi_date,
    route,
)

# ---------------------------------------------------------------------------
# Load real vibration scenarios from the local scenarios file
# ---------------------------------------------------------------------------

_SCENARIO_FILE = Path(__file__).parent.parent / "scenarios/local/vibration_utterance.json"
_SCENARIOS = {s["id"]: s["text"] for s in json.loads(_SCENARIO_FILE.read_text())}

# ---------------------------------------------------------------------------
# Stub LLM used in integration tests — records the last prompt it received
# ---------------------------------------------------------------------------

class _EchoLLM(LLMBackend):
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        return "ok"


# ---------------------------------------------------------------------------
# has_multi_date
# ---------------------------------------------------------------------------

def test_multi_date_fires_on_versus():
    assert has_multi_date("Compare power output in January versus March") is True

def test_multi_date_fires_on_compared_to():
    assert has_multi_date("Efficiency in Q1 compared to Q2") is True

def test_multi_date_fires_on_between_and():
    assert has_multi_date("Show vibration data between 2024-01-01 and 2024-06-01") is True

def test_multi_date_fires_on_two_month_names():
    assert has_multi_date("What changed between January and March?") is True

def test_multi_date_fires_on_two_iso_dates():
    assert has_multi_date("Get readings from 2024-01-15 and 2024-07-20") is True

def test_multi_date_fires_on_mixed_date_tokens():
    assert has_multi_date("Compare January data with Q2 results") is True

def test_multi_date_silent_on_single_month():
    assert has_multi_date("Get data from January 2024") is False

def test_multi_date_silent_on_no_date():
    assert has_multi_date("List all sensors on WT-105") is False

def test_multi_date_silent_on_between_without_and():
    # "between" alone should not fire without a following "and"
    assert has_multi_date("What happened between the two shutdowns?") is False


# ---------------------------------------------------------------------------
# has_derived_metric
# ---------------------------------------------------------------------------

def test_derived_metric_fires_on_total():
    assert has_derived_metric("What is the total energy production this month?") is True

def test_derived_metric_fires_on_average():
    assert has_derived_metric("What is the average efficiency across all turbines?") is True

def test_derived_metric_fires_on_kwh():
    assert has_derived_metric("How many kWh did WT-105 produce yesterday?") is True

def test_derived_metric_fires_on_estimate():
    assert has_derived_metric("Estimate the remaining useful life of the gearbox") is True

def test_derived_metric_silent_on_raw_retrieval():
    assert has_derived_metric("List all sensors installed on WT-105") is False

def test_derived_metric_silent_on_status_query():
    assert has_derived_metric("What is the current wind speed at the farm?") is False

def test_derived_metric_no_partial_word_match():
    # "totally" contains "total" — word boundary must prevent a match
    assert has_derived_metric("I am totally unsure about this reading") is False


# ---------------------------------------------------------------------------
# has_anomaly_keywords
# ---------------------------------------------------------------------------

def test_anomaly_fires_on_abnormal():
    assert has_anomaly_keywords("Is there an abnormal vibration in WT-105?") is True

def test_anomaly_fires_on_why():
    assert has_anomaly_keywords("Why is the generator temperature so high?") is True

def test_anomaly_fires_on_fault_keyword():
    assert has_anomaly_keywords("Detect any faults in the pitch control system") is True

def test_anomaly_fires_on_out_of_range_phrase():
    assert has_anomaly_keywords("The RPM reading appears to be out of range") is True

def test_anomaly_fires_on_sensor_fault_phrase():
    assert has_anomaly_keywords("A sensor fault was logged on WT-105 at noon") is True

def test_anomaly_silent_on_retrieval_query():
    assert has_anomaly_keywords("Get vibration data for WT-105 from last week") is False

def test_anomaly_silent_on_alarm_list():
    assert has_anomaly_keywords("List all alarm events for WT-105 this week") is False

def test_anomaly_no_partial_word_match():
    # "default" contains "fault" — word boundary must prevent a match
    assert has_anomaly_keywords("The default configuration is active") is False


# ---------------------------------------------------------------------------
# has_multi_asset
# ---------------------------------------------------------------------------

def test_multi_asset_fires_on_two_turbine_ids():
    assert has_multi_asset("Compare the efficiency of WT-101 and WT-105") is True

def test_multi_asset_fires_on_three_ids():
    assert has_multi_asset("Show gearbox faults for WT-101, WT-103, and WT-107") is True

def test_multi_asset_silent_on_single_asset():
    assert has_multi_asset("Get vibration data for WT-105") is False

def test_multi_asset_silent_on_category_reference():
    # "all turbines" is a category, not two specific asset IDs
    assert has_multi_asset("List all turbines at the wind farm") is False

def test_multi_asset_silent_on_repeated_same_id():
    # WT-105 appears twice but is the same asset — set deduplication prevents false positive
    assert has_multi_asset("Compare WT-105 this week with WT-105 last week") is False


# ---------------------------------------------------------------------------
# has_conditional_filter
# ---------------------------------------------------------------------------

def test_conditional_fires_on_operating_hours():
    assert has_conditional_filter("Show power output during operating hours only") is True

def test_conditional_fires_on_operator_expression():
    assert has_conditional_filter("Get tonnage readings when Tonnage > 0") is True

def test_conditional_fires_on_non_zero():
    assert has_conditional_filter("List sensors with non-zero readings") is True

def test_conditional_fires_on_only_when():
    assert has_conditional_filter("Retrieve alarms only when the turbine is running") is True

def test_conditional_fires_on_where_operator():
    assert has_conditional_filter("Show records where RPM != 0") is True

def test_conditional_silent_on_unconditional_retrieval():
    assert has_conditional_filter("Get all power output readings for WT-105") is False

def test_conditional_silent_on_temporal_when():
    # "When did" is a temporal question, not a conditional filter on a secondary signal
    assert has_conditional_filter("When did the last fault occur on WT-105?") is False


# ---------------------------------------------------------------------------
# route() — real vibration scenarios
#
# Expected labels were assigned by inspecting each query against the five
# signal definitions. Scenarios marked True fire at least one signal;
# scenarios marked False fire none with the current keyword set.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id, expected", [
    # Simple knowledge queries — no signal fires
    (301, False),  # "What vibration analysis capabilities are available?"
    (302, False),  # "What bearings are available in the built-in database?"
    (309, False),  # "What vibration sensors are available for Motor_01?"
    # Complex queries — anomaly signal fires
    (320, True),   # "Check for outer race bearing fault..." — keyword: fault
    (321, True),   # "...identify the most likely root cause" — phrase: root cause
    (323, True),   # "...outer race fault is suspected..." — keyword: fault
])
def test_route_on_vibration_scenario(scenario_id: int, expected: bool):
    text = _SCENARIOS[scenario_id]
    assert route(text) is expected, f"Scenario {scenario_id}: {text!r}"


# ---------------------------------------------------------------------------
# Integration: RuleBasedClassifier + ReasonRoutingLLMBackend
# ---------------------------------------------------------------------------

def test_integration_complex_query_appends_think_trigger():
    base = _EchoLLM()
    router = ReasonRoutingLLMBackend(base, RuleBasedClassifier(), think_trigger="</think>")
    router.generate("Why is the vibration on WT-105 abnormal compared to last month?")
    assert base.last_prompt.endswith("</think>")

def test_integration_simple_query_passes_through_unchanged():
    base = _EchoLLM()
    router = ReasonRoutingLLMBackend(base, RuleBasedClassifier(), think_trigger="</think>")
    router.generate("List all sensors on WT-105")
    assert base.last_prompt == "List all sensors on WT-105"

def test_integration_does_not_double_append_existing_trigger():
    base = _EchoLLM()
    router = ReasonRoutingLLMBackend(base, RuleBasedClassifier(), think_trigger="</think>")
    router.generate("Why is the RPM out of range? </think>")
    assert base.last_prompt.count("</think>") == 1
