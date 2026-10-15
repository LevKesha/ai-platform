"""The chat cap reservation is priced from llm.slots, not a family argument."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages"))

from chat_helper.pricing import (  # noqa: E402
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    RATES,
    _input_tokens,
    actual_micro,
    fit_input,
    max_output_tokens,
    rate_family,
    rates_for_slot,
    worst_case_micro,
)
from platform_common.registry import SLOT_NAMES, load_model_mix  # noqa: E402


def test_each_slot_rate_family_matches_its_model_id() -> None:
    mix = load_model_mix(ROOT / "platform-config.yaml")
    priced: list[str] = []
    for name in SLOT_NAMES:
        model_id = mix[f"{name}_bedrock_id"]
        family = rate_family(model_id)
        assert family in model_id
        assert rate_family(mix[f"{name}_anthropic_api_id"]) == family
        if family in RATES:
            assert rates_for_slot(name) == RATES[family]
            priced.append(name)
        else:
            with pytest.raises(ValueError, match=rf"llm\.slots\.{name}"):
                rates_for_slot(name)
    assert priced == ["default", "max", "helper"]


def test_reservation_is_priced_for_the_slot() -> None:
    helper = worst_case_micro("q", [], "system", "helper")
    live = worst_case_micro("q", [], "system", "default")
    assert helper == _manual("helper")
    assert live == _manual("default")
    assert live > helper
    with pytest.raises(ValueError, match=r"llm\.slots\.sonnet"):
        worst_case_micro("q", [], "system", "sonnet")
    with pytest.raises(ValueError, match=r"llm\.slots\.haiku"):
        actual_micro({"input_tokens": 1, "output_tokens": 1}, "haiku")


def test_output_cap_defaults_and_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)
    assert max_output_tokens() == MAX_OUTPUT_TOKENS == 400
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "400")
    assert max_output_tokens() == 400
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "32")
    assert max_output_tokens() == 32
    assert worst_case_micro("q", [], "system", "helper") == _manual("helper")


def test_bedrock_input_is_trimmed_to_about_8k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)
    question, history, system = fit_input("q" * 4000, [{"role": "user", "text": "h" * 4000}], "s" * 80000)
    assert _input_tokens(question, history, system) <= MAX_INPUT_TOKENS


def _manual(slot: str) -> int:
    in_rate, out_rate = rates_for_slot(slot)
    tokens = (len("system") + len("q")) // 4 + 1
    return tokens * in_rate + max_output_tokens() * out_rate
