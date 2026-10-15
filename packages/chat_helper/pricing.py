"""Application spend rates, in integer micro-dollars per token.

The $3 chat cap reservation is priced for the slot about to be called.
The slot's model ID lives in platform-config.yaml llm.slots. The rate family
is the family token inside that model ID. Callers do not pass a family name.
Haiku is 1 in / 5 out. Sonnet-class live answers are 3 in / 15 out.
The reserve uses that slot rate over the bounded input plus the output bound.
The cap itself lives in the chat policy.
"""
from __future__ import annotations

import os
import re

MAX_QUESTION_CHARS = 500
MAX_HISTORY_MESSAGES = 8
# Default only. max_output_tokens() is the owner; Lambda env may set the same number.
MAX_OUTPUT_TOKENS = 400
# About 8k tokens. Every Bedrock call is trimmed to this before converse.
MAX_INPUT_TOKENS = 8000

# Application rates. The family name must occur in the slot's model ID.
RATES = {
    "haiku": (1, 5),
    "sonnet": (3, 15),
}
FAMILIES = ("haiku", "sonnet", "opus")


def bound_question(question: str) -> str:
    return question[:MAX_QUESTION_CHARS]


def bound_history(history: list) -> list[dict[str, str]]:
    kept = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = str(item.get("text") or "")[:MAX_QUESTION_CHARS]
        kept.append({"role": role, "text": text})
    return kept


def _input_tokens(question: str, history: list[dict[str, str]], system: str) -> int:
    chars = len(system) + len(question) + sum(len(item["text"]) for item in history)
    return chars // 4 + 1


def max_output_tokens() -> int:
    """Output cap sent on every Bedrock converse. Env overrides the default."""
    raw = os.environ.get("MAX_OUTPUT_TOKENS", "").strip()
    if raw == "":
        return MAX_OUTPUT_TOKENS
    value = int(raw)
    if value < 1:
        raise ValueError("MAX_OUTPUT_TOKENS must be a positive integer")
    return value


def fit_input(
    question: str, history: list[dict[str, str]], system: str
) -> tuple[str, list[dict[str, str]], str]:
    """Trim history, then the question, then the system text until the estimate fits."""
    kept = list(history)
    while _input_tokens(question, kept, system) > MAX_INPUT_TOKENS and kept:
        kept = kept[1:]
    while _input_tokens(question, kept, system) > MAX_INPUT_TOKENS and question:
        question = question[: len(question) // 2]
    while _input_tokens(question, kept, system) > MAX_INPUT_TOKENS and system:
        system = system[: len(system) // 2]
    return question, kept, system


def rate_family(model_id: str) -> str:
    """The one family token inside a slot model ID."""
    found = [
        family
        for family in FAMILIES
        if re.search(rf"(?:^|[^a-z]){family}(?:[^a-z]|$)", model_id)
    ]
    if len(found) != 1:
        raise ValueError(f"model id {model_id!r} must name one rate family")
    return found[0]


def rates_for_slot(slot: str) -> tuple[int, int]:
    """Rates for one llm.slots entry. The model ID selects the family."""
    from platform_common.registry import load_model_mix

    mix = load_model_mix()
    key = f"{slot}_bedrock_id"
    if key not in mix:
        raise ValueError(f"llm.slots.{slot} is required")
    model_id = mix[key]
    family = rate_family(model_id)
    if family not in RATES:
        raise ValueError(f"llm.slots.{slot} model {model_id} has no application rate")
    return RATES[family]


def worst_case_micro(question: str, history: list[dict[str, str]], system: str, slot: str) -> int:
    in_rate, out_rate = rates_for_slot(slot)
    tokens = _input_tokens(question, history, system)
    return tokens * in_rate + max_output_tokens() * out_rate


def actual_micro(usage: dict, slot: str) -> int:
    in_rate, out_rate = rates_for_slot(slot)
    return int(usage.get("input_tokens", 0)) * in_rate + int(usage.get("output_tokens", 0)) * out_rate
