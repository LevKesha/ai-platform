"""Application spend rates, in integer micro-dollars per token.

Haiku is 1 in / 5 out. Sonnet-class live answers are 3 in / 15 out.
The reserve uses the rate of the model about to be called, over the bounded
input plus the output bound. The cap itself lives in the chat policy.
"""
from __future__ import annotations

MAX_QUESTION_CHARS = 500
MAX_HISTORY_MESSAGES = 8
MAX_OUTPUT_TOKENS = 400

RATES = {
    "haiku": (1, 5),
    "sonnet": (3, 15),
}


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


def worst_case_micro(question: str, history: list[dict[str, str]], system: str, kind: str) -> int:
    in_rate, out_rate = RATES[kind]
    tokens = _input_tokens(question, history, system)
    return tokens * in_rate + MAX_OUTPUT_TOKENS * out_rate


def actual_micro(usage: dict, kind: str) -> int:
    in_rate, out_rate = RATES[kind]
    return int(usage.get("input_tokens", 0)) * in_rate + int(usage.get("output_tokens", 0)) * out_rate
