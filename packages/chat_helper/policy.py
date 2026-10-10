"""The one chat policy. Lambda, UI, and tests read this file."""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED = (
    "turn_limit",
    "live_answers",
    "cap_micro",
    "chat_ttl_hours",
    "unanswered_ttl_days",
    "privacy",
    "opener",
    "suggested",
    "label_cv",
    "label_live",
    "off_topic",
    "not_in_cv",
    "closed",
    "link_text",
    "link_href",
    "switch",
    "placeholder",
    "bubble_name",
)


def find_policy(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        candidate = parent / "olympus" / "chat-policy.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("olympus/chat-policy.json")


def load_policy(path: str | Path | None = None) -> dict:
    file = Path(path) if path else find_policy()
    data = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("chat policy must be an object")
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError("chat policy missing " + ", ".join(missing))
    return data
