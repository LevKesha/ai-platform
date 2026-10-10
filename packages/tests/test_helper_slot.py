"""Helper slot is the proven Haiku id from platform-config.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from platform_common.registry import load_model_mix  # noqa: E402

HELPER_BEDROCK_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_helper_slot_is_proven_haiku() -> None:
    mix = load_model_mix(ROOT / "platform-config.yaml")
    assert mix["helper_bedrock_id"] == HELPER_BEDROCK_ID
    assert mix["helper_litellm_model_name"] == HELPER_BEDROCK_ID
    assert mix["helper_anthropic_api_id"] == HELPER_BEDROCK_ID
    assert mix["model_id"] == mix["default_bedrock_id"]
    assert mix["helper_bedrock_id"] != mix["default_bedrock_id"]
    assert mix["helper_bedrock_id"] != mix["max_bedrock_id"]
