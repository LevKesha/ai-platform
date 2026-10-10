"""Helper slot is required and pinned to the proven Haiku id."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from platform_common.registry import load_model_mix  # noqa: E402

HELPER_BEDROCK_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_platform_config",
        ROOT / "scripts" / "check-platform-config.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_helper_slot_is_proven_haiku() -> None:
    mix = load_model_mix(ROOT / "platform-config.yaml")
    assert mix["helper_bedrock_id"] == HELPER_BEDROCK_ID
    assert mix["helper_litellm_model_name"] == HELPER_BEDROCK_ID
    assert mix["helper_anthropic_api_id"] == HELPER_BEDROCK_ID
    assert mix["model_id"] == mix["default_bedrock_id"]
    assert mix["helper_bedrock_id"] != mix["default_bedrock_id"]
    assert mix["helper_bedrock_id"] != mix["max_bedrock_id"]


def test_helper_slot_is_required(tmp_path: Path) -> None:
    text = (ROOT / "platform-config.yaml").read_text(encoding="utf-8")
    stripped = re.sub(r"\n    helper:\n(?:      .+\n)+", "\n", text)
    assert "helper:" not in stripped
    path = tmp_path / "platform-config.yaml"
    path.write_text(stripped, encoding="utf-8")
    with pytest.raises(ValueError, match=r"llm\.slots\.helper is required"):
        load_model_mix(path)


def test_deprecated_id_only_allowed_in_denylist() -> None:
    checker = _checker()
    deprecated = [str(item) for item in checker.load_doc()["deprecated_model_ids"]]
    config = (ROOT / "platform-config.yaml").read_text(encoding="utf-8")
    assert checker.deprecated_hits(checker.without_denylist(config), deprecated) == []
    sample = "model_id: " + deprecated[0]
    assert checker.deprecated_hits(sample, deprecated) == [deprecated[0]]


def test_checker_passes_and_ignores_cursor_page() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check-platform-config.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cursor/index.html" not in proc.stdout + proc.stderr
