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


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_platform_config",
        ROOT / "scripts" / "check-platform-config.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_helper_slot_is_read_from_the_config() -> None:
    mix = load_model_mix(ROOT / "platform-config.yaml")
    helper = mix["helper_bedrock_id"]
    assert mix["helper_litellm_model_name"] == helper
    assert mix["helper_anthropic_api_id"] == helper
    assert mix["model_id"] == mix["default_bedrock_id"]
    assert helper != mix["default_bedrock_id"]
    assert helper != mix["max_bedrock_id"]
    assert mix["helper_invoke"] == "direct-bedrock"
    checker = (ROOT / "scripts" / "check-platform-config.py").read_text(encoding="utf-8")
    assert helper not in checker
    assert helper not in Path(__file__).read_text(encoding="utf-8")


def test_litellm_rows_are_default_and_max_only() -> None:
    checker = _checker()
    mix = load_model_mix(ROOT / "platform-config.yaml")
    rows = checker.litellm_want_rows(mix)
    assert len(rows) == 2
    blob = " ".join(name + " " + model for name, model in rows)
    assert mix["helper_bedrock_id"] not in blob
    assert mix["helper_invoke"] == "direct-bedrock"


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


def test_edge_timeouts_stay_out_of_the_model_mix() -> None:
    from platform_common.registry import _parse_simple

    checker = _checker()
    text = (ROOT / "platform-config.yaml").read_text(encoding="utf-8")
    parsed = _parse_simple(text)
    doc = checker.load_doc()
    mix = load_model_mix(ROOT / "platform-config.yaml")
    assert parsed["aws"]["region"] == doc["aws"]["region"]
    assert parsed["llm"]["model_id"] == mix["model_id"]
    assert "edge" not in mix
    assert parsed["edge"]["olympus_admin"]["cf_origin_read_timeout"] == "120"
    assert parsed["edge"]["olympus_admin"]["alb_idle_timeout"] == "180"
    want_cf, want_idle = checker.olympus_admin_timeouts(doc)
    assert (want_cf, want_idle) == (120, 180)
    covered = {(repo, rel, kind) for repo, rel, kind in checker.CHECKS}
    assert ("llm-cost", "README.md", "readme_ssot_model") in covered
    assert ("agent-api", "k8s/helm/templates/ingress.yaml", "alb_idle") in covered
    assert ("llm-cost", "k8s/helm/litellm/templates/ingress.yaml", "alb_idle") in covered
    assert ("llm-cost", "k8s/helm/headroom-savings/templates/ingress.yaml", "alb_idle") in covered
    sample = "load-balancer-attributes: idle_timeout.timeout_seconds=180"
    assert checker.idle_timeout_seconds(sample) == ["180"]
    assert checker.idle_timeout_seconds("# idle_timeout.timeout_seconds=120\n") == []


def test_readme_ssot_model_reads_the_backticked_line() -> None:
    checker = _checker()
    want = load_model_mix(ROOT / "platform-config.yaml")["model_id"]
    text = f"SSOT text model (from platform-config.yaml):\n`{want}`\n"
    assert checker.readme_ssot_model(text) == want
    table = f"| `CLAUDE_MODEL_ID` | `{want}` |"
    assert checker.readme_default(table, "CLAUDE_MODEL_ID") == want
    assert checker.readme_ssot_model(table) is None
    assert checker.readme_default(text, "CLAUDE_MODEL_ID") is None
    covered = {(repo, rel, kind) for repo, rel, kind in checker.CHECKS}
    assert ("ai-platform", "claude-router/README.md", "readme_claude") in covered
    assert ("llm-cost", "README.md", "readme_ssot_model") in covered


def test_readme_ssot_model_fails_on_a_wrong_id() -> None:
    checker = _checker()
    want = load_model_mix(ROOT / "platform-config.yaml")["model_id"]
    text = "SSOT text model (from platform-config.yaml):\n`not-the-slot`\n"
    got = checker.readme_ssot_model(text)
    assert got == "not-the-slot"
    assert got != want


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
