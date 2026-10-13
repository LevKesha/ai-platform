"""Classifier self-test. The live probe is the workflow step, not this test."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _smoke():
    path = ROOT / "scripts" / "smoke-direct-alb.py"
    spec = importlib.util.spec_from_file_location("smoke_direct_alb", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_alb_classifier_self_test() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke-direct-alb.py"), "--self-test-only"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "self-test ok" in proc.stdout


def test_probe_checks_https_then_port_80(monkeypatch) -> None:
    smoke = _smoke()
    seen: list[str] = []

    def fake_curl(_curl: str, url: str) -> tuple[int, str]:
        seen.append(url)
        return 28, "000"

    monkeypatch.setattr(smoke, "curl_once", fake_curl)
    monkeypatch.setattr(smoke.shutil, "which", lambda name: "curl")
    smoke.probe("alb.example")
    assert seen == ["https://alb.example/", "http://alb.example:80/"]


def test_https_failure_does_not_continue_to_port_80(monkeypatch) -> None:
    smoke = _smoke()
    seen: list[str] = []

    def fake_curl(_curl: str, url: str) -> tuple[int, str]:
        seen.append(url)
        return 0, "200"

    monkeypatch.setattr(smoke, "curl_once", fake_curl)
    monkeypatch.setattr(smoke.shutil, "which", lambda name: "curl")
    try:
        smoke.probe("alb.example")
    except SystemExit:
        pass
    else:
        raise AssertionError("https 200 must fail closed")
    assert seen == ["https://alb.example/"]
