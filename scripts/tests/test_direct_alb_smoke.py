"""Classifier self-test. The live probe is the workflow step, not this test."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_direct_alb_classifier_self_test() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke-direct-alb.py"), "--self-test-only"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "self-test ok" in proc.stdout
