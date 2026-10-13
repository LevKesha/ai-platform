"""Workflow-literal scan self-test runs in CI; the full scan needs sibling clones."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_workflow_literal_self_test() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check-workflow-literals.py"), "--self-test-only"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "self-test ok" in proc.stdout
