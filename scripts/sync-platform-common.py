#!/usr/bin/env python3
"""Vendor packages/platform_common into sibling service build contexts.

SSOT: ai-platform/packages/platform_common/
Do not edit copies under agent-api / rag-service / mcp-server / agents / claude-router.

Usage:
  python scripts/sync-platform-common.py          # copy
  python scripts/sync-platform-common.py --check  # exit 1 if drift
"""
from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "platform_common"
PARENT = ROOT.parent

TARGETS = [
    PARENT / "agent-api" / "platform_common",
    PARENT / "rag-service" / "platform_common",
    PARENT / "mcp-server" / "platform_common",
    ROOT / "claude-router" / "platform_common",
    ROOT / "agents" / "support-runbook-copilot" / "platform_common",
]


def sync_one(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SRC, dest)
    for p in dest.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    print(f"synced {dest}")


def check_one(dest: Path) -> bool:
    if not dest.is_dir():
        print(f"MISSING {dest}", file=sys.stderr)
        return False
    cmp = filecmp.dircmp(SRC, dest, ignore=["__pycache__"])
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        print(f"DRIFT {dest}", file=sys.stderr)
        print(
            f"  left_only={cmp.left_only} right_only={cmp.right_only} diff={cmp.diff_files}",
            file=sys.stderr,
        )
        return False
    print(f"ok {dest}")
    return True


def main() -> int:
    if not SRC.is_dir():
        print(f"missing SSOT {SRC}", file=sys.stderr)
        return 1
    check = "--check" in sys.argv
    ok = True
    for dest in TARGETS:
        if not dest.parent.is_dir():
            print(f"skip (no parent): {dest}")
            continue
        if check:
            ok = check_one(dest) and ok
        else:
            sync_one(dest)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
