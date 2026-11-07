#!/usr/bin/env python3
"""Vendor packages/platform_common into sibling service build contexts.

SSOT: ai-platform/packages/platform_common/
Do not edit copies under agent-api / rag-service / mcp-server / agents / claude-router / llm-cost/spend.

Usage:
  python scripts/sync-platform-common.py          # copy
  python scripts/sync-platform-common.py --check  # exit 1 if drift
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "platform_common"
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from platform_repos import PLATFORM_REPOS  # noqa: E402

# Repos that vendor platform_common. The other PLATFORM_REPOS must not grow a copy.
VENDOR_REPOS = ("agent-api", "rag-service", "mcp-server", "llm-cost")

TARGETS = [
    PARENT / "agent-api" / "platform_common",
    PARENT / "rag-service" / "platform_common",
    PARENT / "mcp-server" / "platform_common",
    ROOT / "claude-router" / "platform_common",
    ROOT / "agents" / "support-runbook-copilot" / "platform_common",
    PARENT / "llm-cost" / "spend" / "platform_common",
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
    require = os.environ.get("PLATFORM_CONFIG_REQUIRE_SIBLINGS") == "1"
    ok = True
    for dest in TARGETS:
        if not dest.parent.is_dir():
            print(f"skip (no parent): {dest}")
            if require:
                ok = False
            continue
        if check:
            ok = check_one(dest) and ok
        else:
            sync_one(dest)
    for name in PLATFORM_REPOS:
        if name in VENDOR_REPOS or name == "ai-platform":
            continue
        repo = PARENT / name
        if not repo.is_dir():
            if require:
                print(f"MISSING {repo}", file=sys.stderr)
                ok = False
            else:
                print(f"skip (no repo): {name}")
            continue
        extra = repo / "platform_common"
        if extra.is_dir():
            print(f"DRIFT unexpected {extra}", file=sys.stderr)
            ok = False
        else:
            print(f"ok no vendor {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
