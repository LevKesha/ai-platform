#!/usr/bin/env python3
"""Fail if sibling repos drift from platform-config.yaml LLM SSOT."""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "platform-config.yaml"
PARENT = ROOT.parent

# Paths relative to each sibling repo (or ai-platform itself)
CHECKS: list[tuple[str, str, str]] = [
    # (repo_dir, relative_file, pattern_kind)
    ("ai-platform", "claude-router/k8s/deployment.yaml", "claude"),
    ("ai-platform", "claude-router/app/main.py", "claude_default"),
    ("ai-platform", "agents/support-runbook-copilot/k8s/deployment.yaml", "claude"),
    ("ai-platform", "agents/support-runbook-copilot/app/main.py", "claude_default"),
    ("ai-platform", "agents/support-runbook-copilot/agent-spec.yaml", "spec_model"),
    ("agent-api", "agent-spec.yaml", "spec_model"),
    ("agent-api", "k8s/helm/values.yaml", "helm_claude"),
    ("agent-api", ".env.example", "env_claude"),
    ("rag-service", "agent-spec.yaml", "spec_model"),
    ("rag-service", "k8s/helm/values.yaml", "helm_both"),
    ("rag-service", ".env.example", "env_both"),
    ("infrastructure", "charts/rag-service/values.yaml", "helm_both"),
    ("llm-cost", "k8s/helm/litellm/files/litellm_config.yaml", "litellm_model"),
]


def load_ssot() -> dict:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return data["llm"]


def find_model(text: str, env_var: str) -> str | None:
    # K8s: - name: VAR \n value: id
    m = re.search(
        rf"name:\s*{re.escape(env_var)}\s*\n\s*value:\s*([^\s#]+)",
        text,
    )
    if m:
        return m.group(1).rstrip("\"'")
    # Active (non-comment) env / yaml assignment
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(rf"{re.escape(env_var)}\s*[:=]\s*[\"']?([^\s\"'#]+)", s)
        if m:
            return m.group(1).rstrip("\"'")
    return None


def find_embedding(text: str, env_var: str) -> str | None:
    m = re.search(
        rf"name:\s*{re.escape(env_var)}\s*\n\s*value:\s*([^\s#]+)",
        text,
    )
    if m:
        return m.group(1).rstrip("\"'")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(rf"{re.escape(env_var)}\s*[:=]\s*[\"']?([^\s\"'#]+)", s)
        if m:
            return m.group(1).rstrip("\"'")
    return None


def find_default_getenv(text: str, env_var: str) -> str | None:
    m = re.search(
        rf'getenv\(\s*["\']{re.escape(env_var)}["\']\s*,\s*["\']([^"\']+)["\']',
        text,
    )
    return m.group(1) if m else None


def main() -> int:
    llm = load_ssot()
    want_claude = llm["model_id"]
    want_embed = llm["embedding_model_id"]
    claude_var = llm["env_var"]
    embed_var = llm["embedding_env_var"]
    errors: list[str] = []

    # Forbidden: stale tree under ai-platform
    stale = ROOT / "rag-service"
    if stale.exists():
        errors.append(f"stale tree must be removed: {stale}")

    for repo, rel, kind in CHECKS:
        path = (ROOT if repo == "ai-platform" else PARENT / repo) / rel
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if kind in ("claude", "helm_claude", "env_claude", "spec_model", "helm_both", "env_both"):
            got = find_model(text, claude_var)
            if kind == "spec_model":
                m = re.search(r"^\s*model:\s*([^\s#]+)", text, re.M)
                got = m.group(1) if m else None
            if got != want_claude:
                errors.append(f"{path}: {claude_var}/model want={want_claude!r} got={got!r}")
        if kind == "litellm_model":
            m = re.search(r"model_name:\s*([^\s#]+)", text)
            got = m.group(1) if m else None
            if got != want_claude:
                errors.append(f"{path}: model_name want={want_claude!r} got={got!r}")
            continue
        if kind == "claude_default":
            got = find_default_getenv(text, claude_var)
            if got != want_claude:
                errors.append(f"{path}: getenv default want={want_claude!r} got={got!r}")
        if kind in ("helm_both", "env_both"):
            got_e = find_embedding(text, embed_var)
            if got_e != want_embed:
                errors.append(f"{path}: {embed_var} want={want_embed!r} got={got_e!r}")

    if errors:
        print("SSOT drift detected:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: platform-config.yaml LLM SSOT matches checked consumers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
