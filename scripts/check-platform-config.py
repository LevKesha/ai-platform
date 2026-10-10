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
sys.path.insert(0, str(ROOT / "packages"))

from platform_common.registry import SLOT_NAMES, load_model_mix  # noqa: E402

# Proven helper slot. Access gate: Converse in eu-central-1 returned OK.
HELPER_BEDROCK_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

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
    ("llm-cost", "k8s/helm/litellm/files/litellm_config.yaml", "litellm_model"),
]

# HEADROOM_PROBE_MODEL must stay on the default slot until Phase C.
PROBE_CHECKS: list[tuple[str, str, str]] = [
    ("llm-cost", "k8s/helm/headroom-savings/values.yaml", "helm"),
    ("llm-cost", "savings/app.py", "getenv"),
    ("llm-cost", "scripts/develop_headroom_traffic.py", "getenv"),
    ("llm-cost", "scripts/probe_compress.py", "getenv"),
    ("agent-api", "scripts/develop_cv_jobs_headroom_traffic.py", "getenv"),
]

# Anthropic API short ids (not Bedrock). Default slot only.
SHORT_ID_CHECKS: list[tuple[str, str, str]] = [
    ("agent-api", "llm/providers/anthropic.py", r'model_id:\s*str\s*=\s*"([^"]+)"'),
    ("agent-api", "llm/providers/factory.py", r'anthropic_model_id:\s*str\s*=\s*"([^"]+)"'),
    ("agent-api", "orchestrator/config/settings.py", r'anthropic_model_id:\s*str\s*=\s*"([^"]+)"'),
]

OLYMPUS_MODEL_ID_FILES = (
    "olympus/console-data.js",
    "olympus/public-data.js",
)


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
        rf'(?:getenv|environ\.get)\(\s*["\']{re.escape(env_var)}["\']\s*,\s*["\']([^"\']+)["\']',
        text,
    )
    return m.group(1) if m else None


def litellm_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    name: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.search(r"\bmodel_name:\s*([^\s#]+)", s)
        if m:
            name = m.group(1)
            continue
        m = re.search(r"\bmodel:\s*([^\s#]+)", s)
        if m and name:
            rows.append((name, m.group(1)))
            name = None
    return rows


def repo_path(repo: str, rel: str) -> Path:
    return (ROOT if repo == "ai-platform" else PARENT / repo) / rel


def js_string_field(text: str, field: str) -> list[str]:
    """Quoted JS field. Boundary so maxModelId is not read as modelId."""
    return re.findall(
        rf'(?<![A-Za-z0-9_]){re.escape(field)}:\s*"([^"]+)"',
        text,
    )


def main() -> int:
    llm = load_ssot()
    want_claude = llm["model_id"]
    want_embed = llm["embedding_model_id"]
    claude_var = llm["env_var"]
    embed_var = llm["embedding_env_var"]
    errors: list[str] = []

    try:
        mix = load_model_mix(CONFIG)
    except (OSError, KeyError, ValueError) as exc:
        print(f"SSOT drift detected:\n  - registry mix: {exc}")
        return 1

    slots = llm.get("slots") or {}
    for name in SLOT_NAMES:
        slot = slots.get(name) or {}
        for field in ("litellm_model_name", "bedrock_id", "anthropic_api_id"):
            if mix.get(f"{name}_{field}") != slot.get(field):
                errors.append(
                    f"registry loader {name}.{field} want={slot.get(field)!r} got={mix.get(f'{name}_{field}')!r}"
                )
    if mix["model_id"] != want_claude:
        errors.append(f"registry loader model_id want={want_claude!r} got={mix['model_id']!r}")
    if want_claude != (slots.get("default") or {}).get("bedrock_id"):
        errors.append("llm.model_id must alias slots.default.bedrock_id")
    if mix.get("helper_bedrock_id") != HELPER_BEDROCK_ID:
        errors.append(
            f"llm.slots.helper.bedrock_id want={HELPER_BEDROCK_ID!r} got={mix.get('helper_bedrock_id')!r}"
        )
    if (slots.get("helper") or {}).get("bedrock_id") != HELPER_BEDROCK_ID:
        errors.append(
            f"platform-config helper bedrock_id want={HELPER_BEDROCK_ID!r} got={(slots.get('helper') or {}).get('bedrock_id')!r}"
        )

    want_short = mix["default_anthropic_api_id"]

    # Forbidden: stale tree under ai-platform
    stale = ROOT / "rag-service"
    if stale.exists():
        errors.append(f"stale tree must be removed: {stale}")

    for repo, rel, kind in CHECKS:
        path = repo_path(repo, rel)
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
            rows = litellm_rows(text)
            want_rows = {
                (
                    mix["default_litellm_model_name"],
                    f"bedrock/converse/{mix['default_bedrock_id']}",
                ),
                (
                    mix["max_litellm_model_name"],
                    f"bedrock/converse/{mix['max_bedrock_id']}",
                ),
            }
            if set(rows) != want_rows:
                errors.append(f"{path}: model_list want={sorted(want_rows)!r} got={rows!r}")
            continue
        if kind == "claude_default":
            got = find_default_getenv(text, claude_var)
            if got != want_claude:
                errors.append(f"{path}: getenv default want={want_claude!r} got={got!r}")
        if kind in ("helm_both", "env_both"):
            got_e = find_embedding(text, embed_var)
            if got_e != want_embed:
                errors.append(f"{path}: {embed_var} want={want_embed!r} got={got_e!r}")

    for repo, rel, kind in PROBE_CHECKS:
        path = repo_path(repo, rel)
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        got = find_model(text, "HEADROOM_PROBE_MODEL") if kind == "helm" else find_default_getenv(text, "HEADROOM_PROBE_MODEL")
        if got != want_claude:
            errors.append(f"{path}: HEADROOM_PROBE_MODEL want={want_claude!r} got={got!r}")

    for repo, rel, pattern in SHORT_ID_CHECKS:
        path = repo_path(repo, rel)
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        found = re.findall(pattern, text)
        if found != [want_short]:
            errors.append(f"{path}: anthropic short id want={want_short!r} got={found!r}")

    for rel in OLYMPUS_MODEL_ID_FILES:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        found = js_string_field(text, "modelId")
        if found != [want_claude]:
            errors.append(f"{path}: modelId want={want_claude!r} got={found!r}")
        max_found = js_string_field(text, "maxModelId")
        want_max = mix["max_bedrock_id"]
        if rel == "olympus/console-data.js":
            if max_found != [want_max]:
                errors.append(f"{path}: maxModelId want={[want_max]!r} got={max_found!r}")
            slot = js_string_field(text, "maxSlot")
            if slot != [mix["max_litellm_model_name"]]:
                errors.append(
                    f"{path}: maxSlot want={[mix['max_litellm_model_name']]!r} got={slot!r}"
                )
        elif max_found and max_found != [want_max]:
            errors.append(f"{path}: maxModelId want={[want_max]!r} got={max_found!r}")

    cursor_html = ROOT / "olympus" / "cursor" / "index.html"
    if not cursor_html.exists():
        errors.append(f"missing: {cursor_html}")
    else:
        cursor_text = cursor_html.read_text(encoding="utf-8")
        ids = re.findall(r"eu\.anthropic\.[A-Za-z0-9._:-]+", cursor_text)
        want_ids = [want_claude, mix["max_bedrock_id"]]
        if ids != want_ids:
            errors.append(f"{cursor_html}: bedrock ids want={want_ids!r} got={ids!r}")
        if mix["max_litellm_model_name"] not in cursor_text:
            errors.append(
                f"{cursor_html}: missing max slot name {mix['max_litellm_model_name']!r}"
            )

    if errors:
        print("SSOT drift detected:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: platform-config.yaml LLM SSOT matches checked consumers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
