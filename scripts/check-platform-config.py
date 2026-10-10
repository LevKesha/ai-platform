#!/usr/bin/env python3
"""Fail if sibling repos drift from platform-config.yaml LLM SSOT."""
from __future__ import annotations

import re
import subprocess
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

# Paths relative to each sibling repo (or ai-platform itself)
CHECKS: list[tuple[str, str, str]] = [
    # (repo_dir, relative_file, pattern_kind)
    ("ai-platform", "claude-router/k8s/deployment.yaml", "claude"),
    ("ai-platform", "claude-router/app/main.py", "claude_default"),
    ("ai-platform", "agents/support-runbook-copilot/k8s/deployment.yaml", "claude"),
    ("ai-platform", "agents/support-runbook-copilot/app/main.py", "claude_default"),
    ("ai-platform", "agents/support-runbook-copilot/agent-spec.yaml", "spec_model"),
    ("ai-platform", "agents/support-runbook-copilot/.env.example", "env_claude"),
    ("ai-platform", "claude-router/README.md", "readme_claude"),
    ("ai-platform", "n8n-config/headroom-demo-workflow.json", "quoted_model"),
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


def load_doc() -> dict:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("platform-config.yaml: expected a mapping")
    return data


def load_ssot() -> dict:
    return load_doc()["llm"]


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


def litellm_want_rows(mix: dict[str, str]) -> set[tuple[str, str]]:
    """LiteLLM routes default and max only. Helper is direct Bedrock, not a row."""
    return {
        (
            mix["default_litellm_model_name"],
            f"bedrock/converse/{mix['default_bedrock_id']}",
        ),
        (
            mix["max_litellm_model_name"],
            f"bedrock/converse/{mix['max_bedrock_id']}",
        ),
    }


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


def resolve(repo: str, rel: str) -> Path | None:
    """None when a sibling clone is not mounted. A present clone with a missing file still fails."""
    if repo == "ai-platform":
        return ROOT / rel
    base = PARENT / repo
    if not base.is_dir():
        return None
    return base / rel


def slot_requirement_errors(slots: object) -> list[str]:
    """default, max, and helper are the same required shape."""
    if not isinstance(slots, dict):
        return ["llm.slots is required"]
    errors: list[str] = []
    for name in SLOT_NAMES:
        slot = slots.get(name)
        if not isinstance(slot, dict):
            errors.append(f"llm.slots.{name} is required")
            continue
        for field in ("litellm_model_name", "bedrock_id", "anthropic_api_id"):
            if not slot.get(field):
                errors.append(f"llm.slots.{name}.{field} is required")
        if name in {"default", "max"} and not slot.get("label"):
            errors.append(f"llm.slots.{name}.label is required")
        if name == "helper" and slot.get("invoke") != "direct-bedrock":
            errors.append("llm.slots.helper.invoke must be direct-bedrock")
    return errors


def without_denylist(text: str) -> str:
    kept: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if not skipping and re.match(r"^deprecated_model_ids:\s*$", line):
            skipping = True
            continue
        if skipping:
            if re.match(r"^\S", line):
                skipping = False
            else:
                continue
        kept.append(line)
    return "".join(kept)


def deprecated_hits(text: str, deprecated: list[str]) -> list[str]:
    found: list[str] = []
    for dep in deprecated:
        if re.search(rf"(?<![\w.]){re.escape(dep)}(?![\w])", text):
            found.append(dep)
    return found


def scan_deprecated(deprecated: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        return ["git ls-files failed"]
    errors: list[str] = []
    config_text = CONFIG.read_text(encoding="utf-8")
    for dep in deprecated_hits(without_denylist(config_text), deprecated):
        errors.append(f"platform-config.yaml: deprecated model id outside denylist: {dep}")
    for raw in proc.stdout.decode().split("\0"):
        if not raw or raw == "platform-config.yaml":
            continue
        path = ROOT / raw
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for dep in deprecated_hits(text, deprecated):
            errors.append(f"{raw}: deprecated model id {dep}")
    return errors


def quoted_models(text: str) -> list[str]:
    return re.findall(r"""(?<![A-Za-z0-9_])model\s*:\s*['"]([^'"]+)['"]""", text)


def readme_default(text: str, env_var: str) -> str | None:
    match = re.search(rf"`{re.escape(env_var)}`\s*\|\s*`([^`]+)`", text)
    return match.group(1) if match else None


def js_string_field(text: str, field: str) -> list[str]:
    """Quoted JS field. Boundary so maxModelId is not read as modelId."""
    return re.findall(
        rf'(?<![A-Za-z0-9_]){re.escape(field)}:\s*"([^"]+)"',
        text,
    )


def main() -> int:
    doc = load_doc()
    llm = doc["llm"]
    want_claude = llm["model_id"]
    want_embed = llm["embedding_model_id"]
    claude_var = llm["env_var"]
    embed_var = llm["embedding_env_var"]
    errors: list[str] = []
    slots = llm.get("slots") or {}
    errors.extend(slot_requirement_errors(slots))
    deprecated = doc.get("deprecated_model_ids")
    if not isinstance(deprecated, list) or not deprecated:
        errors.append("deprecated_model_ids is required")
    else:
        errors.extend(scan_deprecated([str(item) for item in deprecated]))
    if errors:
        print("SSOT drift detected:")
        for e in errors:
            print(f"  - {e}")
        return 1

    try:
        mix = load_model_mix(CONFIG)
    except (OSError, KeyError, ValueError) as exc:
        print(f"SSOT drift detected:\n  - registry mix: {exc}")
        return 1

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
    helper = slots.get("helper") or {}
    if mix.get("helper_bedrock_id") != helper.get("bedrock_id"):
        errors.append(
            "llm.slots.helper.bedrock_id "
            f"want={helper.get('bedrock_id')!r} got={mix.get('helper_bedrock_id')!r}"
        )

    want_short = mix["default_anthropic_api_id"]

    # Forbidden: stale tree under ai-platform
    stale = ROOT / "rag-service"
    if stale.exists():
        errors.append(f"stale tree must be removed: {stale}")

    skipped: set[str] = set()
    for repo, rel, kind in CHECKS:
        path = resolve(repo, rel)
        if path is None:
            skipped.add(repo)
            continue
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
            want_rows = litellm_want_rows(mix)
            if set(rows) != want_rows:
                errors.append(f"{path}: model_list want={sorted(want_rows)!r} got={rows!r}")
            continue
        if kind == "claude_default":
            got = find_default_getenv(text, claude_var)
            if got != want_claude:
                errors.append(f"{path}: getenv default want={want_claude!r} got={got!r}")
        if kind == "readme_claude":
            got = readme_default(text, claude_var)
            if got != want_claude:
                errors.append(f"{path}: {claude_var} want={want_claude!r} got={got!r}")
        if kind == "quoted_model":
            found_models = quoted_models(text)
            if found_models != [want_claude]:
                errors.append(f"{path}: quoted model want={[want_claude]!r} got={found_models!r}")
        if kind in ("helm_both", "env_both"):
            got_e = find_embedding(text, embed_var)
            if got_e != want_embed:
                errors.append(f"{path}: {embed_var} want={want_embed!r} got={got_e!r}")

    for repo, rel, kind in PROBE_CHECKS:
        path = resolve(repo, rel)
        if path is None:
            skipped.add(repo)
            continue
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        got = find_model(text, "HEADROOM_PROBE_MODEL") if kind == "helm" else find_default_getenv(text, "HEADROOM_PROBE_MODEL")
        if got != want_claude:
            errors.append(f"{path}: HEADROOM_PROBE_MODEL want={want_claude!r} got={got!r}")

    for repo, rel, pattern in SHORT_ID_CHECKS:
        path = resolve(repo, rel)
        if path is None:
            skipped.add(repo)
            continue
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
            label = js_string_field(text, "modelLabel")
            want_label = str((slots.get("default") or {}).get("label"))
            if label != [want_label]:
                errors.append(f"{path}: modelLabel want={[want_label]!r} got={label!r}")
            max_label = js_string_field(text, "maxModelLabel")
            want_max_label = str((slots.get("max") or {}).get("label"))
            if max_label != [want_max_label]:
                errors.append(f"{path}: maxModelLabel want={[want_max_label]!r} got={max_label!r}")
        elif max_found and max_found != [want_max]:
            errors.append(f"{path}: maxModelId want={[want_max]!r} got={max_found!r}")

    if errors:
        print("SSOT drift detected:")
        for e in errors:
            print(f"  - {e}")
        return 1
    if skipped:
        print("skip siblings (clone not present): " + ", ".join(sorted(skipped)))
    print("OK: platform-config.yaml LLM SSOT matches checked consumers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
