"""Read the platform model mix from platform-config.yaml.

SSOT: ai-platform/platform-config.yaml llm.model_id and llm.slots.
Stdlib only so Bedrock callers do not gain a YAML dependency.
"""
from __future__ import annotations

from pathlib import Path

SLOT_FIELDS = ("litellm_model_name", "bedrock_id", "anthropic_api_id")
# Active values must not be the Phase C targets. The comment in
# platform-config.yaml names eu.anthropic.claude-opus-4-8 for that later flip.
_PHASE_C_MARKERS = ("opus-4-8", "opus-5", "claude-sonnet-5", "claude-opus-5")


def find_platform_config(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        candidate = parent / "platform-config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("platform-config.yaml not found above platform_common")


def _parse_simple(text: str) -> dict:
    """Indent-based map parser. Skips comments and list items."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        rest = rest.strip()
        if " #" in rest:
            rest = rest.split(" #", 1)[0].strip()
        parent = stack[-1][1]
        if rest == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
            continue
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in {"'", '"'}:
            rest = rest[1:-1]
        parent[key] = rest
    return root


def _phase_c_marker(value: str) -> str | None:
    for marker in _PHASE_C_MARKERS:
        if marker in value:
            return marker
    return None


def load_model_mix(config_path: str | Path | None = None) -> dict[str, str]:
    """Return the live default/max mix. Does not select a Phase C id."""
    path = Path(config_path) if config_path else find_platform_config()
    parsed = _parse_simple(path.read_text(encoding="utf-8"))
    llm = parsed["llm"]
    slots = llm["slots"]
    mix: dict[str, str] = {"model_id": str(llm["model_id"])}
    for name in ("default", "max"):
        slot = slots[name]
        for field in SLOT_FIELDS:
            mix[f"{name}_{field}"] = str(slot[field])
    for value in mix.values():
        marker = _phase_c_marker(value)
        if marker:
            raise ValueError(f"Phase C id is not live ({marker} in {value})")
    if mix["model_id"] != mix["default_bedrock_id"]:
        raise ValueError("llm.model_id must alias slots.default.bedrock_id")
    if mix["default_litellm_model_name"] == mix["max_litellm_model_name"]:
        raise ValueError("slots need distinct litellm_model_name values")
    return mix
