"""LevKesha repos the SSOT checks must cover.

GitHub is the set (`gh repo list LevKesha`). This tuple is the check's copy.
A name on GitHub and missing here fails the check. A name here and missing
on GitHub fails the check.
"""
from __future__ import annotations

OWNER = "LevKesha"

# Keep this equal to `gh repo list LevKesha`. Alphabetical.
PLATFORM_REPOS = (
    "agent-api",
    "ai-platform",
    "athena-labs",
    "cicd",
    "infrastructure",
    "llm-cost",
    "mcp-server",
    "rag-service",
)


def repo_set_errors(expected: set[str], live: set[str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(expected - live)
    extra = sorted(live - expected)
    if missing:
        errors.append("SSOT repo list has names GitHub does not: " + ", ".join(missing))
    if extra:
        errors.append("GitHub repos missing from the SSOT check: " + ", ".join(extra))
    return errors
