#!/usr/bin/env python3
"""Fail when olympus portal files contain secret chrome.

Scans assignment and credential-value shapes. A bare Authorization, Bearer,
or API_KEY mention is not secret chrome.

  python3 scripts/check-portal-secrets.py
  python3 scripts/check-portal-secrets.py --self-test
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLYMPUS = ROOT / "olympus"
TEXT_SUFFIXES = {".html", ".js", ".css", ".json", ".md", ".svg", ".sh"}

# Left-hand secret name, then a quoted or compact value.
ASSIGN = re.compile(
    r"""(?ix)
    (?P<name>api[_-]?key|client[_-]?secret|secret(?:[_-]?key)?|password|passwd|
            access[_-]?token|auth(?:orization)?|bearer|token)
    \s*[:=]\s*
    (?P<quote>['"])(?P<value>[^'"]*)(?P=quote)
    """
)
BEARER_VALUE = re.compile(
    r"""(?ix)authorization\s*[:=]\s*['"]?\s*bearer\s+(?P<value>\S+)"""
)
TOKEN = re.compile(
    r"""(?x)
    (?:
        sk-[A-Za-z0-9_\-]{12,}
      | AKIA[0-9A-Z]{16}
      | eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}
      | -----BEGIN\ (?:RSA\ |OPENSSH\ |EC\ )?PRIVATE\ KEY-----
    )
    """
)
PLACEHOLDER = re.compile(
    r"""(?ix)
    ^(
        change-?me.*
      | changeme.*
      | example.*
      | placeholder.*
      | redacted.*
      | your[-_ ].*
      | xxx+
      | todo
      | insert.*
      | <[^>]*>
      | \$\{[^}]+\}
      | \*+
      | bearer
    )$
    """
)
COMPACT = re.compile(r"^[A-Za-z0-9+/=_\-.]{12,}$")


def _placeholder(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    if PLACEHOLDER.match(text):
        return True
    if "change-me" in text.lower() or "changeme" in text.lower():
        return True
    return False


def _secret_value(value: str) -> bool:
    text = value.strip().strip("'\"")
    if _placeholder(text):
        return False
    if TOKEN.search(text):
        return True
    if re.search(r"\s", text):
        return False
    return bool(COMPACT.fullmatch(text))


def findings_in(text: str) -> list[str]:
    found: list[str] = []
    for match in ASSIGN.finditer(text):
        if _secret_value(match.group("value")):
            found.append(f"{match.group('name')} assignment")
    for match in BEARER_VALUE.finditer(text):
        if _secret_value(match.group("value")):
            found.append("authorization bearer value")
    # Credential literals that are not bare words.
    for match in TOKEN.finditer(text):
        if _placeholder(match.group(0)):
            continue
        snippet = match.group(0)
        if any(snippet in item for item in found):
            continue
        # Skip when the token sits only inside an already-recorded assignment span.
        found.append(f"credential literal {snippet[:12]}")
    return found


def scan_tree(root: Path) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f"missing portal tree: {root}"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root.parents[0]).as_posix()
        for item in findings_in(text):
            errors.append(f"{rel}: {item}")
    return errors


def self_test() -> None:
    # Build secret-shaped fixtures at runtime so the source never contains
    # a contiguous token that GitGuardian / secret scanners flag.
    fake = "sk-" + ("ab" * 10)
    prose = "The Authorization header uses Bearer auth and an API_KEY name."
    if findings_in(prose):
        raise SystemExit(f"bare words must not match: {findings_in(prose)}")
    placeholder = 'const API_KEY = "change-me-not-a-real-key";'
    if findings_in(placeholder):
        raise SystemExit(f"placeholder assignment must not match: {findings_in(placeholder)}")
    assigned = f'const API_KEY = "{fake}";'
    if not findings_in(assigned):
        raise SystemExit("token assignment must match")
    bearer = f"Authorization: Bearer {fake}"
    if not any("bearer" in item or "credential" in item for item in findings_in(bearer)):
        raise SystemExit(f"bearer value must match: {findings_in(bearer)}")
    label = "<label>API_KEY</label>"
    if findings_in(label):
        raise SystemExit("label text must not match")
    print("self-test ok", flush=True)


def main() -> int:
    if "--self-test" in sys.argv:
        self_test()
        return 0
    self_test()
    errors = scan_tree(OLYMPUS)
    if errors:
        print("portal secrets failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("OK: olympus has no secret chrome")
    return 0


if __name__ == "__main__":
    sys.exit(main())
