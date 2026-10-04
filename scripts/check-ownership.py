#!/usr/bin/env python3
"""Fail unless every tracked path has exactly one owner.

Map: ownership.json. This script is the only ownership gate.
It does not see who typed a hunk, a Security stop, or a Grok override.

  python3 scripts/check-ownership.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "ownership.json"
MAP_NAME = "ownership.json"

REQUIRED_ROLES = {
    "grok": "committer",
    "eng": "writer",
    "product": "owner",
    "design": "owner",
    "platform": "owner",
    "content": "owner",
    "security": "reviewer",
}
NO_PATHS = frozenset({"grok", "security"})
ONE_TREE = frozenset({"product", "design", "content"})
PROPOSE_ONLY_DIRS = frozenset({"helm", "charts", "terraform"})
CI_PROVES_KEYS = ("hunk_author", "security_stop", "override")


def propose_only_class(path: str) -> bool:
    """Workflows, this map, Helm, and Terraform are not autonomous app code."""
    if path == MAP_NAME or path.startswith(".github/workflows/"):
        return True
    parts = path.split("/")
    if PROPOSE_ONLY_DIRS.intersection(parts[:-1]):
        return True
    name = parts[-1]
    if name in {"Chart.yaml", "Chart.lock"}:
        return True
    return name.endswith(".tf") or name.endswith(".tfvars")


def matches(prefix: str, path: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix


def second_map(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if name == "CODEOWNERS":
        return True
    return name == MAP_NAME and path != MAP_NAME


def load_map() -> dict:
    try:
        data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ownership map: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("ownership map: expected an object")
    return data


def tracked_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode() or "git ls-files failed")
    raw = proc.stdout.decode()
    return [p for p in raw.split("\0") if p]


def policy_errors(doc: dict) -> list[str]:
    errors: list[str] = []
    seats = doc.get("seats")
    if not isinstance(seats, dict) or set(seats) != set(REQUIRED_ROLES):
        errors.append("seats must be exactly grok, eng, product, design, platform, content, security")
        return errors
    for name, role in REQUIRED_ROLES.items():
        seat = seats.get(name)
        if not isinstance(seat, dict) or seat.get("role") != role:
            errors.append(f"{name} role must be {role}")
    security = seats["security"]
    if security.get("verdicts") != ["pass", "fail", "stop"]:
        errors.append("security verdicts must be pass, fail, stop")

    review = doc.get("review")
    if not isinstance(review, dict):
        errors.append("review rule missing")
        return errors
    if review.get("stop_blocks_merge") is not True:
        errors.append("review.stop_blocks_merge must be true")
    if review.get("eng_cannot_close_stop") is not True:
        errors.append("review.eng_cannot_close_stop must be true")
    override = review.get("override")
    if (
        not isinstance(override, dict)
        or override.get("by") != "grok"
        or override.get("named") is not True
        or override.get("expires") is not True
    ):
        errors.append("review.override must be a named Grok acceptance with expiry")
    proves = review.get("ci_proves")
    if not isinstance(proves, dict):
        errors.append("review.ci_proves missing")
    else:
        for key in CI_PROVES_KEYS:
            if proves.get(key) is not False:
                errors.append(f"review.ci_proves.{key} must be false")

    sr = doc.get("security_review")
    if not isinstance(sr, dict):
        errors.append("security_review missing")
        return errors
    reviewers = sr.get("reviewers")
    if not isinstance(reviewers, list) or any(not isinstance(login, str) or not login for login in reviewers):
        errors.append("security_review.reviewers must be a list of github logins")
    paths = sr.get("paths")
    if not isinstance(paths, list) or not paths:
        errors.append("security_review.paths missing")
    else:
        for entry in paths:
            if not isinstance(entry, dict):
                errors.append("security_review path must be an object")
                continue
            if entry.get("when") not in {"any", "hunk"} or not isinstance(entry.get("path"), str) or not entry["path"]:
                errors.append(f"security_review path needs path and when any|hunk: {entry!r}")
    if not isinstance(sr.get("hunk_signals"), list) or not sr["hunk_signals"]:
        errors.append("security_review.hunk_signals missing")
    return errors


def rule_errors(doc: dict, tracked: list[str]) -> list[str]:
    errors: list[str] = []
    seats = doc.get("seats")
    if not isinstance(seats, dict):
        return ["rules: seats missing"]
    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        return ["rules missing"]

    flat: list[tuple[str, str, str | None]] = []
    by_owner: dict[str, list[dict]] = {name: [] for name in seats}
    for rule in rules:
        if not isinstance(rule, dict):
            errors.append("rule is not an object")
            continue
        owner = rule.get("owner")
        prefixes = rule.get("prefixes")
        disposition = rule.get("disposition")
        if owner not in seats:
            errors.append(f"unknown owner: {owner}")
            continue
        if owner in NO_PATHS:
            errors.append(f"{owner} owns a path")
        if disposition not in (None, "propose-only"):
            errors.append(f"bad disposition for {owner}: {disposition}")
        if not isinstance(prefixes, list) or not prefixes:
            errors.append(f"{owner} rule has no prefixes")
            continue
        by_owner.setdefault(owner, []).append(rule)
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
                errors.append(f"bad prefix: {prefix!r}")
                continue
            if "*" in prefix or "?" in prefix or "\\" in prefix:
                errors.append(f"globs are not supported: {prefix}")
                continue
            flat.append((prefix, owner, disposition))

    for seat in ONE_TREE:
        owned = by_owner.get(seat, [])
        if len(owned) > 1:
            errors.append(f"{seat} has more than one tree")
            continue
        if not owned:
            continue
        prefixes = owned[0].get("prefixes")
        if not isinstance(prefixes, list) or len(prefixes) != 1 or not str(prefixes[0]).endswith("/"):
            errors.append(f"{seat} must own one directory tree")

    seen: set[str] = set()
    for prefix, owner, _disposition in flat:
        if prefix in seen:
            errors.append(f"duplicate prefix: {prefix}")
        seen.add(prefix)
        if not any(matches(prefix, path) for path in tracked):
            errors.append(f"rule matches nothing: {prefix} ({owner})")

    for path in tracked:
        if second_map(path):
            errors.append(f"second ownership file: {path}")
        hits = [(owner, disposition) for prefix, owner, disposition in flat if matches(prefix, path)]
        if not hits:
            errors.append(f"unowned: {path}")
            continue
        if len(hits) > 1:
            names = ", ".join(owner for owner, _disposition in hits)
            errors.append(f"two owners: {path}: {names}")
            continue
        owner, disposition = hits[0]
        if owner in NO_PATHS:
            errors.append(f"{owner} owns a path: {path}")
        if propose_only_class(path) and disposition != "propose-only":
            errors.append(f"propose-only path not marked propose-only: {path} ({owner})")
        if disposition == "propose-only" and not propose_only_class(path):
            errors.append(f"propose-only rule covers app path: {path}")
    return errors


def check(tracked: list[str], doc: dict) -> list[str]:
    return policy_errors(doc) + rule_errors(doc, tracked)


def _fixture() -> tuple[list[str], dict]:
    tracked = [
        "app/main.py",
        "site/index.html",
        "pkg/lib.py",
        ".github/workflows/ci.yml",
        "ownership.json",
    ]
    doc = {
        "seats": {
            "grok": {"role": "committer"},
            "eng": {"role": "writer"},
            "product": {"role": "owner"},
            "design": {"role": "owner"},
            "platform": {"role": "owner"},
            "content": {"role": "owner"},
            "security": {"role": "reviewer", "verdicts": ["pass", "fail", "stop"]},
        },
        "review": {
            "stop_blocks_merge": True,
            "eng_cannot_close_stop": True,
            "override": {"by": "grok", "named": True, "expires": True},
            "ci_proves": {"hunk_author": False, "security_stop": False, "override": False},
        },
        "security_review": {
            "reviewers": [],
            "paths": [{"path": "edge/", "when": "any"}],
            "hunk_signals": ["role-arn"],
        },
        "rules": [
            {"owner": "eng", "prefixes": ["app/"]},
            {"owner": "content", "prefixes": ["site/"]},
            {
                "owner": "platform",
                "disposition": "propose-only",
                "prefixes": [".github/workflows/", "ownership.json"],
            },
            {"owner": "platform", "prefixes": ["pkg/"]},
        ],
    }
    return tracked, doc


def _expect(errors: list[str], text: str) -> None:
    if not any(text in error for error in errors):
        raise SystemExit(f"self-test missing {text!r} in {errors}")


def self_test() -> None:
    tracked, doc = _fixture()
    clean = check(tracked, doc)
    if clean:
        raise SystemExit(f"self-test clean fixture failed: {clean}")

    _expect(check(tracked + ["orphan.txt"], doc), "unowned: orphan.txt")

    two = json.loads(json.dumps(doc))
    two["rules"].append({"owner": "platform", "prefixes": ["app/"]})
    _expect(check(tracked, two), "two owners: app/main.py")

    autonomous = json.loads(json.dumps(doc))
    autonomous["rules"][2].pop("disposition")
    _expect(
        check(tracked, autonomous),
        "propose-only path not marked propose-only: .github/workflows/ci.yml",
    )

    helm = json.loads(json.dumps(doc))
    _expect(
        check(tracked + ["pkg/helm/Chart.yaml"], helm),
        "propose-only path not marked propose-only: pkg/helm/Chart.yaml",
    )

    coded = json.loads(json.dumps(doc))
    coded["rules"][0]["disposition"] = "propose-only"
    _expect(check(tracked, coded), "propose-only rule covers app path: app/main.py")

    stopped = json.loads(json.dumps(doc))
    stopped["review"]["ci_proves"]["security_stop"] = True
    _expect(check(tracked, stopped), "review.ci_proves.security_stop must be false")

    sec = json.loads(json.dumps(doc))
    sec["rules"].append({"owner": "security", "prefixes": ["site/"]})
    _expect(check(tracked, sec), "security owns a path")

    trees = json.loads(json.dumps(doc))
    trees["rules"].append({"owner": "content", "prefixes": ["extra/"]})
    extra = tracked + ["extra/a.txt"]
    _expect(check(extra, trees), "content has more than one tree")

    _expect(check(tracked + ["CODEOWNERS"], doc), "unowned: CODEOWNERS")
    owners = json.loads(json.dumps(doc))
    owners["rules"].append({"owner": "eng", "prefixes": ["CODEOWNERS"]})
    _expect(check(tracked + ["CODEOWNERS"], owners), "second ownership file: CODEOWNERS")
    print("self-test ok", flush=True)


def main() -> int:
    self_test()
    doc = load_map()
    tracked = tracked_paths()
    errors = check(tracked, doc)
    if errors:
        print("ownership check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(tracked)} tracked paths, exactly one owner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
