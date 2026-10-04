#!/usr/bin/env python3
"""Fail when a diff touches auth, secrets, identity, or the public edge and no Security review artifact exists.

Policy: ownership.json security_review. Security owns no paths.
reviewers is the only list of GitHub logins that can record pass, fail, or stop.
An empty list fails closed. This script does not invent a login.

A named Grok acceptance with an expiry is not accepted here.

  python3 scripts/check-security-review.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "ownership.json"
VERDICT_PREFIX = "security-review:"
VERDICTS = ("pass", "fail", "stop")


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"git failed: {' '.join(args)}")
    return proc


def load_map_text(text: str) -> dict:
    data = json.loads(text)
    sr = data.get("security_review")
    if not isinstance(sr, dict):
        raise SystemExit("security_review missing from ownership map")
    return sr


def load_head_policy() -> dict:
    return load_map_text(MAP_PATH.read_text(encoding="utf-8"))


def load_sha_policy(sha: str) -> dict | None:
    proc = git(["git", "show", f"{sha}:ownership.json"], check=False)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    sr = data.get("security_review")
    return sr if isinstance(sr, dict) else None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _path_entries(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    entries = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("path"), str) and item.get("when") in {"any", "hunk"}:
            entries.append({"path": item["path"], "when": item["when"]})
    return entries


def union_policy(base: dict | None, head: dict) -> dict:
    """Reviewers and skip lists stay on the base map once it exists, so a PR cannot widen who may pass."""
    if not base:
        return head
    by_path: dict[str, dict] = {}
    for entry in _path_entries(base.get("paths")) + _path_entries(head.get("paths")):
        prev = by_path.get(entry["path"])
        if prev is None or entry["when"] == "any" or prev["when"] != "any":
            by_path[entry["path"]] = entry
    signals = list(dict.fromkeys(_str_list(base.get("hunk_signals")) + _str_list(head.get("hunk_signals"))))
    basenames = list(dict.fromkeys(_str_list(base.get("always_basenames")) + _str_list(head.get("always_basenames"))))
    suffixes = list(dict.fromkeys(_str_list(base.get("always_suffixes")) + _str_list(head.get("always_suffixes"))))
    return {
        "reviewers": _str_list(base.get("reviewers")),
        "paths": list(by_path.values()),
        "hunk_signals": signals,
        "skip_prefixes": _str_list(base.get("skip_prefixes")),
        "skip_except_prefixes": _str_list(base.get("skip_except_prefixes")),
        "skip_suffixes": _str_list(base.get("skip_suffixes")),
        "skip_paths": _str_list(base.get("skip_paths")),
        "always_basenames": basenames,
        "always_suffixes": suffixes,
    }


def parse_diff(text: str) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    path: str | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            marker = " b/"
            path = line.split(marker, 1)[1] if marker in line else None
            if path:
                files.setdefault(path, [])
            continue
        if path is None or line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            files[path].append(line[1:])
    return files


def _listed(path: str, policy: dict) -> dict | None:
    found = None
    for entry in policy.get("paths", []):
        prefix = entry["path"]
        if prefix.endswith("/") and path.startswith(prefix):
            found = entry
        elif path == prefix:
            found = entry
    return found


def _skipped(path: str, policy: dict) -> bool:
    if path in set(policy.get("skip_paths", [])):
        return True
    if any(path.endswith(suffix) for suffix in policy.get("skip_suffixes", [])):
        return True
    for prefix in policy.get("skip_prefixes", []):
        if path.startswith(prefix) and not any(path.startswith(extra) for extra in policy.get("skip_except_prefixes", [])):
            return True
    return False


def _signal(lines: list[str], policy: dict) -> bool:
    blob = "\n".join(lines).lower()
    return any(signal.lower() in blob for signal in policy.get("hunk_signals", []))


def needs_review(path: str, lines: list[str], policy: dict) -> bool:
    name = path.rsplit("/", 1)[-1]
    if name in set(policy.get("always_basenames", [])):
        return True
    if any(path.endswith(suffix) and not path.endswith(f"{suffix}.example") for suffix in policy.get("always_suffixes", [])):
        return True
    if _skipped(path, policy):
        return False
    entry = _listed(path, policy)
    if entry and entry["when"] == "any":
        return True
    if entry and entry["when"] == "hunk":
        if not lines:
            return True
        return _signal(lines, policy)
    return _signal(lines, policy)


def required_paths(files: dict[str, list[str]], policy: dict) -> list[str]:
    return sorted(path for path, lines in files.items() if needs_review(path, lines, policy))


def unmatched_rules(tracked: list[str], policy: dict) -> list[str]:
    errors = []
    for entry in policy.get("paths", []):
        prefix = entry["path"]
        hit = any(path.startswith(prefix) if prefix.endswith("/") else path == prefix for path in tracked)
        if not hit:
            errors.append(f"security_review path matches nothing: {prefix}")
    return errors


def _verdict_in(body: str) -> str | None:
    found = None
    for line in body.splitlines():
        text = line.strip().lower()
        if not text.startswith(VERDICT_PREFIX):
            continue
        value = text[len(VERDICT_PREFIX) :].strip()
        if value in VERDICTS:
            found = value
    return found


def latest_verdict(artifacts: list[dict], reviewers: list[str]) -> str | None:
    allowed = set(reviewers)
    chosen: tuple[str, str] | None = None
    for artifact in artifacts:
        user = artifact.get("user") or {}
        login = user.get("login") if isinstance(user, dict) else None
        if login not in allowed:
            continue
        if artifact.get("state") == "DISMISSED":
            continue
        verdict = _verdict_in(str(artifact.get("body") or ""))
        if verdict is None:
            continue
        stamp = str(artifact.get("submitted_at") or artifact.get("created_at") or "")
        if chosen is None or stamp >= chosen[0]:
            chosen = (stamp, verdict)
    return None if chosen is None else chosen[1]


def evaluate(files: dict[str, list[str]], artifacts: list[dict], policy: dict) -> list[str]:
    paths = required_paths(files, policy)
    if not paths:
        return []
    reviewers = _str_list(policy.get("reviewers"))
    if not reviewers:
        listed = ", ".join(paths)
        return [
            f"security review missing: {listed}",
            "security_review.reviewers is empty; no GitHub login can record pass, fail, or stop",
        ]
    verdict = latest_verdict(artifacts, reviewers)
    if verdict == "pass":
        return []
    if verdict in {"fail", "stop"}:
        return [f"security review {verdict}: {', '.join(paths)}"]
    return [f"security review missing: {', '.join(paths)}"]


def _policy() -> dict:
    return {
        "reviewers": [],
        "skip_prefixes": ["olympus/"],
        "skip_except_prefixes": ["olympus/auth/"],
        "skip_suffixes": [".md"],
        "skip_paths": ["ownership.json", "scripts/check-security-review.py"],
        "always_basenames": ["secrets.yaml", "secrets.yml"],
        "always_suffixes": [".env"],
        "hunk_signals": ["cognito", "cloudfront", "alb.ingress", "role-arn", "secretkeyref"],
        "paths": [
            {"path": "olympus/auth/", "when": "any"},
            {"path": "n8n/k8s/ingress.yaml", "when": "any"},
            {"path": "claude-router/k8s/deployment.yaml", "when": "hunk"},
        ],
    }


def _expect(errors: list[str], text: str) -> None:
    if not any(text in error for error in errors):
        raise SystemExit(f"self-test missing {text!r} in {errors}")


def self_test() -> None:
    policy = _policy()
    portal = {"olympus/index.html": ["cloudfront and cognito copy"]}
    if required_paths(portal, policy):
        raise SystemExit("portal copy must not require security review")
    docs = {"n8n/SECURITY.md": ["alb.ingress.kubernetes.io/scheme: internet-facing"]}
    if required_paths(docs, policy):
        raise SystemExit("markdown must not require security review")
    auth = {"olympus/auth/hosted-ui.css": ["color: #fff;"]}
    _expect(evaluate(auth, [], policy), "security review missing: olympus/auth/hosted-ui.css")
    quiet = {"claude-router/k8s/deployment.yaml": ["replicas: 2"]}
    if required_paths(quiet, policy):
        raise SystemExit("replica hunk must not require security review")
    identity = {"claude-router/k8s/deployment.yaml": ["eks.amazonaws.com/role-arn: arn:aws:iam::1:role/x"]}
    _expect(evaluate(identity, [], policy), "security review missing: claude-router/k8s/deployment.yaml")
    ingress = {"n8n/k8s/ingress.yaml": ["# comment only"]}
    _expect(evaluate(ingress, [], policy), "security review missing: n8n/k8s/ingress.yaml")
    secret = {"n8n/k8s/secrets.yaml": ["key: value"]}
    _expect(evaluate(secret, [], policy), "security review missing: n8n/k8s/secrets.yaml")
    example = {"agents/support-runbook-copilot/.env.example": ["CLAUDE_MODEL_ID=x"]}
    if required_paths(example, policy):
        raise SystemExit(".env.example must not require security review")
    dot = {"agents/support-runbook-copilot/.env": ["TOKEN=1"]}
    _expect(evaluate(dot, [], policy), "security review missing: agents/support-runbook-copilot/.env")

    allowed = dict(policy)
    allowed["reviewers"] = ["security-bot"]
    passing = [{"user": {"login": "security-bot"}, "body": "security-review: pass", "submitted_at": "2026-10-04T00:00:01Z"}]
    if evaluate(ingress, passing, allowed):
        raise SystemExit("listed pass must satisfy the review")
    other = [{"user": {"login": "eng"}, "state": "APPROVED", "body": "security-review: pass", "submitted_at": "2026-10-04T00:00:02Z"}]
    _expect(evaluate(ingress, other, allowed), "security review missing")
    stopped = passing + [
        {"user": {"login": "security-bot"}, "body": "security-review: stop", "submitted_at": "2026-10-04T00:00:03Z"}
    ]
    _expect(evaluate(ingress, stopped, allowed), "security review stop")
    cleared = stopped + [
        {"user": {"login": "security-bot"}, "body": "security-review: pass", "submitted_at": "2026-10-04T00:00:04Z"}
    ]
    if evaluate(ingress, cleared, allowed):
        raise SystemExit("a later pass from the listed login must clear stop")
    forged = [
        {
            "user": {"login": "eng"},
            "body": "grok-acceptance: Grok\nexpires: 2026-10-05\nsecurity-review: pass",
            "submitted_at": "2026-10-04T00:00:05Z",
        }
    ]
    _expect(evaluate(ingress, forged, allowed), "security review missing")

    head = dict(policy)
    head["reviewers"] = ["attacker"]
    base = dict(policy)
    base["reviewers"] = ["security-bot"]
    merged = union_policy(base, head)
    if merged["reviewers"] != ["security-bot"]:
        raise SystemExit("head must not add a reviewer")
    attack = [{"user": {"login": "attacker"}, "body": "security-review: pass", "submitted_at": "2026-10-04T00:00:06Z"}]
    _expect(evaluate(ingress, attack, merged), "security review missing")
    print("self-test ok", flush=True)


def diff_text() -> str:
    base = os.environ.get("BASE_SHA")
    head = os.environ.get("HEAD_SHA")
    if not base:
        proc = git(["git", "merge-base", "origin/main", "HEAD"])
        base = proc.stdout.strip()
    if head:
        git(["git", "cat-file", "-e", f"{head}^{{commit}}"], check=False)
        if git(["git", "cat-file", "-e", f"{head}^{{commit}}"], check=False).returncode != 0:
            git(["git", "fetch", "origin", head], check=False)
        if git(["git", "cat-file", "-e", f"{base}^{{commit}}"], check=False).returncode != 0:
            git(["git", "fetch", "origin", base], check=False)
        proc = git(["git", "diff", "--unified=0", base, head])
    else:
        proc = git(["git", "diff", "--unified=0", base])
    return proc.stdout


def fetch_artifacts() -> list[dict]:
    number = os.environ.get("PR_NUMBER")
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not number or not token or not repo:
        return []
    artifacts: list[dict] = []
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-platform-security-review",
    }
    for path in (f"pulls/{number}/reviews", f"issues/{number}/comments"):
        url = f"https://api.github.com/repos/{repo}/{path}?per_page=100"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        if isinstance(payload, list):
            artifacts.extend(item for item in payload if isinstance(item, dict))
    return artifacts


def main() -> int:
    self_test()
    base_sha = os.environ.get("BASE_SHA")
    if not base_sha:
        base_sha = git(["git", "merge-base", "origin/main", "HEAD"]).stdout.strip()
    policy = union_policy(load_sha_policy(base_sha), load_head_policy())
    files = parse_diff(diff_text())
    try:
        artifacts = fetch_artifacts()
    except Exception as exc:
        print(f"security review failed: could not read reviews: {exc}", file=sys.stderr)
        return 1
    tracked = [line for line in git(["git", "ls-files"]).stdout.splitlines() if line]
    errors = unmatched_rules(tracked, policy) + evaluate(files, artifacts, policy)
    if errors:
        print("security review failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    if required_paths(files, policy):
        print("OK: security review pass")
    else:
        print("OK: no security review required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
