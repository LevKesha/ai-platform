#!/usr/bin/env python3
"""Workflow literals: infrastructure-owned values must not be restated in CI YAML.

Owner: infrastructure outputs exported to platform-infra.dev.json
(aws_account_id, aws_region, cluster_name, github_oidc_role_arn). Workflows
read them through secrets (AWS_OIDC_ROLE_ARN, AWS_REGION, EKS_CLUSTER_NAME).

Scans .github/workflows/*.yml|*.yaml in every platform repo (sibling clones
next to ai-platform, cicd included) for:
  - an IAM role ARN
  - an AWS account id (the exported one, or any 12-digit id inside an ARN or ECR host)
  - an AWS region name
  - the exported cluster name (also as a prefix, e.g. <cluster>-rag-db)
Full-line YAML comments are ignored. Exit 1 on any finding, or when a repo is
missing and --require-all is set.

Usage: python3 scripts/check-workflow-literals.py [--require-all] [--self-test-only]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
INFRA = ROOT / "platform-infra.dev.json"
REPOS = ("ai-platform", "agent-api", "llm-cost", "cicd", "infrastructure", "rag-service", "mcp-server")

ROLE_ARN = re.compile(r"arn:aws:iam::\d{12}:role/[\w+=,.@/-]+")
ACCOUNT_IN_ARN_OR_ECR = re.compile(r"(?:arn:aws:[a-z0-9-]*:[a-z0-9-]*:)(\d{12})|\b(\d{12})\.dkr\.ecr\.")
REGION = re.compile(r"\b(?:us|eu|ap|sa|ca|me|af|il|mx)-(?:north|south|east|west|central|northeast|southeast|northwest|southwest)-\d\b")


def owner_values(path: Path = INFRA) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {k: str(data.get(k) or "") for k in ("aws_account_id", "aws_region", "cluster_name")}
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise SystemExit(f"platform-infra.dev.json missing {missing}")
    return out


def scan_text(text: str, owners: dict[str, str]) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    account = owners["aws_account_id"]
    cluster = re.compile(rf"(?<![\w.-]){re.escape(owners['cluster_name'])}(?![\w])")
    for no, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        role = ROLE_ARN.search(line)
        if role:
            findings.append((no, "role-arn", role.group(0)))
        if account in line or ACCOUNT_IN_ARN_OR_ECR.search(line):
            findings.append((no, "account-id", line.strip()))
        region = REGION.search(line)
        if region:
            findings.append((no, "region", region.group(0)))
        if cluster.search(line):
            findings.append((no, "cluster", line.strip()))
    return findings


def workflow_files(repo: Path) -> list[Path]:
    wf = repo / ".github" / "workflows"
    if not wf.is_dir():
        return []
    return sorted([*wf.glob("*.yml"), *wf.glob("*.yaml")])


def self_test() -> None:
    acct = "1234" + "5678" + "9012"
    owners = {"aws_account_id": acct, "aws_region": "xx" + "-test-1", "cluster_name": "unit" + "-cluster"}
    role = "arn:" + "aws:iam::" + acct + ":role/" + "Deploy"
    region = "eu" + "-central-" + "1"
    bad = "\n".join([
        "env:",
        "  ROLE: " + role,
        "  REGION: " + region,
        "  CLUSTER: " + owners["cluster_name"],
        "  DB: " + owners["cluster_name"] + "-rag-db",
        "  REG: " + acct + ".dkr.ecr." + region + ".amazonaws.com",
    ])
    kinds = {k for _, k, _ in scan_text(bad, owners)}
    if kinds != {"role-arn", "account-id", "region", "cluster"}:
        raise SystemExit(f"self-test: expected all four kinds, got {sorted(kinds)}")
    good = "\n".join([
        "# ROLE example " + role,
        "env:",
        "  ROLE: ${{ secrets.AWS_OIDC_ROLE_ARN }}",
        "  REGION: ${{ secrets.AWS_REGION }}",
        "  CLUSTER: ${{ secrets.EKS_CLUSTER_NAME }}",
        "  NAME: my-unit-clusterless",
    ])
    if scan_text(good, owners):
        raise SystemExit(f"self-test: clean text flagged: {scan_text(good, owners)}")
    print("self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test_only:
        return 0
    owners = owner_values()
    failed = False
    missing: list[str] = []
    for name in REPOS:
        repo = ROOT if name == "ai-platform" else PARENT / name
        if repo != ROOT and not (repo / ".git").exists():
            missing.append(name)
            continue
        hits = []
        for path in workflow_files(repo):
            for no, kind, text in scan_text(path.read_text(encoding="utf-8", errors="replace"), owners):
                hits.append(f"  {name}/{path.relative_to(repo).as_posix()}:{no}: {kind}: {text[:120]}")
        if hits:
            failed = True
            print(f"FAIL {name}: {len(hits)} workflow literal(s)")
            print("\n".join(hits))
        else:
            print(f"PASS {name}")
    if missing:
        print("skip repos (clone not present): " + ", ".join(missing))
        if args.require_all:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())