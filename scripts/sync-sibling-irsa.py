#!/usr/bin/env python3
"""Sync IRSA / ECR values in sibling service repos from platform-infra.dev.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
INFRA = ROOT / "platform-infra.dev.json"


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    for pattern, repl in replacements:
        text, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"{path}: failed to patch {pattern!r}")
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"updated {path}")
    else:
        print(f"unchanged {path}")


def main() -> int:
    data = json.loads(INFRA.read_text(encoding="utf-8"))
    jobs = [
        (
            PARENT / "agent-api/k8s/helm/values.yaml",
            [
                (r"(roleArn:\s*).+", rf"\g<1>{data['agent_api_irsa_arn']}"),
                (
                    r"(repository:\s*).+",
                    rf"\g<1>{data['ecr_agent_api_repository_url']}",
                ),
            ],
        ),
        (
            PARENT / "agent-api/agent-spec.yaml",
            [(r"(iamRole:\s*).+", rf"\g<1>{data['agent_api_irsa_arn']}")],
        ),
        (
            PARENT / "rag-service/k8s/helm/values.yaml",
            [
                (r"(roleArn:\s*).+", rf"\g<1>{data['rag_service_irsa_arn']}"),
                (
                    r"(repository:\s*).+",
                    rf"\g<1>{data['ecr_rag_service_repository_url']}",
                ),
            ],
        ),
        (
            PARENT / "rag-service/agent-spec.yaml",
            [(r"(iamRole:\s*).+", rf"\g<1>{data['rag_service_irsa_arn']}")],
        ),
        (
            PARENT / "mcp-server/k8s/helm/values.yaml",
            [
                (r"(roleArn:\s*).+", rf"\g<1>{data['mcp_service_irsa_arn']}"),
                (
                    r"(repository:\s*).+",
                    rf"\g<1>{data['ecr_mcp_server_repository_url']}",
                ),
            ],
        ),
        (
            PARENT / "mcp-server/agent-spec.yaml",
            [(r"(iamRole:\s*).+", rf"\g<1>{data['mcp_service_irsa_arn']}")],
        ),
        (
            PARENT / "infrastructure/charts/rag-service/values.yaml",
            [
                (
                    r"(eks\.amazonaws\.com/role-arn:\s*).+",
                    rf"\g<1>{data['rag_service_irsa_arn']}",
                ),
                (
                    r"(repository:\s*).+",
                    rf"\g<1>{data['ecr_rag_service_repository_url']}",
                ),
            ],
        ),
    ]
    for path, reps in jobs:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        patch_file(path, reps)
    print("OK: sibling IRSA/ECR synced from platform-infra.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
