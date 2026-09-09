#!/usr/bin/env python3
"""Verify consumers match platform-infra.dev.json (infra SSOT snapshot)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
INFRA = ROOT / "platform-infra.dev.json"


def must_contain(path: Path, needle: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing {path}")
        return
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        errors.append(f"{path}: expected {needle!r}")


def main() -> int:
    data = json.loads(INFRA.read_text(encoding="utf-8"))
    subnets = ",".join(data["public_subnet_ids"])
    errors: list[str] = []

    must_contain(ROOT / "n8n/k8s/ingress.yaml", data["n8n_acm_certificate_arn"], errors)
    must_contain(ROOT / "n8n/k8s/ingress.yaml", subnets, errors)
    must_contain(ROOT / "n8n/k8s/ingress.yaml", data["n8n_domain_name"], errors)
    must_contain(
        ROOT / "claude-router/k8s/deployment.yaml",
        data["agent_api_irsa_arn"],
        errors,
    )
    must_contain(
        ROOT / "agents/support-runbook-copilot/k8s/deployment.yaml",
        data["agent_api_irsa_arn"],
        errors,
    )

    for path, key in [
        (PARENT / "agent-api/k8s/helm/values.yaml", "agent_api_irsa_arn"),
        (PARENT / "rag-service/k8s/helm/values.yaml", "rag_service_irsa_arn"),
        (PARENT / "mcp-server/k8s/helm/values.yaml", "mcp_service_irsa_arn"),
    ]:
        must_contain(path, data[key], errors)

    if errors:
        print("infra SSOT drift:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: consumers match platform-infra.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
