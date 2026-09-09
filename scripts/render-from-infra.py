#!/usr/bin/env python3
"""Render ai-platform k8s manifests from platform-infra.dev.json (TF SSOT snapshot)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "platform-infra.dev.json"


def load_infra() -> dict:
    data = json.loads(INFRA.read_text(encoding="utf-8"))
    required = [
        "public_subnet_ids",
        "n8n_acm_certificate_arn",
        "n8n_domain_name",
        "agent_api_irsa_arn",
        "ecr_claude_router_repository_url",
        "ecr_support_runbook_copilot_repository_url",
    ]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise SystemExit(f"platform-infra.dev.json missing: {missing}")
    return data


def render_ingress(data: dict) -> None:
    tpl = (ROOT / "n8n/k8s/ingress.yaml.tpl").read_text(encoding="utf-8")
    subnets = ",".join(data["public_subnet_ids"])
    out = (
        tpl.replace("__PUBLIC_SUBNET_IDS__", subnets)
        .replace("__N8N_ACM_CERTIFICATE_ARN__", data["n8n_acm_certificate_arn"])
        .replace("__N8N_DOMAIN_NAME__", data["n8n_domain_name"])
    )
    path = ROOT / "n8n/k8s/ingress.yaml"
    path.write_text(out, encoding="utf-8")
    print(f"wrote {path}")


def patch_role_arn(text: str, role_arn: str) -> str:
    text = re.sub(
        r"(eks\.amazonaws\.com/role-arn:\s*).+",
        rf"\g<1>{role_arn}",
        text,
    )
    text = re.sub(
        r"(roleArn:\s*).+",
        rf"\g<1>{role_arn}",
        text,
    )
    text = re.sub(
        r"(iamRole:\s*).+",
        rf"\g<1>{role_arn}",
        text,
    )
    return text


def patch_ecr_image(text: str, repo_url: str) -> str:
    # Replace registry/repo prefix before :tag
    return re.sub(
        r"image:\s+\S+/(\S+):(\S+)",
        rf"image: {repo_url}:\2",
        text,
        count=1,
    )


def render_ai_platform_agents(data: dict) -> None:
    role = data["agent_api_irsa_arn"]
    targets = [
        (
            ROOT / "claude-router/k8s/deployment.yaml",
            data["ecr_claude_router_repository_url"],
        ),
        (
            ROOT / "agents/support-runbook-copilot/k8s/deployment.yaml",
            data["ecr_support_runbook_copilot_repository_url"],
        ),
    ]
    for path, ecr in targets:
        text = path.read_text(encoding="utf-8")
        text = patch_role_arn(text, role)
        text = patch_ecr_image(text, ecr)
        path.write_text(text, encoding="utf-8")
        print(f"updated {path}")

    spec = ROOT / "agents/support-runbook-copilot/agent-spec.yaml"
    text = patch_role_arn(spec.read_text(encoding="utf-8"), role)
    spec.write_text(text, encoding="utf-8")
    print(f"updated {spec}")


def main() -> int:
    data = load_infra()
    render_ingress(data)
    render_ai_platform_agents(data)
    print("OK: rendered from platform-infra.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
