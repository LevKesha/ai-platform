#!/usr/bin/env python3
"""Render ai-platform k8s manifests from platform-infra.dev.json (TF SSOT snapshot)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from cloudfront_sg import require_sg_id

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


def n8n_sg_annotations(data: dict) -> str:
    """Pin the CloudFront-only group. An empty id must not render a world-open ingress."""
    try:
        sg = require_sg_id(data, "n8n_cloudfront_sg_id")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return (
        f"    alb.ingress.kubernetes.io/security-groups: {sg}\n"
        '    alb.ingress.kubernetes.io/manage-backend-security-group-rules: "true"\n'
    )


def ingress_text(data: dict) -> str:
    tpl = (ROOT / "n8n/k8s/ingress.yaml.tpl").read_text(encoding="utf-8")
    subnets = ",".join(data["public_subnet_ids"])
    return (
        tpl.replace("__PUBLIC_SUBNET_IDS__", subnets)
        .replace("__N8N_ACM_CERTIFICATE_ARN__", data["n8n_acm_certificate_arn"])
        .replace("__N8N_DOMAIN_NAME__", data["n8n_domain_name"])
        .replace("    __N8N_CF_SG_ANNOTATIONS__\n", n8n_sg_annotations(data))
    )


def store_rendered(path: Path, new: str, check: bool) -> None:
    """Write the render, or fail when the committed file differs. Does not write in --check."""
    rendered = new.replace("\r\n", "\n")
    if check:
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        rel = path.relative_to(ROOT).as_posix()
        if current != rendered:
            raise SystemExit(f"drift: {rel}")
        print(f"ok {rel}", flush=True)
        return
    path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {path}", flush=True)


def render_ingress(data: dict, *, check: bool = False) -> None:
    store_rendered(ROOT / "n8n/k8s/ingress.yaml", ingress_text(data), check)


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


def render_ai_platform_agents(data: dict, *, check: bool = False) -> None:
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
        store_rendered(path, text, check)

    spec = ROOT / "agents/support-runbook-copilot/agent-spec.yaml"
    text = patch_role_arn(spec.read_text(encoding="utf-8"), role)
    store_rendered(spec, text, check)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a render would change a file, or if the n8n CloudFront SG id is missing or empty. Does not write.",
    )
    args = parser.parse_args()
    data = load_infra()
    render_ingress(data, check=args.check)
    render_ai_platform_agents(data, check=args.check)
    if args.check:
        print("OK: render --check")
    else:
        print("OK: rendered from platform-infra.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
