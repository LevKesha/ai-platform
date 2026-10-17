#!/usr/bin/env python3
"""Verify consumers match platform-infra.dev.json (infra SSOT snapshot).

--live reads AWS and compares it to the snapshot. It does not rewrite the
snapshot and it does not treat Helm values as the subnet source. Stale JSON
must fail even when a chart still has the same old ids.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from cloudfront_sg import require_sg_id

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
INFRA = ROOT / "platform-infra.dev.json"

PATH_INGRESS_VALUES = (
    "agent-api/k8s/helm/values.yaml",
    "llm-cost/k8s/helm/litellm/values.yaml",
    "llm-cost/k8s/helm/headroom-savings/values.yaml",
)


def must_contain(path: Path, needle: str, errors: list[str]) -> None:
    if not path.exists():
        message = f"missing {path}"
        if message not in errors:
            errors.append(message)
        return
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        errors.append(f"{path}: expected {needle!r}")


def same_ids(left: list[str], right: list[str]) -> bool:
    return sorted(left) == sorted(right)


def self_test() -> None:
    live = ["subnet-bbbb", "subnet-aaaa"]
    stale = ["subnet-old1", "subnet-old2"]
    if same_ids(stale, live):
        raise SystemExit("stale ids must not match live")
    if not same_ids(list(reversed(live)), live):
        raise SystemExit("order must not matter")
    for key in ("n8n_cloudfront_sg_id", "olympus_admin_cloudfront_sg_id"):
        for data in ({}, {key: None}, {key: ""}, {key: "   "}):
            try:
                require_sg_id(data, key)
            except ValueError as exc:
                if "missing or empty" not in str(exc):
                    raise SystemExit(f"{key} blank must say missing or empty, got {exc}") from exc
            else:
                raise SystemExit(f"{key} must fail when missing or empty: {data!r}")
        if require_sg_id({key: "sg-abc"}, key) != "sg-abc":
            raise SystemExit(f"{key} must accept an sg- id")
    print("self-test ok", flush=True)


def file_errors(data: dict) -> list[str]:
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

    for rel in PATH_INGRESS_VALUES:
        path = PARENT / rel
        for subnet in data["public_subnet_ids"]:
            must_contain(path, subnet, errors)
        must_contain(path, data["litellm_acm_certificate_arn"], errors)
        must_contain(path, data["litellm_cognito_user_pool_arn"], errors)
    errors.extend(cloudfront_sg_errors(data))
    return errors


def cloudfront_sg_errors(data: dict) -> list[str]:
    """Both CloudFront SG ids are required. Empty used to omit the annotation and reopen the ALB."""
    errors: list[str] = []
    n8n_sg = _sg_id(data, "n8n_cloudfront_sg_id", errors)
    if n8n_sg:
        n8n = ROOT / "n8n/k8s/ingress.yaml"
        must_contain(n8n, f"alb.ingress.kubernetes.io/security-groups: {n8n_sg}", errors)
        must_contain(
            n8n,
            'alb.ingress.kubernetes.io/manage-backend-security-group-rules: "true"',
            errors,
        )

    admin_sg = _sg_id(data, "olympus_admin_cloudfront_sg_id", errors)
    if admin_sg:
        for rel in PATH_INGRESS_VALUES:
            must_contain(PARENT / rel, admin_sg, errors)
    return errors


def _sg_id(data: dict, key: str, errors: list[str]) -> str:
    try:
        return require_sg_id(data, key)
    except ValueError as exc:
        errors.append(str(exc))
        return ""


def aws_text(args: list[str]) -> str:
    aws = shutil.which("aws")
    if not aws:
        raise SystemExit("aws CLI ABSENT")
    proc = subprocess.run([aws, *args], capture_output=True)
    if proc.returncode != 0:
        raise SystemExit("aws call failed")
    return proc.stdout.decode("utf-8", "replace").strip()


def live_errors(data: dict) -> list[str]:
    """Compare AWS to the snapshot. Never write the snapshot."""
    region = data.get("aws_region") or "eu-central-1"
    errors: list[str] = []

    alb = aws_text(
        [
            "elbv2",
            "describe-load-balancers",
            "--region",
            region,
            "--load-balancer-arns",
            data["litellm_alb_arn"],
            "--query",
            "LoadBalancers[0].AvailabilityZones[].SubnetId",
            "--output",
            "json",
        ]
    )
    live_subnets = json.loads(alb)
    json_subnets = list(data["public_subnet_ids"])
    print("JSON_SUBNETS", " ".join(json_subnets), flush=True)
    print("LIVE_ALB_SUBNETS", " ".join(live_subnets), flush=True)
    match = same_ids(json_subnets, live_subnets)
    print("SUBNET_MATCH", match, flush=True)
    if not match:
        errors.append("public_subnet_ids != live olympus-admin ALB subnets")

    pool = data["litellm_cognito_user_pool_id"]
    client = data["litellm_cognito_user_pool_client_id"]
    domain = data.get("litellm_cognito_custom_domain") or data["litellm_cognito_user_pool_domain"]
    got_pool = aws_text(
        [
            "cognito-idp",
            "describe-user-pool",
            "--region",
            region,
            "--user-pool-id",
            pool,
            "--query",
            "UserPool.Id",
            "--output",
            "text",
        ]
    )
    got_client = aws_text(
        [
            "cognito-idp",
            "describe-user-pool-client",
            "--region",
            region,
            "--user-pool-id",
            pool,
            "--client-id",
            client,
            "--query",
            "UserPoolClient.ClientId",
            "--output",
            "text",
        ]
    )
    got_domain_pool = aws_text(
        [
            "cognito-idp",
            "describe-user-pool-domain",
            "--region",
            region,
            "--domain",
            domain,
            "--query",
            "DomainDescription.UserPoolId",
            "--output",
            "text",
        ]
    )
    pool_ok = got_pool == pool
    client_ok = got_client == client
    domain_ok = got_domain_pool == pool
    print("COGNITO_POOL_MATCH", pool_ok, flush=True)
    print("COGNITO_CLIENT_MATCH", client_ok, flush=True)
    print("COGNITO_DOMAIN_MATCH", domain_ok, flush=True)
    if not pool_ok:
        errors.append("litellm_cognito_user_pool_id != live user pool")
    if not client_ok:
        errors.append("litellm_cognito_user_pool_client_id != live app client")
    if not domain_ok:
        errors.append("cognito domain does not belong to the snapshot user pool")
    return errors


def main() -> int:
    self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Compare the snapshot to live ALB subnets and Cognito. Does not edit files.",
    )
    args = parser.parse_args()
    data = json.loads(INFRA.read_text(encoding="utf-8"))
    errors = live_errors(data) if args.live else []
    errors.extend(file_errors(data))
    if errors:
        print("infra SSOT drift:")
        for error in errors:
            print(f"  - {error}")
        return 1
    if args.live:
        print("OK: live AWS matches platform-infra.dev.json and consumers")
    else:
        print("OK: consumers match platform-infra.dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
