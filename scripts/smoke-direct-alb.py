#!/usr/bin/env python3
"""Probe the n8n ALB DNS directly. Timeout or 403 means the CloudFront-only lock holds.

One attempt. This does not apply manifests and does not retry.
DNS comes from n8n_alb_arn in platform-infra.dev.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "platform-infra.dev.json"


def classify(returncode: int, http_code: str) -> None:
    code = (http_code or "").strip()
    if returncode == 28:
        return
    if returncode == 0 and code == "403":
        return
    raise SystemExit(
        f"direct ALB expected timeout or 403, got http {code or 'none'} exit {returncode}"
    )


def self_test() -> None:
    classify(28, "000")
    classify(0, "403")
    for returncode, http_code in ((0, "200"), (0, "401"), (60, "000"), (7, "000")):
        try:
            classify(returncode, http_code)
        except SystemExit:
            continue
        raise SystemExit(f"must fail http {http_code} exit {returncode}")
    print("self-test ok", flush=True)


def alb_dns(data: dict) -> str:
    arn = data.get("n8n_alb_arn")
    if not isinstance(arn, str) or not arn.strip():
        raise SystemExit("n8n_alb_arn is missing or empty")
    aws = shutil.which("aws")
    if not aws:
        raise SystemExit("aws CLI ABSENT")
    region = data.get("aws_region") or "eu-central-1"
    proc = subprocess.run(
        [
            aws,
            "elbv2",
            "describe-load-balancers",
            "--region",
            region,
            "--load-balancer-arns",
            arn.strip(),
            "--query",
            "LoadBalancers[0].DNSName",
            "--output",
            "text",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit("aws call failed")
    dns = proc.stdout.decode("utf-8", "replace").strip()
    if not dns or dns == "None":
        raise SystemExit("n8n ALB DNS is missing")
    return dns


def probe(dns: str) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise SystemExit("curl ABSENT")
    proc = subprocess.run(
        [
            curl,
            "-sS",
            "-o",
            os.devnull,
            "-w",
            "%{http_code}",
            "--connect-timeout",
            "8",
            "--max-time",
            "12",
            "-k",
            f"https://{dns}/",
        ],
        capture_output=True,
        text=True,
    )
    print(f"direct_alb_dns={dns} http={proc.stdout.strip() or 'none'} exit={proc.returncode}", flush=True)
    classify(proc.returncode, proc.stdout)


def main() -> int:
    self_test()
    if "--self-test-only" in sys.argv[1:]:
        return 0
    data = json.loads(INFRA.read_text(encoding="utf-8"))
    probe(alb_dns(data))
    print("OK: direct n8n ALB timed out or returned 403")
    return 0


if __name__ == "__main__":
    sys.exit(main())
