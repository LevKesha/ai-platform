#!/usr/bin/env python3
"""Import orchestrator-workflow.json into n8n (PUT + activate). Phase 2+ updates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKFLOW_JSON = os.path.join(SCRIPT_DIR, "orchestrator-workflow.json")
DEFAULT_HOST = "https://n8n.levkesha.com"
DEFAULT_WORKFLOW_ID = "JW6BZO6PZnExaOa5"
WEBHOOK_PATH = "/webhook/ai-orchestrator"
READ_ONLY_FIELDS = ("id", "active", "versionId", "meta", "tags")

SMOKE_PAYLOADS = {
    "rag": {"mode": "rag", "input": "test"},
    "runbook": {"mode": "runbook", "input": "test"},
}


def load_and_prepare(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    for key in READ_ONLY_FIELDS:
        wf.pop(key, None)
    for key in list(wf.get("connections", {})):
        if key.endswith(".error"):
            del wf["connections"][key]
    return wf


def api_request(
    host: str,
    api_key: str,
    method: str,
    path: str,
    body: dict | None = None,
) -> dict:
    url = f"{host.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "X-N8N-API-KEY": api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"{method} {url} failed: HTTP {exc.code}\n{detail}") from exc


def put_workflow(host: str, api_key: str, workflow_id: str, wf: dict) -> dict:
    return api_request(host, api_key, "PUT", f"/api/v1/workflows/{workflow_id}", wf)


def activate_workflow(host: str, api_key: str, workflow_id: str) -> dict:
    return api_request(host, api_key, "POST", f"/api/v1/workflows/{workflow_id}/activate")


def smoke_webhook(host: str, mode: str) -> None:
    payload = SMOKE_PAYLOADS[mode]
    url = f"{host.rstrip('/')}{WEBHOOK_PATH}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"Smoke {mode} failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Smoke {mode} failed: {exc.reason}") from exc

    if body.get("mode_used") != mode:
        raise SystemExit(f"Smoke {mode}: expected mode_used={mode!r}, got {body!r}")
    if "answer" not in body:
        raise SystemExit(f"Smoke {mode}: missing answer in {body!r}")
    print(f"smoke {mode}: ok (mode_used={body['mode_used']!r})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PUT orchestrator-workflow.json to n8n and activate.",
    )
    parser.add_argument(
        "--host",
        help="Override N8N_HOST (e.g. http://127.0.0.1:5678 for kubectl port-forward)",
    )
    parser.add_argument(
        "--workflow-id",
        default=os.environ.get("N8N_WORKFLOW_ID", DEFAULT_WORKFLOW_ID),
        help=f"Target workflow id (default: {DEFAULT_WORKFLOW_ID})",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=DEFAULT_WORKFLOW_JSON,
        help="Path to workflow JSON (default: orchestrator-workflow.json beside this script)",
    )
    parser.add_argument(
        "--smoke",
        choices=["runbook", "rag", "all"],
        help="Optional webhook smoke test after activate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = args.host or os.environ.get("N8N_HOST", DEFAULT_HOST)
    api_key = os.environ.get("N8N_API_KEY")
    if not api_key:
        raise SystemExit("N8N_API_KEY is required")

    wf = load_and_prepare(args.json_path)
    print(f"PUT workflow {args.workflow_id} -> {host}")
    result = put_workflow(host, api_key, args.workflow_id, wf)
    print(f"updated: {result.get('name', '?')} (id={result.get('id', args.workflow_id)})")

    print("POST activate")
    active = activate_workflow(host, api_key, args.workflow_id)
    print(f"active: {active.get('active', True)}")

    if args.smoke:
        modes = ["rag", "runbook"] if args.smoke == "all" else [args.smoke]
        for mode in modes:
            smoke_webhook(host, mode)


if __name__ == "__main__":
    main()
