"""Bedrock Anthropic (Claude) invoke helpers."""
from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"


def bedrock_runtime_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def invoke_claude(
    model_id: str,
    region: str,
    *,
    messages: list[dict[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    tools: list | None = None,
    tool_choice: dict | None = None,
    guardrail_config: dict | None = None,
    client=None,
) -> dict[str, Any]:
    """Invoke Claude on Bedrock; return parsed response body."""
    body: dict[str, Any] = {
        "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    runtime = client or bedrock_runtime_client(region)
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "body": json.dumps(body),
    }
    if guardrail_config:
        kwargs["guardrailConfig"] = guardrail_config

    try:
        response = runtime.invoke_model(**kwargs)
        return json.loads(response["body"].read())
    except (BotoCoreError, ClientError):
        logger.exception("Bedrock Claude invoke failed")
        raise


def claude_text(result: dict[str, Any]) -> str:
    """Extract first text block from an Anthropic Messages response."""
    return result["content"][0]["text"].strip()
