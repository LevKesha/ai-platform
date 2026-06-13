import json
import os
import re

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")

app = FastAPI(title="claude-router", version="0.1.0")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

SYSTEM = (
    "Route user input to exactly one mode: rag or agent. "
    "rag: factual lookup, document/knowledge retrieval, Q&A over stored content. "
    "agent: actions, tool use, multi-step tasks, orchestration, or open-ended reasoning. "
    "Reply with only one word: rag or agent."
)


class RouteRequest(BaseModel):
    task: str = "classify_route"
    allowed_modes: list[str] = ["rag", "agent"]
    input: str


class RouteResponse(BaseModel):
    mode: str
    route: str


def invoke_claude(user_input: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"Classify:\n\n{user_input}"}],
    }
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    return json.loads(response["body"].read())["content"][0]["text"].strip()


def pick_mode(text: str, allowed: list[str]) -> str:
    lowered = text.lower()
    for mode in allowed:
        if re.search(rf"\b{re.escape(mode)}\b", lowered):
            return mode
    return allowed[0]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input required")
    if req.task != "classify_route":
        raise HTTPException(status_code=400, detail=f"unsupported task: {req.task}")

    allowed = [m.lower() for m in req.allowed_modes] or ["rag", "agent"]
    try:
        mode = pick_mode(invoke_claude(req.input), allowed)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=503, detail=f"bedrock error: {exc}")

    return RouteResponse(mode=mode, route=mode)
