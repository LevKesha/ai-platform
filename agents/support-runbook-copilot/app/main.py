import os
import uuid

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from platform_common.bedrock import claude_text, invoke_claude

MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")

app = FastAPI(title="support-runbook-copilot", version="0.1.0")

SYSTEM = (
    "You are an incident response copilot. Given an incident summary, produce a concise "
    "numbered action plan for on-call engineers. Include immediate mitigation, investigation "
    "steps, and communication. Be specific and practical. Output plain text with numbered steps."
)


class InvokeRequest(BaseModel):
    incident_summary: str | None = None
    input: str | None = Field(None, description="Alias for incident_summary (n8n compat)")


class InvokeResponse(BaseModel):
    ok: bool = True
    action_plan: str
    meta: dict


def resolve_summary(req: InvokeRequest) -> str:
    summary = (req.incident_summary or req.input or "").strip()
    if not summary:
        raise HTTPException(status_code=400, detail="incident_summary is required")
    if len(summary) > 8000:
        raise HTTPException(status_code=400, detail="incident_summary too long (max 8000)")
    return summary


def invoke_bedrock(incident_summary: str) -> str:
    result = invoke_claude(
        MODEL_ID,
        AWS_REGION,
        messages=[{"role": "user", "content": f"Incident:\n\n{incident_summary}"}],
        system=SYSTEM,
        max_tokens=1200,
        temperature=0.2,
    )
    return claude_text(result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    summary = resolve_summary(req)
    request_id = str(uuid.uuid4())
    try:
        plan = invoke_bedrock(summary)
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(status_code=503, detail=f"bedrock error: {exc}") from exc

    return InvokeResponse(
        action_plan=plan,
        meta={"model": MODEL_ID, "request_id": request_id},
    )
