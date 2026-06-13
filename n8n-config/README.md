# Theme 4 - n8n Orchestrator Workflow

This folder contains the n8n Cloud artifact for Theme 4:

- `orchestrator-workflow.json` - import this workflow into n8n

## Outcome

One webhook endpoint orchestrates:

- `rag-service` (`/query`)
- `agent-api` (`/agent`)
- `auto` mode via Claude classification, then route to RAG or Agent

Output is normalized to:

```json
{
  "mode_used": "rag|agent",
  "answer": "...",
  "meta": {}
}
```

## Workflow Layout

1. **Webhook** `POST /ai-orchestrator` (stable `webhookId` in JSON for API import registration)
2. **Validate Input** (Function)
3. **IF mode**
   - `rag` -> HTTP `rag-service /query`
   - `agent` -> HTTP `agent-api /agent`
   - `auto` -> Claude classify -> IF result -> rag/agent HTTP
4. **Normalize output** (Set)
5. **Respond to Webhook**
6. **Error branch** -> normalized error response (validation/webhook errors via **Format Error**; HTTP failures via **Format HTTP Error**)

## Webhook ID

The Webhook Trigger node includes a stable `webhookId` (`a1b2c3d4-e5f6-7890-abcd-orchestr8wh01`) so REST API imports register the same production webhook path consistently. Do not change it unless you intentionally want a new webhook registration.

## Error Response Shape

Success (HTTP 200):

```json
{ "mode_used": "rag|agent", "answer": "...", "meta": {} }
```

HTTP Request failures route to **Format HTTP Error** → **Webhook Response Error** (intended HTTP 400):

```json
{
  "ok": false,
  "error": "human-readable message",
  "meta": { "source": "n8n-orchestrator", "node": "Call rag-service /query" }
}
```

Validation errors (`invalid` mode, missing input) use **Format Error** → **Webhook Response Error**.

**n8n 2.12.3 note:** the error respond branch may return HTTP 200 with an empty body (RespondToWebhook `getParentNodes` bug). `rag` / `agent` success paths work. Upgrade n8n or use the UI to fix error respond wiring if you need HTTP 400 for validation errors.

## Required n8n Variables / Credentials

Configure in n8n Cloud:

- `RAG_SERVICE_BASE` (example: `https://rag-service.dev.example.com`)
- `AGENT_API_BASE` (example: `https://agent-api.dev.example.com`)
- `CLAUDE_ROUTER_URL` (optional direct classifier endpoint)
- `SERVICE_API_KEY` (if services require bearer auth)

Use HTTP header auth in HTTP Request nodes where needed:

- `Authorization: Bearer {{$env.SERVICE_API_KEY}}`

## Webhook Security

- Use **Production URL** for real calls; Test URL only for manual testing.
- Enable webhook auth (header or basic auth) in n8n.
- Add input length validation and mode allow-list.
- Keep business logic heavy lifting in your Python services, not in n8n scripts.

## Example Requests

RAG mode:

```bash
curl -X POST "$N8N_WEBHOOK_URL/ai-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{"mode":"rag","input":"Summarize latest ECS cluster issues"}'
```

Agent mode:

```bash
curl -X POST "$N8N_WEBHOOK_URL/ai-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{"mode":"agent","input":"List S3 buckets and inspect alarms"}'
```

Auto mode:

```bash
curl -X POST "$N8N_WEBHOOK_URL/ai-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{"mode":"auto","input":"Why did ingestion fail after deploy?"}'
```

## Notes

- `mcp-server` is part of Theme 4 scope and can be added as another HTTP branch later.
- Current workflow focuses on the required mini-project path (rag/agent/auto).

