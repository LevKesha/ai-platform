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

1. **Webhook** `POST /ai-orchestrator`
2. **Validate Input** (Function)
3. **IF mode**
   - `rag` -> HTTP `rag-service /query`
   - `agent` -> HTTP `agent-api /agent`
   - `auto` -> Claude classify -> IF result -> rag/agent HTTP
4. **Normalize output** (Set)
5. **Respond to Webhook**
6. **Error branch** -> normalized error response

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

