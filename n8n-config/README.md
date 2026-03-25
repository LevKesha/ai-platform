# Theme 4 — n8n Orchestrator Workflow

Artifact for **self-hosted n8n** (EKS) or n8n Cloud:

- **`orchestrator-workflow.json`** — import into n8n (or sync via GitOps).

## Outcome

One webhook orchestrates:

- **`rag-service`** → `POST /query`
- **`agent-api`** → `POST /agent`
- **`auto`** → classifier HTTP node, then branch to RAG or Agent

Normalized JSON responses (shape depends on branch; see nodes **Normalize** / **Respond to Webhook**).

## Workflow layout (current repo)

1. **Webhook** — `POST` path `ai-orchestrator`; response mode **Respond to Webhook**.
2. **Validate Input** — **Code** node (v2): reads `body` or top-level JSON; `mode` ∈ `rag`, `agent`, `auto`; validates `input`.
3. **IF mode = rag / agent / auto** — chained **If** nodes.
4. **HTTP Request** nodes — `POST`, JSON body; **in-cluster URLs** (example dev namespace):  
   `http://rag-service.dev.svc.cluster.local/query`,  
   `http://agent-api.dev.svc.cluster.local/agent`,  
   `http://claude-router.dev.svc.cluster.local/route` (adjust if your Services differ).
5. **Normalize** + **Respond to Webhook** / error branch.

### Critical: If node version

Workflow JSON uses **If node `typeVersion: 1`** with **`conditions.string`** (value1 / operation / value2).  
**If v2** expects the newer **filter** condition format; **legacy `conditions.string` under v2 is not evaluated correctly** and can route all traffic to the wrong branch. Re-import from this repo after edits.

### n8n v2 and `$env`

Self-hosted n8n v2 may block **`$env.*`** in expressions unless you relax security. This workflow uses **static cluster DNS URLs** in HTTP nodes instead of `$env.RAG_SERVICE_BASE`.

## URLs (test vs production)

- **Test:** `/webhook-test/ai-orchestrator` (while the workflow is listening in test mode).
- **Production:** `/webhook/ai-orchestrator` (active workflow).

Full URL: `http://<n8n-host>:5678/webhook/...` (or HTTPS via LB/Ingress).

## Webhook security

- Prefer **Production** URL for real traffic; test URL only for manual runs.
- Add auth (header / basic) on the webhook in n8n for anything beyond dev.
- Keep heavy logic in **rag-service** / **agent-api**; n8n validates and routes only.

## Example requests

```bash
curl -X POST "http://<n8n-host>:5678/webhook/ai-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{"mode":"rag","input":"Your question"}'

curl -X POST "http://<n8n-host>:5678/webhook/ai-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{"mode":"agent","input":"Your goal"}'
```

## Cross-repo reference

- Infra / destroy / Theme 4 gate: **`infrastructure` repo** → `docs/THEME4_SESSION_2026-03-25.md`
- agent-api MCP / Bedrock: **`agent-api` repo** → `docs/THEME4_AGENT_MCP_2026-03-25.md`
