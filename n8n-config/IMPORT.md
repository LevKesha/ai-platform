# Orchestrator import runbook (Phase 2+)

Update the live n8n orchestrator from `orchestrator-workflow.json` via REST API.

## Prerequisites

- Owner account claimed in n8n UI
- API key in env with scopes: `workflow:read`, `workflow:update`, `workflow:activate`
- Workflow id `JW6BZO6PZnExaOa5` (override with `N8N_WORKFLOW_ID` if needed)
- Python 3.10+ (stdlib only)

## Environment

```powershell
$env:N8N_HOST = "https://n8n.levkesha.com"   # default
$env:N8N_API_KEY = "<from n8n Settings → API>"
$env:N8N_WORKFLOW_ID = "JW6BZO6PZnExaOa5"    # optional
```

From ai-platform repo root:

```powershell
cd C:\Users\zxcv0\PycharmProjects\ai-platform
python n8n-config/import-orchestrator.py
```

## WAF 403 fallback (port-forward)

`n8n-dev` WAF may block `PUT /api/v1/workflows/*` on the public hostname until an allow rule lands in Terraform.

```powershell
kubectl -n n8n port-forward svc/n8n 5678:5678
python n8n-config/import-orchestrator.py --host http://127.0.0.1:5678
```

Re-run smoke against production URL after import (webhook is registered on the live instance):

```powershell
python n8n-config/import-orchestrator.py --smoke all
```

## Smoke tests

```powershell
python n8n-config/import-orchestrator.py --smoke rag
python n8n-config/import-orchestrator.py --smoke runbook
python n8n-config/import-orchestrator.py --smoke all
```

Expected success shape:

```json
{ "mode_used": "rag|runbook", "answer": "...", "meta": {} }
```

Manual curl (production webhook):

```powershell
curl -X POST "$env:N8N_HOST/webhook/ai-orchestrator" `
  -H "Content-Type: application/json" `
  -d '{"mode":"runbook","input":"test"}'
```

## Known issues

| Issue | Workaround |
|-------|------------|
| WAF 403 on PUT | Use `--host http://127.0.0.1:5678` with kubectl port-forward, or wait for WAF allow rule in infrastructure TF |
| PUT 400 read-only field | Script strips `id`, `active`, `versionId`, `meta`, `tags` automatically |
| PUT invalid connections | Script removes `.error` connection keys |
| Set node `?? {}` | n8n Set nodes do not support JS nullish coalescing; use `$json.field \|\| {}` or IF branches instead |

## Checklist

```
- [ ] Edit orchestrator-workflow.json
- [ ] N8N_API_KEY set (never commit)
- [ ] python n8n-config/import-orchestrator.py
- [ ] --smoke runbook (and rag) pass
- [ ] Report workflow id and webhook URL
```
