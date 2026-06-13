# claude-router

Minimal Bedrock classifier for n8n orchestrator **auto** mode. Routes user input to `rag` or `agent`.

## API

- `GET /health` — liveness
- `POST /route` — classify input

```json
{"task":"classify_route","allowed_modes":["rag","agent"],"input":"What is our refund policy?"}
```

Response: `{"mode":"rag","route":"rag"}` (either field works for n8n IF node).

## Env

| Variable | Default |
|----------|---------|
| `CLAUDE_MODEL_ID` | `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `AWS_REGION` | `eu-central-1` |

Uses IRSA (`AgentApiIRSA-dev`) for Bedrock invoke. Trust policy must include `system:serviceaccount:dev:claude-router`.

## K8s

Namespace `dev`, ClusterIP **port 8000** (not internet-exposed):

`http://claude-router.dev.svc.cluster.local:8000/route`
