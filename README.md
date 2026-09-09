# ai-platform
AI Platform POC — agent-api, rag-service, mcp-server, n8n orchestration on AWS EKS

## Platform SSOT

- **`platform-config.yaml`** — single source of truth for Bedrock model IDs, embedding model, AWS region/account defaults.
- Sibling service repos (`agent-api`, `rag-service`, `mcp-server`) must mirror those IDs in `agent-spec.yaml`, Helm values, and `.env.example`.
- Drift check: `python scripts/check-platform-config.py` (requires PyYAML; expects sibling clones next to this repo).

## Theme 4 status

- `n8n-config/` contains the orchestrator workflow JSON + usage docs.
- `n8n/` contains a self-hosted EKS baseline for n8n + Postgres.
