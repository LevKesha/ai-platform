# ai-platform
AI Platform POC — agent-api, rag-service, mcp-server, n8n orchestration on AWS EKS

## Platform SSOT

- **`platform-config.yaml`** — Bedrock model IDs / embedding model (LLM SSOT).
- **`platform-infra.dev.json`** — infra IDs snapshot (ACM, subnets, IRSA, ECR). **Owner:** `LevKesha/infrastructure` terraform outputs; refresh via `../infrastructure/scripts/export-platform-outputs.py`.
- **`packages/platform_common/`** — shared JSON logging, AWS text summaries, Bedrock Claude invoke. Vendor into service repos: `python scripts/sync-platform-common.py` (edit SSOT only; do not hand-edit copies).
- Render k8s from snapshot: `python scripts/render-from-infra.py`
- Sync sibling helm/agent-spec IRSA+ECR: `python scripts/sync-sibling-irsa.py`
- Drift checks: `python scripts/check-platform-config.py` and `python scripts/check-infra-consumers.py`

## Theme 4 status

- `n8n-config/` contains the orchestrator workflow JSON + usage docs.
- `n8n/` contains a self-hosted EKS baseline for n8n + Postgres.
