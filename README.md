# ai-platform
AI Platform POC — agent-api, rag-service, mcp-server, n8n orchestration on AWS EKS

## Platform SSOT

- **`platform-config.yaml`** — Bedrock model IDs / embedding model (LLM SSOT).
- **`packages/chat_helper/pricing.py`** — chat caps. `MAX_OUTPUT_TOKENS` defaults to 400 and reads env `MAX_OUTPUT_TOKENS`. `fit_input` trims input to `MAX_INPUT_TOKENS` (8000). Infrastructure sets the Lambda env to 400 and reserved concurrency to 2. The account-wide Bedrock alarm (Sum > 30 `Invocations` per 300 seconds) lives in `LevKesha/infrastructure` and counts every Bedrock call, including chat, RAG, and LiteLLM.
- **`platform-infra.dev.json`** — infra IDs snapshot (ACM, subnets, IRSA, ECR). **Owner:** `LevKesha/infrastructure` terraform outputs; refresh via `../infrastructure/scripts/export-platform-outputs.py`.
- **`packages/platform_common/`** — shared JSON logging, AWS text summaries, Bedrock Claude invoke. Vendor into service repos: `python scripts/sync-platform-common.py` (edit SSOT only; do not hand-edit copies).
- Render k8s from snapshot: `python scripts/render-from-infra.py`
- Sync sibling helm/agent-spec IRSA+ECR: `python scripts/sync-sibling-irsa.py`
- Drift checks: `python scripts/check-platform-config.py` and `python scripts/check-infra-consumers.py`. The repo set is `scripts/platform_repos.py` (every `LevKesha` repo). Locally a sibling clone that is not present is skipped; CI (`platform-config-siblings` job) clones `agent-api`, `athena-labs`, `cicd`, `infrastructure`, `llm-cost`, `mcp-server`, and `rag-service`, and runs with `PLATFORM_CONFIG_REQUIRE_SIBLINGS=1`, so a missing clone fails. That job needs the Actions secret `SIBLING_REPOS_READ_TOKEN`: a read-only token (`contents: read`) limited to those sibling repos, as a fine-grained PAT or GitHub App token. The job fails until the secret exists. Outside Actions, `check-platform-config.py` also fails if `gh repo list LevKesha` disagrees with `platform_repos.py`.
- Path ownership: `python3 scripts/check-ownership.py` (`ownership.json`)
- Security review for auth, secrets, identity, and the public edge: `python3 scripts/check-security-review.py`

## Theme 4 status

- `n8n-config/` contains the orchestrator workflow JSON + usage docs.
- `n8n/` contains a self-hosted EKS baseline for n8n + Postgres.
