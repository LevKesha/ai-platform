# platform_common

Shared Python helpers for LevKesha AI services.

**SSOT location:** this directory (in `LevKesha/ai-platform`).

| Module | Responsibility |
|--------|----------------|
| `logging.py` | JSON structured logging + request_id context |
| `aws_summaries.py` | ECS / S3 / CloudWatch text summaries |
| `bedrock.py` | Claude on Bedrock invoke helpers |

## Consumers

Vendored (copied) into Docker build contexts:

- `../agent-api/platform_common`
- `../rag-service/platform_common`
- `../mcp-server/platform_common`
- `claude-router/platform_common`
- `agents/support-runbook-copilot/platform_common`

```bash
python scripts/sync-platform-common.py
python scripts/sync-platform-common.py --check
```

Edit **here only**, then sync. Do not hand-edit vendored trees.
