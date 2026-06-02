# Agent Platform — Mini Project Game Plan

> **Goal:** Build a production-minded AI agent platform on Kubernetes (local → EKS),
> with a mandatory "ready-to-shift" design so any agent or service can be lifted
> to ECS (or another runtime) with zero code changes.

---

## Guiding Principles

1. **Agent = image + spec + config** — the agent contract is the product, not the orchestrator.
2. **Runtime-agnostic first** — no Kubernetes-only objects inside agent code. K8s/ECS adapters live outside.
3. **AWS-native from day one** — ECR for images, Bedrock for LLM, Secrets Manager for secrets, CloudWatch for logs/metrics.
4. **HTTPS only** — all agent invocation over HTTPS. No plain HTTP between services.
5. **Every service must have** a `Dockerfile`, a `/health` endpoint, env-based config, and a defined port. No exceptions.

---

## Repo Inventory & Role in This Plan

| Repo | Language | Private | Role in Plan | Action |
|---|---|---|---|---|
| `ai-platform` | Python | No | **Core monorepo** — n8n orchestration, rag-service, home of GAMEPLAN | Primary workspace |
| `agent-api` | Python | Yes | **Core** — agent gateway, has Dockerfile + k8s/ + app/ + llm/ | Add `ecs/` stubs, add `agent-spec.yaml` |
| `rag-service` | Python | Yes | **Core** — RAG backend, clean Dockerfile, env-based config | Add `k8s/` manifests + `ecs/` stubs |
| `mcp-server` | Python | No | **Core** — MCP tool server, has Dockerfile + k8s/ + tools/ | Add `ecs/` stubs, verify `/health` |
| `infrastructure` | HCL | Yes | **Supporting** — Terraform; add ECR repos, IAM roles, Secrets Manager | Add `ecr.tf`, `iam.tf` for agents |
| `cicd` | Shell | Yes | **Supporting** — CI/CD pipelines; extend to build/push agent images to ECR | Add ECR push step per agent |
| `microservices` | — | Yes | **Reference** — existing service patterns to reuse in agent adapters | Read-only reference |
| `peakyblinders` | Dockerfile | No | **Reference** — container build patterns | Read-only reference |
| `moviescicd` | Python | No | **Reference** — CI/CD patterns for Python services | Read-only reference |
| `APIs` | Python | No | **Reference** — FastAPI patterns reusable in new agent | Read-only reference |
| `azurepipeline` | HCL | No | **Not relevant** — Azure-specific, skip | No action |
| `WorldOfGames` | Python | No | **Not relevant** — old learning project, skip | No action |

---

## Repo Structure (target)
ai-platform/ ← primary workspace
├── agents/
│ └── support-runbook-copilot/ ← first agent (starter use case)
│ ├── Dockerfile
│ ├── app/
│ ├── agent-spec.yaml ← agent contract (single source of truth)
│ ├── k8s/ ← K8s adapter
│ │ ├── deployment.yaml
│ │ ├── service.yaml
│ │ ├── hpa.yaml
│ │ └── secret.yaml
│ └── ecs/ ← ECS adapter (ready-to-shift)
│ ├── task-definition.json
│ └── service.json
├── rag-service/ ← existing ✅ — add k8s/ + ecs/
├── n8n/
│ └── k8s/ ✅ exists → add ecs/
├── infra/ ← mirrors infrastructure repo
│ ├── ecr.tf
│ ├── iam.tf
│ ├── eks.tf
│ └── secrets.tf
└── GAMEPLAN.md


Cross-repo dependencies:
- `agent-api` → add `ecs/` stubs + `agent-spec.yaml`
- `mcp-server` → add `ecs/` stubs, verify `/health`
- `infrastructure` → add ECR repos + IAM task roles for agents
- `cicd` → add ECR image push step per agent

---

## Agent Spec Format

Every agent must have an `agent-spec.yaml` at its root.
Both K8s and ECS adapters are derived from it.

```yaml
apiVersion: agents.platform/v1alpha1
kind: Agent
metadata:
  name: support-runbook-copilot
  version: "1.0.0"
  description: >
    Advises on incidents by retrieving runbooks and generating a suggested action plan.

spec:
  runtime:
    image: <ECR_ACCOUNT>.dkr.ecr.eu-central-1.amazonaws.com/agents/support-runbook-copilot:1.0.0
    port: 8080
    healthcheck:
      path: /health
      intervalSeconds: 30

  invocation:
    mode: api
    transport:
      protocol: https
      port: 443
      tls:
        required: true
        minVersion: "1.2"
    auth:
      type: bearer_token
    endpoint: /invoke
    method: POST

  llm:
    provider: bedrock
    model: anthropic.claude-3-5-sonnet
    temperature: 0.2
    maxTokens: 1200

  security:
    runAsNonRoot: true
    secrets:
      - name: bedrock-credentials
    networkPolicy:
      egress:
        - bedrock.amazonaws.com

  scalability:
    min_instances: 2
    max_instances: 10
    scale_metric: cpu
    scale_target: 60

  resources:
    cpu: "500m"
    memory: "1Gi"

  observability:
    logging:
      level: info
      format: json
    metrics:
      enabled: true
    audit:
      logInputs: true
      logToolCalls: true
      redactSecrets: true
```

---

## Phase 1 — Foundation

### 1.1 Local dev environment
- [ ] Install `kind` or `minikube` for local K8s cluster
- [ ] Configure `kubectl` context for local cluster
- [ ] Confirm AWS CLI profile with Bedrock + ECR access (`eu-central-1`)

### 1.2 ECR setup (extends `infrastructure` repo)
- [ ] Add `ecr.tf` — one ECR repo per service: `rag-service`, `agent-api`, `mcp-server`, `support-runbook-copilot`
- [ ] Add lifecycle policy: keep last 5 tagged images, delete untagged after 1 day
- [ ] Test: build `rag-service` image → push to ECR → pull back

### 1.3 Standardize existing services

**rag-service** (in `ai-platform` and `rag-service` repos):
- [ ] Confirm `/health` endpoint in `app/main.py` — add if missing
- [ ] Add `rag-service/k8s/`: `deployment.yaml`, `service.yaml`, `hpa.yaml`, `secret.yaml`
- [ ] Add `rag-service/ecs/`: `task-definition.json` + `service.json` stubs
- [ ] Add `agent-spec.yaml`

**agent-api** (existing repo — already has Dockerfile + k8s/):
- [ ] Add `ecs/` stubs to match existing Dockerfile
- [ ] Add `agent-spec.yaml`
- [ ] Verify `/health` endpoint exists

**mcp-server** (existing repo — already has Dockerfile + k8s/):
- [ ] Add `ecs/` stubs
- [ ] Add `agent-spec.yaml`
- [ ] Verify `/health` endpoint exists

---

## Phase 2 — First Agent

### 2.1 support-runbook-copilot (new agent)
- [ ] Create `agents/support-runbook-copilot/`
- [ ] Write `agent-spec.yaml` (use template above)
- [ ] Build minimal FastAPI app (`app/main.py`):
  - `POST /invoke` — accepts `incident_summary`, calls Bedrock Claude, returns action plan
  - `GET /health` — returns `{"status": "ok"}`
- [ ] `Dockerfile` — mirror `rag-service` pattern (python:3.11-slim, uvicorn, port 8080)
- [ ] `requirements.txt` + `.env.example`

### 2.2 K8s adapter for the agent
- [ ] `k8s/deployment.yaml` — driven by `scalability` block in agent-spec
- [ ] `k8s/service.yaml`
- [ ] `k8s/hpa.yaml` — min/max from agent-spec, CPU target 60%
- [ ] Deploy to local kind cluster and test `POST /invoke` end-to-end

### 2.3 ECS adapter stub (ready-to-shift)
- [ ] `ecs/task-definition.json` — translate Dockerfile + env + port from agent-spec
- [ ] `ecs/service.json` — translate `scalability` block into ECS autoscaling config

---

## Phase 3 — Observability & Security

- [ ] All services log JSON to stdout (CloudWatch picks up on both EKS and ECS)
- [ ] Add CloudWatch log group per service in `infra/`
- [ ] IAM task roles — least-privilege: Bedrock + ECR + Secrets Manager (extends `infrastructure` repo)
- [ ] `runAsNonRoot: true` in all K8s deployments
- [ ] NetworkPolicy restricting egress per agent spec
- [ ] Basic CloudWatch dashboard: invocations, latency p99, error rate
- [ ] Extend `cicd` repo — add ECR image build + push step per agent on merge to main

---

## Ready-to-Shift Checklist

Before marking any service shiftable to ECS:

- [ ] No K8s-specific objects inside application code
- [ ] All config from env vars only
- [ ] Single process per container (no sidecars baked in)
- [ ] `/health` responds within 5s
- [ ] `ecs/task-definition.json` exists and matches current Dockerfile
- [ ] `ecs/service.json` has correct `minCapacity`, `maxCapacity`, scaling metric
- [ ] Image in ECR with versioned tag (not `latest`)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Orchestrator | Kubernetes (EKS) | Primary; local kind for dev |
| Shift target | Amazon ECS on EC2 | Same image, different adapter |
| Image registry | Amazon ECR | Works identically on EKS and ECS |
| LLM provider | Amazon Bedrock | Already in rag-service; consistent pattern |
| Secret management | AWS Secrets Manager | Works on EKS (IRSA) and ECS (task role) identically |
| Config pattern | Env vars only | Required for runtime portability |
| Invocation | HTTPS API (mode: api) | Consistent across all agents |
| Autoscaling signal | CPU 60% | HPA on K8s → target-tracking policy on ECS |
| Logging | JSON to stdout | CloudWatch on both platforms, no agent change needed |

---

## Current Status Across All Repos

| Service | Dockerfile | Health endpoint | Env config | K8s manifests | ECS stubs | agent-spec |
|---|---|---|---|---|---|---|
| rag-service | ✅ | ❓ verify | ✅ | ❌ add | ❌ add | ❌ add |
| agent-api | ✅ | ❓ verify | ✅ | ✅ exists | ❌ add | ❌ add |
| mcp-server | ✅ | ❓ verify | ❓ verify | ✅ exists | ❌ add | ❌ add |
| n8n | upstream image | N/A | ✅ | ✅ exists | ❌ add | N/A |
| support-runbook-copilot | ❌ create | ❌ create | ❌ create | ❌ create | ❌ create | ❌ create |