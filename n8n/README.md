# n8n on EKS (self-hosted) + RDS PostgreSQL

## Files

- `k8s/base.yaml`: namespace, n8n deployment, ClusterIP service
- `k8s/ingress.yaml`: ALB ingress (HTTPS) for `n8n.levkesha.com`
- `k8s/secrets.example.yaml`: secrets template for n8n + external RDS credentials

## Cheap-but-safe RDS profile (dev)

Recommended starting point for personal dev:

- Engine: **PostgreSQL** (managed by RDS)
- Class: **burstable smallest practical** for your region/account
- Deployment: **Single-AZ** (dev)
- Storage: **gp3**, small baseline (for n8n metadata this is usually enough)
- Backups: **enabled** (short retention, e.g. 3-7 days)
- Public access: **No**
- Network: allow inbound 5432 only from worker node/private subnet SGs
- Monitoring: basic CloudWatch alarms (CPU, storage, freeable memory)

Cost controls:

- avoid Multi-AZ in dev
- avoid provisioned IOPS unless needed
- right-size instance/storage after 1-2 weeks of usage metrics

## Deploy

### Prerequisites

1. Copy `k8s/secrets.example.yaml` to `k8s/secrets.yaml` (do not commit).
2. Set RDS endpoint/port/db/user/password, stable `N8N_ENCRYPTION_KEY`, `N8N_HOST`, and `WEBHOOK_URL`.
3. Confirm `k8s/ingress.yaml` has the ACM cert ARN and public subnet annotation (required when EKS nodes are in private subnets).

### Apply order

Apply in this order so secrets exist before the deployment starts and the ALB is created last:

```bash
kubectl apply -f n8n/k8s/secrets.yaml
kubectl apply -f n8n/k8s/base.yaml
kubectl apply -f n8n/k8s/ingress.yaml
```

### Verify rollout

```bash
kubectl -n n8n get pods,svc,ingress
kubectl -n n8n rollout status deploy/n8n
```

### DNS

Create a Route53 alias (or CNAME) for `n8n.levkesha.com` to the ALB hostname from `kubectl -n n8n get ingress n8n`, then open `https://n8n.levkesha.com`.

### Post-deploy (first login)

1. **Claim owner** — open `https://n8n.levkesha.com`, complete the owner signup form (first user becomes instance owner).
2. **Create API key** — Settings → n8n API → create a key for automation/CI if needed.
3. **Import orchestrator workflow** — Workflows → Import from File → select `n8n-config/orchestrator-workflow.json` from this repo. Configure credentials/variables per `n8n-config/README.md`, then activate the workflow.

## Recommended next hardening

- Restrict source CIDRs / add WAF.
- Keep secrets out of git (or use External Secrets + AWS Secrets Manager).
- Enforce TLS validation to RDS by mounting AWS RDS CA bundle and enabling strict cert checks.
