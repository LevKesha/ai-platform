# n8n on EKS (self-hosted) + RDS PostgreSQL

## Files

- `k8s/base.yaml`: namespace, n8n PVC, n8n deployment/service
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

1. Update secrets:
   - copy `k8s/secrets.example.yaml` to `k8s/secrets.yaml`
   - set RDS endpoint/port/db/user/password
   - set stable `N8N_ENCRYPTION_KEY`
   - set `N8N_HOST` and `WEBHOOK_URL`
2. Apply manifests:

```bash
kubectl apply -f n8n/k8s/secrets.yaml
kubectl apply -f n8n/k8s/base.yaml
```

3. Check rollout:

```bash
kubectl -n n8n get pods,svc
kubectl -n n8n rollout status deploy/n8n
```

4. Access n8n:
   - use `EXTERNAL-IP` from `kubectl -n n8n get svc n8n`
   - open `http://<external-ip>:5678` (or DNS target)

## Recommended next hardening

- Move from `Service type: LoadBalancer` to Ingress + TLS.
- Restrict source CIDRs / add WAF.
- Keep secrets out of git (or use External Secrets + AWS Secrets Manager).
- Enforce TLS validation to RDS by mounting AWS RDS CA bundle and enabling strict cert checks.
