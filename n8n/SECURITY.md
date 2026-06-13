# Security model

## Public edge (internet)

- **Only n8n** is reachable from the internet via the ALB.
- **HTTPS:443** serves traffic; **HTTP:80** redirects to HTTPS (`ssl-redirect: '443'`).
- HSTS enabled on the HTTPS listener.
- Backends (`rag-service`, `agent-api`, `claude-router`, `mcp-server`) use **ClusterIP** — not routable from the internet.

Optional hardening on `n8n/k8s/ingress.yaml`:

```yaml
# Restrict to office/VPN egress IP(s):
alb.ingress.kubernetes.io/inbound-cidrs: "203.0.113.0/32"
```

For fully private access, change `alb.ingress.kubernetes.io/scheme` to `internal` and reach n8n via VPN/bastion (public webhooks will not work without a private path).

## In-cluster (VPC only)

- n8n calls backends over **plain HTTP on port 8000** on Kubernetes DNS names (e.g. `http://rag-service.dev.svc.cluster.local:8000/query`).
- This traffic stays inside the cluster network; it is **not** exposed as a public URL.
- `k8s/network-policies/dev-backends.yaml` limits ingress to those pods: only `n8n` namespace (and `agent-api` → `mcp-server` within `dev`).

## Service ports

All app Services expose **port 8000** (app port), not port 80. Workflow HTTP Request nodes must include `:8000` in URLs.
