# Rendered from platform-infra.dev.json (infra SSOT). Do not hand-edit ARNs/subnets.
# Refresh: python scripts/render-from-infra.py
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: n8n
  namespace: n8n
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/subnets: __PUBLIC_SUBNET_IDS__
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS":443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS13-1-2-2021-06
    alb.ingress.kubernetes.io/listener-attributes.HTTPS-443: routing.http.response.strict_transport_security.header_value=max-age=31536000;includeSubDomains
    alb.ingress.kubernetes.io/certificate-arn: __N8N_ACM_CERTIFICATE_ARN__
    alb.ingress.kubernetes.io/healthcheck-path: /
    alb.ingress.kubernetes.io/success-codes: '200,401,403'
    alb.ingress.kubernetes.io/load-balancer-attributes: idle_timeout.timeout_seconds=120
spec:
  ingressClassName: alb
  rules:
    - host: __N8N_DOMAIN_NAME__
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: n8n
                port:
                  number: 5678
