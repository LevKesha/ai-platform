/** Demo fixtures for the read-only console. Not live telemetry. */
window.OLYMPUS_CONSOLE = {
  banner: "Demo Mode – Read-Only",
  env: {
    cluster: "dev-cluster",
    region: "eu-central-1",
    branch: "dev",
    main: "parked",
    productionCluster: false,
  },
  views: [
    { id: "topology", label: "Platform Topology" },
    { id: "configuration", label: "Configuration" },
    { id: "delivery", label: "Delivery" },
    { id: "infrastructure", label: "Infrastructure" },
    { id: "spend", label: "Services & Spend" },
    { id: "headroom", label: "Headroom Demo" },
    { id: "litellm", label: "LiteLLM Admin UI" },
  ],
  headroom: {
    webhookUrl: "https://n8n.levkesha.com/webhook/headroom-demo",
    title: "Headroom compress probe",
    lede: "This button runs a live in-cluster /v1/compress on the Headroom sidecar. The browser calls n8n; n8n calls ClusterIP LiteLLM:8787. No public Headroom URL.",
    honesty: [
      "Not Theseus. Not /agent. Research/Perplexity does not hit this path.",
      "Payload is a fixed tool-heavy JSON dump (same idea as llm-cost/scripts/probe_compress.py).",
      "applied_guardrails ≠ a promised 90%. Read tokens_before / tokens_after from this click.",
      "If the webhook or :8787 port is not up yet, the run fails honestly — no fixture ratio.",
    ],
  },
  litellm: {
    adminUrl: "https://olympus.levkesha.com/litellm/ui",
    webhookUrl: "https://n8n.levkesha.com/webhook/litellm-demo",
    title: "LiteLLM Admin UI",
    lede: "Open the real LiteLLM Admin at olympus.levkesha.com/litellm/ui. Cognito login gates access; the Service stays ClusterIP. Optional health probe still goes Olympus → n8n → :4000 /health/liveliness.",
    honesty: [
      "Cognito-gated path under Olympus — not anonymous public Admin. Not Theseus. Not /agent.",
      "Path publishes :4000 only. Port 8787 is Headroom ClusterIP — not on olympus.levkesha.com/litellm.",
      "No master key in Olympus git. Break-glass: kubectl port-forward svc/litellm 4000:4000.",
      "Health probe fails honestly if n8n or :4000 is down — no invented status.",
    ],
    screenshare: {
      command: "kubectl -n llm-cost port-forward svc/litellm 4000:4000",
      localUrl: "http://127.0.0.1:4000/ui",
    },
  },
  topology: {
    note: "Four Helm-deployed services on EKS. n8n is on its own host (n8n path-under-proxy limit). Other services are ClusterIP.",
    services: [
      {
        name: "agent-api",
        role: "FastAPI Bedrock /agent gateway",
        edge: "ClusterIP",
        public: false,
      },
      {
        name: "rag-service",
        role: "Ingestion + semantic search + Claude on EKS",
        edge: "ClusterIP",
        public: false,
      },
      {
        name: "mcp-server",
        role: "MCP tools/resources/prompts; IRSA",
        edge: "ClusterIP",
        public: false,
      },
      {
        name: "n8n",
        role: "Workflow automation. Own host; login; screenshare demo.",
        edge: "Public ALB",
        public: true,
        url: "https://n8n.levkesha.com",
      },
    ],
  },
  configuration: {
    ssotFile: "platform-config.yaml",
    ssotRepo: "LevKesha/ai-platform",
    modelId: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    notes: [
      "Bedrock model ID is owned by platform-config.yaml.",
      "/agent stays unwired from LiteLLM.",
      "LiteLLM hop on agent-api only when LITELLM_BASE_URL is set.",
    ],
  },
  delivery: {
    cicd: "Reusable GitHub Actions: ECR + Helm deploy SSOT (private cicd repo).",
    infraWorkflows: "terraform-*.yml GitHub Actions live in infrastructure (private).",
    note: "No trigger, rollback, or mutate controls in this console.",
  },
  infrastructure: {
    iac: "Terraform (HCL) and Helm charts in the private infrastructure repo.",
    cluster: "dev-cluster",
    region: "eu-central-1",
    activeBranch: "dev",
    parkedBranch: "main",
    productionCluster: false,
  },
  spend: {
    surfaces: [
      {
        name: "LiteLLM",
        access: "EKS ClusterIP / laptop port-forward",
        note: "Not a public URL.",
      },
      {
        name: "Headroom sidecar",
        access: "ClusterIP with LiteLLM",
        note: "Do not claim Headroom token savings on Theseus.",
      },
      {
        name: "Theseus",
        access: "ClusterIP / laptop port-forward",
        note: "GitHub-as-user on agent-api. No Headroom savings claim.",
      },
      {
        name: "Spend",
        access: "ClusterIP / laptop port-forward",
        note: "Demo data — not live telemetry.",
      },
    ],
    hops: [
      { name: "CLI build hop", path: "laptop compose" },
      { name: "Platform proxy", path: "EKS ClusterIP" },
    ],
  },
  empty: {
    privateRepo:
      "Private repository. No public GitHub link. Evidence is described here; source is not linked.",
    noPublicEndpoint:
      "No public endpoint. This surface is ClusterIP or laptop port-forward only.",
    demoData: "Demo data — read-only fixture. Not live telemetry.",
    n8nUnavailable:
      "n8n host unavailable. This site does not invent a fallback URL or a /n8n path. Retry https://n8n.levkesha.com or continue with static evidence.",
  },
};
