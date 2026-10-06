/** Demo fixtures for the read-only console. Not live telemetry. */
window.OLYMPUS_CONSOLE = {
  views: [
    { id: "topology", label: "Platform Topology" },
    { id: "configuration", label: "Configuration" },
    { id: "delivery", label: "Delivery" },
    { id: "infrastructure", label: "Infrastructure" },
    { id: "cvjobs", label: "CV×Jobs" },
    { id: "spend", label: "Services & Spend" },
    { id: "headroom", label: "Headroom Admin" },
    { id: "litellm", label: "LiteLLM Admin UI" },
    { id: "n8n", label: "n8n" },
  ],
  cvjobs: {
    demoUrl: "https://olympus.levkesha.com/cv-jobs/",
    title: "CV×Jobs",
    lede: "Matcher ↔ Editor · fixture-first",
    chip: "Not LinkedIn auto-apply",
  },
  headroom: {
    adminUrl: "https://olympus.levkesha.com/headroom",
    webhookUrl: "https://n8n.levkesha.com/webhook/headroom-demo",
    title: "Headroom savings Admin",
  },
  n8n: {
    title: "n8n",
    url: "https://n8n.levkesha.com",
    sell: "Workflows on their own host — open them, don’t rebuild them.",
    demo: "Demo Mode — Read-Only",
    workflows: [
      {
        name: "orchestrator-workflow",
        note: "One webhook routes rag, agent, or auto into the cluster services.",
      },
      {
        name: "litellm-demo",
        note: "Webhook health check for LiteLLM.",
      },
      {
        name: "headroom-demo",
        note: "Webhook probe into Headroom compress.",
      },
    ],
  },
  litellm: {
    adminUrl: "https://olympus.levkesha.com/litellm/ui",
    webhookUrl: "https://n8n.levkesha.com/webhook/litellm-demo",
    title: "LiteLLM Admin UI",
    screenshare: {
      command: "kubectl -n llm-cost port-forward svc/litellm 4000:4000",
      localUrl: "http://127.0.0.1:4000/ui",
    },
  },
  topology: {
    note: "Four Helm services on EKS. n8n on its own host.",
    services: [
      {
        name: "agent-api",
        role: "FastAPI Bedrock /agent",
        edge: "ClusterIP",
        public: false,
      },
      {
        name: "rag-service",
        role: "ingest + search · Claude on EKS",
        edge: "ClusterIP",
        public: false,
      },
      {
        name: "mcp-server",
        role: "tools · resources · prompts · IRSA",
        edge: "ClusterIP",
        public: false,
      },
      {
        name: "n8n",
        role: "workflows · login · screenshare",
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
        name: "CV×Jobs Demo",
        access: "Cognito at olympus.levkesha.com/cv-jobs → agent-api :8000 (/cv-jobs + /v1/demo)",
      },
      {
        name: "LiteLLM",
        access: "EKS ClusterIP / laptop port-forward",
      },
      {
        name: "Headroom savings Admin",
        access: "Cognito at olympus.levkesha.com/headroom → ClusterIP :8790 → :8787",
      },
      {
        name: "Headroom sidecar",
        access: "ClusterIP with LiteLLM (:8787)",
      },
      {
        name: "agent-api · GitHub",
        access: "ClusterIP / laptop port-forward",
      },
      {
        name: "Spend",
        access: "ClusterIP / laptop port-forward",
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
    n8nOfflineTitle: "n8n is offline",
    n8nOfflineBody:
      "Host unreachable. Continue with Architecture — no invented `/n8n` path.",
  },
};
