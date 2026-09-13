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
  ],
  topology: {
    note: "Four Helm-deployed services on EKS. Only n8n has a public edge.",
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
        role: "Workflow automation",
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
      "n8n public edge unavailable. This site does not invent a fallback URL. Retry https://n8n.levkesha.com or continue with static evidence.",
  },
};
