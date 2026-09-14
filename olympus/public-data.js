/** Verified inventory only. No invented URLs, metrics, or public edges. */
window.OLYMPUS_PUBLIC = {
  brand: "The Olympus",
  positioning:
    "Senior AI-driven cloud and platform engineering: AWS/EKS orchestration, Bedrock configuration as SSOT, Helm, Terraform, GitHub Actions, and controlled workflow automation. Not a traditional project gallery.",
  thesis:
    "Operate the boundary between AI services, Kubernetes, IaC, delivery automation, and read-only interview evidence.",
  inventoryDate: "2026-09-13",
  intendedHost: {
    hostname: "olympus.levkesha.com",
    status: "live",
  },
  publicEdge: {
    label: "n8n workflow automation",
    note: "Own host. n8n 2.25 has no supported reverse-proxy path under Olympus. Login-gated; interview demo is screenshare.",
    url: "https://n8n.levkesha.com",
  },
  flagship: {
    name: "ai-platform",
    visibility: "Verified public",
    url: "https://github.com/LevKesha/ai-platform",
    summary:
      "EKS orchestration of agent-api, rag-service, mcp-server, and n8n. Bedrock SSOT via platform-config.yaml. Only public platform repository.",
    components: ["agent-api", "rag-service", "mcp-server", "n8n"],
    ssot: "platform-config.yaml",
    cluster: "dev-cluster",
    region: "eu-central-1",
    iacBranch: "dev (main parked)",
  },
  privateRepos: [
    {
      name: "infrastructure",
      summary:
        "Terraform (HCL), charts/, terraform-*.yml GHA, Bedrock budget TF. Active IaC branch is dev; main is parked. No production cluster.",
      tags: ["Terraform", "Helm charts", "GHA"],
    },
    {
      name: "cicd",
      summary: "Reusable GitHub Actions: ECR + Helm deploy SSOT.",
      tags: ["GHA", "ECR", "Helm"],
    },
    {
      name: "agent-api",
      summary:
        "FastAPI Bedrock /agent gateway; Helm/EKS; local orchestrator Research→UX→Frontend→Engineering; Theseus GitHub-as-user. LiteLLM hop only when LITELLM_BASE_URL is set. /agent stays unwired from LiteLLM.",
      tags: ["FastAPI", "Helm"],
    },
    {
      name: "rag-service",
      summary: "Ingestion + semantic search + Claude on EKS; Helm. README empty.",
      tags: ["EKS", "Helm"],
    },
    {
      name: "mcp-server",
      summary: "MCP tools/resources/prompts; Helm; IRSA.",
      tags: ["MCP", "IRSA"],
    },
    {
      name: "llm-cost",
      summary: "LiteLLM ClusterIP + Headroom sidecar + spend; Helm. Not public.",
      tags: ["LiteLLM", "Headroom"],
    },
    {
      name: "microservices",
      summary: "Older services. Helm charts live in infrastructure.",
      tags: ["Helm"],
    },
  ],
  skipped: ["peakyblinders", "moviescicd", "APIs", "azurepipeline", "WorldOfGames"],
  modelId: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
  honesty: [
    "Do not claim Headroom token savings on Theseus.",
    "/agent stays unwired from LiteLLM.",
    "Infra main is parked; live IaC is branch dev; no production cluster.",
    "LiteLLM, Headroom, Theseus, and spend are ClusterIP / laptop port-forward — not public.",
    "CLI build hop is laptop compose; platform proxy is EKS ClusterIP.",
    "n8n stays on n8n.levkesha.com at / — n8n 2.25 does not support a reverse-proxy path under olympus.levkesha.com.",
    "n8n editor demo is screenshare; login required. Not an anonymous portfolio app.",
    "Headroom demo on Olympus POSTs n8n /webhook/headroom-demo → ClusterIP :8787 /v1/compress. Not a public Headroom edge.",
    "LiteLLM Admin UI demo on Olympus POSTs n8n /webhook/litellm-demo → ClusterIP :4000. Real /ui is port-forward screenshare. Not a public LiteLLM URL; not an iframe; no master key.",
  ],
};
