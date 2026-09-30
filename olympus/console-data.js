/** Demo fixtures for the read-only console. Not live telemetry. */
window.OLYMPUS_CONSOLE = {
  banner: "Demo Mode — Read-Only",
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
    { id: "cvjobs", label: "CV×Jobs" },
    { id: "spend", label: "Services & Spend" },
    { id: "headroom", label: "Headroom Admin" },
    { id: "litellm", label: "LiteLLM Admin UI" },
  ],
  cvjobs: {
    demoUrl: "https://olympus.levkesha.com/cv-jobs/",
    title: "CV×Jobs",
    lede: "Matcher ↔ Editor · fixture-first",
    chip: "Not LinkedIn auto-apply",
    honesty: [
      "Not LinkedIn auto-apply. Not resume.io Job Tracker sync.",
      "Provenance fixture|user_pasted only — live → 422. Evidence-lock + 3-round feedback cap.",
      "Demo /v1/demo/headroom = measured tokens on this CV+job — not Theseus, not Headroom savings ledger.",
      "Develop traffic should also hit /headroom compress (cv_jobs_* profiles) to fill savings rows.",
    ],
  },
  headroom: {
    adminUrl: "https://olympus.levkesha.com/headroom",
    webhookUrl: "https://n8n.levkesha.com/webhook/headroom-demo",
    title: "Headroom savings Admin",
    lede: "Primary: Cognito-gated Admin at olympus.levkesha.com/headroom → thin ClusterIP savings API → Headroom :8787. Secondary n8n demo stays for screenshare without Cognito.",
    honesty: [
      "Cognito path under Olympus — not anonymous public Headroom. Not Theseus. Not LiteLLM spend.",
      "Admin UI + /headroom/api/* share Cognito with /litellm (same ALB group olympus-admin).",
      "Payload is Anthropic tool_result + log-like tool output (aligns with llm-cost/scripts/probe_compress.py).",
      "JSON-stringified PR-array dumps often router:noop on Headroom 0.37.0 — probe uses compressable log lines.",
      "applied_guardrails ≠ a promised 90%. Read tokens_before / tokens_after from the Admin or this demo click.",
      "n8n demo fails honestly if webhook or :8787 is down — no fixture ratio.",
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
        note: "Fixture-first. Demo headroom ≠ Headroom savings ledger.",
      },
      {
        name: "LiteLLM",
        access: "EKS ClusterIP / laptop port-forward",
        note: "Not a public URL.",
      },
      {
        name: "Headroom savings Admin",
        access: "Cognito at olympus.levkesha.com/headroom → ClusterIP :8790 → :8787",
        note: "Not Theseus. Not LiteLLM spend. Ledger is Headroom tokens only.",
      },
      {
        name: "Headroom sidecar",
        access: "ClusterIP with LiteLLM (:8787)",
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
    n8nOfflineTitle: "n8n is offline",
    n8nOfflineBody:
      "Host unreachable. Continue with Architecture and Selected Work — no invented `/n8n` path.",
  },
};
