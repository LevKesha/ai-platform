/** Demo fixtures for the read-only console. Not live telemetry. */
window.OLYMPUS_CONSOLE = {
  views: [
    { id: "glossary", label: "Glossary" },
    { id: "cvjobs", label: "CV×Jobs" },
    { id: "spend", label: "AI spend control" },
    { id: "n8n", label: "n8n" },
  ],
  cvjobs: {
    title: "CV×Jobs",
    subline: "Score a CV against the jobs you paste, then tailor it for the one you want.",
    badges: ["Behind sign-in", "Not LinkedIn auto-apply"],
    cta: "Request a walkthrough",
    ctaHref: "/",
  },
  spend: {
    title: "AI spend control",
    subline: "Every model call goes through one gateway, gets trimmed, and gets counted.",
    badge: "Behind sign-in",
    lines: [
      {
        label: "LiteLLM",
        text: "the gateway every model call goes through. It counts the spend.",
      },
      {
        label: "Headroom",
        text: "the sidecar beside LiteLLM. It compresses prompts.",
      },
    ],
  },
  n8n: {
    title: "n8n",
    subline: "The automations that tie the platform together.",
    pill: "Asleep. Request a look.",
    cta: "Request a look",
    ctaHref: "#sleep-portal",
    workflows: [
      {
        label: "One request in, the right service answers",
        name: "orchestrator-workflow",
      },
      {
        label: "A quick health check on the model gateway",
        name: "litellm-demo",
      },
      {
        label: "A live test of prompt compression",
        name: "headroom-demo",
      },
    ],
  },
  edgePaths: ["/cv-jobs", "/v1/demo", "/headroom", "/v1/hr", "/litellm", "/litellm/ui"],
  empty: {
    privateRepo:
      "Private repository. No public GitHub link. Evidence is described here; source is not linked.",
  },
};
