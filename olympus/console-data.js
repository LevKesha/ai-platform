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
    subline: "Every AI request goes through one place, gets shortened, and gets counted.",
    badge: "Behind sign-in",
    lines: [
      {
        label: "LiteLLM",
        text: "Every AI request goes through here. This is where spend is counted.",
      },
      {
        label: "Headroom",
        text: "Sits beside that gateway and shortens long prompts.",
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
        label: "A quick check that the AI gateway is running",
        name: "litellm-demo",
      },
      {
        label: "A live test that long prompts get shortened",
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
