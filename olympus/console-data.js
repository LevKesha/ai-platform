/** Demo fixtures for the read-only console. Not live telemetry. */
window.OLYMPUS_CONSOLE = {
  verifiedOn: "2026-10-17",
  wakeWithin: "about 7 min",
  views: [
    { id: "glossary", label: "Glossary" },
    { id: "cvjobs", label: "CV×Jobs" },
    { id: "spend", label: "AI spend control" },
    { id: "n8n", label: "n8n" },
    { id: "litellm", label: "LiteLLM" },
    { id: "dashboards", label: "Live dashboards" },
  ],
  cvjobs: {
    title: "CV×Jobs",
    subline: "Score a CV against the jobs you paste, then tailor it for the one you want.",
    badges: ["Behind sign-in"],
    look: "Request a look",
    lookHref: "console.html?state=form",
  },
  spend: {
    title: "AI spend control",
    subline: "Every CV×Jobs AI request goes through one place and gets counted.",
    badge: "Behind sign-in",
    look: "Request a look",
    lookHref: "console.html?state=form",
    lines: [
      {
        label: "Routes",
        text: "Wake-only.",
      },
      {
        label: "Admin",
        text: "admin · Lev only",
      },
      {
        label: "LiteLLM",
        text: "Every CV×Jobs AI request goes through here. This is where its spend is counted.",
      },
      {
        label: "Headroom",
        text: "Sits beside that gateway and trims big prompts.",
      },
    ],
  },
  n8n: {
    title: "n8n",
    subline: "The automations that tie the platform together.",
    pill: "Request a look.",
    cta: "Request a look",
    ctaHref: "console.html?state=form",
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
  litellm: {
    title: "LiteLLM",
    subline: "Every CV×Jobs AI request goes through here.",
    look: "Request a look",
    lookHref: "console.html?state=form",
  },
  dashboards: {
    title: "Live dashboards",
    note: "Demoed live by Lev.",
    cta: "Request a look",
    ctaHref: "console.html?state=form",
    tiles: [
      { label: "LiteLLM", href: "/litellm", login: "admin login" },
      { label: "n8n", href: "https://n8n.levkesha.com", login: "admin login" },
      { label: "CV×Jobs", href: "/cv-jobs/", login: "Sign in" },
      { label: "Headroom", href: "/headroom", login: "Sign in" },
    ],
  },
  edgePaths: ["/cv-jobs", "/v1/demo", "/headroom", "/v1/hr"],
  empty: {
    privateRepo:
      "Private repository. No public GitHub link. Evidence is described here; source is not linked.",
  },
};
