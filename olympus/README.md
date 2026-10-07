# The Olympus

Static interview evidence site. Vanilla HTML/CSS/JS. No build step, no bundler, no framework.

## Open locally

From this folder, open `index.html` in a browser (double-click or File → Open). Relative links work without a server.

Optional local server from the `olympus` directory:

```text
python -m http.server 8080
```

Then open `http://127.0.0.1:8080/`.

## Hosting

Public static host: `https://olympus.levkesha.com` (S3 + CloudFront).

n8n runtime: `https://n8n.levkesha.com` (own host at `/`). n8n 2.25 has no supported reverse-proxy path under Olympus. Login-gated; interview demo is screenshare.

## Cognito brand shell

Assets for `auth.olympus.levkesha.com` live in `auth/`. Colors only: background `#0a0a0a`, surface `#141414`, border `#2a2a2a`, text `#f5f5f5` / `#a3a3a3`, accent `#2dd4bf`. No logo file.

`auth/apply-brand-shell.sh` reads the pool from `describe-user-pool-domain` and applies either classic `SetUICustomization` (managed-login version 1) or managed-login branding settings (version 2) to every app client on that pool. It does not contain pool or client IDs. Do not run it from the Olympus S3 deploy. The Cognito module is in LevKesha/infrastructure (`cognito-alb-auth`); that module still documents the prefix domain and is not edited here.

Hosted UI and managed login cannot set the strings "The Olympus" or "Sign in to Olympus". Those stay Cognito defaults until AWS exposes editable heading text.

## Pages

| File | Purpose |
|------|---------|
| `index.html` | Positioning |
| `architecture.html` | One column. Five existing diagrams, copy under each. |
| `cursor/index.html` | Place A. Production alias key `cursor` (see Deploy Olympus). Top nav item. |
| `proof.html` | SolarEdge case, timeline rail, skills/certs chips |
| `evidence.html` | Inventory table |
| `console.html` | Demo console (no auth, read-only) |
| `404.html` | Empty / not-found |

## Console

`console.js` switches views from `console-data.js`. Views: Configuration, Delivery, Infrastructure, Services & Spend, **CV×Jobs Demo**, **Headroom Admin**, **LiteLLM Admin UI**. Unknown hashes, including the removed `#topology`, open Configuration. Banner is Demo Mode – Read-Only except live probes and Cognito-gated Admin links. CV×Jobs primary CTA opens `https://olympus.levkesha.com/cv-jobs/` (Cognito → agent-api). Headroom primary CTA opens `https://olympus.levkesha.com/headroom` (Cognito → savings API → `:8787`); secondary demo POSTs `https://n8n.levkesha.com/webhook/headroom-demo`. LiteLLM primary CTA opens `https://olympus.levkesha.com/litellm/ui`; secondary health probe POSTs `https://n8n.levkesha.com/webhook/litellm-demo`. Break-glass: `kubectl -n llm-cost port-forward svc/litellm 4000:4000`. No invented ratios/health on failure.

Verified copy lives in `public-data.js`. Do not invent metrics or extra public URLs.
