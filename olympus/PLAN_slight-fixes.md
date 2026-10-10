# Olympus slight fixes — plan only

Status: **plan**. No Architecture viewer, Console n8n, content-strip, or Overview-title code in this change. No merge. No deploy.

Parallel, out of scope: Secrets Manager work on [ai-platform#26](https://github.com/LevKesha/ai-platform/pull/26) (head `1a786ab9`, still draft). Agent `bc-b0f448c8`. Do not merge #26 from this plan.

Skills used as helpers (GitHub is SSOT): `frontend-engineer`, `ui-ux-engineer`, `olympus-edge`, `cv-jobs-poc`. Edge contract: n8n stays on `https://n8n.levkesha.com`. `N8N_EDITOR_BASE_URL` is that host (`n8n/k8s/secrets.yaml`). Olympus static is S3 + CloudFront. Console is vanilla HTML/JS, no backend.

## #26 impact

Checked `origin/main...1a786ab9`. Files: `scripts/inject-secrets-from-sm.ps1`, `scripts/inject-secrets-from-sm.sh`, `n8n-config/IMPORT.md`, `n8n-config/README.md`, `n8n-config/import-orchestrator.py`, `n8n/README.md`. **No `olympus/` HTML, CSS, or JS.**

| Slight fix | Blocked or changed by #26? |
|---|---|
| A Architecture full-view | No. Static page + CSS. Diagrams may show SM **names**, not values, and must not teach `.env` as SSOT. |
| B Console in-portal n8n | No file conflict. Hard rule below. Do not read `N8N_API_KEY` in the browser. |
| C Content strip | No. Copy lock is Content. Import/API sentences, if any are added later, point at `scripts/inject-secrets-from-sm.ps1` / `.sh`. No paste-key-into-chat. |
| D Overview title | No. Hold for Lev. |

Nothing in #26 blocks this portal plan as scoped.

**Console n8n hard rule.** The browser must never hold `N8N_API_KEY`: not in `console-data.js`, `localStorage`, or any client env. The key stays in Secrets Manager (`dev-cluster-n8n/api-key`) and server-side inject only. A Console n8n view is status / UI / a server-side proxy that never ships the key to the browser.

**Also locked for later copy and diagrams**

- SM names are allowed. Secret values are not.
- Do not present a gitignored `.env` as SSOT.
- `olympus/auth/` stays gated. The content-strip PR does not edit it.
- Legacy `.env` keys `LITELLM_SPEND_KEY` and `LITELLM_DB_PASSWORD` do not change this plan. Console must not fetch them from the browser.

**Soft-warn backlog (not portal PRs)**

1. Terraform does not own SM names `dev-cluster-n8n/api-key` and `dev-cluster-agent-api/local-orchestrator-keys`.
2. Leftover legacy `.env` keys `LITELLM_SPEND_KEY` and `LITELLM_DB_PASSWORD`.
3. Bash inject (`scripts/inject-secrets-from-sm.sh`) writes SM payloads to `mktemp` files (`chmod 600`, `rm -f` on the normal path). A kill before `rm` can leave them. No portal change.

## A — Architecture full-size view

Today: `olympus/architecture.html` links each poster to the same PNG. The browser opens that file at native size. On-page column is `max-width: 72rem` (`olympus/styles.css` line 1293; an earlier `max-width: none` at line 521 loses). Do not change `72rem`. Do not shrink on-page images.

Intrinsic sizes (px): `edge-three-band-v4.png` 5200×3500, `k8s-internal-hero-v4.png` 3200×1720, `b1-cv-jobs-seq-v4.png` 3200×1562, `b2-cost-spend-loop-v4.png` 2800×1562, `b3-soft-down-restore-v4.png` 3000×1962. All are wider than a normal window, so the browser image viewer fits them to the viewport. A larger export does not make that fitted view bigger.

| Option | Effect | Call |
|---|---|---|
| CSS full-view page | Same PNG, width about 110–115vw, page scroll. On-page column unchanged. | **Recommend.** Design confirms the percent. Default to hold: 112.5% until Design measures. |
| Larger export assets | Sharper pixels only. Native viewer still fits the window. | Not the size lever. |
| Zoom on `.arch-block img` | Changes the on-page column image. | Reject. |

Later Eng files (not this PR): new full-view page under `olympus/`, a small CSS rule that does not touch `.page-architecture main`, and `architecture.html` anchors pointed at that page instead of the raw PNG. Same asset files.

## B — Console n8n inside the portal

Job: an n8n section under Console whose primary action is not `target="_blank"` to `https://n8n.levkesha.com`.

| Option | Tradeoff | Call |
|---|---|---|
| Console-native status view | Stays on the static site. No key in the client. Shows host, login-gated note, and that workflows are not listed without a server. | **Recommend for the slight fix.** |
| iframe of `n8n.levkesha.com` | 2026-10-06: `GET /` returned the editor HTML with no `X-Frame-Options` and no CSP. `GET /rest/settings` returned JSON with no CORS header. Clickjacking and third-party cookies are still open risks. A framed editor must still never receive `N8N_API_KEY` from Olympus JS. | Not the first PR. Needs Lev. |
| Path proxy under `olympus.levkesha.com` | Conflicts with the locked edge contract (n8n path-under-proxy limit; editor base URL is the n8n host). Would also put Cognito on the Olympus ALB in front of n8n. | Reject unless Lev overrides the edge contract in infrastructure. |
| Browser call to n8n REST with the API key | Ships `N8N_API_KEY` to the client. | **Forbidden.** |

Later Eng files, after Lev accepts the native status view: `olympus/console-data.js` (new view id, no key), `olympus/console.js` (renderer; primary control stays in the panel), `olympus/console.html` only if the noscript line must mention the view. Global nav `target="_blank"` links are a separate Lev choice; do not silently retarget every header in the same PR.

### External n8n links (inventory)

Same header link, `target="_blank"`:

- `olympus/index.html:21`
- `olympus/proof.html:21`
- `olympus/architecture.html:21`
- `olympus/cursor/index.html:21`
- `olympus/evidence.html:21`
- `olympus/console.html:21`
- `olympus/404.html:21`

Also:

- `olympus/evidence.html:100` body link, `target="_blank"`
- `olympus/console.js:63` public service URL (n8n is the only `public: true` row), `target="_blank"`
- `olympus/console.js:115` “Retry n8n.levkesha.com”, `target="_blank"`
- `olympus/console-data.js:67` topology URL (data)
- `olympus/public-data.js:16` public-edge URL (data)
- `olympus/console-data.js:29` and `:34` webhook URLs (POST probes, not nav)

`olympus/README.md:21` and `:45` document the host. Not a visitor control.

## C — Content strip (words only, after Content lock)

Eng does not rewrite copy until Content locks delete/replace text. Proposed owner for every hit below: **Content**. Eng applies the locked text later. `olympus/auth/` is not part of that PR.

Grep of `olympus/` for honesty / Theseus / CLI. There is **no** visitor-page string `AWS CLI`.

### Honesty notes

- `olympus/cursor/index.html:47` `<ul class="honesty">` through the list items at lines 48–50
- `olympus/styles.css:1453` and `:1462` — style only; drop if the list is removed
- `olympus/cursor/index.html:52` personal note repeats the same idea. Content decides whether that sentence stays.

No other page renders `class="honesty"`.

### Theseus

- `olympus/public-data.js:45` agent-api summary
- `olympus/evidence.html:68` agent-api cell
- `olympus/cursor/index.html:69` subcommand `theseus`
- `olympus/cursor/index.html:72` “Theseus (CLI)”
- `olympus/cursor/index.html:77` “Theseus github-only”
- `olympus/console-data.js:113` spend surface name `Theseus`
- `olympus/architecture.svg:3` desc, and `:51` label `Theseus · spend`

`architecture.svg` is not referenced by any HTML page. Raster text is not greppable. An image read of `olympus/diagrams/edge-three-band-v4.png` showed a cluster-internal label “Theseus · spend”. Design confirms before any PNG edit. Other PNGs were not pixel-read in this pass.

### AWS CLI vs other CLI

Operator script only (do not edit; `olympus/auth/` stays gated):

- `olympus/auth/apply-brand-shell.sh:21` echo `aws CLI required`

Page strings that say CLI and are **not** AWS CLI (Content: leave unless Lev meant every CLI mention):

- `olympus/cursor/index.html:36` alt text “CLI orchestrator”
- `olympus/cursor/index.html:68` summary “CLI / models”
- `olympus/cursor/index.html:72` “Theseus (CLI)” (also a Theseus hit)
- `olympus/cursor/index.html:73` “CLI orchestrator”
- `olympus/console-data.js:122` hop name `CLI build hop`

## D — Overview title

Hold. Content/Design own it. Current hero is `olympus/index.html:28`: “A cloud engineer who owns what he runs.” Document title `olympus/index.html:6` is “Olympus — Overview”. No Eng PR until Lev picks the replacement.

## Order of PRs

Independent. Do not bundle.

1. **This PR** — plan markdown only. Draft. No merge.
2. **A** — after Design names the percent (hold 112.5% until then). Eng, `olympus/` full-view page. No `72rem` change. No new PNG required.
3. **B** — after Lev accepts the Console-native status view (or explicitly picks iframe / proxy). Eng. Hard rule: no `N8N_API_KEY` in the browser. Path-proxy needs an edge-contract override and an infrastructure change; it is not this PR.
4. **C** — after Content locks the replacement strings. Eng applies words only. Skip `olympus/auth/`.
5. **D** — no PR until Lev picks the title. Fold into C only if that lock includes the title.

Deploy of Olympus stays on the existing `main` workflow after a later merge. Not this run.

## Risks

- Treating a bigger PNG as the full-size lever does nothing while the browser fits the image to the window.
- An iframe looks available because `GET /` sends no frame denial. That is not an auth or clickjacking pass.
- A path under `olympus.levkesha.com/n8n` breaks the locked n8n host contract.
- Putting the API key in Console JS undoes the #26 security pass.
- Deleting every “CLI” string would remove orchestrator inventory Content did not mark as AWS CLI.
- Editing `olympus/auth/` in the copy PR crosses the gate.
