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

Only public runtime edge: `https://n8n.levkesha.com`.

## Pages

| File | Purpose |
|------|---------|
| `index.html` | Positioning |
| `architecture.html` | Topology + `architecture.svg` |
| `projects.html` | ai-platform featured; private evidence cards |
| `evidence.html` | Inventory table |
| `console.html` | Demo console (no auth, read-only) |
| `404.html` | Empty / not-found |

## Console

`console.js` switches views from `console-data.js`. Views: Platform Topology, Configuration, Delivery, Infrastructure, Services & Spend. Banner is Demo Mode – Read-Only. No mutation controls.

Verified copy lives in `public-data.js`. Do not invent metrics or extra public URLs.
