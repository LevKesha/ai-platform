# Agent instructions

- **GitHub is SSOT** for infra IDs, model IDs, CI secrets, and shared code (`platform-config.yaml`, `platform-infra.dev.json`, terraform outputs, `cicd/docs/ci-secrets.md`).
- Personal Cursor skills (`~/.cursor/skills`) are **non-authoritative helpers**. If a skill disagrees with GitHub, follow GitHub, then update the skill.
- For debugging (errors, failed tests, regressions, “fix” / “why is this broken”), use **root-cause-first**: reproduce → name the mechanism → reject band-aids → fix the source → verify.
- Do **not** copy a personal skill library into this repo.
