# Agent instructions

- **GitHub is SSOT** for infra IDs, model IDs, CI secrets, and shared code (`platform-config.yaml`, `platform-infra.dev.json`, terraform outputs, `cicd/docs/ci-secrets.md`).
- Personal Cursor skills (`~/.cursor/skills`) are **non-authoritative helpers**. If a skill disagrees with GitHub, follow GitHub, then update the skill.
- For debugging (errors, failed tests, regressions, “fix” / “why is this broken”), use **root-cause-first**: reproduce → name the mechanism → reject band-aids → fix the source → verify.
- Do **not** copy a personal skill library into this repo.
- Path ownership is `ownership.json` (exactly one owner per tracked path). Claim new tracked paths there in the same change. Check: `python3 scripts/check-ownership.py`.
- Security review (`python3 scripts/check-security-review.py`) is required only for auth, secrets, identity, and public-edge diffs. Ordinary Olympus portal copy is outside that set. `security_review.reviewers` lists GitHub logins; it is empty until a real login is added.
