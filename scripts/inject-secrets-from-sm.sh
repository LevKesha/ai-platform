#!/usr/bin/env bash
# Load dev secrets from AWS Secrets Manager into this process.
# Prints env names and PRESENT/ABSENT only. Never prints values.
#
# Durable SSOT is Secrets Manager. Source this file so exports stick:
#   source scripts/inject-secrets-from-sm.sh
#   source scripts/inject-secrets-from-sm.sh --dry-run
#
# Secrets:
#   dev-cluster-n8n/api-key                       -> N8N_API_KEY
#   dev-cluster-agent-api/local-orchestrator-keys -> JSON keys as env vars
set +x

REGION="${AWS_REGION:-eu-central-1}"
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
fi

if [[ "${BASH_SOURCE[0]}" == "$0" && "$DRY" -eq 0 ]]; then
  echo "source this script to inject into the current shell (or pass --dry-run)" >&2
  exit 2
fi

N8N_SECRET="dev-cluster-n8n/api-key"
ORCH_SECRET="dev-cluster-agent-api/local-orchestrator-keys"

# Prefer the Windows py launcher. Git Bash often resolves python3 to a
# Store alias that is not a real interpreter.
if command -v py >/dev/null 2>&1; then
  run_py() { py -3 "$@"; }
elif command -v python >/dev/null 2>&1; then
  run_py() { python "$@"; }
elif command -v python3 >/dev/null 2>&1; then
  run_py() { python3 "$@"; }
else
  echo "python ABSENT" >&2
  exit 1
fi

_fetch() {
  local name="$1" dest="$2"
  if ! aws secretsmanager get-secret-value \
      --region "$REGION" \
      --secret-id "$name" \
      --query SecretString \
      --output text >"$dest" 2>/dev/null; then
    : >"$dest"
    return 1
  fi
  return 0
}

_apply() {
  local name="$1" plain="$2" expected="$3"
  local raw exp
  raw="$(mktemp)"
  exp="$(mktemp)"
  chmod 600 "$raw" "$exp"
  if ! _fetch "$name" "$raw"; then
    echo "$name ABSENT"
    # shellcheck disable=SC2086
    for key in $expected; do
      echo "$key ABSENT"
    done
    rm -f "$raw" "$exp"
    return 0
  fi
  echo "$name PRESENT"
  SM_IN="$raw" SM_OUT="$exp" SM_PLAIN="$plain" SM_DRY="$DRY" SM_EXPECTED="$expected" run_py - <<'PY'
import json, os, sys
raw_path = os.environ["SM_IN"]
out_path = os.environ["SM_OUT"]
plain = os.environ.get("SM_PLAIN") or ""
dry = os.environ.get("SM_DRY") == "1"
expected = [k for k in (os.environ.get("SM_EXPECTED") or "").split() if k]
try:
    raw = open(raw_path, encoding="utf-8").read().strip()
    mapping = {}
    if raw.startswith("{"):
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("not-object")
        for key, val in obj.items():
            if "PASSWORD" in key or "DATABASE_URL" in key:
                print(f"{key} SKIP")
                continue
            mapping[key] = "" if val is None else str(val)
    elif plain and raw:
        mapping[plain] = raw
    elif not raw:
        print("SECRET ABSENT")
    else:
        raise ValueError("not-json")
    seen = set()
    lines = []
    for key in expected:
        seen.add(key)
        val = mapping.get(key, "")
        present = bool(val)
        print(f"{key} {'PRESENT' if present else 'ABSENT'}")
        if present and not dry:
            esc = val.replace("'", "'\"'\"'")
            lines.append(f"export {key}='{esc}'")
    for key, val in mapping.items():
        if key in seen:
            continue
        present = bool(val)
        print(f"{key} {'PRESENT' if present else 'ABSENT'}")
        if present and not dry:
            esc = val.replace("'", "'\"'\"'")
            lines.append(f"export {key}='{esc}'")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
except Exception:
    print("SECRET parse-failed")
    try:
        open(out_path, "w", encoding="utf-8").close()
    except OSError:
        pass
    sys.exit(0)
PY
  # shellcheck disable=SC1090
  if [[ "$DRY" -eq 0 ]]; then
    # shellcheck disable=SC1090
    source "$exp"
  fi
  rm -f "$raw" "$exp"
}

_apply "$N8N_SECRET" "N8N_API_KEY" "N8N_API_KEY"
_apply "$ORCH_SECRET" "" "OPENAI_API_KEY ANTHROPIC_API_KEY ANTHROPIC_WORKSPACE_ID PERPLEXITY_API_KEY LITELLM_API_KEY LITELLM_ENGINEERING_API_KEY LITELLM_RESEARCH_API_KEY LITELLM_BASE_URL"
