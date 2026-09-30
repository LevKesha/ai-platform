#!/usr/bin/env bash
# Apply the Olympus Cognito brand shell to the live custom domain.
# Discovers the user pool from the domain. Does not create pools, clients,
# domains, or Terraform resources, and does not embed resource IDs.
#
# Not wired to deploy-olympus. Lev/Platform run this after AWS credentials
# are available. The cognito-alb-auth module in LevKesha/infrastructure still
# declares the amazoncognito.com prefix; Platform reconciles that module.
# This repo only ships the shell assets and this apply path.
#
# AWS does not allow arbitrary hosted-UI or managed-login page text, so
# "The Olympus" and "Sign in to Olympus" are not sent. No logo image is
# uploaded (wordmark art is not invented). Colors only.
set -euo pipefail

DOMAIN="auth.olympus.levkesha.com"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CSS_FILE="${ROOT}/hosted-ui.css"
SETTINGS_FILE="${ROOT}/managed-login-settings.json"

command -v aws >/dev/null 2>&1 || { echo "aws CLI required" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 1; }
[[ -f "$CSS_FILE" && -f "$SETTINGS_FILE" ]] || { echo "brand assets missing" >&2; exit 1; }

DESC="$(aws cognito-idp describe-user-pool-domain --domain "$DOMAIN")"
POOL="$(jq -r '.DomainDescription.UserPoolId // empty' <<<"$DESC")"
VERSION="$(jq -r '.DomainDescription.ManagedLoginVersion // empty' <<<"$DESC")"
if [[ -z "$POOL" || "$POOL" == "null" ]]; then
  echo "No UserPoolId for ${DOMAIN}. Refusing to guess an ID." >&2
  exit 1
fi

echo "domain=${DOMAIN} managed-login-version=${VERSION}"

if [[ "$VERSION" == "1" ]]; then
  CSS="$(cat "$CSS_FILE")"
  aws cognito-idp set-ui-customization \
    --user-pool-id "$POOL" \
    --client-id ALL \
    --css "$CSS"
  echo "Applied classic Hosted UI CSS to ClientId ALL."
  exit 0
fi

if [[ "$VERSION" != "2" ]]; then
  echo "Unexpected ManagedLoginVersion '${VERSION}'. Stop. Do not guess a branding API." >&2
  exit 1
fi

mapfile -t CLIENTS < <(aws cognito-idp list-user-pool-clients \
  --user-pool-id "$POOL" \
  --query 'UserPoolClients[].ClientId' \
  --output text | tr '\t' '\n' | sed '/^$/d')

if [[ ${#CLIENTS[@]} -eq 0 ]]; then
  echo "No app clients on the pool for ${DOMAIN}. Refusing to invent a client ID." >&2
  exit 1
fi

for CLIENT in "${CLIENTS[@]}"; do
  BRAND_ID="$(aws cognito-idp describe-managed-login-branding-by-client \
    --user-pool-id "$POOL" \
    --client-id "$CLIENT" \
    --query 'ManagedLoginBranding.ManagedLoginBrandingId' \
    --output text 2>/dev/null || true)"
  if [[ -z "$BRAND_ID" || "$BRAND_ID" == "None" ]]; then
    aws cognito-idp create-managed-login-branding \
      --user-pool-id "$POOL" \
      --client-id "$CLIENT" \
      --settings "file://${SETTINGS_FILE}"
    echo "Created managed-login style for one app client."
  else
    aws cognito-idp update-managed-login-branding \
      --user-pool-id "$POOL" \
      --managed-login-branding-id "$BRAND_ID" \
      --settings "file://${SETTINGS_FILE}"
    echo "Updated managed-login style for one app client."
  fi
done
