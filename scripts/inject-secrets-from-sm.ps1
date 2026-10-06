# Load dev secrets from AWS Secrets Manager into this process.
# Prints env names and PRESENT/ABSENT only. Never prints values.
#
# Durable SSOT is Secrets Manager (eu-central-1). This script is the
# ephemeral process-env inject. A gitignored .env is not SSOT.
#
#   . .\scripts\inject-secrets-from-sm.ps1
#   .\scripts\inject-secrets-from-sm.ps1 -DryRun
#
# Secrets:
#   dev-cluster-n8n/api-key                         -> N8N_API_KEY
#   dev-cluster-agent-api/local-orchestrator-keys   -> JSON keys as env vars

[CmdletBinding()]
param(
    [string]$Region = "eu-central-1",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-PSDebug -Off

$N8nSecret = "dev-cluster-n8n/api-key"
$OrchestratorSecret = "dev-cluster-agent-api/local-orchestrator-keys"
$OrchestratorKeys = @(
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_WORKSPACE_ID",
    "PERPLEXITY_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_ENGINEERING_API_KEY",
    "LITELLM_RESEARCH_API_KEY",
    "LITELLM_BASE_URL"
)

function Get-SmRaw {
    param([string]$SecretId)
    $err = New-TemporaryFile
    try {
        $out = & aws secretsmanager get-secret-value `
            --region $Region `
            --secret-id $SecretId `
            --query SecretString `
            --output text 2>$err
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return ("$out").Trim()
    } finally {
        Remove-Item -LiteralPath $err -Force -ErrorAction SilentlyContinue
    }
}

function Convert-SecretMap {
    param(
        [string]$Raw,
        [string]$PlainKey
    )
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return $null
    }
    $trim = $Raw.Trim()
    if ($trim.StartsWith("{")) {
        try {
            $obj = $trim | ConvertFrom-Json
        } catch {
            return $null
        }
        $map = @{}
        foreach ($prop in $obj.PSObject.Properties) {
            if ($prop.Name -match "PASSWORD|DATABASE_URL") {
                $script:Skipped += $prop.Name
                continue
            }
            $map[$prop.Name] = [string]$prop.Value
        }
        return ,$map
    }
    if ($PlainKey) {
        return @{ $PlainKey = $trim }
    }
    return $null
}

function Publish-EnvName {
    param(
        [string]$Name,
        [AllowEmptyString()][string]$Value
    )
    $present = -not [string]::IsNullOrWhiteSpace($Value)
    if ($present -and -not $DryRun) {
        Set-Item -Path "Env:$Name" -Value $Value
    }
    if ($present) {
        Write-Output "$Name PRESENT"
    } else {
        Write-Output "$Name ABSENT"
    }
}

function Publish-Secret {
    param(
        [string]$SecretId,
        [string]$PlainKey,
        [string[]]$ExpectedKeys
    )
    $raw = Get-SmRaw -SecretId $SecretId
    if ($null -eq $raw) {
        Write-Output "$SecretId ABSENT"
        foreach ($key in $ExpectedKeys) {
            Write-Output "$key ABSENT"
        }
        return
    }
    Write-Output "$SecretId PRESENT"
    $script:Skipped = @()
    $map = Convert-SecretMap -Raw $raw -PlainKey $PlainKey
    foreach ($skipped in $script:Skipped) {
        Write-Output "$skipped SKIP"
    }
    if ($null -eq $map) {
        Write-Output "$SecretId parse-failed"
        foreach ($key in $ExpectedKeys) {
            Write-Output "$key ABSENT"
        }
        return
    }
    $seen = @{}
    foreach ($key in $ExpectedKeys) {
        $seen[$key] = $true
        $val = $null
        if ($map.ContainsKey($key)) { $val = $map[$key] }
        Publish-EnvName -Name $key -Value $val
    }
    foreach ($key in @($map.Keys)) {
        if ($seen.ContainsKey($key)) { continue }
        Publish-EnvName -Name $key -Value $map[$key]
    }
}

Publish-Secret -SecretId $N8nSecret -PlainKey "N8N_API_KEY" -ExpectedKeys @("N8N_API_KEY")
Publish-Secret -SecretId $OrchestratorSecret -PlainKey "" -ExpectedKeys $OrchestratorKeys
