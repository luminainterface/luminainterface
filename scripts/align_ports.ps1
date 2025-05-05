# PowerShell script for port alignment
$ErrorActionPreference = "Stop"

# Configuration
$UI_ENV_FILE = Join-Path -Path $PSScriptRoot -ChildPath ".." | Join-Path -ChildPath "ui" | Join-Path -ChildPath ".env"
$COMPOSE_FILE = Join-Path -Path $PSScriptRoot -ChildPath ".." | Join-Path -ChildPath "docker-compose.yml"
$BACKUP_SUFFIX = ".bak"

Write-Host "Starting port alignment check..."

# Create backup of UI env file if it exists
if (Test-Path $UI_ENV_FILE) {
    Copy-Item $UI_ENV_FILE "${UI_ENV_FILE}${BACKUP_SUFFIX}"
}

# Create UI directory if it doesn't exist
$uiDir = Split-Path -Parent $UI_ENV_FILE
if (-not (Test-Path $uiDir)) {
    New-Item -ItemType Directory -Path $uiDir -Force | Out-Null
}

# Extract ports from docker-compose.yml
Write-Host "Reading service ports from docker-compose.yml..."
if (-not (Test-Path $COMPOSE_FILE)) {
    Write-Host "[ERROR] docker-compose.yml not found at: $COMPOSE_FILE"
    exit 1
}

$composeContent = Get-Content $COMPOSE_FILE -Raw

$API_PORT = if ($composeContent -match 'hub-api:[\s\S]*?(\d+):8000') { $matches[1] } else { "8000" }
$REDIS_PORT = if ($composeContent -match 'redis:[\s\S]*?(\d+):6379') { $matches[1] } else { "6379" }
$VECTOR_PORT = if ($composeContent -match 'vector-db:[\s\S]*?(\d+):6333') { $matches[1] } else { "6333" }
$LLM_PORT = if ($composeContent -match 'llm-engine:[\s\S]*?(\d+):11434') { $matches[1] } else { "11434" }

# Update or create UI environment file
Write-Host "Updating UI environment variables..."
@"
NEXT_PUBLIC_API_URL=http://localhost:${API_PORT}
NEXT_PUBLIC_REDIS_URL=redis://localhost:${REDIS_PORT}
NEXT_PUBLIC_VECTOR_URL=http://localhost:${VECTOR_PORT}
NEXT_PUBLIC_LLM_URL=http://localhost:${LLM_PORT}
"@ | Set-Content $UI_ENV_FILE

Write-Host "Port alignment complete!"
Write-Host "Environment file updated at: $UI_ENV_FILE"
Write-Host "Backup created at: ${UI_ENV_FILE}${BACKUP_SUFFIX}"

# Verify the update
Write-Host "Current port configuration:"
Get-Content $UI_ENV_FILE 