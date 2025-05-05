# PowerShell script for system verification
$ErrorActionPreference = "Stop"

Write-Host "Starting system verification..."

# 1. Pull & boot
Write-Host "Pulling latest images and starting services..."
docker compose pull
docker compose up -d

# 2. Ports -> UI
Write-Host "Aligning ports..."
& "$PSScriptRoot\align_ports.ps1"

# 3. Smoke
Write-Host "Running smoke tests..."
& "$PSScriptRoot\smoke.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke tests failed"
    exit 1
}

# 4. Dashboards
Write-Host "Opening monitoring dashboard..."
Start-Process "http://localhost:3000/d/graph-ops"

Write-Host "System verification complete!" 