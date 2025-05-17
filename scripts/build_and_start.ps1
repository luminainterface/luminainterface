# Lumina System Boot & Build Script (PowerShell) (Updated for Docker Compose V2)
# -----------------------------------------------
# Modular, menu-driven, skips failed builds, and prints a summary of failures.

$ErrorActionPreference = "Stop"

# Ensure we're in the right directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath/..

Write-Host "Starting build script in $(Get-Location)" -ForegroundColor Cyan
Write-Host "Docker Compose file exists: $(Test-Path docker-compose.yml)" -ForegroundColor Cyan

# Verify Docker Compose V2 is installed
try {
    $composeVersion = docker compose version --short
    if (-not ($composeVersion -match "^v2")) {
        Write-Host "Warning: Docker Compose V2 not detected. Current version: $composeVersion" -ForegroundColor Yellow
        Write-Host "Please install Docker Compose V2 for best compatibility" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error checking Docker Compose version: $_" -ForegroundColor Red
    exit 1
}

$ComposeFile = "docker-compose.yml"
$LogDir = "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Define service groups with their corresponding profiles
$Stages = @(
    @{
        Name = "Infrastructure"
        Profile = "infra"
        Services = @("redis", "neo4j", "qdrant", "ollama")
    },
    @{
        Name = "Monitoring"
        Profile = "monitoring"
        Services = @("prometheus", "grafana", "promtail", "loki", "jaeger", "alertmanager")
    },
    @{
        Name = "Core Services"
        Profile = "core"
        Services = @("concept-dict", "concept-analytics", "concept-analyzer", "graph-api", "learning-graph")
    },
    @{
        Name = "Ingestion"
        Profile = "ingestion"
        Services = @("crawler", "event-mux", "batch-embedder", "output-engine")
    },
    @{
        Name = "Chat & LLM"
        Profile = "chat"
        Services = @("dual-chat-router", "masterchat-core", "masterchat-llm", "output-engine")
    },
    @{
        Name = "UI"
        Profile = "ui"
        Services = @("ui")
    }
)

$FailedBuilds = @()
$FailedStarts = @()

function BuildAndStart($Services, $StageName, $Profile) {
    Write-Host "`nBuilding stage: $StageName (Profile: $Profile)" -ForegroundColor Yellow
    Write-Host "Services to build: $($Services -join ', ')" -ForegroundColor Yellow
    
    if ($Services.Count -eq 0) { 
        Write-Host "No services to build in stage: $StageName" -ForegroundColor Yellow
        return 
    }
    
    # First, pull any required images
    Write-Host "Pulling required images for profile: $Profile" -ForegroundColor Cyan
    docker compose -f $ComposeFile --profile $Profile pull
    
    foreach ($svc in $Services) {
        $logFile = Join-Path $LogDir "build_${svc}_output.log"
        Write-Host "`nBuilding service: $svc" -ForegroundColor Cyan
        
        try {
            # Use Docker Compose V2 for build with profile
            $buildOutput = & docker compose -f $ComposeFile --profile $Profile build $svc 2>&1
            $buildOutput | Out-File -FilePath $logFile -Append
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Build failed for service: $svc" -ForegroundColor Red
                Write-Host "See $logFile for details" -ForegroundColor Red
                $FailedBuilds += $svc
                continue
            }
            
            Write-Host "Starting service: $svc" -ForegroundColor Cyan
            # Use Docker Compose V2 for up with profile
            $startOutput = & docker compose -f $ComposeFile --profile $Profile up -d $svc 2>&1
            $startOutput | Out-File -FilePath $logFile -Append
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Start failed for service: $svc" -ForegroundColor Red
                Write-Host "See $logFile for details" -ForegroundColor Red
                $FailedStarts += $svc
            } else {
                Write-Host "Successfully started $svc" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Exception occurred for service $svc : $_" -ForegroundColor Red
            Write-Host "See $logFile for details" -ForegroundColor Red
            $FailedBuilds += $svc
        }
    }
}

function WaitForHealthyContainers($Services, $timeoutSeconds = 120) {
    $startTime = Get-Date
    $unhealthy = $Services
    while ($unhealthy.Count -gt 0) {
        $unhealthy = @()
        foreach ($svc in $Services) {
            $status = docker ps --filter "name=$svc" --format "{{.Names}}:{{.Status}}"
            if ($status -match ":(healthy|Up)") {
                continue
            } else {
                $unhealthy += $svc
            }
        }
        if ($unhealthy.Count -eq 0) { break }
        $elapsed = (Get-Date) - $startTime
        if ($elapsed.TotalSeconds -gt $timeoutSeconds) {
            Write-Host "Timeout waiting for services to become healthy: $($unhealthy -join ', ')" -ForegroundColor Yellow
            break
        }
        Write-Host "Waiting for services to become healthy: $($unhealthy -join ', ')..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
    if ($unhealthy.Count -eq 0) {
        Write-Host "All services in this stage are healthy or running." -ForegroundColor Green
    }
}

function ShowMenu {
    Write-Host "`nAvailable stages:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $Stages.Count; $i++) {
        Write-Host ("  {0}. {1}" -f ($i+1), $Stages[$i].Name) -ForegroundColor White
    }
    Write-Host "`nEnter your choice(s) (comma-separated, Enter for ALL)" -ForegroundColor Cyan
    $choice = Read-Host
    return $choice
}

function Main {
    try {
        Write-Host "`nStarting build process..." -ForegroundColor Cyan
        $choice = ShowMenu
        
        $selected = @()
        if ([string]::IsNullOrWhiteSpace($choice)) {
            Write-Host "No choice provided, selecting all stages" -ForegroundColor Yellow
            $selected = 0..($Stages.Count-1)
        } else {
            Write-Host "Processing user choices: $choice" -ForegroundColor Yellow
            $indices = $choice -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -match "^\d+$" }
            foreach ($idx in $indices) {
                $i = [int]$idx - 1
                if ($i -ge 0 -and $i -lt $Stages.Count) {
                    $selected += $i
                }
            }
        }
        
        Write-Host "Selected stages: $($selected -join ', ')" -ForegroundColor Yellow
        foreach ($i in $selected) {
            $stage = $Stages[$i]
            BuildAndStart $stage.Services $stage.Name $stage.Profile
            WaitForHealthyContainers $stage.Services
            Write-Host "Pausing for 15 seconds to allow services to stabilize..." -ForegroundColor Cyan
            Start-Sleep -Seconds 15
        }
        
        if ($FailedBuilds.Count -gt 0) {
            Write-Host "`nThe following services FAILED TO BUILD and were skipped:" -ForegroundColor Red
            $FailedBuilds | Sort-Object -Unique | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        }
        if ($FailedStarts.Count -gt 0) {
            Write-Host "`nThe following services FAILED TO START:" -ForegroundColor Red
            $FailedStarts | Sort-Object -Unique | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        }
        
        Write-Host "`nBuild process completed. Check the logs directory for detailed output." -ForegroundColor Cyan
    }
    catch {
        Write-Host "An error occurred: $_" -ForegroundColor Red
        Write-Host "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
    }
}

Main
