# Colors for output
$RED = [System.ConsoleColor]::Red
$GREEN = [System.ConsoleColor]::Green
$YELLOW = [System.ConsoleColor]::Yellow

Write-Host "🔍 Lumina Debug Tool" -ForegroundColor $YELLOW
Write-Host "====================="

# Check Docker services
Write-Host "`nChecking Docker services..." -ForegroundColor $YELLOW
docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}" | Select-String -Pattern "lumina|redis|qdrant|prometheus|grafana"

# Check Prometheus targets
Write-Host "`nChecking Prometheus targets..." -ForegroundColor $YELLOW
try {
    $targets = Invoke-RestMethod -Uri "http://localhost:9091/api/v1/targets" -Method Get
    $targets.data.activeTargets | Where-Object { $_.health -eq "down" } | ForEach-Object {
        Write-Host "DOWN: $($_.labels.job) - $($_.lastError)" -ForegroundColor $RED
    }
} catch {
    Write-Host "✗ Failed to check Prometheus targets" -ForegroundColor $RED
}

# Check Redis
Write-Host "`nChecking Redis..." -ForegroundColor $YELLOW
try {
    $redis = docker exec redis redis-cli -p 6379 ping
    if ($redis -eq "PONG") {
        Write-Host "✓ Redis is responding" -ForegroundColor $GREEN
    }
} catch {
    Write-Host "✗ Redis not responding" -ForegroundColor $RED
}

# Check Qdrant
Write-Host "`nChecking Qdrant..." -ForegroundColor $YELLOW
try {
    $qdrant = Invoke-RestMethod -Uri "http://localhost:6335/health" -Method Get
    Write-Host "✓ Qdrant is responding" -ForegroundColor $GREEN
} catch {
    Write-Host "✗ Qdrant not responding" -ForegroundColor $RED
}

# Check service health endpoints
Write-Host "`nChecking service health..." -ForegroundColor $YELLOW
$services = @(
    "graph-api:8201"
    "masterchat:8301"
    "crawler:8401"
    "event-mux:8101"
    "learning-graph:8601"
    "concept-analyzer:8501"
    "action-handler:8701"
    "trend-analyzer:8801"
)

foreach ($service in $services) {
    $name, $port = $service -split ":"
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -Method Get -ErrorAction Stop
        Write-Host "✓ $name" -ForegroundColor $GREEN
    } catch {
        Write-Host "✗ $name" -ForegroundColor $RED
    }
}

# Check Grafana datasource
Write-Host "`nChecking Grafana datasource..." -ForegroundColor $YELLOW
try {
    $datasources = Invoke-RestMethod -Uri "http://localhost:3000/api/datasources" -Method Get -Headers @{
        "Authorization" = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("admin:lumina"))
    }
    $datasources | Where-Object { $_.type -eq "prometheus" } | ForEach-Object {
        Write-Host "Prometheus: $($_.url) - $($_.access)"
    }
} catch {
    Write-Host "✗ Failed to check Grafana datasource" -ForegroundColor $RED
}

# Check Alertmanager silences
Write-Host "`nChecking Alertmanager silences..." -ForegroundColor $YELLOW
try {
    $silences = Invoke-RestMethod -Uri "http://localhost:9093/api/v2/silences" -Method Get
    $silences | ForEach-Object {
        Write-Host "$($_.status) - $($_.comment)"
    }
} catch {
    Write-Host "✗ Failed to check Alertmanager silences" -ForegroundColor $RED
}

# Check memory usage
Write-Host "`nChecking memory usage..." -ForegroundColor $YELLOW
docker stats --no-stream --format "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}"

# Check WebSocket connections
Write-Host "`nChecking WebSocket connections..." -ForegroundColor $YELLOW
try {
    $wsConnections = Get-NetTCPConnection -LocalPort 8101 -State Established | Measure-Object | Select-Object -ExpandProperty Count
    Write-Host "Active WebSocket connections: $wsConnections"
} catch {
    Write-Host "✗ Failed to check WebSocket connections" -ForegroundColor $RED
}

Write-Host "`nDebug complete!" -ForegroundColor $YELLOW 