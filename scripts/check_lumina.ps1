# Colors for output
$RED = [System.ConsoleColor]::Red
$GREEN = [System.ConsoleColor]::Green
$YELLOW = [System.ConsoleColor]::Yellow

Write-Host "🔍 Lumina Debug Tool" -ForegroundColor $YELLOW
Write-Host "====================="

# Check Docker services
Write-Host "`nChecking Docker services..." -ForegroundColor $YELLOW
docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}" | Select-String -Pattern "lumina|redis|qdrant|prometheus|grafana"

# Check Redis
Write-Host "`nChecking Redis..." -ForegroundColor $YELLOW
try {
    $redis = docker exec nnprojecthome-redis-1 redis-cli -p 6379 ping
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
    @{name="graph-api"; port=8201},
    @{name="masterchat"; port=8301},
    @{name="crawler"; port=8401},
    @{name="event-mux"; port=8101},
    @{name="learning-graph"; port=8601},
    @{name="concept-analyzer"; port=8501},
    @{name="action-handler"; port=8701},
    @{name="trend-analyzer"; port=8801}
)

foreach ($service in $services) {
    Write-Host "Checking $($service.name)..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.port)/health" -Method Get -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host " OK" -ForegroundColor $GREEN
        } else {
            Write-Host " Failed (Status: $($response.StatusCode))" -ForegroundColor $RED
        }
    } catch {
        Write-Host " Not responding" -ForegroundColor $RED
    }
}

# Check memory usage
Write-Host "`nChecking memory usage..." -ForegroundColor $YELLOW
docker stats --no-stream --format "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}"

Write-Host "`nDebug complete!" -ForegroundColor $YELLOW 