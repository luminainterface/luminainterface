Write-Host "Lumina Debug Tool"
Write-Host "================"

# Check Docker services
Write-Host "`nChecking Docker services..."
docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}" | Select-String -Pattern "lumina|redis|qdrant|prometheus|grafana"

# Check Redis
Write-Host "`nChecking Redis..."
try { 
    $redis = docker exec nnprojecthome-redis-1 redis-cli -p 6379 ping
    if ($redis -eq "PONG") { Write-Host "Redis OK" }
} catch { Write-Host "Redis not responding" }

# Check services
$services = @(
    @{name="graph-api"; port="8201"},
    @{name="masterchat"; port="8301"},
    @{name="crawler"; port="8401"},
    @{name="event-mux"; port="8101"}
)

Write-Host "`nChecking services..."
foreach ($svc in $services) {
    Write-Host "Checking $($svc.name)..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($svc.port)/health" -ErrorAction Stop
        Write-Host " OK"
    } catch {
        Write-Host " Failed"
    }
}

Write-Host "`nChecking memory usage..."
docker stats --no-stream --format "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}"

Write-Host "`nDebug complete!" 