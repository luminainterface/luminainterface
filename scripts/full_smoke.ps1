# PowerShell smoke test for Lumina
$ErrorActionPreference = "Stop"

# Configurable ports (match docker-compose)
$GRAPH_API = if ($env:GRAPH_API) { $env:GRAPH_API } else { "http://localhost:8201" }
$MASTERCHAT = if ($env:MASTERCHAT) { $env:MASTERCHAT } else { "http://localhost:8301" }
$EVENT_MUX_WS = if ($env:EVENT_MUX_WS) { $env:EVENT_MUX_WS } else { "ws://localhost:8101/ws" }
$CRAWLER_METRICS = if ($env:CRAWLER_METRICS) { $env:CRAWLER_METRICS } else { "http://localhost:8401/metrics" }

Write-Host ">> Smoke test: Lumina end-to-end" -ForegroundColor Green

# 1. enqueue a tiny crawl task
Write-Host "  * enqueue crawl... " -NoNewline
$body = @{
    crawl = @("Ping")
    hops = 0
    max_nodes = 5
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "$MASTERCHAT/tasks" -Method Post -Body $body -ContentType "application/json" | Out-Null
    Write-Host "done"
} catch {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# 2. wait for a node.add event via event-mux logs
Write-Host "  * wait for node.add event " -NoNewline
$found = $false
for ($i = 1; $i -le 12; $i++) {
    $logs = docker compose logs --tail=20 event-mux
    if ($logs -match "node.add") {
        Write-Host "OK" -ForegroundColor Green
        $found = $true
        break
    }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 5
}

if (-not $found) {
    Write-Host "TIMEOUT" -ForegroundColor Red
    exit 1
}

# 3. check metrics endpoints
Write-Host "  * crawler /metrics reachable " -NoNewline
try {
    $metrics = Invoke-RestMethod -Uri $CRAWLER_METRICS
    if ($metrics -match "crawler_pages_total") {
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host "  * graph-api /metrics.summary reachable " -NoNewline
try {
    $summary = Invoke-RestMethod -Uri "$GRAPH_API/metrics/summary" | ConvertFrom-Json
    if ($summary.nodes) {
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# 4. health endpoints
foreach ($svc in @("graph-api", "masterchat")) {
    $url = if ($svc -eq "graph-api") { $GRAPH_API } else { $MASTERCHAT }
    Write-Host "  * $svc health " -NoNewline
    try {
        Invoke-RestMethod -Uri "$url/health" | Out-Null
        Write-Host "OK" -ForegroundColor Green
    } catch {
        Write-Host "FAIL" -ForegroundColor Red
        Write-Host $_.Exception.Message
        exit 1
    }
}

Write-Host "[OK] Smoke test passed - pipeline alive" -ForegroundColor Green 