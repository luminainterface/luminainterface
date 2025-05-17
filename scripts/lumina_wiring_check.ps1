# Lumina System Wiring & Integration Automated Check (PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File scripts\lumina_wiring_check.ps1

$ports = @(8140, 8000, 8401, 8311, 8301, 8906, 8501, 8201)
$summary = @()

function Pass($msg) { Write-Host "`u2714 $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "`u274C $msg" -ForegroundColor Red }

# 1. Service Health & Metrics
foreach ($port in $ports) {
    try {
        Invoke-WebRequest -Uri "http://localhost:$port/health" -UseBasicParsing -TimeoutSec 3 | Out-Null
        Pass "/health on :$port"
    } catch {
        Fail "/health on :$port"; $summary += "/health fail on :$port"
    }
    try {
        Invoke-WebRequest -Uri "http://localhost:$port/metrics" -UseBasicParsing -TimeoutSec 3 | Out-Null
        Pass "/metrics on :$port"
    } catch {
        Fail "/metrics on :$port"; $summary += "/metrics fail on :$port"
    }
}

# 2. Prometheus Targets
try {
    $prom = Invoke-WebRequest -Uri "http://localhost:9090/targets" -UseBasicParsing -TimeoutSec 3
    if ($prom.Content -match 'UP') {
        Pass "Prometheus targets show services as UP"
    } else {
        Fail "Prometheus targets not all UP"; $summary += "Prometheus targets not all UP"
    }
} catch {
    Fail "Prometheus not reachable"; $summary += "Prometheus not reachable"
}

# 3. Grafana Dashboards
try {
    $graf = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 3
    if ($graf.Content -match 'database') {
        Pass "Grafana is running"
    } else {
        Fail "Grafana not reachable"; $summary += "Grafana not reachable"
    }
} catch {
    Fail "Grafana not reachable"; $summary += "Grafana not reachable"
}

# 4. Loki Logs
try {
    $loki = Invoke-WebRequest -Uri "http://localhost:3100/ready" -UseBasicParsing -TimeoutSec 3
    if ($loki.Content -match 'ready') {
        Pass "Loki is running"
    } else {
        Fail "Loki not reachable"; $summary += "Loki not reachable"
    }
} catch {
    Fail "Loki not reachable"; $summary += "Loki not reachable"
}

# 5. Jaeger Tracing
try {
    $jaeger = Invoke-WebRequest -Uri "http://localhost:16686" -UseBasicParsing -TimeoutSec 3
    if ($jaeger.Content -match '<title>Jaeger UI') {
        Pass "Jaeger UI is running"
    } else {
        Fail "Jaeger not reachable"; $summary += "Jaeger not reachable"
    }
} catch {
    Fail "Jaeger not reachable"; $summary += "Jaeger not reachable"
}

# 6. Redis Pub/Sub Channel (requires redis-cli in PATH)
if (Get-Command redis-cli -ErrorAction SilentlyContinue) {
    $channels = & redis-cli --raw PUBSUB CHANNELS
    if ($channels -match 'crawl_request') {
        Pass "Redis pub/sub channel 'crawl_request' is active"
    } else {
        Fail "Redis pub/sub channel 'crawl_request' not found"; $summary += "Redis pub/sub channel missing"
    }
} else {
    Fail "redis-cli not installed, skipping Redis check"; $summary += "redis-cli missing"
}

# 7. Print Summary
if ($summary.Count -eq 0) {
    Write-Host "`nAll checks passed!" -ForegroundColor Green
} else {
    Write-Host "`nSome checks failed:" -ForegroundColor Red
    foreach ($item in $summary) { Write-Host "- $item" }
} 
 