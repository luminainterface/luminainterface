# PowerShell script for smoke testing
$ErrorActionPreference = "Stop"

# Configuration
$MAX_RETRIES = 5
$RETRY_DELAY = 10
$API_KEY = if ($env:LUMINA_API_KEY) { $env:LUMINA_API_KEY } else { "test-key" }
$PROMETHEUS_URL = "http://localhost:9091"
$REDIS_URL = "localhost:6381"
$MASTERCHAT_URL = "http://localhost:8301"
$GRAPH_API_URL = "http://localhost:8201"
$EVENT_MUX_URL = "http://localhost:8101"
$CRAWLER_URL = "http://localhost:8401"
$VECTOR_DB_URL = "http://localhost:6335"
$LLM_ENGINE_URL = "http://localhost:11436"

# Load environment variables if .env exists
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

Write-Host "Starting smoke tests..."

# Function to test an endpoint with retries
function Test-Endpoint {
    param (
        [string]$url,
        [string]$name
    )
    
    Write-Host "Testing $name at $url..."
    $retries = 0
    
    while ($retries -lt $MAX_RETRIES) {
        try {
            $response = Invoke-WebRequest -Uri $url -Method Get -ErrorAction Stop -TimeoutSec 10
            Write-Host "[OK] $name is healthy (Status: $($response.StatusCode))"
                return $true
        }
        catch {
            $retries++
            Write-Host "[DEBUG] Error details: $($_.Exception.Message)"
            if ($retries -lt $MAX_RETRIES) {
                Write-Host "[RETRY] Attempt $retries/$MAX_RETRIES for $name"
                Start-Sleep -Seconds $RETRY_DELAY
            }
        }
    }
    
    Write-Host "[ERROR] $name is not responding after $MAX_RETRIES attempts"
    return $false
}

# Function to check Prometheus metrics
function Check-PrometheusMetrics {
    Write-Host "Checking Prometheus metrics..."
    
    # Wait for Prometheus to be ready
    Start-Sleep -Seconds 10
    
    try {
        # Check planner_p95 metric
        $response = Invoke-WebRequest -Uri "$PROMETHEUS_URL/api/v1/query?query=planner_p95" -Method Get -TimeoutSec 10
        $result = $response.Content | ConvertFrom-Json
        $plannerP95 = $result.data.result[0].value[1]
        
        if ($null -eq $plannerP95) {
            Write-Host "[ERROR] planner_p95 metric not found"
            return $false
        }
        Write-Host "[OK] planner_p95 metric found: $plannerP95"
        
        # Check other critical metrics
        $metrics = @("event_processing_latency_seconds", "redis_operations_total", "active_connections")
        foreach ($metric in $metrics) {
            $response = Invoke-WebRequest -Uri "$PROMETHEUS_URL/api/v1/query?query=$metric" -Method Get -TimeoutSec 10
            $result = $response.Content | ConvertFrom-Json
            if ($null -eq $result.data.result[0]) {
                Write-Host "[ERROR] $metric not found"
                return $false
            }
        }
        Write-Host "[OK] All required metrics are present"
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to check Prometheus metrics: $_"
        return $false
    }
}

# Function to check Redis lag
function Check-RedisLag {
    Write-Host "Checking Redis lag..."
    
    # Wait for Redis to be ready
    Start-Sleep -Seconds 5
    
    try {
        # Get Redis stream info
        $streamInfo = docker compose exec redis redis-cli xinfo stream graph_stream 2>$null
        if (-not $streamInfo) {
            Write-Host "[WARN] No graph_stream found, creating test stream..."
            docker compose exec redis redis-cli xadd graph_stream * test test > $null
            $streamInfo = docker compose exec redis redis-cli xinfo stream graph_stream
        }
        
        # Extract consumer lag
        $consumerLag = ($streamInfo | Select-String "pending").ToString().Split()[1]
        if (-not $consumerLag) {
            $consumerLag = "0"
        }
        
        # Check if lag is within acceptable range
        if ([int]$consumerLag -gt 100) {
            Write-Host "[ERROR] Redis lag too high: $consumerLag"
            return $false
        }
        
        Write-Host "[OK] Redis lag is acceptable: $consumerLag"
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to check Redis lag: $_"
        return $false
    }
}

# Function to check comprehensive health
function Check-Health {
    Write-Host "Checking comprehensive health..."
    
    try {
        $response = Invoke-WebRequest -Uri "$MASTERCHAT_URL/health" -Method Get -TimeoutSec 10
        $health = $response.Content | ConvertFrom-Json
        
        # Parse and verify health components
        $components = @("redis", "duckdb", "event_processor", "metrics_collector")
        foreach ($component in $components) {
            if ($health.services.$component -ne "ok") {
                Write-Host "[ERROR] $component health check failed"
                return $false
            }
        }
        
        # Check overall status
        if ($health.status -ne "ok") {
            Write-Host "[ERROR] Overall health status is not ok"
            return $false
        }
        
        Write-Host "[OK] All health checks passed"
        return $true
    }
    catch {
        Write-Host "[ERROR] Health check failed: $_"
        return $false
    }
}

# Function to test Redis
function Test-Redis {
    Write-Host "Testing Redis..."
    $retries = 0
    
    while ($retries -lt $MAX_RETRIES) {
        try {
            $result = docker compose exec redis redis-cli ping
            if ($result -eq "PONG") {
                Write-Host "[OK] Redis is healthy"
                return $true
            }
            throw "Redis did not respond with PONG"
        }
        catch {
            $retries++
            Write-Host "[DEBUG] Error details: $_"
            if ($retries -lt $MAX_RETRIES) {
                Write-Host "[RETRY] Attempt $retries/$MAX_RETRIES for Redis"
                Start-Sleep -Seconds $RETRY_DELAY
            }
        }
    }
    
    Write-Host "[ERROR] Redis is not responding after $MAX_RETRIES attempts"
    return $false
}

# Main test sequence
Write-Host "Checking service health..."
$services = docker compose ps --format "{{.Name}}: {{.State}}"
if (-not $?) {
    Write-Host "[ERROR] Docker Compose services are not running"
    exit 1
}

Write-Host "Found services:"
$services | ForEach-Object { Write-Host "[INFO] $_" }

# Run all checks
$endpoints = @(
    @{Url="$GRAPH_API_URL/health"; Name="Graph API"},
    @{Url="$EVENT_MUX_URL/health"; Name="Event Mux"},
    @{Url="$CRAWLER_URL/health"; Name="Crawler"},
    @{Url="$MASTERCHAT_URL/health"; Name="MasterChat"},
    @{Url="$VECTOR_DB_URL/collections"; Name="Vector DB"},
    @{Url="$LLM_ENGINE_URL/api/tags"; Name="LLM Engine"}
)

foreach ($endpoint in $endpoints) {
    if (-not (Test-Endpoint -url $endpoint.Url -name $endpoint.Name)) {
    exit 1
}
}

# Test Redis separately
if (-not (Test-Redis)) { exit 1 }

# Run enhanced checks
if (-not (Check-PrometheusMetrics)) { exit 1 }
if (-not (Check-RedisLag)) { exit 1 }
if (-not (Check-Health)) { exit 1 }

# Test event processing
Write-Host "Checking event processing..."
try {
$body = @{
    type = "smoke_test"
    data = @{
        test = $true
    }
} | ConvertTo-Json

    Invoke-WebRequest -Uri "$EVENT_MUX_URL/events/test" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10 | Out-Null

# Wait briefly for event processing
    Start-Sleep -Seconds 5

# Verify event was processed
    $response = Invoke-WebRequest -Uri "$EVENT_MUX_URL/events/latest" -Method Get -TimeoutSec 10
    if ($response.Content -match "smoke_test") {
        Write-Host "[OK] Event processing verified"
    }
    else {
        Write-Host "[ERROR] Event processing failed"
        exit 1
    }
}
catch {
    Write-Host "[ERROR] Event processing failed: $_"
    exit 1
}

Write-Host "All smoke tests passed!" 