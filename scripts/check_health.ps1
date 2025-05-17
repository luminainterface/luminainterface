# PowerShell script for checking service health
$ErrorActionPreference = "Stop"

$services = @{
    "graph-api" = "8201"
    "event-mux" = "8101"
    "masterchat" = "8301"
    "crawler" = "8401"
}

Write-Host "🏥 Checking service health..."

foreach ($svc in $services.GetEnumerator()) {
    $name = $svc.Key
    $port = $svc.Value
    
    Write-Host "Checking $name... " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ OK" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed (Status: $($response.StatusCode))" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Failed (Error: $($_.Exception.Message))" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n✨ All services are healthy!" 