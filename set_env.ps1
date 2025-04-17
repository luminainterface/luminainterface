# Set environment variables
$env:LUMINA_API_KEY = "test-api-key"
$env:LUMINA_MODEL_PATH = Join-Path $PSScriptRoot "models"
$env:PYTHONPATH = "$PSScriptRoot;$PSScriptRoot\src"

# Create required directories if they don't exist
$directories = @("logs", "data", "models", "src/v5", "src/v6", "src/v7", "src/v7_5", "src/v8", "src/v9", "src/v10")
foreach ($dir in $directories) {
    $path = Join-Path $PSScriptRoot $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force
    }
}

Write-Host "Environment variables set:"
Write-Host "LUMINA_API_KEY: $env:LUMINA_API_KEY"
Write-Host "LUMINA_MODEL_PATH: $env:LUMINA_MODEL_PATH"
Write-Host "PYTHONPATH: $env:PYTHONPATH" 
$env:LUMINA_API_KEY = "test-api-key"
$env:LUMINA_MODEL_PATH = Join-Path $PSScriptRoot "models"
$env:PYTHONPATH = "$PSScriptRoot;$PSScriptRoot\src"

# Create required directories if they don't exist
$directories = @("logs", "data", "models", "src/v5", "src/v6", "src/v7", "src/v7_5", "src/v8", "src/v9", "src/v10")
foreach ($dir in $directories) {
    $path = Join-Path $PSScriptRoot $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force
    }
}

Write-Host "Environment variables set:"
Write-Host "LUMINA_API_KEY: $env:LUMINA_API_KEY"
Write-Host "LUMINA_MODEL_PATH: $env:LUMINA_MODEL_PATH"
Write-Host "PYTHONPATH: $env:PYTHONPATH" 