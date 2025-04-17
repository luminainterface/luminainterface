Write-Host "Starting V6 Portal of Contradiction with Language Module..." -ForegroundColor Cyan

# Create necessary directories if they don't exist
if (-not (Test-Path -Path "data/memory/language_memory")) {
    New-Item -Path "data/memory/language_memory" -ItemType Directory -Force
    Write-Host "Created language memory directory" -ForegroundColor Green
}

if (-not (Test-Path -Path "data/neural_linguistic")) {
    New-Item -Path "data/neural_linguistic" -ItemType Directory -Force
    Write-Host "Created neural linguistic directory" -ForegroundColor Green
}

if (-not (Test-Path -Path "data/v10")) {
    New-Item -Path "data/v10" -ItemType Directory -Force
    Write-Host "Created consciousness mirror directory" -ForegroundColor Green
}

if (-not (Test-Path -Path "data/central_language")) {
    New-Item -Path "data/central_language" -ItemType Directory -Force
    Write-Host "Created central language node directory" -ForegroundColor Green
}

# Start the V6 Portal
python src/v6/launcher.py

# Pause to see any error messages
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 