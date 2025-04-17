# Run all CI/CD setup and verification steps

# Set execution policy for current session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Clear any existing virtual environment
Write-Host "Cleaning up any existing virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Remove-Item -Recurse -Force venv
}

# Run setup script
Write-Host "`nRunning setup script..." -ForegroundColor Cyan
try {
    .\setup_ci.ps1
    if (-not $?) {
        Write-Host "Setup script failed. Please check the output above." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error running setup script: $_" -ForegroundColor Red
    exit 1
}

# Run verification script
Write-Host "`nRunning verification script..." -ForegroundColor Cyan
try {
    .\verify_setup.ps1
    if (-not $?) {
        Write-Host "Verification failed. Please check the output above." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error running verification script: $_" -ForegroundColor Red
    exit 1
}

# Run CI/CD pipeline
Write-Host "`nRunning CI/CD pipeline..." -ForegroundColor Cyan
try {
    .\run_ci.ps1
    if (-not $?) {
        Write-Host "CI/CD pipeline failed. Please check the output above." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error running CI/CD pipeline: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nAll steps completed successfully!" -ForegroundColor Green
Write-Host "Your CI/CD environment is now set up and ready to use." -ForegroundColor Green 