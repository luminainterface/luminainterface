# Verification script for Lumina CI/CD setup

Write-Host "Verifying Lumina CI/CD setup..." -ForegroundColor Cyan

# Check virtual environment
Write-Host "`nChecking virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment not found" -ForegroundColor Red
    exit 1
}

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version
if ($?) {
    Write-Host "✓ Python version: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found" -ForegroundColor Red
    exit 1
}

# Check installed packages
Write-Host "`nChecking installed packages..." -ForegroundColor Yellow
$requiredPackages = @("pytest", "black", "flake8", "mypy", "pytest-cov")
foreach ($package in $requiredPackages) {
    $installed = pip show $package
    if ($?) {
        Write-Host "✓ $package is installed" -ForegroundColor Green
    } else {
        Write-Host "✗ $package is not installed" -ForegroundColor Red
        exit 1
    }
}

# Check test files
Write-Host "`nChecking test files..." -ForegroundColor Yellow
$versions = @("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7_5", "v8", "v9", "v10", "v11", "v12")
foreach ($version in $versions) {
    $testFile = "src/tests/test_${version}_*.py"
    if (Test-Path $testFile) {
        Write-Host "✓ Test files exist for $version" -ForegroundColor Green
    } else {
        Write-Host "✗ Test files not found for $version" -ForegroundColor Red
    }
}

# Check coverage files
Write-Host "`nChecking coverage files..." -ForegroundColor Yellow
if (Test-Path "src/coverage.xml") {
    Write-Host "✓ Coverage file exists" -ForegroundColor Green
} else {
    Write-Host "✗ Coverage file not found" -ForegroundColor Red
}

# Check version bridge
Write-Host "`nChecking version bridge..." -ForegroundColor Yellow
if (Test-Path "src/version_bridge_integration.py") {
    Write-Host "✓ Version bridge file exists" -ForegroundColor Green
} else {
    Write-Host "✗ Version bridge file not found" -ForegroundColor Red
}

# Final status
Write-Host "`nSetup Verification Complete!" -ForegroundColor Cyan
Write-Host "If you see any red '✗' marks, please run the setup script again."
Write-Host "To run the CI/CD pipeline, use: .\run_ci.ps1" 