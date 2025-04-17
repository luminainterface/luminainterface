# Run script for Lumina CI/CD pipeline

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Run linting
Write-Host "Running linting checks..."
black --check src/
if (-not $?) {
    Write-Error "Black linting failed"
    exit 1
}

flake8 src/
if (-not $?) {
    Write-Error "Flake8 linting failed"
    exit 1
}

mypy src/
if (-not $?) {
    Write-Error "Mypy type checking failed"
    exit 1
}

# Run tests for all versions
$versions = @("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7_5", "v8", "v9", "v10", "v11", "v12")
foreach ($version in $versions) {
    Write-Host "Testing version $version..."
    cd src
    pytest tests/test_${version}_*.py --cov=$version --cov-report=xml
    if (-not $?) {
        Write-Error "Tests failed for version $version"
        exit 1
    }
    cd ..
}

# Run integration tests
Write-Host "Running integration tests..."
cd src
pytest tests/test_integration_*.py --cov=integration --cov-report=xml
if (-not $?) {
    Write-Error "Integration tests failed"
    exit 1
}
cd ..

# Run version bridge tests
Write-Host "Running version bridge tests..."
cd src
python version_bridge_integration.py
if (-not $?) {
    Write-Error "Version bridge tests failed"
    exit 1
}
cd ..

Write-Host "All tests passed successfully!"
Write-Host "CI/CD pipeline completed successfully." 