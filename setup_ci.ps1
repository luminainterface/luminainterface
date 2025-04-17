# Setup script for Lumina CI/CD environment

# Check if Python is installed
$pythonVersion = python --version
if (-not $?) {
    Write-Error "Python is not installed. Please install Python 3.8 or higher."
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run linting
Write-Host "Running linting checks..."
black --check src/
flake8 src/
mypy src/

# Run tests for all versions
$versions = @("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7_5", "v8", "v9", "v10", "v11", "v12")
foreach ($version in $versions) {
    Write-Host "Testing version $version..."
    cd src
    pytest tests/test_${version}_*.py --cov=$version --cov-report=xml
    cd ..
}

# Run integration tests
Write-Host "Running integration tests..."
cd src
pytest tests/test_integration_*.py --cov=integration --cov-report=xml
cd ..

# Run version bridge tests
Write-Host "Running version bridge tests..."
cd src
python version_bridge_integration.py
cd ..

Write-Host "CI/CD setup complete!"
Write-Host "To run the pipeline locally, use: .\run_ci.ps1" 