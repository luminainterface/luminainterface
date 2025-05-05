# Check if Docker is running
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running. Please start Docker Desktop first."
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path .env)) {
    @"
OLLAMA_URL=http://llm-engine:11434
OLLAMA_MODEL=phi2
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://localhost:6379
EMBEDDING_CACHE_SIZE=1000
CRAWL_RATE_LIMIT=5/minute
SUMMARISE_RATE_LIMIT=10/minute
QA_RATE_LIMIT=20/minute
MISTRAL_API_KEY=your_api_key
MISTRAL_MODEL=mistral-medium
LLM_TEMP=0.3
GOAL_FILE=goal_lattice.yml
"@ | Out-File -FilePath .env -Encoding UTF8
    Write-Host "Created .env file. Please update MISTRAL_API_KEY with your actual key."
}

# Install Python dependencies
Write-Host "Installing Python dependencies..."
pip install -e .

# Start Redis if not running
$redisStatus = docker ps -q -f name=redis
if (-not $redisStatus) {
    Write-Host "Starting Redis..."
    docker run -d --name redis -p 6379:6379 redis:latest
}

# Start Qdrant if not running
$qdrantStatus = docker ps -q -f name=qdrant
if (-not $qdrantStatus) {
    Write-Host "Starting Qdrant..."
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
}

# Start Ollama if not running
$ollamaStatus = docker ps -q -f name=ollama
if (-not $ollamaStatus) {
    Write-Host "Starting Ollama..."
    docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
}

# Start MasterChat
Write-Host "Starting MasterChat..."
$env:PYTHONPATH = "."
python run.py

# Wait for services to be ready
Write-Host "Waiting for services to be ready..."
Start-Sleep -Seconds 10

# Test MasterChat health endpoint
$healthResponse = Invoke-RestMethod -Uri "http://localhost:8300/health" -Method Get
Write-Host "MasterChat health status: $($healthResponse.status)"

Write-Host "Services started successfully!"
Write-Host "You can now use MasterChat at http://localhost:8300"
Write-Host "API documentation available at http://localhost:8300/docs" 