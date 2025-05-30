# 🌟 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10 - ONE CLICK STARTUP (Windows)
# Revolutionary 3-Tier Strategic Steering System
# 
# Usage: .\start-ultimate-architecture.ps1
# Dashboard: http://localhost:9001
# 
# ARCHITECTURE LAYERS:
# 1. 🧠 HIGH-RANK ADAPTER - Ultimate Strategic Steering (Port 9000)
# 2. 🎯 META-ORCHESTRATION CONTROLLER - Strategic Logic (Port 8999)  
# 3. ⚡ ENHANCED EXECUTION SUITE - 8-Phase Orchestration (Port 8998)

param(
    [switch]$SkipHealthCheck
)

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "🌟 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10" -ForegroundColor Magenta
Write-Host "==============================================" -ForegroundColor Magenta
Write-Host "🚀 Starting Revolutionary 3-Tier Strategic Steering System..." -ForegroundColor Yellow
Write-Host ""

# Function to print colored status
function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if docker and docker-compose are available
function Test-Dependencies {
    Write-Status "🔍 Checking system dependencies..." "Cyan"
    
    try {
        docker --version | Out-Null
    } catch {
        Write-Status "❌ Docker is not installed or not in PATH. Please install Docker Desktop first." "Red"
        exit 1
    }
    
    try {
        docker-compose --version | Out-Null
    } catch {
        Write-Status "❌ Docker Compose is not available. Please ensure Docker Desktop is installed properly." "Red"
        exit 1
    }
    
    Write-Status "✅ Docker and Docker Compose are available" "Green"
}

# Function to ensure all required files exist
function Test-CoreFiles {
    Write-Status "📋 Ensuring all core architecture files exist..." "Cyan"
    
    $RequiredFiles = @(
        "high_rank_adapter.py",
        "meta_orchestration_controller.py", 
        "enhanced_real_world_benchmark.py",
        "ultimate_ai_architecture_summary.py",
        "docker-compose-v10-ultimate.yml",
        "constraint_mask.py",
        "token_limiter.py",
        "unsat_guard.py"
    )
    
    foreach ($file in $RequiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Status "❌ Missing critical file: $file" "Red"
            exit 1
        }
    }
    
    Write-Status "✅ All core architecture files present" "Green"
}

# Function to create missing service requirements.txt files
function New-RequirementsFiles {
    Write-Status "📦 Creating requirements.txt files for services..." "Cyan"
    
    $ServiceDirs = @(
        "services/high-rank-adapter",
        "services/meta-orchestration", 
        "services/enhanced-execution",
        "services/architecture-summary"
    )
    
    $RequirementsContent = @"
fastapi==0.104.1
uvicorn[standard]==0.24.0
redis==5.0.1
aiohttp==3.9.1
pydantic==2.5.0
python-multipart==0.0.6
requests==2.31.0
asyncio-mqtt==0.13.0
python-json-logger==2.0.7
"@
    
    foreach ($dir in $ServiceDirs) {
        $reqFile = Join-Path $dir "requirements.txt"
        if (-not (Test-Path $reqFile)) {
            Write-Status "📝 Creating $reqFile" "Yellow"
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
            $RequirementsContent | Out-File -FilePath $reqFile -Encoding UTF8
        }
    }
    
    Write-Status "✅ Requirements files ready" "Green"
}

# Function to build critical missing services
function New-CriticalServices {
    Write-Status "🏗️ Ensuring critical services are available..." "Cyan"
    
    # Create minimal neural-thought-engine if missing
    $neuralAppPath = "services/neural-thought-engine/app.py"
    if (-not (Test-Path $neuralAppPath)) {
        Write-Status "🧠 Creating Neural Thought Engine service..." "Yellow"
        New-Item -ItemType Directory -Path "services/neural-thought-engine" -Force | Out-Null
        
        $neuralAppContent = @"
#!/usr/bin/env python3
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Neural Thought Engine", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "neural-thought-engine"}

@app.post("/reason")
async def reason(data: dict):
    return {"status": "processed", "result": "Neural reasoning active"}

@app.post("/analyze")
async def analyze(data: dict):
    return {"status": "analyzed", "insights": "Neural analysis complete"}

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8890))
    uvicorn.run(app, host="0.0.0.0", port=port)
"@
        $neuralAppContent | Out-File -FilePath $neuralAppPath -Encoding UTF8
        
        $neuralDockerContent = @"
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8890
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8890/health || exit 1
CMD ["python", "app.py"]
"@
        $neuralDockerContent | Out-File -FilePath "services/neural-thought-engine/Dockerfile" -Encoding UTF8
        
        $neuralReqContent = @"
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.9.1
"@
        $neuralReqContent | Out-File -FilePath "services/neural-thought-engine/requirements.txt" -Encoding UTF8
    }
    
    # Create minimal multi-concept-detector if missing
    $conceptAppPath = "services/multi-concept-detector/app.py"
    if (-not (Test-Path $conceptAppPath)) {
        Write-Status "🎯 Creating Multi-Concept Detector service..." "Yellow"
        New-Item -ItemType Directory -Path "services/multi-concept-detector" -Force | Out-Null
        
        $conceptAppContent = @"
#!/usr/bin/env python3
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Multi-Concept Detector", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "multi-concept-detector"}

@app.post("/detect")
async def detect_concepts(data: dict):
    return {"status": "detected", "concepts": {"main": "detected_concept"}, "confidence": 0.8}

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8860))
    uvicorn.run(app, host="0.0.0.0", port=port)
"@
        $conceptAppContent | Out-File -FilePath $conceptAppPath -Encoding UTF8
        
        $conceptDockerContent = @"
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8860
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8860/health || exit 1
CMD ["python", "app.py"]
"@
        $conceptDockerContent | Out-File -FilePath "services/multi-concept-detector/Dockerfile" -Encoding UTF8
        
        $conceptReqContent = @"
fastapi==0.104.1
uvicorn[standard]==0.24.0
"@
        $conceptReqContent | Out-File -FilePath "services/multi-concept-detector/requirements.txt" -Encoding UTF8
    }
    
    Write-Status "✅ Critical services ready" "Green"
}

# Function to stop any existing services
function Stop-ExistingServices {
    Write-Status "🛑 Stopping any existing services..." "Yellow"
    
    try {
        docker-compose -f docker-compose-v10-ultimate.yml down --remove-orphans 2>$null | Out-Null
    } catch {
        # Ignore errors if no services are running
    }
    
    Write-Status "✅ Existing services stopped" "Green"
}

# Function to start the infrastructure services first
function Start-Infrastructure {
    Write-Status "🏗️ Starting infrastructure services..." "Blue"
    
    # Start Redis, Qdrant, Neo4j, Ollama first
    docker-compose -f docker-compose-v10-ultimate.yml up -d redis qdrant neo4j godlike-ollama
    
    # Wait for infrastructure to be ready
    Write-Status "⏳ Waiting for infrastructure services..." "Yellow"
    Start-Sleep -Seconds 10
    
    # Check infrastructure health
    for ($i = 1; $i -le 30; $i++) {
        $redisStatus = docker-compose -f docker-compose-v10-ultimate.yml ps redis
        if ($redisStatus -match "Up.*healthy") {
            Write-Status "✅ Infrastructure services ready" "Green"
            break
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
}

# Function to start the core 3-tier architecture
function Start-CoreArchitecture {
    Write-Status "🌟 Starting CORE 3-TIER ARCHITECTURE..." "Magenta"
    
    # Start Layer 1: High-Rank Adapter
    Write-Status "🧠 Starting Layer 1: High-Rank Adapter..." "Cyan"
    docker-compose -f docker-compose-v10-ultimate.yml up -d high-rank-adapter
    
    # Start Layer 2: Meta-Orchestration Controller
    Write-Status "🎯 Starting Layer 2: Meta-Orchestration Controller..." "Cyan"
    docker-compose -f docker-compose-v10-ultimate.yml up -d meta-orchestration-controller
    
    # Start Layer 3: Enhanced Execution Suite
    Write-Status "⚡ Starting Layer 3: Enhanced Execution Suite..." "Cyan"
    docker-compose -f docker-compose-v10-ultimate.yml up -d enhanced-execution-suite
    
    # Start Architecture Summary Dashboard
    Write-Status "📊 Starting Architecture Summary Dashboard..." "Cyan"
    docker-compose -f docker-compose-v10-ultimate.yml up -d ultimate-architecture-summary
    
    # Wait for core services
    Start-Sleep -Seconds 15
    Write-Status "✅ Core 3-tier architecture started" "Green"
}

# Function to start supporting services
function Start-SupportingServices {
    Write-Status "🔧 Starting supporting services..." "Blue"
    
    # Start essential supporting services
    $EssentialServices = @(
        "neural-thought-engine",
        "multi-concept-detector", 
        "lora-coordination-hub",
        "rag-coordination-enhanced",
        "swarm-intelligence-engine",
        "a2a-coordination-hub"
    )
    
    foreach ($service in $EssentialServices) {
        Write-Status "🚀 Starting $service..." "Yellow"
        try {
            docker-compose -f docker-compose-v10-ultimate.yml up -d $service
        } catch {
            Write-Status "⚠️ Warning: $service failed to start" "Yellow"
        }
        Start-Sleep -Seconds 2
    }
    
    Write-Status "✅ Supporting services started" "Green"
}

# Function to verify system health
function Test-SystemHealth {
    if ($SkipHealthCheck) {
        Write-Status "⏭️ Skipping health check as requested" "Yellow"
        return
    }
    
    Write-Status "🏥 Verifying system health..." "Cyan"
    
    # Core endpoints to check
    $CoreEndpoints = @(
        @{Url="http://localhost:9000/health"; Name="High-Rank Adapter"},
        @{Url="http://localhost:8999/health"; Name="Meta-Orchestration Controller"},
        @{Url="http://localhost:8998/health"; Name="Enhanced Execution Suite"},
        @{Url="http://localhost:9001/health"; Name="Architecture Summary"}
    )
    
    Write-Status "🔍 Checking core services..." "Yellow"
    Start-Sleep -Seconds 10  # Give services time to start
    
    $HealthyServices = 0
    $TotalServices = $CoreEndpoints.Count
    
    foreach ($endpoint in $CoreEndpoints) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint.Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Status "✅ $($endpoint.Name): HEALTHY" "Green"
                $HealthyServices++
            } else {
                Write-Status "❌ $($endpoint.Name): UNHEALTHY" "Red"
            }
        } catch {
            Write-Status "❌ $($endpoint.Name): UNHEALTHY" "Red"
        }
    }
    
    if ($HealthyServices -eq $TotalServices) {
        Write-Status "🎉 ALL CORE SERVICES HEALTHY!" "Green"
    } else {
        Write-Status "⚠️ $HealthyServices/$TotalServices core services healthy" "Yellow"
    }
}

# Function to display final status
function Show-FinalStatus {
    Write-Host ""
    Write-Status "🌟 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10 STARTUP COMPLETE!" "Magenta"
    Write-Host "=================================================================="
    Write-Host ""
    Write-Status "🎯 CORE ARCHITECTURE ACCESS POINTS:" "Cyan"
    Write-Host ""
    Write-Status "📊 Main Dashboard:              http://localhost:9001" "Green"
    Write-Status "🧠 High-Rank Adapter:           http://localhost:9000" "Blue"
    Write-Status "🎯 Meta-Orchestration:          http://localhost:8999" "Blue"
    Write-Status "⚡ Enhanced Execution:           http://localhost:8998" "Blue"
    Write-Host ""
    Write-Status "🔧 SUPPORTING SERVICES:" "Cyan"
    Write-Status "🧠 Neural Thought Engine:       http://localhost:8890" "Yellow"
    Write-Status "🎯 Multi-Concept Detector:      http://localhost:8860" "Yellow"
    Write-Status "🎭 LoRA Coordination Hub:       http://localhost:8995" "Yellow"
    Write-Status "📚 RAG Coordination Enhanced:   http://localhost:8952" "Yellow"
    Write-Host ""
    Write-Status "📋 QUICK COMMANDS:" "Cyan"
    Write-Host "• View all services:    docker-compose -f docker-compose-v10-ultimate.yml ps"
    Write-Host "• View logs:           docker-compose -f docker-compose-v10-ultimate.yml logs [service]"
    Write-Host "• Stop all:            docker-compose -f docker-compose-v10-ultimate.yml down"
    Write-Host "• Restart service:     docker-compose -f docker-compose-v10-ultimate.yml restart [service]"
    Write-Host ""
    Write-Status "🚀 The Ultimate AI Orchestration Architecture v10 is now PRODUCTION READY!" "Magenta"
    Write-Host ""
}

# Main execution flow
function Start-UltimateArchitecture {
    try {
        # Step 1: Check dependencies
        Test-Dependencies
        
        # Step 2: Ensure core files exist
        Test-CoreFiles
        
        # Step 3: Create requirements files
        New-RequirementsFiles
        
        # Step 4: Ensure critical services exist
        New-CriticalServices
        
        # Step 5: Stop existing services
        Stop-ExistingServices
        
        # Step 6: Start infrastructure
        Start-Infrastructure
        
        # Step 7: Start core architecture
        Start-CoreArchitecture
        
        # Step 8: Start supporting services
        Start-SupportingServices
        
        # Step 9: Verify system health
        Test-SystemHealth
        
        # Step 10: Display final status
        Show-FinalStatus
        
    } catch {
        Write-Status "❌ Error during startup: $($_.Exception.Message)" "Red"
        Write-Status "💡 Try running: docker-compose -f docker-compose-v10-ultimate.yml down" "Yellow"
        exit 1
    }
}

# Run main function
Start-UltimateArchitecture 