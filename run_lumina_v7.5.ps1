Write-Host ""
Write-Host "======================================="
Write-Host "    LUMINA v7.5 Integrated Launcher    "
Write-Host "======================================="
Write-Host ""

# Ensure necessary directories exist
$directories = @(
    "data", "data\neural", "data\memory", "data\consciousness", "data\knowledge",
    "data\v7", "data\v7.5", "data\autowiki", "data\onsite_memory", "data\conversations",
    "data\breath", "logs", "logs\neural", "logs\memory", "logs\consciousness",
    "logs\knowledge", "logs\v7", "logs\v7.5", "logs\system", "logs\database", "logs\autowiki"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# Load API key and LLM parameters from .env file
if (Test-Path ".env") {
    Write-Host "Loading API key from .env file..."
    $envContent = Get-Content ".env"
    
    $apiKeyLine = $envContent | Where-Object { $_ -match "MISTRAL_API_KEY=(.+)" }
    if ($apiKeyLine) {
        $env:MISTRAL_API_KEY = $matches[1]
        $maskedKey = $env:MISTRAL_API_KEY.Substring(0, 4) + "..." + $env:MISTRAL_API_KEY.Substring($env:MISTRAL_API_KEY.Length - 4)
        Write-Host "Using Mistral API Key: $maskedKey"
    }
    
    Write-Host "LLM Parameters:"
    $llmParams = $envContent | Where-Object { $_ -match "LLM_" }
    foreach ($param in $llmParams) {
        Write-Host "  $param"
        if ($param -match "(.+)=(.+)") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

# Set environment variables
$env:PYTHONPATH = Get-Location
$env:LUMINA_HOME = Get-Location
$env:LUMINA_DATA_DIR = Join-Path (Get-Location) "data"
$env:LUMINA_LOG_DIR = Join-Path (Get-Location) "logs"
$env:LUMINA_PORT = "7500"
$env:LUMINA_GUI_FRAMEWORK = "PySide6"
$env:LUMINA_ENABLE_AUTOWIKI = "true"
$env:LUMINA_ENABLE_DREAMMODE = "true"
$env:LUMINA_ENABLE_NEURAL_SEED = "true"
$env:LUMINA_HOLOGRAPHIC_PORT = "7505"
$env:LUMINA_CHAT_PORT = "7510"
$env:LUMINA_DATABASE_PORT = "7515"
$env:LUMINA_MONITOR_PORT = "7520"
$env:LUMINA_AUTOWIKI_PORT = "7525"

# Copy conversation_flow.py if needed
if (Test-Path "src\v7.5\conversation_flow.py") {
    if (-not (Test-Path "src\v7_5\conversation_flow.py")) {
        Write-Host "Copying conversation_flow.py to v7_5 directory..."
        if (-not (Test-Path "src\v7_5")) {
            New-Item -Path "src\v7_5" -ItemType Directory | Out-Null
        }
        Copy-Item "src\v7.5\conversation_flow.py" "src\v7_5\" -ErrorAction SilentlyContinue
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Failed to copy conversation_flow.py"
        }
    }
}

# Check for module path variations
$v7_5_dot = Test-Path "src\v7.5"
$v7_5_underscore = Test-Path "src\v7_5"

if ($v7_5_dot) {
    Write-Host "Found v7.5 module path (dot notation)"
}

if ($v7_5_underscore) {
    Write-Host "Found v7_5 module path (underscore notation)"
}

if (-not ($v7_5_dot -or $v7_5_underscore)) {
    Write-Host "WARNING: Could not find v7.5 module path! Some components may not work."
}

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host $pythonVersion
}
catch {
    Write-Host "Error: Python is not installed or not in the PATH."
    Write-Host "Please install Python 3.8 or newer and try again."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if PySide6 is installed
try {
    python -c "import PySide6" 2>&1 | Out-Null
}
catch {
    Write-Host "PySide6 is not installed. Installing..."
    python -m pip install PySide6
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install PySide6. Please install it manually:"
        Write-Host "pip install PySide6"
        Read-Host "Press Enter to continue"
    }
}

# Functions for starting components
function Start-NeuralSeed {
    Write-Host "Starting Neural Seed System..."
    # Check if neural_seed exists in src/neural directory
    if (Test-Path "src\neural\seed.py") {
        Write-Host "Found neural seed in src\neural directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python src\neural\seed.py --log-level=info --port=$env:LUMINA_PORT; Read-Host 'Press Enter to close'`""
        return
    }
    
    # Try the direct seed.py in src
    if (Test-Path "src\seed.py") {
        Write-Host "Found seed.py in src directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python src\seed.py --log-level=info --port=$env:LUMINA_PORT; Read-Host 'Press Enter to close'`""
        return
    }
    
    # Last resort - try module import approach
    if ($v7_5_dot) {
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python -m src.v7.5.neural_seed --log-level=info --port=$env:LUMINA_PORT; Read-Host 'Press Enter to close'`""
    }
    elseif ($v7_5_underscore) {
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python -m src.v7_5.neural_seed --log-level=info --port=$env:LUMINA_PORT; Read-Host 'Press Enter to close'`""
    }
    else {
        Write-Host "WARNING: Could not find neural_seed module!"
    }
}

function Start-HolographicUI {
    Write-Host "Starting Holographic UI..."
    try {
        python run_holographic_frontend.py --gui-framework=$env:LUMINA_GUI_FRAMEWORK --port=$env:LUMINA_HOLOGRAPHIC_PORT
    }
    catch {
        Write-Host "Failed to start Holographic UI."
        Write-Host "Check that PySide6 is installed and run_holographic_frontend.py exists."
    }
}

function Start-ChatInterface {
    Write-Host "Starting Chat Interface..."
    
    # First try to use run_v7_5.bat if it exists
    if (Test-Path "run_v7_5.bat") {
        Write-Host "Using run_v7_5.bat to start chat interface..."
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `".\run_v7_5.bat --from-holographic`""
        return
    }
    
    # Try direct file path approach
    if (Test-Path "src\v7.5\lumina_frontend.py") {
        Write-Host "Found lumina_frontend.py in v7.5 directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7.5 && python lumina_frontend.py --port=$env:LUMINA_CHAT_PORT; Read-Host 'Press Enter to close'`""
    }
    elseif (Test-Path "src\v7_5\lumina_frontend.py") {
        if ((Get-Item "src\v7_5\lumina_frontend.py").Length -gt 100) {
            Write-Host "Found lumina_frontend.py in v7_5 directory"
            Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7_5 && python lumina_frontend.py --port=$env:LUMINA_CHAT_PORT; Read-Host 'Press Enter to close'`""
        }
        else {
            Write-Host "lumina_frontend.py exists but appears to be empty in v7_5 directory"
            # Check for central_language_node.py as an alternative
            if (Test-Path "src\central_language_node.py") {
                Write-Host "Using central_language_node.py as alternative chat interface"
                Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python src\central_language_node.py --port=$env:LUMINA_CHAT_PORT; Read-Host 'Press Enter to close'`""
            }
            else {
                Write-Host "ERROR: Could not find a valid chat interface module!"
            }
        }
    }
    else {
        Write-Host "WARNING: Could not find lumina_frontend.py directly."
        
        # Try to find alternatives in root src
        if (Test-Path "src\chat_with_system.py") {
            Write-Host "Found chat_with_system.py as alternative"
            Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python src\chat_with_system.py --port=$env:LUMINA_CHAT_PORT; Read-Host 'Press Enter to close'`""
            return
        }
        
        # Try module import approach as last resort
        Write-Host "Trying module import approach..."
        if ($v7_5_dot) {
            Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python -m src.v7.5.lumina_frontend --port=$env:LUMINA_CHAT_PORT; Read-Host 'Press Enter to close'`""
        }
        elseif ($v7_5_underscore) {
            Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python -m src.v7_5.lumina_frontend --port=$env:LUMINA_CHAT_PORT; Read-Host 'Press Enter to close'`""
        }
        else {
            Write-Host "ERROR: Failed to start Chat Interface!"
        }
    }
}

function Start-SystemMonitor {
    Write-Host "Starting System Monitor..."
    
    if (Test-Path "src\v7.5\system_monitor.py") {
        Write-Host "Found system_monitor.py in v7.5 directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7.5 && python system_monitor.py --port=$env:LUMINA_MONITOR_PORT; Read-Host 'Press Enter to close'`""
    }
    elseif (Test-Path "src\v7_5\system_monitor.py") {
        Write-Host "Found system_monitor.py in v7_5 directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7_5 && python system_monitor.py --port=$env:LUMINA_MONITOR_PORT; Read-Host 'Press Enter to close'`""
    }
    # Try monitoring directory if exists
    elseif (Test-Path "src\monitoring") {
        $monitoringFiles = Get-ChildItem -Path "src\monitoring" -Filter "*.py"
        if ($monitoringFiles.Count -gt 0) {
            Write-Host "Found monitoring files in src\monitoring"
            Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python $($monitoringFiles[0].FullName) --port=$env:LUMINA_MONITOR_PORT; Read-Host 'Press Enter to close'`""
        }
        else {
            Write-Host "WARNING: Could not find system monitor module!"
        }
    }
    else {
        Write-Host "WARNING: Could not find system_monitor module!"
    }
}

function Start-DatabaseConnector {
    Write-Host "Starting Database Connector..."
    
    if (Test-Path "src\v7.5\database_connector.py") {
        Write-Host "Found database_connector.py in v7.5 directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7.5 && python database_connector.py --port=$env:LUMINA_DATABASE_PORT; Read-Host 'Press Enter to close'`""
    }
    elseif (Test-Path "src\v7_5\database_connector.py") {
        Write-Host "Found database_connector.py in v7_5 directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7_5 && python database_connector.py --port=$env:LUMINA_DATABASE_PORT; Read-Host 'Press Enter to close'`""
    }
    # Check for database-related files in root src
    elseif (Test-Path "src\memory_api_server.py") {
        Write-Host "Using memory_api_server.py as alternative database connector"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python src\memory_api_server.py --port=$env:LUMINA_DATABASE_PORT; Read-Host 'Press Enter to close'`""
    }
    elseif (Test-Path "src\verify_database_connections.py") {
        Write-Host "Using verify_database_connections.py as alternative"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python src\verify_database_connections.py --port=$env:LUMINA_DATABASE_PORT; Read-Host 'Press Enter to close'`""
    }
    else {
        Write-Host "WARNING: Could not find database_connector module!"
    }
}

function Start-Autowiki {
    Write-Host "Starting Autowiki..."
    
    if (Test-Path "src\v7_5\autowiki.py") {
        Write-Host "Found autowiki.py in v7_5 directory"
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7_5 && python autowiki.py --port=$env:LUMINA_AUTOWIKI_PORT; Read-Host 'Press Enter to close'`""
        return
    }
    
    # Look for autowiki in v7.5 if exists
    if (Test-Path "src\v7.5") {
        $wikiFiles = Get-ChildItem -Path "src\v7.5" -Filter "*wiki*.py"
        if ($wikiFiles.Count -gt 0) {
            Write-Host "Found autowiki module in v7.5: $($wikiFiles[0].Name)"
            Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"cd src\v7.5 && python $($wikiFiles[0].Name) --port=$env:LUMINA_AUTOWIKI_PORT; Read-Host 'Press Enter to close'`""
            return
        }
    }
    
    # Try module import approach as last resort
    if ($v7_5_dot) {
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python -m src.v7.5.autowiki --port=$env:LUMINA_AUTOWIKI_PORT; Read-Host 'Press Enter to close'`""
    }
    elseif ($v7_5_underscore) {
        Start-Process -WindowStyle Normal powershell -ArgumentList "-Command `"python -m src.v7_5.autowiki --port=$env:LUMINA_AUTOWIKI_PORT; Read-Host 'Press Enter to close'`""
    }
    else {
        Write-Host "WARNING: Could not find autowiki module!"
    }
}

function Run-ComponentTests {
    Write-Host "Running component tests..."
    if (Test-Path "src\component_test.py") {
        python "src\component_test.py"
    }
    else {
        Write-Host "WARNING: Component test script not found!"
    }
}

# Main menu loop
while ($true) {
    Clear-Host
    Write-Host ""
    Write-Host "LUMINA V7.5 System"
    Write-Host "=================="
    Write-Host ""
    Write-Host "[1] Start Complete Holographic System"
    Write-Host "[2] Start Dashboard Panels"
    Write-Host "[3] Start Holographic UI Only"
    Write-Host "[4] Start Chat Interface Only"
    Write-Host "[5] Start System Monitor Only"
    Write-Host "[6] Start Database Connector Only"
    Write-Host "[7] Start AutoWiki Module Only"
    Write-Host "[8] Run Component Tests"
    Write-Host "[9] Open Documentation"
    Write-Host "[Q] Quit"
    Write-Host ""
    
    $choice = Read-Host "Enter your choice"
    
    switch ($choice) {
        "1" {
            Write-Host "Starting all LUMINA components..."
            Start-NeuralSeed
            Start-Sleep -Seconds 3
            Start-DatabaseConnector
            Start-Sleep -Seconds 2
            Start-Autowiki
            Start-Sleep -Seconds 2
            Start-ChatInterface
            Start-Sleep -Seconds 2
            Start-SystemMonitor
            Start-Sleep -Seconds 2
            Start-HolographicUI
        }
        "2" {
            Write-Host "Starting Dashboard Panels..."
            Start-SystemMonitor
        }
        "3" {
            Start-HolographicUI
        }
        "4" {
            Start-ChatInterface
        }
        "5" {
            Start-SystemMonitor
        }
        "6" {
            Start-DatabaseConnector
        }
        "7" {
            Start-Autowiki
        }
        "8" {
            Run-ComponentTests
            Read-Host "Press Enter to continue"
        }
        "9" {
            if (Test-Path "docs\index.html") {
                Start-Process "docs\index.html"
            }
            else {
                Write-Host "Documentation not found."
                Read-Host "Press Enter to continue"
            }
        }
        { $_ -eq "q" -or $_ -eq "Q" } {
            Write-Host "Exiting LUMINA V7.5 System..."
            exit 0
        }
        default {
            Write-Host "Invalid choice!"
            Read-Host "Press Enter to continue"
        }
    }
} 