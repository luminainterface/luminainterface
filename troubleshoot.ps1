# Central Node Monitor Troubleshooter
# PowerShell script for diagnosing and fixing common issues

# Set up logging
$logDir = "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

function Write-Log {
    param($message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $message" | Out-File -Append "$logDir\troubleshoot.log"
    Write-Host $message
}

# Function to check Python installation
function Test-PythonInstallation {
    try {
        $pythonVersion = python --version 2>&1
        Write-Log "Python installation verified: $pythonVersion"
        return $true
    }
    catch {
        Write-Log "ERROR: Python is not installed or not in PATH"
        return $false
    }
}

# Function to check dependencies
function Test-Dependencies {
    Write-Log "Checking Python dependencies..."
    $requiredPackages = @("PySide6", "psutil", "numpy", "pyqtgraph", "matplotlib", "networkx")
    $installedPackages = pip list | Select-String -Pattern ($requiredPackages -join "|")
    
    if ($installedPackages.Count -lt $requiredPackages.Count) {
        Write-Log "Installing required dependencies..."
        pip install $requiredPackages
    }
    Write-Log "Dependencies verified"
}

# Function to fix timer initialization
function Fix-TimerInitialization {
    Write-Log "Fixing timer initialization in central_node_monitor.py..."
    $filePath = "src\central_node_monitor.py"
    $content = Get-Content $filePath
    $content = $content -replace 'self\.timer\.stop\(\)', 'if hasattr(self, "update_timer"): self.update_timer.stop()'
    $content | Set-Content $filePath
    Write-Log "Timer initialization fix applied"
}

# Function to check file structure
function Test-FileStructure {
    Write-Log "Verifying file structure..."
    $requiredFiles = @(
        "src\language\background_learning_engine.py",
        "src\central_node.py"
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Log "ERROR: $file not found"
            return $false
        }
    }
    Write-Log "File structure verified"
    return $true
}

# Function to run the monitor
function Start-Monitor {
    Write-Log "Starting central node monitor..."
    try {
        python src\central_node_monitor.py 2> "$logDir\monitor_error.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Log "ERROR: Monitor failed to start. Check $logDir\monitor_error.log for details"
            Get-Content "$logDir\monitor_error.log"
            return $false
        }
        return $true
    }
    catch {
        Write-Log "ERROR: Failed to start monitor: $_"
        return $false
    }
}

# Main execution
Write-Host "=== Central Node Monitor Troubleshooter ===" -ForegroundColor Blue
Write-Host ""

# Run checks
if (-not (Test-PythonInstallation)) { exit 1 }
Test-Dependencies
if (-not (Test-FileStructure)) { exit 1 }
Fix-TimerInitialization

# Ask user if they want to run the monitor
$runMonitor = Read-Host "Do you want to run the monitor now? (y/n)"
if ($runMonitor -eq "y") {
    Start-Monitor
}

Write-Host ""
Write-Host "Troubleshooting complete. Check $logDir\troubleshoot.log for details." -ForegroundColor Green
Read-Host "Press Enter to continue..." 