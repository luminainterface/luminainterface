@echo off
setlocal EnableDelayedExpansion

:: Set console colors for better visibility
color 0F

echo [92m=== LUMINA Central Node Monitor Debug Launcher ===[0m
echo.

:: Set Python path
set PYTHONPATH=%~dp0;%~dp0src

:: Create required directories
echo Creating required directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "assets" mkdir assets

:: Check Python installation
echo Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo [91mError: Python is not installed or not in PATH[0m
    pause
    exit /b 1
)

:: Check required packages
echo Checking required packages...
python -c "import PySide6" > nul 2>&1
if errorlevel 1 (
    echo [91mError: PySide6 is not installed. Installing...[0m
    pip install PySide6
)

python -c "import numpy" > nul 2>&1
if errorlevel 1 (
    echo [91mError: numpy is not installed. Installing...[0m
    pip install numpy
)

:: Check UI components
echo Checking UI components...
if not exist "src\ui\components\modern_metrics_card.py" (
    echo [91mError: Missing modern_metrics_card.py[0m
    pause
    exit /b 1
)

if not exist "src\ui\components\modern_log_viewer.py" (
    echo [91mError: Missing modern_log_viewer.py[0m
    pause
    exit /b 1
)

:: Run system diagnostics
echo Running system diagnostics...
python src/backend_diagnostics.py
if errorlevel 1 (
    echo [91mError: System diagnostics failed. Please check logs/diagnostics.log[0m
    pause
    exit /b 1
)

:: Start monitor in debug mode
echo [92mStarting LUMINA Central Node Monitor in debug mode...[0m
echo [93mDebug output will be shown below:[0m
echo.

set PYTHONPATH=%~dp0;%~dp0src
set DEBUG=1
python src/central_node_monitor.py

if errorlevel 1 (
    echo.
    echo [91mError: Monitor crashed. Check the error message above.[0m
    echo Error code: %errorlevel%
    echo.
    echo Debug information:
    echo - Python path: %PYTHONPATH%
    echo - Working directory: %CD%
    echo - Debug mode: enabled
    echo.
    echo Please check:
    echo 1. All required packages are installed
    echo 2. UI components are present
    echo 3. Theme files are properly configured
    echo 4. Log files for detailed error messages
    pause
) else (
    echo [92mMonitor closed successfully[0m
)

endlocal 