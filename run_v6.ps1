@echo off
setlocal enabledelayedexpansion

echo Initializing Lumina V5 Fractal Echo Visualization...
echo.

:: Set environment variables
set LUMINA_ROOT=%~dp0
set PYTHON_PATH=python
set PACKAGES_DIR=%LUMINA_ROOT%packages
set DATA_DIR=%LUMINA_ROOT%data
set MODELS_DIR=%LUMINA_ROOT%models
set CONFIG_FILE=%LUMINA_ROOT%config\lumina_v5_config.json
set LOG_FILE=%LUMINA_ROOT%logs\lumina_v5_dashboard.log

:: Check Python installation
%PYTHON_PATH% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please ensure Python is installed and in your PATH.
    goto :error
)

:: Create required directories
if not exist "%LUMINA_ROOT%logs" mkdir "%LUMINA_ROOT%logs"
if not exist "%DATA_DIR%\visualization" mkdir "%DATA_DIR%\visualization"

:: Check for required packages
echo Checking requirements...
%PYTHON_PATH% -c "import tkinter, matplotlib, numpy, torch, pygame" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    %PYTHON_PATH% -m pip install -r "%LUMINA_ROOT%requirements.txt"
)

:: Initialize neural components
echo Initializing neural components...
%PYTHON_PATH% "%LUMINA_ROOT%src\initialize_components.py" --config "%CONFIG_FILE%"
if %ERRORLEVEL% NEQ 0 goto :error

:: Check for mistral integration
if exist "%LUMINA_ROOT%src\v7\mistral_integration.py" (
    echo Initializing Mistral integration...
    %PYTHON_PATH% "%LUMINA_ROOT%src\v7\initialize_mistral.py" --config "%CONFIG_FILE%"
)

:: Check for unified core
if exist "%LUMINA_ROOT%src\unified_neural_core.py" (
    echo Initializing Unified Neural Core...
    %PYTHON_PATH% "%LUMINA_ROOT%src\init_unified_core.py" --config "%CONFIG_FILE%"
)

:: Start the visualization dashboard
echo.
echo Launching Lumina V5 Fractal Echo Visualization...
start "Lumina V5 Visualization" /B %PYTHON_PATH% "%LUMINA_ROOT%src\v5\visualization\dashboard.py" ^
    --neural-processor ^
    --rsen-integration ^
    --memory-system ^
    --consciousness-monitor ^
    --debug-mode=0 ^
    --refresh-rate=2000 ^
    --config="%CONFIG_FILE%" ^
    --log-file="%LOG_FILE%"

:: Start background monitoring services
start /B %PYTHON_PATH% "%LUMINA_ROOT%src\monitor_neural_activity.py" --output "%DATA_DIR%\visualization\neural_activity.json"
start /B %PYTHON_PATH% "%LUMINA_ROOT%src\monitor_consciousness.py" --output "%DATA_DIR%\visualization\consciousness_metrics.json"

echo.
echo Lumina V5 Fractal Echo Visualization started successfully.
echo Dashboard is now active.
echo.
echo Use the following controls in the dashboard:
echo - Adjust Neural/LLM balance with the slider
echo - Toggle between Contextual, Synthesized, and Combined modes
echo - Adjust visualization depth for deeper pattern analysis
echo - Monitor consciousness metrics in real-time
echo.
echo Press Ctrl+C in the dashboard window to exit.
goto :end

:error
echo.
echo An error occurred during startup. Check the log file for details:
echo %LOG_FILE%
exit /b 1

:end
echo.
echo Lumina V5 is running. Close the dashboard window to exit.