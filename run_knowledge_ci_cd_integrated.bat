@echo off
echo Starting Integrated Knowledge CI/CD System for v8

:: Set environment variables
set PYTHONPATH=%~dp0
set MISTRAL_API_KEY=nLKZEpq29OihnaArxV7s6KtzsNEiky2A

:: Create necessary directories if they don't exist
if not exist "data" mkdir data
if not exist "data\backups" mkdir data\backups
if not exist "logs" mkdir logs

:: Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

:: Check for required packages
echo Checking dependencies...
python -c "import sqlite3, uuid, argparse, threading" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    pip install argparse
)

:: Check for Flask (needed for health monitoring)
python -c "import flask" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing Flask for health monitoring...
    pip install flask
)

:: Set health check port (change as needed)
set HEALTH_PORT=8765

:: Run the integrated system (in a separate process)
echo Running Knowledge CI/CD Integrated System...
echo Health monitoring available at http://localhost:%HEALTH_PORT%/health
start /b "" python src/v8/knowledge_ci_cd_integrated.py --health-check-port=%HEALTH_PORT% %*

:: Wait for 30 seconds then continue
echo Running for 30 seconds...
timeout /t 30 /nobreak

:: Send exit signal
echo Sending stop signal to Knowledge CI/CD Integrated System...
echo Completed test run. If you want to run longer, use run_full_knowledge_ci_cd.bat instead.

exit /b 0 