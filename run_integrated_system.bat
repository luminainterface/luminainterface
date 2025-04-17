@echo off
title Integrated v7 Holographic and v8 Knowledge CI/CD System

:: Set the current working directory to the script's directory
cd /d "%~dp0"

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

:: Set environment variables
set V8_PORT=8765
set V7_DATA_PATH=data\v7
set V8_DATA_PATH=data\v8
set SYNC_INTERVAL=30

echo ===============================================
echo Integrated v7 Holographic and v8 Knowledge CI/CD System
echo ===============================================
echo.
echo This script will start:
echo 1. The v8 Knowledge CI/CD System (Health Check Server)
echo 2. The v8 to v7 Knowledge Bridge
echo 3. The v7 Holographic System
echo.
echo Press Ctrl+C to stop all services
echo.

:: Create required directories
if not exist "%V7_DATA_PATH%\neural_seed" mkdir "%V7_DATA_PATH%\neural_seed"
if not exist "%V8_DATA_PATH%\knowledge" mkdir "%V8_DATA_PATH%\knowledge"
if not exist "%V8_DATA_PATH%\metrics" mkdir "%V8_DATA_PATH%\metrics"
if not exist "%V8_DATA_PATH%\temple" mkdir "%V8_DATA_PATH%\temple"

:: Start the v8 Health Check Server in a new window
start "V8 Health Check Server" cmd /c "python src\v8\health_check_server.py --port %V8_PORT% --check-interval %SYNC_INTERVAL% 2>&1 | tee logs\v8_health_check_server.log"

:: Wait a moment for the health check server to start
echo Starting v8 Health Check Server on port %V8_PORT%...
timeout /t 3 /nobreak > nul

:: Start the v8 to v7 Knowledge Bridge in a new window
start "V8 to V7 Knowledge Bridge" cmd /c "python src\bridge\v8_to_v7_knowledge_bridge.py --v8-port %V8_PORT% --knowledge-path %V7_DATA_PATH%\neural_seed --sync-interval %SYNC_INTERVAL% 2>&1 | tee logs\v8_to_v7_bridge.log"

:: Wait a moment for the bridge to start
echo Starting v8 to v7 Knowledge Bridge...
timeout /t 2 /nobreak > nul

:: Start the v7 Holographic System
echo Starting v7 Holographic System...
call run_v7_holographic.bat

:: Wait for user to press Ctrl+C
echo.
echo All systems started. Press Ctrl+C to stop all services.
pause > nul 