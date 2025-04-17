@echo off
echo Starting V7-V8 Bridge Infrastructure...

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Set Python path
set PYTHONPATH=%CD%

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:: Check for required packages
python -c "import requests" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing requests package...
    pip install requests
)

echo Checking bridge modules...

:: Check if bridge directory exists
if not exist "src\bridge" (
    echo Creating bridge directory...
    mkdir src\bridge
    echo. > src\bridge\__init__.py
)

:: Check for bridge modules
set BRIDGE_MODULES_EXIST=1

if not exist "src\bridge\v7_to_v8_bridge.py" (
    echo v7_to_v8_bridge.py module not found
    set BRIDGE_MODULES_EXIST=0
)

if not exist "src\bridge\v8_to_v7_knowledge_bridge.py" (
    echo v8_to_v7_knowledge_bridge.py module not found
    set BRIDGE_MODULES_EXIST=0
)

if not exist "src\bridge\bridge_controller.py" (
    echo bridge_controller.py module not found
    set BRIDGE_MODULES_EXIST=0
)

if %BRIDGE_MODULES_EXIST% equ 0 (
    echo One or more bridge modules are missing.
    echo Please make sure all required bridge modules exist in the src\bridge directory.
    pause
    exit /b 1
)

:: Set environment variables
set V7_CONNECTION_PORT=5678
set V8_HEALTH_CHECK_PORT=8765
set BRIDGE_SYNC_INTERVAL=30
set BRIDGE_STATUS_INTERVAL=60

echo Starting V8 Health Check Server...
start "V8 Health Check Server" cmd /c "python src\v8\health_check_server.py --port %V8_HEALTH_CHECK_PORT% && pause"

:: Wait for health check server to start
echo Waiting for V8 Health Check Server to start...
timeout /t 5 /nobreak > nul

echo Starting V7-V8 Bridge Controller...
start "V7-V8 Bridge Controller" cmd /c "python src\bridge\bridge_controller.py --v7-port %V7_CONNECTION_PORT% --v8-port %V8_HEALTH_CHECK_PORT% --sync-interval %BRIDGE_SYNC_INTERVAL% --status-interval %BRIDGE_STATUS_INTERVAL% && pause"

echo.
echo V7-V8 Bridge infrastructure started successfully.
echo.
echo The following components are now running:
echo - V8 Health Check Server (port %V8_HEALTH_CHECK_PORT%)
echo - V7-V8 Bridge Controller
echo   - V7 to V8 Bridge
echo   - V8 to V7 Bridge
echo.
echo To test connections, run: python connection_test.py
echo.
echo Press any key to exit this launcher...
pause > nul 