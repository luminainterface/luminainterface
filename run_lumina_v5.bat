@echo off
echo Starting Lumina V5 Neural Network System...
echo.

:: Set up Python environment
set PYTHONPATH=%~dp0
cd %~dp0

:: Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

:: Create necessary directories if they don't exist
if not exist logs mkdir logs
if not exist data mkdir data

:: Start the dashboard in the background
start "Lumina V5 Dashboard" python src/visualization/create_dashboard.py

:: Start the main Lumina V5 system
echo Starting Lumina V5 core system...
python src/v5/launch_lumina.py --ui --verbose

:: If the system exits, wait for user input
echo.
echo Lumina V5 system has stopped.
pause 