@echo off
setlocal enabledelayedexpansion

:: Set up environment variables
set PYTHONPATH=%CD%
set LUMINA_HOME=%CD%

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
)

:: Start the frontend system
echo Starting Lumina Frontend System...
python src/main.py
if errorlevel 1 (
    echo Failed to start the frontend system
    pause
    exit /b 1
)

:: Cleanup
deactivate
endlocal 