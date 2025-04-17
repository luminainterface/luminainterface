@echo off
echo Starting Neural Network Backend System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create necessary directories if they don't exist
if not exist src\integration\data (
    mkdir src\integration\data
)
if not exist src\integration\logs (
    mkdir src\integration\logs
)

REM Set up Python environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Add the current directory to PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Create database if it doesn't exist
if not exist src\integration\data\signals.db (
    echo Creating signals database...
    python -c "from src.integration.database import init_db; init_db()"
)

REM Set up logging
echo Configuring logging...
python -c "import logging; logging.basicConfig(filename='src/integration/logs/backend.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')"

REM Start the backend system
echo Starting backend system...
python src/integration/backend.py

REM Deactivate virtual environment
deactivate

echo.
echo Backend system stopped.
pause 