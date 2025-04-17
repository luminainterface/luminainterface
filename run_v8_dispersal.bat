@echo off
echo Starting LUMINA V8 Seed Dispersal System...

:: Create necessary directories if they don't exist
if not exist "data" mkdir data
if not exist "data\seed" mkdir data\seed
if not exist "logs" mkdir logs

:: Set Python path to include the project root for proper imports
set PYTHONPATH=%CD%

:: Set environment variables
set SEED_DATA_DIR=data/seed
set GUI_FRAMEWORK=PySide6

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check for dependencies
echo Checking system dependencies...
python -c "import PySide6" >nul 2>&1
if %errorlevel% neq 0 (
    echo PySide6 is not installed.
    echo Installing required dependencies...
    pip install PySide6
)

:: Start the seed system if not already running
start /B python src/seed.py

:: Wait a moment for seed system to initialize
timeout /t 3 >nul

:: Start the V8 dispersal system
echo Starting V8 Seed Dispersal System...
python -m src.v8.seed_dispersal_system

echo V8 Seed Dispersal System has been closed.
pause 