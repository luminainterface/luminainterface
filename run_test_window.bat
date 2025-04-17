@echo off
echo Starting Neural Network Visualization Test Window...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create src directory structure if it doesn't exist
if not exist src\frontend\ui\components\widgets (
    mkdir src\frontend\ui\components\widgets
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

REM Run the test window
echo Starting visualization...
python src/frontend/ui/test_window.py

REM Deactivate virtual environment
deactivate

echo.
echo Test window closed.
pause 