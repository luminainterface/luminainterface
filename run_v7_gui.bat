@echo off
echo ===================================
echo   LUMINA V7 Enhanced GUI Launcher
echo ===================================
echo.

:: Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "data\neural_linguistic" mkdir data\neural_linguistic

:: Set PYTHONPATH to include current directory and src
set PYTHONPATH=%CD%
set PYTHONPATH=%PYTHONPATH%;%CD%\src
echo Set PYTHONPATH to: %PYTHONPATH%

:: Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python and make sure it's in your PATH
    echo.
    pause
    exit /b 1
)

:: Verify PySide6 is installed
echo Checking for PySide6...
python -c "import PySide6" >nul 2>nul
if %errorlevel% neq 0 (
    echo PySide6 not found. Installing...
    python -m pip install PySide6
)

:: Run the v7 launcher
echo.
echo Launching LUMINA V7 GUI...
echo.
python src/v7/run_v7.py
set LAUNCH_RESULT=%errorlevel%

if %LAUNCH_RESULT% neq 0 (
    echo.
    echo Launch failed with exit code %LAUNCH_RESULT%
    echo Check logs/v7_runtime.log for details
    echo.
    pause
    exit /b %LAUNCH_RESULT%
)

echo.
echo LUMINA V7 closed successfully.
echo Press any key to exit...
pause >nul 