@echo off
echo ===================================
echo   LUMINA V7.0.0.1 Direct Launcher
echo ===================================
echo.

:: Set error handling
setlocal enabledelayedexpansion

:: Set PYTHONPATH to include current directory
set PYTHONPATH=%CD%;%PYTHONPATH%
echo Set PYTHONPATH to include current directory

:: Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python and make sure it's in your PATH
    echo.
    pause
    exit /b 1
)

:: Get current directory for Python import path
echo Current directory: %CD%

:: Launch the V7 main window directly
echo Launching LUMINA V7.0.0.1 directly...
echo.
echo === Launch Log ===
python -c "import sys; sys.path.insert(0, '.'); from src.v7.v7_launcher import main; main()" %*
set LAUNCH_RESULT=%errorlevel%

:: Check exit code
if %LAUNCH_RESULT% neq 0 (
    echo.
    echo === Launch failed with exit code %LAUNCH_RESULT% ===
    echo.
    echo Trying fallback launch method...
    
    :: Try fallback approach
    cd src\v7
    python v7_launcher.py %*
    set LAUNCH_RESULT=%errorlevel%
    cd ..\..
    
    if %LAUNCH_RESULT% neq 0 (
        echo.
        echo === Fallback launch failed ===
        echo Try running the repair script:
        echo   repair_dependencies.bat
        echo.
        pause
        exit /b %LAUNCH_RESULT%
    )
)

:: Keep the window open if run directly (not from command line)
echo.
echo Press any key to exit...
pause >nul 