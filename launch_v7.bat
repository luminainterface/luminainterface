@echo off
echo ===================================
echo   LUMINA V7.0.0.1 Launch Script
echo ===================================
echo.

:: Set error handling
setlocal enabledelayedexpansion

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

:: Launch the application with output redirection
echo Launching LUMINA V7.0.0.1...
echo.
echo === Launch Log ===
python src/v7/lumina_v7/launch_v7.py %* 2>&1
set LAUNCH_RESULT=%errorlevel%

:: Check exit code
if %LAUNCH_RESULT% neq 0 (
    echo.
    echo === Launch failed with exit code %LAUNCH_RESULT% ===
    echo If there were dependency errors, try running:
    echo python src/v7/lumina_v7/launch_v7.py --install
    echo.
    pause
    exit /b %LAUNCH_RESULT%
)

:: Keep the window open if run directly (not from command line)
echo.
echo Press any key to exit...
pause >nul 