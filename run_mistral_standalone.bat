@echo off
rem ====================================
rem Standalone Mistral Chat Launcher
rem ====================================

echo Starting Standalone Mistral Chat...

rem Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please install Python and try again.
    pause
    exit /b 1
)

rem Set current directory to script location
cd /d "%~dp0"

rem Create data directories if they don't exist
if not exist "data\onsite_memory" (
    echo Creating onsite memory directory...
    mkdir "data\onsite_memory"
)

rem Check if required modules are installed
python -c "import PySide6" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo PySide6 not installed. Installing...
    python -m pip install PySide6
)

echo Launching Mistral Chat...
echo.

rem Run the application
python run_mistral_standalone.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Application exited with errors.
    pause
    exit /b %ERRORLEVEL%
)

exit /b 0 