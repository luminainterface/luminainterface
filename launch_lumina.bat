@echo off
setlocal enabledelayedexpansion

:: Set title
title LUMINA v7.5 Debug Mode

:: Print banner
echo ================================
echo LUMINA v7.5 Launcher - Debug Mode
echo ================================
echo.

:: Enable debug mode
set PYTHONDEBUG=1
set PYTHONVERBOSE=1

:: Check if Python is installed
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

:: Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo Current directory: %CD%
echo.

:: Set PYTHONPATH to include the project root
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
echo PYTHONPATH: %PYTHONPATH%
echo.

:: Create necessary directories
echo Creating directories...
if not exist "logs" (
    echo Creating logs directory...
    mkdir logs
)
if not exist "data\conversations" (
    echo Creating conversations directory...
    mkdir "data\conversations"
)
if not exist "data\wiki" (
    echo Creating wiki directory...
    mkdir "data\wiki"
)
echo.

:: Check for required packages
echo Checking dependencies...
python -c "import PySide6; print('PySide6 version:', PySide6.__version__)"
if errorlevel 1 (
    echo Installing PySide6...
    python -m pip install PySide6>=6.6.1
)

python -c "import wikipedia; print('Wikipedia version:', wikipedia.__version__)"
if errorlevel 1 (
    echo Installing wikipedia...
    python -m pip install wikipedia>=1.4.0
)

python -c "import qasync; print('qasync version:', qasync.__version__)"
if errorlevel 1 (
    echo Installing qasync...
    python -m pip install qasync>=0.24.0
)

:: Launch LUMINA
echo.
echo Launching LUMINA v7.5 in debug mode...
echo.
python -X dev run_lumina.py

:: If there was an error, pause to show the message
if errorlevel 1 (
    echo.
    echo Error launching LUMINA. See error message above.
    echo.
    echo Debug information:
    echo Python path: %PYTHONPATH%
    echo Current directory: %CD%
    echo.
    pause
)

endlocal 