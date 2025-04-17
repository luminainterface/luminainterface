@echo off
setlocal enabledelayedexpansion

:: Set color codes
set "RED=0C"
set "GREEN=0A"
set "YELLOW=0E"
set "BLUE=09"
set "WHITE=0F"

:: Set title
title Neural Network Node Manager

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [%RED%ERROR%WHITE%] Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Check required packages
echo [%BLUE%INFO%WHITE%] Checking required packages...

:: Check PySide6
pip show PySide6 >nul 2>&1
if %errorlevel% neq 0 (
    echo [%YELLOW%WARNING%WHITE%] PySide6 not found, installing...
    pip install PySide6
)

:: Check NumPy
pip show numpy >nul 2>&1
if %errorlevel% neq 0 (
    echo [%YELLOW%WARNING%WHITE%] NumPy not found, installing...
    pip install numpy
)

:: Check SciPy (required for some processors)
pip show scipy >nul 2>&1
if %errorlevel% neq 0 (
    echo [%YELLOW%WARNING%WHITE%] SciPy not found, installing...
    pip install scipy
)

:: Check if main.py exists
if not exist "main.py" (
    echo [%RED%ERROR%WHITE%] main.py not found in root directory
    pause
    exit /b 1
)

:: Check if node_manager_ui.py exists
if not exist "src\node_manager_ui.py" (
    echo [%RED%ERROR%WHITE%] node_manager_ui.py not found in src directory
    pause
    exit /b 1
)

:: Set Python path to include current directory and src
set PYTHONPATH=%CD%;%CD%\src

:: Launch the node manager UI
echo [%GREEN%SUCCESS%WHITE%] Starting Node Manager UI...
python src\node_manager_ui.py

:: If the UI exits, show message
echo [%BLUE%INFO%WHITE%] Node Manager UI has been closed
pause 