@echo off
REM V7 Memory Node Demo Runner for Windows
REM -------------------------------------
REM This script runs the V7 Memory Node demonstration script
REM with proper environment setup

echo === V7 Memory Node Demo Runner ===

REM Get the script directory
set SCRIPT_DIR=%~dp0

REM Find project root (3 levels up from script directory)
pushd %SCRIPT_DIR%..\..\..
set PROJECT_ROOT=%CD%
echo Project root: %PROJECT_ROOT%

REM Check for virtual environment and activate if found
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

REM Run the memory demo script
echo Starting memory demo...
python src/v7/examples/memory_demo.py %*

REM Check if the demo ran successfully
if %ERRORLEVEL% neq 0 (
    echo Memory demo exited with error code: %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Memory demo completed successfully!
popd
exit /b 0 