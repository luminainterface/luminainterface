@echo off
echo ===================================
echo   LUMINA V7 with Mistral Launcher
echo ===================================
echo.

:: Check for API key parameter
if "%~1"=="" (
    echo Using default Mistral API key.
    set MISTRAL_API_KEY=2AyKmqCkChQ75bseJTLK9QF2AK0aefJPs
) else (
    echo Using provided Mistral API key.
    set MISTRAL_API_KEY=%~1
)

:: Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "data\neural_linguistic" mkdir data\neural_linguistic
if not exist "data\autowiki" mkdir data\autowiki

:: Set PYTHONPATH to include current directory and src
set PYTHONPATH=%CD%
set PYTHONPATH=%PYTHONPATH%;%CD%\src
echo Set PYTHONPATH to: %PYTHONPATH%

:: Configure neural and language weights
set NN_WEIGHT=0.6
set LLM_WEIGHT=0.7
echo Neural Network Weight: %NN_WEIGHT%
echo LLM Weight: %LLM_WEIGHT%

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

:: Run the v7 launcher with Mistral integration
echo.
echo Launching LUMINA V7 with Mistral integration...
echo.

python src/v7/v7_launcher.py --mistral-key %MISTRAL_API_KEY% --mistral-model mistral-medium
set LAUNCH_RESULT=%errorlevel%

if %LAUNCH_RESULT% neq 0 (
    echo.
    echo Launch failed with exit code %LAUNCH_RESULT%
    echo Check logs/v7_launcher.log for details
    echo.
    pause
    exit /b %LAUNCH_RESULT%
)

echo.
echo LUMINA V7 with Mistral closed successfully.
echo Press any key to exit...
pause >nul 