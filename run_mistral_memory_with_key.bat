@echo off
echo Starting Mistral Chat with Onsite Memory...

:: Check for API key parameter
if "%~1"=="" (
    echo Using default Mistral API key.
    set MISTRAL_API_KEY=2AyKmqCkChQ75bseJTLK9QF2AK0aefJPs
    set MOCK_MODE=False
) else (
    echo Using provided Mistral API key.
    set MISTRAL_API_KEY=%~1
    set MOCK_MODE=False
)

:: Create required directories
if not exist "data" mkdir data
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "data\memory" mkdir data\memory
if not exist "data\logs" mkdir data\logs
if not exist "data\neural_linguistic" mkdir data\neural_linguistic

:: Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please install Python and try again.
    pause
    exit /b 1
)

:: Set current directory to script location
cd /d "%~dp0"

:: Check if required modules are installed
python -c "import PySide6" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo PySide6 not installed. Installing...
    python -m pip install PySide6
)

:: Configure environment
set PYTHONPATH=%PYTHONPATH%;%CD%
set PYTHONPATH=%PYTHONPATH%;%CD%\src
set NN_WEIGHT=0.6
set LLM_WEIGHT=0.7

echo Neural Network Weight: %NN_WEIGHT%
echo LLM Weight: %LLM_WEIGHT%
echo.
echo Launching Mistral Chat with Onsite Memory...
echo API Key Status: %MOCK_MODE%
echo.

:: Run the application
python simple_mistral_gui.py

:: Check for errors
if %ERRORLEVEL% neq 0 (
    echo.
    echo Application exited with errors.
    pause
    exit /b %ERRORLEVEL%
)

exit /b 0 