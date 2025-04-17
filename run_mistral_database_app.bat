@echo off
REM Run Mistral Database Application with Neural Network Integration
setlocal

REM Set environment variables
set DATA_DIR=data
set NEURAL_MODELS_DIR=data\model_output
set MISTRAL_MODEL=mistral-medium
set LLM_WEIGHT=0.65
set NN_WEIGHT=0.75
set API_LOG_FILE=logs\api_calls.log
set SYSTEM_LOG_FILE=logs\system.log
set MEMORY_DB=data\onsite_memory\memory.db

REM Create necessary directories if they don't exist
if not exist %DATA_DIR% mkdir %DATA_DIR%
if not exist %NEURAL_MODELS_DIR% mkdir %NEURAL_MODELS_DIR%
if not exist logs mkdir logs
if not exist data\onsite_memory mkdir data\onsite_memory

REM Check for Python installation
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Check for required packages
echo Checking for required packages...
python -c "import torch" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo PyTorch is not installed. Installing...
    pip install torch torchvision torchaudio
)

python -c "import PySide6" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo PySide6 is not installed. Installing...
    pip install PySide6
)

python -c "import sqlite3" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo SQLite3 module is not available
    exit /b 1
)

REM Launch application with appropriate parameters
echo Starting Mistral Chat Application with Neural Integration...
echo.
echo LLM Weight: %LLM_WEIGHT%
echo NN Weight: %NN_WEIGHT%
echo Data Directory: %DATA_DIR%
echo Neural Models: %NEURAL_MODELS_DIR%
echo.

REM Run the application
python simple_mistral_gui.py --data-dir %DATA_DIR% --model %MISTRAL_MODEL% --llm-weight %LLM_WEIGHT% --nn-weight %NN_WEIGHT% --neural-models-dir %NEURAL_MODELS_DIR% --memory-db %MEMORY_DB% --api-log %API_LOG_FILE% --log-file %SYSTEM_LOG_FILE%

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Application terminated with errors.
    echo Check log files for details: %SYSTEM_LOG_FILE%
) else (
    echo Application closed successfully.
)

endlocal 