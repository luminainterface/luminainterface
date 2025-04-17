@echo off
REM Run Mistral Neural Network Demo
setlocal

REM Set environment variables
set DATA_DIR=data
set NEURAL_MODELS_DIR=data\neural_models
set MISTRAL_MODEL=mistral-small
set LLM_WEIGHT=0.65
set NN_WEIGHT=0.35
set SYSTEM_LOG_FILE=logs\system.log

REM Create necessary directories if they don't exist
if not exist %DATA_DIR% mkdir %DATA_DIR%
if not exist %NEURAL_MODELS_DIR% mkdir %NEURAL_MODELS_DIR%
if not exist logs mkdir logs

REM Check for Python installation
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Check for required packages
echo Checking for required packages...
python -c "import numpy" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo NumPy is not installed. Installing...
    pip install numpy
)

python -c "import torch" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo PyTorch is not installed. Installing...
    pip install torch
)

REM Check if Mistral API key is set
if "%MISTRAL_API_KEY%"=="" (
    echo MISTRAL_API_KEY environment variable is not set.
    echo You can either:
    echo 1. Set it before running this script, or
    echo 2. Run in mock mode without the API.
    echo.
    set /p MOCK_MODE="Run in mock mode? (Y/N): "
    
    if /i "%MOCK_MODE%"=="Y" (
        set MOCK_ARG=--mock
    ) else (
        set /p MISTRAL_API_KEY="Enter your Mistral API key: "
        set MOCK_ARG=
    )
) else (
    set MOCK_ARG=
    echo Using Mistral API key from environment variable.
)

REM Launch neural demo with appropriate parameters
echo Starting Mistral Neural Network Demo...
echo.
echo LLM Weight: %LLM_WEIGHT%
echo NN Weight: %NN_WEIGHT%
echo Model: %MISTRAL_MODEL%
echo Data Directory: %DATA_DIR%
echo Neural Models: %NEURAL_MODELS_DIR%
echo.

REM Run the application
python mistral_neural_demo.py --model %MISTRAL_MODEL% --llm-weight %LLM_WEIGHT% --nn-weight %NN_WEIGHT% --model-dir %NEURAL_MODELS_DIR% %MOCK_ARG%

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Application terminated with errors.
    echo Check log files for details: %SYSTEM_LOG_FILE%
) else (
    echo Application closed successfully.
)

endlocal 