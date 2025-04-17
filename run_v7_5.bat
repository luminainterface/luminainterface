@echo off
setlocal enabledelayedexpansion

REM Check if being called from holographic system
if "%1"=="--from-holographic" (
    echo LUMINA v7.5 Chat Interface running as part of Holographic System
    set FROM_HOLOGRAPHIC=true
    
    REM Inherit environment variables if called from holographic system
    if not "%MISTRAL_API_KEY%"=="" (
        echo Using API key from holographic system
        goto start_frontend
    )
) else (
    echo Starting LUMINA v7.5 - Integrated Neural Interface
    echo ================================================
    echo.
)

REM Create necessary directories
if not exist data mkdir data
if not exist logs mkdir logs
if not exist data\breath mkdir data\breath
if not exist data\consciousness mkdir data\consciousness
if not exist data\autowiki mkdir data\autowiki
if not exist data\onsite_memory mkdir data\onsite_memory
if not exist data\conversations mkdir data\conversations

REM Set PYTHONPATH to include the current directory
set PYTHONPATH=%CD%

REM Copy the conversation_flow.py module if needed
if not exist src\v7_5\conversation_flow.py (
    echo Copying conversation_flow.py to v7_5 directory...
    copy src\v7.5\conversation_flow.py src\v7_5\ > nul
    if not errorlevel 0 (
        echo Warning: Failed to copy conversation_flow.py
    )
)

REM Load API key from .env file
echo Loading API key from .env file...
FOR /F "tokens=2 delims==" %%a in ('type .env ^| findstr "MISTRAL_API_KEY"') do (
    set MISTRAL_API_KEY=%%a
)

echo Using Mistral API Key: %MISTRAL_API_KEY:~0,4%...%MISTRAL_API_KEY:~-4%

REM Print LLM parameters
echo LLM Parameters:
FOR /F "tokens=1,2 delims==" %%a in ('type .env ^| findstr "LLM_"') do (
    echo   %%a=%%b
)

REM Print Neural Network parameters
echo Neural Network Parameters:
FOR /F "tokens=1,2 delims==" %%a in ('type .env ^| findstr "NN_"') do (
    echo   %%a=%%b
)

echo.
echo Default values if not specified in .env:
echo   LLM_MODEL=mistral-medium
echo   LLM_WEIGHT=0.7
echo   NN_WEIGHT=0.3
echo   LLM_TEMPERATURE=0.7
echo   LLM_TOP_P=0.9
echo   LLM_TOP_K=50

echo All systems ready. Starting LUMINA v7.5...

:start_frontend
python src/v7.5/lumina_frontend.py

if ERRORLEVEL 1 (
  echo.
  echo Error: LUMINA v7.5 exited with code %ERRORLEVEL%
  echo Check the logs for more information.
) else (
  echo.
  echo LUMINA v7.5 session ended successfully.
)

pause 