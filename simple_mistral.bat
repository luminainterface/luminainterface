@echo off
echo Starting Mistral Chat with Onsite Memory...

:: Create data directories if they don't exist
if not exist "data" mkdir data
if not exist "data\onsite_memory" mkdir data\onsite_memory

:: Run the application
python simple_mistral_gui.py

:: Check for errors
if %ERRORLEVEL% neq 0 (
    echo Application exited with errors
    pause
) 