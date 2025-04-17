@echo off
echo Starting Mistral AI Chat App...

rem Create necessary data directories
if not exist "data\onsite_memory" (
    echo Creating onsite memory directory...
    mkdir "data\onsite_memory"
)

python run_mistral_app.py
if %ERRORLEVEL% neq 0 (
    echo Application exited with errors
    pause
) 