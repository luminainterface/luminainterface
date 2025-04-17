@echo off
setlocal

:: Set Python path
set PYTHONPATH=%~dp0

:: Create required directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models

:: Run system diagnostics
python src/backend_diagnostics.py
if errorlevel 1 (
    echo Error: System diagnostics failed. Please check logs/diagnostics.log
    pause
    exit /b 1
)

:: Start Node Manager UI
echo Starting LUMINA Node Manager...
python src/node_manager_ui.py
if errorlevel 1 (
    echo Error: Node Manager failed to start. Please check logs/node_manager_ui.log
    pause
    exit /b 1
)

endlocal 