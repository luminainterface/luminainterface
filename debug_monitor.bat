@echo off
echo Starting LUMINA Monitor in Debug Mode...
echo.

:: Set Python path
set PYTHONPATH=%~dp0;%~dp0src

:: Enable debug logging
set LUMINA_DEBUG=1
set LUMINA_LOG_LEVEL=DEBUG

:: Create necessary directories if they don't exist
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python not found in PATH
    pause
    exit /b 1
)

echo.
echo Starting monitor with debug options...
python src/central_node_monitor.py --debug --log-level=DEBUG --troubleshoot

if errorlevel 1 (
    echo Error: Monitor failed to start
    echo Check logs/central_node_monitor.log for details
    pause
)

pause 