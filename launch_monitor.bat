@echo off
echo Starting LUMINA Central Node Monitor...

REM Set Python path
set PYTHONPATH=%~dp0;%~dp0src

REM Create necessary directories if they don't exist
if not exist logs mkdir logs
if not exist data mkdir data
if not exist models mkdir models
if not exist assets mkdir assets

REM Run system diagnostics
echo Running system diagnostics...
python src/backend_diagnostics.py
if %ERRORLEVEL% neq 0 (
    echo System diagnostics failed. Please check logs for details.
    echo See logs/diagnostics.log for more information.
    pause
    exit /b 1
)

REM Start the Central Node Monitor
echo Starting monitor...
python src/central_node_monitor.py
if %ERRORLEVEL% neq 0 (
    echo Monitor failed to start. Please check logs for details.
    echo See logs/central_node_monitor.log for more information.
    pause
    exit /b 1
)

exit /b 0 