@echo off
echo Starting LUMINA v7.5 - Integrated Neural Interface
echo ================================================

REM Create required directories if they don't exist
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "data\consciousness" mkdir data\consciousness
if not exist "data\autowiki" mkdir data\autowiki
if not exist "data\breath" mkdir data\breath

REM Set environment variables
set PYTHONPATH=%CD%
set DASHBOARD_PORT=8765
set METRICS_DB_PATH=data\neural_metrics.db
set GUI_FRAMEWORK=pyside6
set MISTRAL_API_KEY=nLKZEpq29OihnaArxV7s6KtzsNEiky2A

REM Check if PySide6 is installed, install if not
python -c "import PySide6" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing PySide6...
    pip install PySide6 pyqtgraph matplotlib numpy pandas
)

REM Set V7 feature flags
set ENABLE_NODE_CONSCIOUSNESS=true
set ENABLE_AUTOWIKI=true
set ENABLE_DREAM_MODE=true
set ENABLE_BREATH_DETECTION=true
set ENABLE_ONSITE_MEMORY=true

echo All systems ready. Starting LUMINA v7.5...
python src/v7.5/lumina_frontend.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: LUMINA v7.5 exited with code %ERRORLEVEL%
    echo Check the logs for more information.
    pause
) else (
    echo.
    echo LUMINA v7.5 session completed.
) 