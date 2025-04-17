@echo off
echo Starting LUMINA V7 Dashboard...

:: Set environment variables
set PYTHONPATH=%CD%;%CD%\src
set DASHBOARD_PORT=5679
set METRICS_DB_PATH=data/neural_metrics.db
set V7_CONNECTION_PORT=5678
set GUI_FRAMEWORK=PySide6
set NN_WEIGHT=0.6
set LLM_WEIGHT=0.7
set CONFIG_DIR=config

:: If Mistral API key is available in .env, use it
if exist .env (
    for /f "tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="MISTRAL_API_KEY" (
            set MISTRAL_API_KEY=%%b
            echo Using Mistral API key from .env file
        )
    )
)

:: Create necessary directories
if not exist data\neural_metrics mkdir data\neural_metrics
if not exist logs mkdir logs
if not exist config mkdir config
if not exist output\exports mkdir output\exports

:: Check if configuration files exist, create them if not
if not exist %CONFIG_DIR%\dashboard_config.json (
    echo Configuration files not found. Please run install_dashboard_dependencies.bat first.
    echo.
    echo Or run: python src/visualization/create_default_configs.py
    pause
    exit /b 1
)

:: Check for dependencies
echo Checking dashboard dependencies...
python src/visualization/check_dashboard_requirements.py --install --gui-framework %GUI_FRAMEWORK%
if errorlevel 1 (
    echo Failed to install required dependencies.
    echo Please run install_dashboard_dependencies.bat to install all required packages.
    pause
    exit /b 1
)

:: Start the dashboard
echo Launching Dashboard...
echo Using configuration files from %CONFIG_DIR%

:: If using virtual environment
if exist venv (
    call venv\Scripts\activate.bat
)

python src/visualization/create_dashboard_qt.py ^
  --v7-port %V7_CONNECTION_PORT% ^
  --db-path %METRICS_DB_PATH% ^
  --config-dir %CONFIG_DIR% ^
  --nn-weight %NN_WEIGHT% ^
  --llm-weight %LLM_WEIGHT% ^
  --gui-framework %GUI_FRAMEWORK%

:: If using mock mode
:: python src/visualization/create_dashboard_qt.py --mock --config-dir %CONFIG_DIR%

:: Deactivate virtual environment if used
if exist venv (
    call venv\Scripts\deactivate.bat
)

echo Dashboard has been closed.
pause 