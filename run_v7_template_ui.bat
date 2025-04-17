@echo off
echo ===================================
echo   LUMINA V7 Plugin Template Launcher
echo ===================================
echo.

:: Create necessary directories
if not exist "logs" mkdir logs
if not exist "plugins" mkdir plugins
if not exist "data\consciousness" mkdir data\consciousness

:: Set PYTHONPATH to include current directory and src
set PYTHONPATH=%CD%
set PYTHONPATH=%PYTHONPATH%;%CD%\src
echo Set PYTHONPATH to: %PYTHONPATH%

:: Set plugin configuration environment variables
set TEMPLATE_PLUGINS_ENABLED=true
set TEMPLATE_PLUGINS_DIRS=plugins;src\v7\plugins;src\plugins
set TEMPLATE_AUTO_LOAD_PLUGINS=consciousness_system_plugin.py;mistral_plugin.py;neural_network_plugin.py
set TEMPLATE_TITLE=LUMINA V7 Enhanced System
echo Plugin configuration set.

:: Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python and make sure it's in your PATH
    echo.
    pause
    exit /b 1
)

:: Verify PySide6 is installed
echo Checking for PySide6...
python -c "import PySide6" >nul 2>nul
if %errorlevel% neq 0 (
    echo PySide6 not found. Installing...
    python -m pip install PySide6
)

:: Check for required packages
echo Checking for required packages...
python -c "import numpy" >nul 2>nul
if %errorlevel% neq 0 (
    echo NumPy not found. Installing...
    python -m pip install numpy
)

:: Run the template application
echo.
echo Launching LUMINA V7 Plugin Template...
echo Loading plugins: %TEMPLATE_AUTO_LOAD_PLUGINS%
echo.
python v7_pyside6_template.py --plugins-enabled --auto-load-plugins
set LAUNCH_RESULT=%errorlevel%

if %LAUNCH_RESULT% neq 0 (
    echo.
    echo Launch failed with exit code %LAUNCH_RESULT%
    echo Check logs/v7_template.log for details
    echo.
    pause
    exit /b %LAUNCH_RESULT%
)

echo.
echo LUMINA V7 Plugin Template closed successfully.
echo Press any key to exit...
pause >nul 