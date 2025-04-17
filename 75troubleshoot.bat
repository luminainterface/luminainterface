@echo off
echo LUMINA v7.5 Troubleshooting Script
echo ================================
echo.

:: Check Python installation
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)
echo.

:: Check required packages
echo Checking required packages...
python -c "import pkg_resources; pkg_resources.require(['PySide6', 'qasync', 'requests', 'beautifulsoup4', 'wikipedia'])"
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install required packages
        exit /b 1
    )
)
echo.

:: Create required directories
echo Setting up required directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist models mkdir models
if not exist cache mkdir cache
if not exist data\wiki mkdir data\wiki
if not exist data\exports mkdir data\exports
if not exist data\conversations mkdir data\conversations
if not exist data\version_cache mkdir data\version_cache

:: Initialize required files
if not exist data\mistral_learning.json (
    echo Creating empty mistral_learning.json...
    echo {"entries": [], "autowiki": []} > data\mistral_learning.json
)

if not exist data\wiki\knowledge_base.json (
    echo Creating empty knowledge_base.json...
    echo {"entries": []} > data\wiki\knowledge_base.json
)

if not exist data\version_cache\compatibility_matrix.json (
    echo Creating compatibility matrix...
    echo {"v7.5": ["v5.0", "v6.0", "v7.0"]} > data\version_cache\compatibility_matrix.json
)
echo.

:: Check PYTHONPATH
echo Checking PYTHONPATH...
set PYTHONPATH=%CD%;%PYTHONPATH%
echo Current directory added to PYTHONPATH
echo.

:: Check source directory structure
echo Checking source directory structure...
if not exist src\v7_5 (
    echo ERROR: src\v7_5 directory not found
    exit /b 1
)

if not exist src\v7_5\minimal_gui.py (
    echo ERROR: minimal_gui.py not found
    exit /b 1
)

if not exist src\v7_5\lumina_core.py (
    echo ERROR: lumina_core.py not found
    exit /b 1
)

if not exist src\v7_5\central_node.py (
    echo ERROR: central_node.py not found
    exit /b 1
)

if not exist src\v7_5\signal_system.py (
    echo ERROR: signal_system.py not found
    exit /b 1
)

if not exist src\v7_5\version_bridge.py (
    echo ERROR: version_bridge.py not found
    exit /b 1
)

if not exist src\v7_5\signal_component.py (
    echo ERROR: signal_component.py not found
    exit /b 1
)

if not exist src\v7_5\version_transform.py (
    echo ERROR: version_transform.py not found
    exit /b 1
)

if not exist src\v7_5\bridge_monitor.py (
    echo ERROR: bridge_monitor.py not found
    exit /b 1
)

echo Source directory structure OK
echo.

:: Set environment variables
echo Setting up environment variables...
set MISTRAL_API_KEY=nLKZEpq29OihnaArxV7s6KtzsNEiky2A
set MODEL_NAME=mistral-medium
set MOCK_MODE=false
set LLM_WEIGHT=0.7
set NN_WEIGHT=0.3
set LLM_TEMPERATURE=0.7
set LLM_TOP_P=0.9
set VERSION_BRIDGE_ENABLED=true
set SIGNAL_SYSTEM_DEBUG=true
set BRIDGE_MONITOR_ENABLED=true
echo Environment variables set
echo.

:: Initialize signal system
echo Initializing signal system...
python -m src.v7_5.init_signal_system
if errorlevel 1 (
    echo ERROR: Failed to initialize signal system
    exit /b 1
)
echo Signal system OK
echo.

:: Initialize version bridge
echo Initializing version bridge...
python -m src.v7_5.init_version_bridge
if errorlevel 1 (
    echo ERROR: Failed to initialize version bridge
    exit /b 1
)
echo Version bridge OK
echo.

:: Check core components
echo Checking core components...
python -m src.v7_5.init_core
if errorlevel 1 (
    echo ERROR: Failed to initialize LUMINA Core
    exit /b 1
)
echo Core components OK
echo.

:: Verify component connections
echo Verifying component connections...
python -m src.v7_5.verify_connections
if errorlevel 1 (
    echo ERROR: Component connection verification failed
    exit /b 1
)
echo Component connections OK
echo.

:: Run application with enhanced logging
echo Running LUMINA v7.5...
echo.
echo [Debug Mode]
set PYTHONPATH=%CD%;%PYTHONPATH%
python -m src.v7_5.minimal_gui

echo.
echo If the application failed to start, check the error messages above
:: Only pause on successful completion
if errorlevel 0 pause 