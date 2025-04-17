@echo off
setlocal enabledelayedexpansion

echo.
echo ===================================
echo LUMINA V7.5 Unified System Launcher
echo ===================================
echo.

REM Define main directories and global system parameters (using quoted paths for safety)
set "MAIN_DIR=%CD%"
set "LOG_DIR=%MAIN_DIR%\logs"
set "DATA_DIR=%MAIN_DIR%\data"
set "KNOWLEDGE_DIR=%DATA_DIR%\knowledge"
set "CACHE_DIR=%DATA_DIR%\cache"
set "NEURAL_DIR=%DATA_DIR%\neural"
set "MEMORY_DIR=%DATA_DIR%\onsite_memory"
set "CONVERSATION_DIR=%DATA_DIR%\conversations"
set "CONSCIOUSNESS_DIR=%DATA_DIR%\consciousness"
set "AUTOWIKI_DIR=%DATA_DIR%\autowiki"
set "BACKUPS_DIR=%DATA_DIR%\backups"

REM Define shared database paths (using quoted paths for safety)
set "SHARED_DB_PATH=%DATA_DIR%\neural_metrics.db"
set "AUTOWIKI_DB_PATH=%KNOWLEDGE_DIR%\wiki_db.sqlite"
set "CONVERSATION_DB_PATH=%DATA_DIR%\conversations.db"
set "MEMORY_DB_PATH=%DATA_DIR%\memory.db"

REM Define network ports for interconnection
set DASHBOARD_PORT=8765
set AUTOWIKI_PORT=7525
set MEMORY_API_PORT=7526
set NEURAL_API_PORT=7527
set CONVERSATION_PORT=7528
set METRICS_PORT=7529
set HEALTH_PORT=8766

REM Create required directories if they don't exist (with error checking)
echo Creating required directories...
call :create_dir "%DATA_DIR%"
call :create_dir "%LOG_DIR%"
call :create_dir "%MEMORY_DIR%"
call :create_dir "%CONSCIOUSNESS_DIR%"
call :create_dir "%AUTOWIKI_DIR%"
call :create_dir "%DATA_DIR%\breath"
call :create_dir "%CONVERSATION_DIR%"
call :create_dir "%NEURAL_DIR%"
call :create_dir "%BACKUPS_DIR%"
call :create_dir "%KNOWLEDGE_DIR%"
call :create_dir "%CACHE_DIR%"

REM Define configuration settings for all components
set ENABLE_NODE_CONSCIOUSNESS=true
set ENABLE_AUTOWIKI=true
set ENABLE_DREAM_MODE=true
set ENABLE_BREATH_DETECTION=true
set ENABLE_ONSITE_MEMORY=true
set SHARED_DB_ENABLED=true
set ENABLE_AUTO_FETCH=true
set AUTO_FETCH_INTERVAL=10
set DEEP_LEARNING_DURATION=5
set NEURAL_INTEGRATION=true
set CHAT_TO_NEURAL=true

REM Check for Python installation and dependencies
echo Checking for Python and required dependencies...

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Verify PIP is available
python -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PIP is not available. Please check your Python installation.
    pause
    exit /b 1
)

REM Update pip to latest version
python -m pip install --upgrade pip >nul

REM Function to check and install Python packages
call :check_package PySide6 "PySide6 pyqtgraph matplotlib numpy pandas"
call :check_package mistralai "mistralai"
call :check_package flask "flask"
call :check_package requests "requests"
call :check_package sqlite3 "Database modules are built-in" nopip

REM Test Mistral API connectivity
echo Testing Mistral API connectivity...
python -c "from mistralai.client import MistralClient; from mistralai.models.chat_completion import ChatMessage; client = MistralClient(api_key='%MISTRAL_API_KEY%'); print('API connection successful!')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not connect to Mistral API. Check your API key and internet connection.
    echo Current API Key: %MISTRAL_API_KEY:~0,4%...%MISTRAL_API_KEY:~-4%
    set /p cont=Continue anyway? (Y/N): 
    if /i "!cont!" NEQ "Y" exit /b 1
) else (
    echo Mistral API connection verified successfully!
)

goto :main_script

REM Function to check and install a package
:check_package
setlocal
set package_name=%~1
set install_command=%~2
set nopip=%~3

echo Checking for %package_name%...
python -c "import %package_name%" 2>nul
if %ERRORLEVEL% NEQ 0 (
    if "%nopip%" == "nopip" (
        echo WARNING: %install_command%
    ) else (
        echo Installing %package_name%...
        python -m pip install %install_command%
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to install %package_name%.
            echo Please install manually with: pip install %install_command%
            pause
        ) else (
            echo Successfully installed %package_name%.
            
            REM If PySide6, update PATH to include scripts directories
            if "%package_name%" == "PySide6" (
                echo Updating PATH for PySide6...
                set "PATH=%PATH%;%APPDATA%\Python\Python310\Scripts"
                set "PATH=%PATH%;%USERPROFILE%\AppData\Local\Programs\Python\Python310\Scripts"
                for /f "tokens=*" %%a in ('where python') do (
                    set "PYTHON_DIR=%%~dpa"
                )
                set "PATH=%PATH%;!PYTHON_DIR!Scripts"
            )
        )
    )
) else (
    echo %package_name% is already installed.
)
endlocal & goto :eof

:create_dir
if not exist %1 (
    mkdir %1
    if !ERRORLEVEL! NEQ 0 (
        echo Error creating directory %1
    ) else (
        echo Created directory %1
    )
) else (
    echo Directory %1 already exists
)
goto :eof

:main_script
REM Load API key from .env file if it exists
if exist .env (
    echo Loading API key from .env file...
    FOR /F "tokens=2 delims==" %%a in ('type .env ^| findstr "MISTRAL_API_KEY"') do (
        set MISTRAL_API_KEY=%%a
    )
) else (
    REM Fallback to hardcoded key if no .env file
    set MISTRAL_API_KEY=nLKZEpq29OihnaArxV7s6KtzsNEiky2A
)

echo Using Mistral API Key: %MISTRAL_API_KEY:~0,4%...%MISTRAL_API_KEY:~-4%

REM Set additional environment variables for system integration
set "PYTHONPATH=%MAIN_DIR%"
set "CONNECTION_STRING=sqlite:///%SHARED_DB_PATH%?check_same_thread=False"
set "GUI_FRAMEWORK=pyside6"

goto :menu

:menu
cls
echo.
echo LUMINA V7.5 System
echo ==================
echo.
echo [1] Start Complete System (All Components)
echo [2] Start Chat Interface Only
echo [3] Start Neural Seed System
echo [4] Start System Monitor
echo [5] Start Database Connector
echo [6] Start AutoWiki
echo [7] Start Holographic UI
echo [8] Start Knowledge CI/CD System
echo [V] Verify System Connections
echo [Q] Quit
echo.
set /p choice=Enter your choice: 

if "%choice%"=="1" (
    call :start_all
    goto menu
) else if "%choice%"=="2" (
    call :start_chat
    goto menu
) else if "%choice%"=="3" (
    call :start_neural_seed
    goto menu
) else if "%choice%"=="4" (
    call :start_system_monitor
    goto menu
) else if "%choice%"=="5" (
    call :start_database
    goto menu
) else if "%choice%"=="6" (
    call :start_autowiki
    goto menu
) else if "%choice%"=="7" (
    call :start_holographic
    goto menu
) else if "%choice%"=="8" (
    call :start_knowledge_cicd
    goto menu
) else if /i "%choice%"=="v" (
    call :verify_connections
    pause
    goto menu
) else if /i "%choice%"=="q" (
    echo Exiting LUMINA V7.5 System...
    exit /b 0
) else (
    echo Invalid choice!
    pause
    goto menu
)

REM Function to verify system connections
:verify_connections
echo.
echo ====================================
echo Verifying System Connection Points
echo ====================================
echo.
echo Database paths:
echo - Shared DB: %SHARED_DB_PATH%
echo - AutoWiki DB: %AUTOWIKI_DB_PATH%
echo - Conversation DB: %CONVERSATION_DB_PATH%
echo - Memory DB: %MEMORY_DB_PATH%
echo.
echo Network ports:
echo - Dashboard: %DASHBOARD_PORT%
echo - AutoWiki: %AUTOWIKI_PORT%
echo - Memory API: %MEMORY_API_PORT%
echo - Neural API: %NEURAL_API_PORT%
echo - Conversation: %CONVERSATION_PORT%
echo - Metrics: %METRICS_PORT%
echo - Health: %HEALTH_PORT%
echo.

REM Check if ports are in use
call :check_port %DASHBOARD_PORT% "Dashboard"
call :check_port %AUTOWIKI_PORT% "AutoWiki"
call :check_port %MEMORY_API_PORT% "Memory API"
call :check_port %NEURAL_API_PORT% "Neural API"
call :check_port %HEALTH_PORT% "Health"

REM Check database files
if exist "%SHARED_DB_PATH%" (
    echo FOUND: Shared database exists
) else (
    echo WARNING: Shared database does not exist yet. It will be created when needed.
)

echo.
echo Verification complete
goto :eof

REM Check if a port is in use
:check_port
setlocal
set port=%~1
set service=%~2
netstat -an | find ":%port% " >nul
if %ERRORLEVEL% EQU 0 (
    echo WARNING: Port %port% (%service%) is already in use!
) else (
    echo OK: Port %port% (%service%) is available
)
endlocal & goto :eof

REM Function to start all components
:start_all
echo Starting all LUMINA components...
call :start_neural_seed
timeout /t 3 /nobreak > nul
call :start_database
timeout /t 2 /nobreak > nul
call :start_autowiki
timeout /t 2 /nobreak > nul
call :start_chat
timeout /t 2 /nobreak > nul
call :start_system_monitor
timeout /t 2 /nobreak > nul
call :start_holographic
timeout /t 2 /nobreak > nul
call :start_knowledge_cicd
goto :eof

REM Function to start the Chat Interface
:start_chat
echo Starting Chat Interface...
call :log_action "Starting Chat Interface"

if exist src\v7.5\lumina_frontend.py (
    echo Found lumina_frontend.py in v7.5 directory
    start "LUMINA Chat Interface" cmd /c "cd src\v7.5 && python lumina_frontend.py --conversation-db="%CONVERSATION_DB_PATH%" --memory-db="%MEMORY_DB_PATH%" --shared-db="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
    goto :chat_started
) else if exist src\v7_5\lumina_frontend.py (
    echo Found lumina_frontend.py in v7_5 directory
    start "LUMINA Chat Interface" cmd /c "cd src\v7_5 && python lumina_frontend.py --conversation-db="%CONVERSATION_DB_PATH%" --memory-db="%MEMORY_DB_PATH%" --shared-db="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
    goto :chat_started
) else (
    echo WARNING: Could not find lumina_frontend.py!
    
    REM Try alternate files
    if exist src\chat_with_system.py (
        echo Found chat_with_system.py, using as alternative
        start "LUMINA Chat Interface" cmd /c "python src\chat_with_system.py --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
        goto :chat_started
    ) else if exist src\central_language_node.py (
        echo Found central_language_node.py, using as alternative
        start "LUMINA Chat Interface" cmd /c "python src\central_language_node.py --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
        goto :chat_started
    ) else (
        echo ERROR: No suitable chat interface module found.
        call :log_action "ERROR: No suitable chat interface module found"
        pause
        goto :eof
    )
)

:chat_started
call :log_action "Chat Interface started"
goto :eof

REM Function to start the Neural Seed System
:start_neural_seed
echo Starting Neural Seed System...
call :log_action "Starting Neural Seed System"

if exist src\neural\seed.py (
    echo Found neural seed in neural directory
    start "LUMINA Neural Seed" cmd /c "python src\neural\seed.py --db-path="%SHARED_DB_PATH%" --port=%NEURAL_API_PORT% --consciousness-dir="%CONSCIOUSNESS_DIR%" --neural-dir="%NEURAL_DIR%" --enable-chat-input=true --process-chat-data=true --listen-mode=true && pause"
    call :log_action "Started Neural Seed from neural directory"
) else if exist src\seed.py (
    echo Found seed.py in src directory
    start "LUMINA Neural Seed" cmd /c "python src\seed.py --db-path="%SHARED_DB_PATH%" --port=%NEURAL_API_PORT% --consciousness-dir="%CONSCIOUSNESS_DIR%" --neural-dir="%NEURAL_DIR%" --enable-chat-input=true --process-chat-data=true --listen-mode=true && pause"
    call :log_action "Started Neural Seed from src directory"
) else (
    echo WARNING: Could not find neural seed module!
    call :log_action "ERROR: Could not find neural seed module"
)
goto :eof

REM Function to start the System Monitor
:start_system_monitor
echo Starting System Monitor...
call :log_action "Starting System Monitor"

if exist src\v7.5\system_monitor.py (
    echo Found system_monitor.py in v7.5 directory
    start "LUMINA System Monitor" cmd /c "cd src\v7.5 && python system_monitor.py --db-path="%SHARED_DB_PATH%" --port=%METRICS_PORT% && pause"
    call :log_action "Started System Monitor from v7.5 directory"
) else if exist src\v7_5\system_monitor.py (
    echo Found system_monitor.py in v7_5 directory
    start "LUMINA System Monitor" cmd /c "cd src\v7_5 && python system_monitor.py --db-path="%SHARED_DB_PATH%" --port=%METRICS_PORT% && pause"
    call :log_action "Started System Monitor from v7_5 directory"
) else if exist src\monitoring (
    echo Checking monitoring directory...
    for %%f in (src\monitoring\*.py) do (
        echo Found %%f
        start "LUMINA System Monitor" cmd /c "python %%f --db-path="%SHARED_DB_PATH%" --port=%METRICS_PORT% && pause"
        call :log_action "Started System Monitor from %%f"
        goto :monitor_found
    )
    echo No monitoring scripts found.
    :monitor_found
) else (
    echo WARNING: Could not find system monitor module!
    call :log_action "ERROR: Could not find system monitor module"
)
goto :eof

REM Logging function
:log_action
echo %date% %time% - %~1 >> "%LOG_DIR%\launcher_log.txt"
goto :eof

REM Function to start the Database Connector
:start_database
echo Starting Database Connector...
call :log_action "Starting Database Connector"

if exist src\v7.5\database_connector.py (
    echo Found database_connector.py in v7.5 directory
    start "LUMINA Database" cmd /c "cd src\v7.5 && python database_connector.py --db-path="%SHARED_DB_PATH%" --memory-db="%MEMORY_DB_PATH%" --conversation-db="%CONVERSATION_DB_PATH%" --port=%MEMORY_API_PORT% && pause"
    call :log_action "Started Database Connector from v7.5 directory"
) else if exist src\v7_5\database_connector.py (
    echo Found database_connector.py in v7_5 directory
    start "LUMINA Database" cmd /c "cd src\v7_5 && python database_connector.py --db-path="%SHARED_DB_PATH%" --memory-db="%MEMORY_DB_PATH%" --conversation-db="%CONVERSATION_DB_PATH%" --port=%MEMORY_API_PORT% && pause"
    call :log_action "Started Database Connector from v7_5 directory"
) else if exist src\memory_api_server.py (
    echo Using memory_api_server.py as alternative
    start "LUMINA Database" cmd /c "python src\memory_api_server.py --db-path="%SHARED_DB_PATH%" --port=%MEMORY_API_PORT% && pause"
    call :log_action "Started memory_api_server.py as alternative"
) else (
    echo WARNING: Could not find database connector module!
    call :log_action "ERROR: Could not find database connector module"
)
goto :eof

REM Function to start AutoWiki
:start_autowiki
echo Starting AutoWiki...
call :log_action "Starting AutoWiki"

REM First check if we have the standalone launcher
if exist run_autowiki_v7.5.bat (
    echo Found standalone AutoWiki launcher
    start "LUMINA AutoWiki" cmd /c "set "SHARED_DB_PATH=%SHARED_DB_PATH%" && set "KNOWLEDGE_DIR=%KNOWLEDGE_DIR%" && set "AUTOWIKI_DB_PATH=%AUTOWIKI_DB_PATH%" && set "AUTOWIKI_PORT=%AUTOWIKI_PORT%" && set "CACHE_DIR=%CACHE_DIR%" && set AUTO_FETCH_INTERVAL=10 && set DEEP_LEARNING_DURATION=5 && run_autowiki_v7.5.bat"
    call :log_action "Started standalone AutoWiki launcher"
    goto :eof
)

REM Check for the module directly
if exist src\v7_5\autowiki.py (
    echo Found autowiki.py in v7_5 directory
    
    start "LUMINA AutoWiki" cmd /c "cd src\v7_5 && python autowiki.py --port=%AUTOWIKI_PORT% --knowledge-dir="%KNOWLEDGE_DIR%" --db-path="%AUTOWIKI_DB_PATH%" --shared-db --shared-db-path="%SHARED_DB_PATH%" --auto-fetch-interval=10 --deep-learning-duration=5 --cache-dir="%CACHE_DIR%" && pause"
    call :log_action "Started AutoWiki from v7_5 directory"
    goto :eof
)

echo Looking for wiki modules...
for %%f in (src\v7.5\*wiki*.py src\v7_5\*wiki*.py src\*wiki*.py) do (
    echo Found %%f
    start "LUMINA AutoWiki" cmd /c "python %%f --port=%AUTOWIKI_PORT% --knowledge-dir="%KNOWLEDGE_DIR%" --db-path="%AUTOWIKI_DB_PATH%" --shared-db --shared-db-path="%SHARED_DB_PATH%" --auto-fetch-interval=10 --deep-learning-duration=5 --cache-dir="%CACHE_DIR%" && pause"
    call :log_action "Started AutoWiki from %%f"
    goto :eof
)

echo WARNING: Could not find autowiki module!
call :log_action "ERROR: Could not find autowiki module"

echo Creating standalone AutoWiki launcher...
echo @echo off > create_autowiki_v7.5.bat
echo setlocal enabledelayedexpansion >> create_autowiki_v7.5.bat
echo echo Creating AutoWiki v7.5 launcher... >> create_autowiki_v7.5.bat
echo copy run_unified_v7.5_fixed.bat run_autowiki_v7.5.bat /Y ^> nul >> create_autowiki_v7.5.bat
echo echo Standalone AutoWiki launcher created. >> create_autowiki_v7.5.bat
echo pause >> create_autowiki_v7.5.bat

start "Creating AutoWiki Launcher" cmd /c "create_autowiki_v7.5.bat"
echo Please run the unified script again after the launcher is created.
call :log_action "Created standalone AutoWiki launcher script"
goto :eof

REM Function to start Holographic UI
:start_holographic
echo Starting Holographic UI...
call :log_action "Starting Holographic UI"

set FOUND_UI=false

REM Try the direct approach first
if exist run_holographic_frontend.py (
    echo Found run_holographic_frontend.py in root directory
    start "LUMINA Holographic UI" cmd /c "python run_holographic_frontend.py --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    call :log_action "Started Holographic UI from root directory"
    goto :eof
)

REM Try common UI directories
if exist src\ui\holographic.py (
    echo Found holographic.py in ui directory
    start "LUMINA Holographic UI" cmd /c "python src\ui\holographic.py --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    call :log_action "Started Holographic UI from ui directory"
    goto :eof
)

if exist src\gui\holographic.py (
    echo Found holographic.py in gui directory
    start "LUMINA Holographic UI" cmd /c "python src\gui\holographic.py --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    call :log_action "Started Holographic UI from gui directory"
    goto :eof
)

REM Look for v7.5 UI files
if exist src\v7.5\ui\holographic.py (
    echo Found holographic.py in v7.5/ui directory
    start "LUMINA Holographic UI" cmd /c "python src\v7.5\ui\holographic.py --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    call :log_action "Started Holographic UI from v7.5/ui directory"
    goto :eof
)

if exist src\v7_5\ui\holographic.py (
    echo Found holographic.py in v7_5/ui directory
    start "LUMINA Holographic UI" cmd /c "python src\v7_5\ui\holographic.py --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    call :log_action "Started Holographic UI from v7_5/ui directory"
    goto :eof
)

REM Search for any UI files that might work
echo Searching for UI files...
for %%f in (src\ui\*.py src\gui\*.py src\v7.5\ui\*.py src\v7_5\ui\*.py) do (
    echo Found potential UI file: %%f
    start "LUMINA Holographic UI" cmd /c "python %%f --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    call :log_action "Started Holographic UI from %%f"
    goto :eof
)

REM Try PySide6 based files directly
if exist src\visualization\*.py (
    for %%f in (src\visualization\*.py) do (
        echo Found visualization file: %%f
        start "LUMINA Holographic UI" cmd /c "python %%f --gui-framework=%GUI_FRAMEWORK% --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
        set FOUND_UI=true
        call :log_action "Started Holographic UI from visualization file: %%f"
        goto :eof
    )
)

if "%FOUND_UI%"=="false" (
    echo WARNING: Could not find holographic UI module. Will create a simple one.
    call :log_action "WARNING: Creating a simple holographic UI fallback"
    
    REM Create a minimal PySide6 launcher file
    echo import sys > holographic_launcher.py
    echo import os >> holographic_launcher.py
    echo import time >> holographic_launcher.py
    echo import argparse >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo # Parse command line arguments >> holographic_launcher.py
    echo parser = argparse.ArgumentParser(description='LUMINA Holographic UI') >> holographic_launcher.py
    echo parser.add_argument('--gui-framework', default='pyside6', help='GUI framework to use') >> holographic_launcher.py
    echo parser.add_argument('--db-path', default='data/neural_metrics.db', help='Path to database') >> holographic_launcher.py
    echo parser.add_argument('--neural-port', type=int, default=7527, help='Neural API port') >> holographic_launcher.py
    echo parser.add_argument('--memory-port', type=int, default=7526, help='Memory API port') >> holographic_launcher.py
    echo parser.add_argument('--metrics-port', type=int, default=7529, help='Metrics port') >> holographic_launcher.py
    echo args = parser.parse_args() >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo try: >> holographic_launcher.py
    echo     from PySide6 import QtWidgets, QtCore, QtGui >> holographic_launcher.py
    echo     print("Using PySide6 for Holographic UI") >> holographic_launcher.py
    echo except ImportError as e: >> holographic_launcher.py
    echo     print(f"PySide6 import error: {e}") >> holographic_launcher.py
    echo     print("Installing PySide6...") >> holographic_launcher.py
    echo     import subprocess >> holographic_launcher.py
    echo     subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"]) >> holographic_launcher.py
    echo     from PySide6 import QtWidgets, QtCore, QtGui >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo class LuminaHolographicUI(QtWidgets.QMainWindow): >> holographic_launcher.py
    echo     def __init__(self, args): >> holographic_launcher.py
    echo         super().__init__() >> holographic_launcher.py
    echo         self.setWindowTitle("LUMINA v7.5 Holographic Interface") >> holographic_launcher.py
    echo         self.resize(800, 600) >> holographic_launcher.py
    echo         self.db_path = args.db_path >> holographic_launcher.py
    echo         self.neural_port = args.neural_port >> holographic_launcher.py
    echo         self.memory_port = args.memory_port >> holographic_launcher.py
    echo         self.metrics_port = args.metrics_port >> holographic_launcher.py
    echo         self.setup_ui() >> holographic_launcher.py
    echo         self.setup_connections() >> holographic_launcher.py
    echo         self.update_timer = QtCore.QTimer(self) >> holographic_launcher.py
    echo         self.update_timer.timeout.connect(self.update_status) >> holographic_launcher.py
    echo         self.update_timer.start(5000)  # Update every 5 seconds >> holographic_launcher.py
    echo         self.update_status() >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo     def setup_ui(self): >> holographic_launcher.py
    echo         self.central_widget = QtWidgets.QWidget() >> holographic_launcher.py
    echo         self.setCentralWidget(self.central_widget) >> holographic_launcher.py
    echo         self.layout = QtWidgets.QVBoxLayout(self.central_widget) >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         # Header >> holographic_launcher.py
    echo         self.header_label = QtWidgets.QLabel("LUMINA v7.5 Holographic Interface") >> holographic_launcher.py
    echo         font = self.header_label.font() >> holographic_launcher.py
    echo         font.setPointSize(20) >> holographic_launcher.py
    echo         font.setBold(True) >> holographic_launcher.py
    echo         self.header_label.setFont(font) >> holographic_launcher.py
    echo         self.header_label.setAlignment(QtCore.Qt.AlignCenter) >> holographic_launcher.py
    echo         self.layout.addWidget(self.header_label) >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         # Status panel >> holographic_launcher.py
    echo         self.status_group = QtWidgets.QGroupBox("System Status") >> holographic_launcher.py
    echo         self.status_layout = QtWidgets.QFormLayout(self.status_group) >> holographic_launcher.py
    echo         self.neural_status = QtWidgets.QLabel("Checking...") >> holographic_launcher.py
    echo         self.memory_status = QtWidgets.QLabel("Checking...") >> holographic_launcher.py
    echo         self.metrics_status = QtWidgets.QLabel("Checking...") >> holographic_launcher.py
    echo         self.db_status = QtWidgets.QLabel("Checking...") >> holographic_launcher.py
    echo         self.status_layout.addRow("Neural API:", self.neural_status) >> holographic_launcher.py
    echo         self.status_layout.addRow("Memory API:", self.memory_status) >> holographic_launcher.py
    echo         self.status_layout.addRow("Metrics API:", self.metrics_status) >> holographic_launcher.py
    echo         self.status_layout.addRow("Database:", self.db_status) >> holographic_launcher.py
    echo         self.layout.addWidget(self.status_group) >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         # System controls >> holographic_launcher.py
    echo         self.control_group = QtWidgets.QGroupBox("System Controls") >> holographic_launcher.py
    echo         self.control_layout = QtWidgets.QVBoxLayout(self.control_group) >> holographic_launcher.py
    echo         self.refresh_btn = QtWidgets.QPushButton("Refresh Status") >> holographic_launcher.py
    echo         self.refresh_btn.clicked.connect(self.update_status) >> holographic_launcher.py
    echo         self.control_layout.addWidget(self.refresh_btn) >> holographic_launcher.py
    echo         self.layout.addWidget(self.control_group) >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         # Log view >> holographic_launcher.py
    echo         self.log_group = QtWidgets.QGroupBox("System Log") >> holographic_launcher.py
    echo         self.log_layout = QtWidgets.QVBoxLayout(self.log_group) >> holographic_launcher.py
    echo         self.log_text = QtWidgets.QTextEdit() >> holographic_launcher.py
    echo         self.log_text.setReadOnly(True) >> holographic_launcher.py
    echo         self.log_layout.addWidget(self.log_text) >> holographic_launcher.py
    echo         self.layout.addWidget(self.log_group) >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo     def setup_connections(self): >> holographic_launcher.py
    echo         self.log("Initializing connections...") >> holographic_launcher.py
    echo         self.log(f"Database path: {self.db_path}") >> holographic_launcher.py
    echo         self.log(f"Neural API port: {self.neural_port}") >> holographic_launcher.py
    echo         self.log(f"Memory API port: {self.memory_port}") >> holographic_launcher.py
    echo         self.log(f"Metrics port: {self.metrics_port}") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo     def update_status(self): >> holographic_launcher.py
    echo         # Check database >> holographic_launcher.py
    echo         if os.path.exists(self.db_path): >> holographic_launcher.py
    echo             self.db_status.setText("Connected") >> holographic_launcher.py
    echo             self.db_status.setStyleSheet("color: green") >> holographic_launcher.py
    echo         else: >> holographic_launcher.py
    echo             self.db_status.setText("Not Found") >> holographic_launcher.py
    echo             self.db_status.setStyleSheet("color: red") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         # Simple port checks - in a real app, would actually try to connect >> holographic_launcher.py
    echo         import socket >> holographic_launcher.py
    echo         def check_port(port): >> holographic_launcher.py
    echo             s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) >> holographic_launcher.py
    echo             try: >> holographic_launcher.py
    echo                 s.connect(('localhost', port)) >> holographic_launcher.py
    echo                 s.close() >> holographic_launcher.py
    echo                 return True >> holographic_launcher.py
    echo             except: >> holographic_launcher.py
    echo                 return False >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         if check_port(self.neural_port): >> holographic_launcher.py
    echo             self.neural_status.setText("Connected") >> holographic_launcher.py
    echo             self.neural_status.setStyleSheet("color: green") >> holographic_launcher.py
    echo         else: >> holographic_launcher.py
    echo             self.neural_status.setText("Not Available") >> holographic_launcher.py
    echo             self.neural_status.setStyleSheet("color: red") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         if check_port(self.memory_port): >> holographic_launcher.py
    echo             self.memory_status.setText("Connected") >> holographic_launcher.py
    echo             self.memory_status.setStyleSheet("color: green") >> holographic_launcher.py
    echo         else: >> holographic_launcher.py
    echo             self.memory_status.setText("Not Available") >> holographic_launcher.py
    echo             self.memory_status.setStyleSheet("color: red") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         if check_port(self.metrics_port): >> holographic_launcher.py
    echo             self.metrics_status.setText("Connected") >> holographic_launcher.py
    echo             self.metrics_status.setStyleSheet("color: green") >> holographic_launcher.py
    echo         else: >> holographic_launcher.py
    echo             self.metrics_status.setText("Not Available") >> holographic_launcher.py
    echo             self.metrics_status.setStyleSheet("color: red") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo         self.log(f"Status updated: {time.strftime('%%Y-%%m-%%d %%H:%%M:%%S')}") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo     def log(self, message): >> holographic_launcher.py
    echo         timestamp = time.strftime("%%Y-%%m-%%d %%H:%%M:%%S") >> holographic_launcher.py
    echo         self.log_text.append(f"{timestamp} - {message}") >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo if __name__ == "__main__": >> holographic_launcher.py
    echo     app = QtWidgets.QApplication(sys.argv) >> holographic_launcher.py
    echo     ui = LuminaHolographicUI(args) >> holographic_launcher.py
    echo     ui.show() >> holographic_launcher.py
    echo     sys.exit(app.exec_()) >> holographic_launcher.py
    
    start "LUMINA Holographic UI" cmd /c "python holographic_launcher.py --db-path="%SHARED_DB_PATH%" --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
)

goto :eof

REM Function to start Knowledge CI/CD System
:start_knowledge_cicd
echo Starting Knowledge CI/CD System...
call :log_action "Starting Knowledge CI/CD System"

REM Check standalone launcher first
if exist run_knowledge_ci_cd_integrated.bat (
    echo Found standalone Knowledge CI/CD launcher
    start "LUMINA Knowledge CI/CD" cmd /c "set "SHARED_DB_PATH=%SHARED_DB_PATH%" && set "KNOWLEDGE_DIR=%KNOWLEDGE_DIR%" && set "HEALTH_PORT=%HEALTH_PORT%" && set "MISTRAL_API_KEY=%MISTRAL_API_KEY%" && run_knowledge_ci_cd_integrated.bat"
    call :log_action "Started standalone Knowledge CI/CD launcher"
    goto :eof
)

REM Look for the appropriate script
if exist src\v8\knowledge_ci_cd_integrated.py (
    echo Found knowledge_ci_cd_integrated.py in v8 directory
    start "LUMINA Knowledge CI/CD" cmd /c "python src\v8\knowledge_ci_cd_integrated.py --health-check-port=%HEALTH_PORT% --db-path="%SHARED_DB_PATH%" --knowledge-dir="%KNOWLEDGE_DIR%" --backup-dir="%BACKUPS_DIR%""
    call :log_action "Started Knowledge CI/CD from v8 directory"
    goto :eof
)

REM Try to find any CI/CD related scripts
for %%f in (src\*knowledge*ci*cd*.py src\v8\*knowledge*.py) do (
    echo Found potential Knowledge CI/CD file: %%f
    start "LUMINA Knowledge CI/CD" cmd /c "python %%f --health-check-port=%HEALTH_PORT% --db-path="%SHARED_DB_PATH%" --knowledge-dir="%KNOWLEDGE_DIR%" --backup-dir="%BACKUPS_DIR%""
    call :log_action "Started Knowledge CI/CD from %%f"
    goto :eof
)

echo WARNING: Could not find Knowledge CI/CD module!
call :log_action "ERROR: Could not find Knowledge CI/CD module"
echo Creating standalone Knowledge CI/CD launcher script...

echo @echo off > knowledge_ci_cd_integrated.py.bat
echo echo Starting Integrated Knowledge CI/CD System for v8 >> knowledge_ci_cd_integrated.py.bat
echo. >> knowledge_ci_cd_integrated.py.bat
echo set "PYTHONPATH=%MAIN_DIR%" >> knowledge_ci_cd_integrated.py.bat
echo set "MISTRAL_API_KEY=%MISTRAL_API_KEY%" >> knowledge_ci_cd_integrated.py.bat
echo set "SHARED_DB_PATH=%SHARED_DB_PATH%" >> knowledge_ci_cd_integrated.py.bat
echo set "KNOWLEDGE_DIR=%KNOWLEDGE_DIR%" >> knowledge_ci_cd_integrated.py.bat
echo set "HEALTH_PORT=%HEALTH_PORT%" >> knowledge_ci_cd_integrated.py.bat
echo. >> knowledge_ci_cd_integrated.py.bat
echo if not exist "data\backups" mkdir "data\backups" >> knowledge_ci_cd_integrated.py.bat
echo. >> knowledge_ci_cd_integrated.py.bat
echo python -c "import flask" ^>nul 2^>^&1 >> knowledge_ci_cd_integrated.py.bat
echo if %%ERRORLEVEL%% NEQ 0 ( >> knowledge_ci_cd_integrated.py.bat
echo     echo Installing Flask for health monitoring... >> knowledge_ci_cd_integrated.py.bat
echo     pip install flask >> knowledge_ci_cd_integrated.py.bat
echo ) >> knowledge_ci_cd_integrated.py.bat
echo. >> knowledge_ci_cd_integrated.py.bat
echo echo Health monitoring will be available at http://localhost:%%HEALTH_PORT%%/health >> knowledge_ci_cd_integrated.py.bat
echo echo Press Ctrl+C to stop the service... >> knowledge_ci_cd_integrated.py.bat
echo pause >> knowledge_ci_cd_integrated.py.bat

echo Knowledge CI/CD standalone launcher created as knowledge_ci_cd_integrated.py.bat
goto :eof

goto menu 