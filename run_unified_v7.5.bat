@echo off
setlocal enabledelayedexpansion

echo.
echo ===================================
echo LUMINA V7.5 Unified System Launcher
echo ===================================
echo.

REM Define main directories and global system parameters
set MAIN_DIR=%CD%
set LOG_DIR=%MAIN_DIR%\logs
set DATA_DIR=%MAIN_DIR%\data
set KNOWLEDGE_DIR=%DATA_DIR%\knowledge
set CACHE_DIR=%DATA_DIR%\cache
set NEURAL_DIR=%DATA_DIR%\neural
set MEMORY_DIR=%DATA_DIR%\onsite_memory
set CONVERSATION_DIR=%DATA_DIR%\conversations
set CONSCIOUSNESS_DIR=%DATA_DIR%\consciousness
set AUTOWIKI_DIR=%DATA_DIR%\autowiki
set BACKUPS_DIR=%DATA_DIR%\backups

REM Define shared database paths
set SHARED_DB_PATH=%DATA_DIR%\neural_metrics.db
set AUTOWIKI_DB_PATH=%KNOWLEDGE_DIR%\wiki_db.sqlite
set CONVERSATION_DB_PATH=%DATA_DIR%\conversations.db
set MEMORY_DB_PATH=%DATA_DIR%\memory.db

REM Define network ports for interconnection
set DASHBOARD_PORT=8765
set AUTOWIKI_PORT=7525
set MEMORY_API_PORT=7526
set NEURAL_API_PORT=7527
set CONVERSATION_PORT=7528
set METRICS_PORT=7529
set HEALTH_PORT=8766

REM Create required directories if they don't exist
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%MEMORY_DIR%" mkdir "%MEMORY_DIR%"
if not exist "%CONSCIOUSNESS_DIR%" mkdir "%CONSCIOUSNESS_DIR%"
if not exist "%AUTOWIKI_DIR%" mkdir "%AUTOWIKI_DIR%"
if not exist "%DATA_DIR%\breath" mkdir "%DATA_DIR%\breath"
if not exist "%CONVERSATION_DIR%" mkdir "%CONVERSATION_DIR%"
if not exist "%NEURAL_DIR%" mkdir "%NEURAL_DIR%"
if not exist "%BACKUPS_DIR%" mkdir "%BACKUPS_DIR%"
if not exist "%KNOWLEDGE_DIR%" mkdir "%KNOWLEDGE_DIR%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"

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
set PYTHONPATH=%MAIN_DIR%
set CONNECTION_STRING=sqlite:///%SHARED_DB_PATH%?check_same_thread=False
set GUI_FRAMEWORK=pyside6
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

REM Check for required packages for Knowledge CI/CD
python -c "import flask" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing Flask for health monitoring...
    pip install flask
)

REM Copy the conversation_flow.py module if needed
if exist src\v7.5\conversation_flow.py (
    if not exist src\v7_5\conversation_flow.py (
        echo Copying conversation_flow.py to v7_5 directory...
        if not exist src\v7_5 mkdir src\v7_5
        copy src\v7.5\conversation_flow.py src\v7_5\ > nul
    )
)

REM Check if PySide6 is installed, install if not
python -m pip install --upgrade pip > nul
echo Checking for PySide6...
python -c "import PySide6" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing PySide6...
    pip install PySide6 pyqtgraph matplotlib numpy pandas
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install PySide6 and dependencies.
        echo Please install them manually with:
        echo pip install PySide6 pyqtgraph matplotlib numpy pandas
        pause
        exit /b 1
    ) else (
        echo PySide6 installed successfully.
        set "PATH=%PATH%;%APPDATA%\Python\Python310\Scripts"
        set "PATH=%PATH%;%USERPROFILE%\AppData\Local\Programs\Python\Python310\Scripts"
    )
) else (
    echo PySide6 is already installed.
)

REM Check if Mistral client is installed, install if not
python -c "import mistralai" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing Mistral AI client...
    pip install mistralai
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install Mistral AI client.
        echo Please install it manually with:
        echo pip install mistralai
        pause
    ) else (
        echo Mistral AI client installed successfully.
        echo Testing API connection...
        python -c "from mistralai.client import MistralClient; from mistralai.models.chat_completion import ChatMessage; client = MistralClient(api_key='%MISTRAL_API_KEY%'); print('API connection successful!')" 2>NUL
        if %ERRORLEVEL% NEQ 0 (
            echo Warning: Could not connect to Mistral API. Check your API key and internet connection.
        ) else (
            echo Mistral API connection verified!
        )
    )
) else (
    echo Mistral AI client is already installed. Testing connection...
    python -c "from mistralai.client import MistralClient; from mistralai.models.chat_completion import ChatMessage; client = MistralClient(api_key='%MISTRAL_API_KEY%'); print('API connection successful!')" 2>NUL
    if %ERRORLEVEL% NEQ 0 (
        echo Warning: Could not connect to Mistral API. Check your API key and internet connection.
    ) else (
        echo Mistral API connection verified!
    )
)

REM Additional useful packages
python -c "import requests" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing additional dependencies...
    pip install requests
)

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
) else if /i "%choice%"=="q" (
    echo Exiting LUMINA V7.5 System...
    exit /b 0
) else (
    echo Invalid choice!
    pause
    goto menu
)

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
if exist src\v7.5\lumina_frontend.py (
    echo Found lumina_frontend.py in v7.5 directory
    start "LUMINA Chat Interface" cmd /c "cd src\v7.5 && python lumina_frontend.py --conversation-db=%CONVERSATION_DB_PATH% --memory-db=%MEMORY_DB_PATH% --shared-db=%SHARED_DB_PATH% --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
) else if exist src\v7_5\lumina_frontend.py (
    echo Found lumina_frontend.py in v7_5 directory
    start "LUMINA Chat Interface" cmd /c "cd src\v7_5 && python lumina_frontend.py --conversation-db=%CONVERSATION_DB_PATH% --memory-db=%MEMORY_DB_PATH% --shared-db=%SHARED_DB_PATH% --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
) else (
    echo WARNING: Could not find lumina_frontend.py!
    
    REM Try alternate files
    if exist src\chat_with_system.py (
        echo Found chat_with_system.py, using as alternative
        start "LUMINA Chat Interface" cmd /c "python src\chat_with_system.py --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
    ) else if exist src\central_language_node.py (
        echo Found central_language_node.py, using as alternative
        start "LUMINA Chat Interface" cmd /c "python src\central_language_node.py --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
    ) else (
        echo ERROR: No suitable chat interface module found.
        pause
    )
)
goto :eof

REM Function to start the Neural Seed System
:start_neural_seed
echo Starting Neural Seed System...
if exist src\neural\seed.py (
    echo Found neural seed in neural directory
    start "LUMINA Neural Seed" cmd /c "python src\neural\seed.py --db-path=%SHARED_DB_PATH% --port=%NEURAL_API_PORT% --consciousness-dir=%CONSCIOUSNESS_DIR% --neural-dir=%NEURAL_DIR% --enable-chat-input=true --process-chat-data=true --listen-mode=true && pause"
) else if exist src\seed.py (
    echo Found seed.py in src directory
    start "LUMINA Neural Seed" cmd /c "python src\seed.py --db-path=%SHARED_DB_PATH% --port=%NEURAL_API_PORT% --consciousness-dir=%CONSCIOUSNESS_DIR% --neural-dir=%NEURAL_DIR% --enable-chat-input=true --process-chat-data=true --listen-mode=true && pause"
) else (
    echo WARNING: Could not find neural seed module!
)
goto :eof

REM Function to start the System Monitor
:start_system_monitor
echo Starting System Monitor...
if exist src\v7.5\system_monitor.py (
    echo Found system_monitor.py in v7.5 directory
    start "LUMINA System Monitor" cmd /c "cd src\v7.5 && python system_monitor.py && pause"
) else if exist src\v7_5\system_monitor.py (
    echo Found system_monitor.py in v7_5 directory
    start "LUMINA System Monitor" cmd /c "cd src\v7_5 && python system_monitor.py && pause"
) else if exist src\monitoring (
    echo Checking monitoring directory...
    for %%f in (src\monitoring\*.py) do (
        echo Found %%f
        start "LUMINA System Monitor" cmd /c "python %%f && pause"
        goto :monitor_found
    )
    echo No monitoring scripts found.
    :monitor_found
) else (
    echo WARNING: Could not find system monitor module!
)
goto :eof

REM Function to start the Database Connector
:start_database
echo Starting Database Connector...
if exist src\v7.5\database_connector.py (
    echo Found database_connector.py in v7.5 directory
    start "LUMINA Database" cmd /c "cd src\v7.5 && python database_connector.py --db-path=%SHARED_DB_PATH% --memory-db=%MEMORY_DB_PATH% --conversation-db=%CONVERSATION_DB_PATH% --port=%MEMORY_API_PORT% && pause"
) else if exist src\v7_5\database_connector.py (
    echo Found database_connector.py in v7_5 directory
    start "LUMINA Database" cmd /c "cd src\v7_5 && python database_connector.py --db-path=%SHARED_DB_PATH% --memory-db=%MEMORY_DB_PATH% --conversation-db=%CONVERSATION_DB_PATH% --port=%MEMORY_API_PORT% && pause"
) else if exist src\memory_api_server.py (
    echo Using memory_api_server.py as alternative
    start "LUMINA Database" cmd /c "python src\memory_api_server.py --db-path=%SHARED_DB_PATH% --port=%MEMORY_API_PORT% && pause"
) else (
    echo WARNING: Could not find database connector module!
)
goto :eof

REM Function to start AutoWiki
:start_autowiki
echo Starting AutoWiki...

REM First check if we have the standalone launcher
if exist run_autowiki_v7.5.bat (
    echo Found standalone AutoWiki launcher
    start "LUMINA AutoWiki" cmd /c "set SHARED_DB_PATH=%SHARED_DB_PATH% && set KNOWLEDGE_DIR=%KNOWLEDGE_DIR% && set AUTOWIKI_DB_PATH=%AUTOWIKI_DB_PATH% && set AUTOWIKI_PORT=%AUTOWIKI_PORT% && set CACHE_DIR=%CACHE_DIR% && set AUTO_FETCH_INTERVAL=10 && set DEEP_LEARNING_DURATION=5 && run_autowiki_v7.5.bat"
    goto :eof
)

REM Check for the module directly
if exist src\v7_5\autowiki.py (
    echo Found autowiki.py in v7_5 directory
    
    REM Create required directories
    if not exist "%KNOWLEDGE_DIR%" mkdir "%KNOWLEDGE_DIR%"
    if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
    
    start "LUMINA AutoWiki" cmd /c "cd src\v7_5 && python autowiki.py --port=%AUTOWIKI_PORT% --knowledge-dir=%KNOWLEDGE_DIR% --db-path=%AUTOWIKI_DB_PATH% --shared-db --shared-db-path=%SHARED_DB_PATH% --auto-fetch-interval=10 --deep-learning-duration=5 && pause"
    goto :eof
)

echo Looking for wiki modules...
for %%f in (src\v7.5\*wiki*.py) do (
    echo Found %%f
    
    REM Create required directories
    if not exist "%KNOWLEDGE_DIR%" mkdir "%KNOWLEDGE_DIR%"
    if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
    
    start "LUMINA AutoWiki" cmd /c "python %%f --port=%AUTOWIKI_PORT% --knowledge-dir=%KNOWLEDGE_DIR% --db-path=%AUTOWIKI_DB_PATH% --shared-db --shared-db-path=%SHARED_DB_PATH% --auto-fetch-interval=10 --deep-learning-duration=5 && pause"
    goto :eof
)

for %%f in (src\v7_5\*wiki*.py) do (
    echo Found %%f
    
    REM Create required directories
    if not exist "%KNOWLEDGE_DIR%" mkdir "%KNOWLEDGE_DIR%"
    if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
    
    start "LUMINA AutoWiki" cmd /c "python %%f --port=%AUTOWIKI_PORT% --knowledge-dir=%KNOWLEDGE_DIR% --db-path=%AUTOWIKI_DB_PATH% --shared-db --shared-db-path=%SHARED_DB_PATH% --auto-fetch-interval=10 --deep-learning-duration=5 && pause"
    goto :eof
)

echo Creating standalone AutoWiki launcher...
echo @echo off > create_autowiki_v7.5.bat
echo setlocal enabledelayedexpansion >> create_autowiki_v7.5.bat
echo echo Creating AutoWiki v7.5 launcher... >> create_autowiki_v7.5.bat
echo copy run_unified_v7.5.bat run_autowiki_v7.5.bat /Y ^> nul >> create_autowiki_v7.5.bat
echo echo Standalone AutoWiki launcher created. >> create_autowiki_v7.5.bat
echo pause >> create_autowiki_v7.5.bat

start "Creating AutoWiki Launcher" cmd /c "create_autowiki_v7.5.bat"
echo WARNING: Could not find autowiki module! Created standalone launcher.
echo Please run the unified script again after the launcher is created.
goto :eof

REM Function to start Holographic UI
:start_holographic
echo Starting Holographic UI...

set FOUND_UI=false

REM Try the direct approach first
if exist run_holographic_frontend.py (
    echo Found run_holographic_frontend.py in root directory
    start "LUMINA Holographic UI" cmd /c "python run_holographic_frontend.py --gui-framework=%GUI_FRAMEWORK% --db-path=%SHARED_DB_PATH% --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    goto :eof
)

REM Try common UI directories
if exist src\ui\holographic.py (
    echo Found holographic.py in ui directory
    start "LUMINA Holographic UI" cmd /c "python src\ui\holographic.py --gui-framework=%GUI_FRAMEWORK% --db-path=%SHARED_DB_PATH% --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    goto :eof
)

if exist src\gui\holographic.py (
    echo Found holographic.py in gui directory
    start "LUMINA Holographic UI" cmd /c "python src\gui\holographic.py --gui-framework=%GUI_FRAMEWORK% --db-path=%SHARED_DB_PATH% --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --metrics-port=%METRICS_PORT% && pause"
    set FOUND_UI=true
    goto :eof
)

REM Look for v7.5 UI files
if exist src\v7.5\ui\holographic.py (
    echo Found holographic.py in v7.5/ui directory
    start "LUMINA Holographic UI" cmd /c "python src\v7.5\ui\holographic.py && pause"
    set FOUND_UI=true
    goto :eof
)

if exist src\v7_5\ui\holographic.py (
    echo Found holographic.py in v7_5/ui directory
    start "LUMINA Holographic UI" cmd /c "python src\v7_5\ui\holographic.py && pause"
    set FOUND_UI=true
    goto :eof
)

REM Search for any UI files that might work
echo Searching for UI files...
for %%f in (src\ui\*.py src\gui\*.py src\v7.5\ui\*.py src\v7_5\ui\*.py) do (
    echo Found potential UI file: %%f
    start "LUMINA Holographic UI" cmd /c "python %%f && pause"
    set FOUND_UI=true
    goto :eof
)

REM Try PySide6 based files directly
if exist src\visualization\*.py (
    for %%f in (src\visualization\*.py) do (
        echo Found visualization file: %%f
        start "LUMINA Holographic UI" cmd /c "python %%f && pause"
        set FOUND_UI=true
        goto :eof
    )
)

if "%FOUND_UI%"=="false" (
    echo WARNING: Could not find holographic UI module!
    echo Creating simple UI launcher...
    
    echo import sys > holographic_launcher.py
    echo try: from PySide6 import QtWidgets, QtCore >> holographic_launcher.py
    echo except ImportError: >> holographic_launcher.py
    echo     print("PySide6 not found. Installing...") >> holographic_launcher.py
    echo     import subprocess >> holographic_launcher.py
    echo     subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"]) >> holographic_launcher.py
    echo     from PySide6 import QtWidgets, QtCore >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo class LuminaHolographicUI(QtWidgets.QMainWindow): >> holographic_launcher.py
    echo     def __init__(self): >> holographic_launcher.py
    echo         super().__init__() >> holographic_launcher.py
    echo         self.setWindowTitle("LUMINA v7.5 Holographic Interface") >> holographic_launcher.py
    echo         self.resize(800, 600) >> holographic_launcher.py
    echo         self.central_widget = QtWidgets.QWidget() >> holographic_launcher.py
    echo         self.setCentralWidget(self.central_widget) >> holographic_launcher.py
    echo         self.layout = QtWidgets.QVBoxLayout(self.central_widget) >> holographic_launcher.py
    echo         self.label = QtWidgets.QLabel("LUMINA v7.5 Holographic Interface") >> holographic_launcher.py
    echo         self.label.setStyleSheet("font-size: 24px; font-weight: bold;") >> holographic_launcher.py
    echo         self.layout.addWidget(self.label) >> holographic_launcher.py
    echo         self.status_label = QtWidgets.QLabel("Connected to neural seed: {}".format("False" if len(sys.argv) < 2 else "True")) >> holographic_launcher.py
    echo         self.layout.addWidget(self.status_label) >> holographic_launcher.py
    echo. >> holographic_launcher.py
    echo if __name__ == "__main__": >> holographic_launcher.py
    echo     app = QtWidgets.QApplication(sys.argv) >> holographic_launcher.py
    echo     ui = LuminaHolographicUI() >> holographic_launcher.py
    echo     ui.show() >> holographic_launcher.py
    echo     sys.exit(app.exec_()) >> holographic_launcher.py
    
    start "LUMINA Holographic UI" cmd /c "python holographic_launcher.py && pause"
)

goto :eof

REM Function to start Knowledge CI/CD System
:start_knowledge_cicd
echo Starting Knowledge CI/CD System...

REM Check standalone launcher first
if exist run_knowledge_ci_cd_integrated.bat (
    echo Found standalone Knowledge CI/CD launcher
    start "LUMINA Knowledge CI/CD" cmd /c "set SHARED_DB_PATH=%SHARED_DB_PATH% && set KNOWLEDGE_DIR=%KNOWLEDGE_DIR% && set HEALTH_PORT=%HEALTH_PORT% && set MISTRAL_API_KEY=%MISTRAL_API_KEY% && run_knowledge_ci_cd_integrated.bat"
    goto :eof
)

REM Look for the appropriate script
if exist src\v8\knowledge_ci_cd_integrated.py (
    echo Found knowledge_ci_cd_integrated.py in v8 directory
    start "LUMINA Knowledge CI/CD" cmd /c "python src\v8\knowledge_ci_cd_integrated.py --health-check-port=%HEALTH_PORT% --db-path=%SHARED_DB_PATH% --knowledge-dir=%KNOWLEDGE_DIR% --backup-dir=%BACKUPS_DIR%"
    goto :eof
)

REM Try to find any CI/CD related scripts
for %%f in (src\*knowledge*ci*cd*.py src\v8\*knowledge*.py) do (
    echo Found potential Knowledge CI/CD file: %%f
    start "LUMINA Knowledge CI/CD" cmd /c "python %%f --health-check-port=%HEALTH_PORT% --db-path=%SHARED_DB_PATH% --knowledge-dir=%KNOWLEDGE_DIR% --backup-dir=%BACKUPS_DIR%"
    goto :eof
)

echo WARNING: Could not find Knowledge CI/CD module!
echo Creating standalone Knowledge CI/CD launcher...

echo @echo off > create_knowledge_cicd_launcher.bat
echo setlocal enabledelayedexpansion >> create_knowledge_cicd_launcher.bat
echo echo Creating Knowledge CI/CD launcher... >> create_knowledge_cicd_launcher.bat
echo copy run_knowledge_ci_cd_integrated.bat run_knowledge_cicd_v7.5.bat /Y ^> nul >> create_knowledge_cicd_launcher.bat
echo echo Standalone Knowledge CI/CD launcher created. >> create_knowledge_cicd_launcher.bat
echo pause >> create_knowledge_cicd_launcher.bat

start "Creating Knowledge CI/CD Launcher" cmd /c "create_knowledge_cicd_launcher.bat"
echo Please run the unified script again after the launcher is created.
goto :eof

goto menu 