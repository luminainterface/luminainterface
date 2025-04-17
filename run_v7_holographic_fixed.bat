@echo off
setlocal enabledelayedexpansion

echo.
echo ===================================
echo LUMINA V7.5 Unified System Launcher
echo ===================================
echo.

REM Create necessary directories if they don't exist
if not exist data mkdir data
if not exist data\neural mkdir data\neural
if not exist data\memory mkdir data\memory
if not exist data\consciousness mkdir data\consciousness
if not exist data\knowledge mkdir data\knowledge
if not exist data\v7 mkdir data\v7
if not exist data\v7.5 mkdir data\v7.5
if not exist data\autowiki mkdir data\autowiki
if not exist data\onsite_memory mkdir data\onsite_memory
if not exist data\conversations mkdir data\conversations
if not exist data\breath mkdir data\breath
if not exist logs mkdir logs
if not exist logs\neural mkdir logs\neural
if not exist logs\memory mkdir logs\memory
if not exist logs\consciousness mkdir logs\consciousness
if not exist logs\knowledge mkdir logs\knowledge
if not exist logs\v7 mkdir logs\v7
if not exist logs\v7.5 mkdir logs\v7.5
if not exist logs\system mkdir logs\system
if not exist logs\database mkdir logs\database
if not exist logs\autowiki mkdir logs\autowiki

REM Load API key from .env file if it exists
if exist .env (
    echo Loading API key from .env file...
    FOR /F "tokens=2 delims==" %%a in ('type .env ^| findstr "MISTRAL_API_KEY"') do (
        set MISTRAL_API_KEY=%%a
    )
    echo Using Mistral API Key: !MISTRAL_API_KEY:~0,4!...!MISTRAL_API_KEY:~-4!
    
    REM Print LLM parameters
    echo LLM Parameters:
    FOR /F "tokens=1,2 delims==" %%a in ('type .env ^| findstr "LLM_"') do (
        set %%a=%%b
        echo   %%a=%%b
    )
)

REM Set up environment variables
set PYTHONPATH=%cd%
set LUMINA_HOME=%cd%
set LUMINA_DATA_DIR=%cd%\data
set LUMINA_LOG_DIR=%cd%\logs
set LUMINA_PORT=7500
set LUMINA_GUI_FRAMEWORK=PySide6
set LUMINA_ENABLE_AUTOWIKI=true
set LUMINA_ENABLE_DREAMMODE=true
set LUMINA_ENABLE_NEURAL_SEED=true
set LUMINA_HOLOGRAPHIC_PORT=7505
set LUMINA_CHAT_PORT=7510
set LUMINA_DATABASE_PORT=7515
set LUMINA_MONITOR_PORT=7520
set LUMINA_AUTOWIKI_PORT=7525

REM Copy the conversation_flow.py module if needed
if exist src\v7.5\conversation_flow.py (
    if not exist src\v7_5\conversation_flow.py (
        echo Copying conversation_flow.py to v7_5 directory...
        if not exist src\v7_5 mkdir src\v7_5
        copy src\v7.5\conversation_flow.py src\v7_5\ > nul
        if errorlevel 1 (
            echo Warning: Failed to copy conversation_flow.py
        )
    )
)

REM Initialize component status
set STATUS_FILE=data\component_status.json
if not exist %STATUS_FILE% (
    echo {> %STATUS_FILE%
    echo   "Neural Seed": "offline",>> %STATUS_FILE%
    echo   "Consciousness": "offline",>> %STATUS_FILE%
    echo   "Holographic UI": "offline",>> %STATUS_FILE%
    echo   "Chat Interface": "offline",>> %STATUS_FILE%
    echo   "Database Connector": "offline",>> %STATUS_FILE%
    echo   "Autowiki": "offline">> %STATUS_FILE%
    echo }>> %STATUS_FILE%
)

REM Check for v7.5 module structure variations (dot vs underscore)
set V7_5_DOT=false
set V7_5_UNDERSCORE=false

if exist src\v7.5 (
    set V7_5_DOT=true
    echo Found v7.5 module path ^(dot notation^)
)

if exist src\v7_5 (
    set V7_5_UNDERSCORE=true
    echo Found v7_5 module path ^(underscore notation^)
)

if "!V7_5_DOT!"=="false" if "!V7_5_UNDERSCORE!"=="false" (
    echo WARNING: Could not find v7.5 module path! Some components may not work.
)

REM Check if Python is installed
python --version 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python 3.8 or newer and try again.
    pause
    exit /b 1
)

REM Check if PySide6 is installed
python -c "import PySide6" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo PySide6 is not installed. Installing...
    pip install PySide6
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install PySide6. Please install it manually:
        echo pip install PySide6
        pause
    )
)

REM Function to start the Neural Seed System
:start_neural_seed
echo Starting Neural Seed System...
if "!V7_5_DOT!"=="true" (
    start "LUMINA Neural Seed" cmd /c "python -m src.v7.5.neural_seed --log-level=info --port=!LUMINA_PORT! && pause"
) else if "!V7_5_UNDERSCORE!"=="true" (
    start "LUMINA Neural Seed" cmd /c "python -m src.v7_5.neural_seed --log-level=info --port=!LUMINA_PORT! && pause"
) else (
    echo WARNING: Could not find neural_seed module!
)
goto :eof

REM Function to start the Holographic Frontend
:start_holographic_ui
echo Starting Holographic UI...
python run_holographic_frontend.py --gui-framework=!LUMINA_GUI_FRAMEWORK! --port=!LUMINA_HOLOGRAPHIC_PORT!
if %ERRORLEVEL% NEQ 0 (
    echo Failed to start Holographic UI.
    echo Check that PySide6 is installed and run_holographic_frontend.py exists.
)
goto :eof

REM Function to start the Chat Interface
:start_chat_interface
echo Starting Chat Interface...

REM First try to use run_v7_5.bat if it exists
if exist run_v7_5.bat (
    echo Using run_v7_5.bat to start chat interface...
    start "LUMINA Chat Interface" cmd /c "run_v7_5.bat --from-holographic"
    goto :eof
)

REM Try direct file path approach (like in run_v7_5.bat)
if exist src\v7.5\lumina_frontend.py (
    start "LUMINA Chat Interface" cmd /c "python src\v7.5\lumina_frontend.py --port=!LUMINA_CHAT_PORT! && pause"
) else if exist src\v7_5\lumina_frontend.py (
    start "LUMINA Chat Interface" cmd /c "python src\v7_5\lumina_frontend.py --port=!LUMINA_CHAT_PORT! && pause"
) else (
    echo WARNING: Could not find lumina_frontend.py directly.
    echo Trying module import approach...
    if "!V7_5_DOT!"=="true" (
        start "LUMINA Chat Interface" cmd /c "python -m src.v7.5.lumina_frontend --port=!LUMINA_CHAT_PORT! && pause"
    ) else if "!V7_5_UNDERSCORE!"=="true" (
        start "LUMINA Chat Interface" cmd /c "python -m src.v7_5.lumina_frontend --port=!LUMINA_CHAT_PORT! && pause"
    ) else (
        echo ERROR: Failed to start Chat Interface!
    )
)
goto :eof

REM Function to start the System Monitor
:start_system_monitor
echo Starting System Monitor...
if "!V7_5_DOT!"=="true" (
    start "LUMINA System Monitor" cmd /c "python -m src.v7.5.system_monitor --port=!LUMINA_MONITOR_PORT! && pause"
) else if "!V7_5_UNDERSCORE!"=="true" (
    start "LUMINA System Monitor" cmd /c "python -m src.v7_5.system_monitor --port=!LUMINA_MONITOR_PORT! && pause"
) else (
    echo WARNING: Could not find system_monitor module!
)
goto :eof

REM Function to start the Database Connector
:start_database_connector
echo Starting Database Connector...
if "!V7_5_DOT!"=="true" (
    start "LUMINA Database Connector" cmd /c "python -m src.v7.5.database_connector --port=!LUMINA_DATABASE_PORT! && pause"
) else if "!V7_5_UNDERSCORE!"=="true" (
    start "LUMINA Database Connector" cmd /c "python -m src.v7_5.database_connector --port=!LUMINA_DATABASE_PORT! && pause"
) else (
    echo WARNING: Could not find database_connector module!
)
goto :eof

REM Function to start the Autowiki
:start_autowiki
echo Starting Autowiki...
if "!V7_5_DOT!"=="true" (
    start "LUMINA Autowiki" cmd /c "python -m src.v7.5.autowiki --port=!LUMINA_AUTOWIKI_PORT! && pause"
) else if "!V7_5_UNDERSCORE!"=="true" (
    start "LUMINA Autowiki" cmd /c "python -m src.v7_5.autowiki --port=!LUMINA_AUTOWIKI_PORT! && pause"
) else (
    echo WARNING: Could not find autowiki module!
)
goto :eof

REM Function to run component tests
:run_component_tests
echo Running component tests...
if exist src\component_test.py (
    python src\component_test.py
) else (
    echo WARNING: Component test script not found!
)
goto :eof

REM Main menu
:main_menu
cls
echo.
echo LUMINA V7.5 System
echo ==================
echo.
echo [1] Start Complete Holographic System
echo [2] Start Dashboard Panels
echo [3] Start Holographic UI Only
echo [4] Start Chat Interface Only
echo [5] Start System Monitor Only
echo [6] Start Database Connector Only
echo [7] Start AutoWiki Module Only
echo [8] Run Component Tests
echo [9] Open Documentation
echo [Q] Quit
echo.
set /p choice=Enter your choice: 

if "%choice%"=="1" (
    echo Starting all LUMINA components...
    call :start_neural_seed
    timeout /t 3 /nobreak > nul
    call :start_database_connector
    timeout /t 2 /nobreak > nul
    call :start_autowiki
    timeout /t 2 /nobreak > nul
    call :start_chat_interface
    timeout /t 2 /nobreak > nul
    start "LUMINA System Monitor" cmd /c "python -m src.v7_5.system_monitor --port=!LUMINA_MONITOR_PORT! && pause"
    timeout /t 2 /nobreak > nul
    python run_holographic_frontend.py --gui-framework=!LUMINA_GUI_FRAMEWORK! --port=!LUMINA_HOLOGRAPHIC_PORT!
    goto main_menu
) else if "%choice%"=="2" (
    echo Starting Dashboard Panels...
    call :start_system_monitor
    goto main_menu
) else if "%choice%"=="3" (
    call :start_holographic_ui
    goto main_menu
) else if "%choice%"=="4" (
    call :start_chat_interface
    goto main_menu
) else if "%choice%"=="5" (
    call :start_system_monitor
    goto main_menu
) else if "%choice%"=="6" (
    call :start_database_connector
    goto main_menu
) else if "%choice%"=="7" (
    call :start_autowiki
    goto main_menu
) else if "%choice%"=="8" (
    call :run_component_tests
    pause
    goto main_menu
) else if "%choice%"=="9" (
    if exist docs (
        start docs\index.html
    ) else (
        echo Documentation not found.
        pause
    )
    goto main_menu
) else if /i "%choice%"=="q" (
    echo Exiting LUMINA V7.5 System...
    exit /b 0
) else (
    echo Invalid choice!
    pause
    goto main_menu
)

goto main_menu

REM Start the main menu
goto main_menu 