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

REM Main menu
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
echo Configuring chat to output to neural seed system
if exist src\v7_5\lumina_gui.py (
    echo Found lumina_gui.py in v7_5 directory
    start "LUMINA Chat Interface" cmd /c "cd src\v7_5 && python lumina_gui.py --conversation-db=%CONVERSATION_DB_PATH% --memory-db=%MEMORY_DB_PATH% --shared-db=%SHARED_DB_PATH% --neural-port=%NEURAL_API_PORT% --memory-port=%MEMORY_API_PORT% --neural-integration=true --output-to-neural=true && pause"
) else (
    echo ERROR: No suitable chat interface module found.
    echo Please ensure lumina_gui.py exists in src\v7_5\lumina_gui.py
)
goto :eof

REM Function to start the Neural Seed System
:start_neural_seed
echo Starting Neural Seed System...
echo Configuring neural seed to process chat data via port %NEURAL_API_PORT%
if exist src\neural\seed.py (
    start "LUMINA Neural Seed" cmd /c "python src\neural\seed.py --db-path=%SHARED_DB_PATH% --port=%NEURAL_API_PORT% --consciousness-dir=%CONSCIOUSNESS_DIR% --neural-dir=%NEURAL_DIR% --enable-chat-input=true --process-chat-data=true --listen-mode=true && pause"
) else if exist src\seed.py (
    start "LUMINA Neural Seed" cmd /c "python src\seed.py --db-path=%SHARED_DB_PATH% --port=%NEURAL_API_PORT% --consciousness-dir=%CONSCIOUSNESS_DIR% --neural-dir=%NEURAL_DIR% --enable-chat-input=true --process-chat-data=true --listen-mode=true && pause"
) else (
    echo WARNING: Could not find neural seed module!
)
goto :eof

REM Function to start the System Monitor
:start_system_monitor
echo Starting System Monitor...
if exist src\v7.5\system_monitor.py (
    start "LUMINA System Monitor" cmd /c "cd src\v7.5 && python system_monitor.py --db-path=%SHARED_DB_PATH% --port=%METRICS_PORT% && pause"
) else if exist src\v7_5\system_monitor.py (
    start "LUMINA System Monitor" cmd /c "cd src\v7_5 && python system_monitor.py --db-path=%SHARED_DB_PATH% --port=%METRICS_PORT% && pause"
) else (
    echo WARNING: Could not find system monitor module!
)
goto :eof

REM Function to start the Database Connector
:start_database
echo Starting Database Connector...
if exist src\v7.5\database_connector.py (
    start "LUMINA Database" cmd /c "cd src\v7.5 && python database_connector.py --db-path=%SHARED_DB_PATH% --memory-db=%MEMORY_DB_PATH% --conversation-db=%CONVERSATION_DB_PATH% --port=%MEMORY_API_PORT% && pause"
) else if exist src\v7_5\database_connector.py (
    start "LUMINA Database" cmd /c "cd src\v7_5 && python database_connector.py --db-path=%SHARED_DB_PATH% --memory-db=%MEMORY_DB_PATH% --conversation-db=%CONVERSATION_DB_PATH% --port=%MEMORY_API_PORT% && pause"
) else if exist src\memory_api_server.py (
    start "LUMINA Database" cmd /c "python src\memory_api_server.py --db-path=%SHARED_DB_PATH% --port=%MEMORY_API_PORT% && pause"
) else (
    echo WARNING: Could not find database connector module!
)
goto :eof

REM Function to start AutoWiki
:start_autowiki
echo Starting AutoWiki...
echo Setting AutoWiki fetch interval to 10 seconds with 5 seconds deep learning
if exist run_autowiki_v7.5.bat (
    start "LUMINA AutoWiki" cmd /c "set SHARED_DB_PATH=%SHARED_DB_PATH% && set KNOWLEDGE_DIR=%KNOWLEDGE_DIR% && set AUTOWIKI_DB_PATH=%AUTOWIKI_DB_PATH% && set AUTOWIKI_PORT=%AUTOWIKI_PORT% && set CACHE_DIR=%CACHE_DIR% && set AUTO_FETCH_INTERVAL=10 && set DEEP_LEARNING_DURATION=5 && run_autowiki_v7.5.bat"
) else if exist src\v7_5\autowiki.py (
    echo Found autowiki.py in v7_5 directory - configuring auto-fetch settings

    REM Create .env file for AutoWiki settings
    echo AUTO_FETCH_INTERVAL=10 > autowiki_settings.env
    echo DEEP_LEARNING_DURATION=5 >> autowiki_settings.env
    echo SHARED_DB_PATH=%SHARED_DB_PATH% >> autowiki_settings.env
    echo AUTOWIKI_DB_PATH=%KNOWLEDGE_DIR%\wiki_db.sqlite >> autowiki_settings.env
    
    REM Launch AutoWiki with auto-fetch enabled (default) and correct data directory
    start "LUMINA AutoWiki" cmd /c "set AUTO_FETCH_INTERVAL=10 && set DEEP_LEARNING_DURATION=5 && cd src\v7_5 && python autowiki.py --data-dir=%KNOWLEDGE_DIR% --port=%AUTOWIKI_PORT% --log-dir=%LOG_DIR% && pause"
) else (
    echo WARNING: Could not find autowiki module!
)
goto :eof

REM Function to start Holographic UI
:start_holographic
echo Starting Holographic UI...
if exist src\v7_5\ui\holographic_frontend.py (
    echo Found holographic_frontend.py in v7_5/ui directory
    start "LUMINA Holographic UI" cmd /c "cd src\v7_5\ui && python holographic_frontend.py --port=5678 --gui-framework=PySide6 && pause"
) else (
    echo WARNING: Could not find holographic UI module!
    echo Please ensure holographic_frontend.py exists in src\v7_5\ui\holographic_frontend.py
)
goto :eof

REM Function to start Knowledge CI/CD System
:start_knowledge_cicd
echo Starting Knowledge CI/CD System...
if exist run_knowledge_ci_cd_integrated.bat (
    start "LUMINA Knowledge CI/CD" cmd /c "set SHARED_DB_PATH=%SHARED_DB_PATH% && set KNOWLEDGE_DIR=%KNOWLEDGE_DIR% && set HEALTH_PORT=%HEALTH_PORT% && set MISTRAL_API_KEY=%MISTRAL_API_KEY% && run_knowledge_ci_cd_integrated.bat"
) else if exist src\v8\knowledge_ci_cd_integrated.py (
    start "LUMINA Knowledge CI/CD" cmd /c "python src\v8\knowledge_ci_cd_integrated.py --health-check-port=%HEALTH_PORT% --db-path=%SHARED_DB_PATH% --knowledge-dir=%KNOWLEDGE_DIR% --backup-dir=%BACKUPS_DIR%"
) else (
    echo WARNING: Could not find Knowledge CI/CD module!
)
goto :eof

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
echo Verification complete
goto :eof 