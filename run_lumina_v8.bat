@echo off
setlocal enabledelayedexpansion
echo.
echo =======================================
echo    LUMINA v8 Seed System Launcher    
echo =======================================
echo.

:: Set error handling
set ERROR_COUNT=0

:: Ensure necessary directories exist
if not exist "data" mkdir data
if not exist "data\neural" mkdir data\neural
if not exist "data\memory" mkdir data\memory
if not exist "data\consciousness" mkdir data\consciousness
if not exist "data\knowledge" mkdir data\knowledge
if not exist "data\v8" mkdir data\v8
if not exist "data\v8\concepts" mkdir data\v8\concepts
if not exist "data\v8\connections" mkdir data\v8\connections
if not exist "data\autowiki" mkdir data\autowiki
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "data\conversations" mkdir data\conversations
if not exist "logs" mkdir logs
if not exist "logs\neural" mkdir logs\neural
if not exist "logs\memory" mkdir logs\memory
if not exist "logs\consciousness" mkdir logs\consciousness
if not exist "logs\knowledge" mkdir logs\knowledge
if not exist "logs\v8" mkdir logs\v8
if not exist "logs\system" mkdir logs\system

:: Load API key from .env file
if exist ".env" (
    echo Loading API key from .env file...
    for /f "tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="MISTRAL_API_KEY" (
            set MISTRAL_API_KEY=%%b
            echo Using Mistral API Key: !MISTRAL_API_KEY:~0,4!...!MISTRAL_API_KEY:~-4!
        )
        echo %%a | findstr /b "LLM_" > nul
        if not errorlevel 1 (
            echo   %%a=%%b
            set %%a=%%b
        )
    )
)

:: Set environment variables
set PYTHONPATH=%CD%
set LUMINA_HOME=%CD%
set LUMINA_DATA_DIR=%CD%\data
set LUMINA_LOG_DIR=%CD%\logs
set LUMINA_PORT=8000
set LUMINA_GUI_FRAMEWORK=PySide6
set LUMINA_ENABLE_AUTOWIKI=true
set LUMINA_ENABLE_DREAMMODE=true
set LUMINA_ENABLE_NEURAL_SEED=true
set LUMINA_TEMPLE_PORT=8005
set LUMINA_SEED_PORT=8010
set LUMINA_ROOT_PORT=8015
set LUMINA_GROWTH_PORT=8020
set LUMINA_BRIDGE_PORT=8025
set LUMINA_CONCEPT_PORT=8030

:: Check for port conflicts
call :check_port_available %LUMINA_TEMPLE_PORT% "Spatial Temple"
call :check_port_available %LUMINA_SEED_PORT% "Seed Dispersal"
call :check_port_available %LUMINA_ROOT_PORT% "Root Connection"
call :check_port_available %LUMINA_GROWTH_PORT% "Auto Growth"
call :check_port_available %LUMINA_BRIDGE_PORT% "Temple Bridge" 
call :check_port_available %LUMINA_CONCEPT_PORT% "SaveConcept"

if %ERROR_COUNT% GTR 0 (
    echo %ERROR_COUNT% port conflicts detected. Please resolve before continuing.
    echo You may need to close other applications or change port numbers in the batch file.
    pause
    exit /b 1
)

:: Set connection variables for components
set TIMESTAMP=%DATE:~-4,4%%DATE:~-7,2%%DATE:~-10,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set "LUMINA_V8_CONNECTIONS_FILE=%CD%\data\v8\connections\system_connections_%TIMESTAMP%.json"
set "LUMINA_V8_SHARED_KEY=v8_system_key_%RANDOM%_%RANDOM%"

:: Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python 3.8 or newer and try again.
    pause
    exit /b 1
)

:: Check if PySide6 is installed
python -c "import PySide6" > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo PySide6 is not installed. Installing...
    python -m pip install PySide6
    if %ERRORLEVEL% neq 0 (
        echo Failed to install PySide6. Please install it manually:
        echo pip install PySide6
        pause
    )
)

:: Generate system connections configuration using Python (more reliable than echo)
echo Generating system connections configuration...
python -c "import json, os, sys; config = {'system_key': os.environ['LUMINA_V8_SHARED_KEY'], 'components': {'spatial_temple': {'port': int(os.environ['LUMINA_TEMPLE_PORT']), 'host': 'localhost'}, 'root_connection': {'port': int(os.environ['LUMINA_ROOT_PORT']), 'host': 'localhost'}, 'auto_growth': {'port': int(os.environ['LUMINA_GROWTH_PORT']), 'host': 'localhost'}, 'seed_dispersal': {'port': int(os.environ['LUMINA_SEED_PORT']), 'host': 'localhost'}, 'temple_bridge': {'port': int(os.environ['LUMINA_BRIDGE_PORT']), 'host': 'localhost'}, 'saveconcept': {'port': int(os.environ['LUMINA_CONCEPT_PORT']), 'host': 'localhost'}}, 'connections': {'spatial_temple': ['temple_bridge', 'root_connection'], 'temple_bridge': ['spatial_temple', 'seed_dispersal', 'saveconcept'], 'seed_dispersal': ['temple_bridge', 'auto_growth'], 'auto_growth': ['seed_dispersal', 'root_connection'], 'root_connection': ['spatial_temple', 'auto_growth'], 'saveconcept': ['temple_bridge']}, 'generated_at': os.environ['TIMESTAMP']}; f = open(os.environ['LUMINA_V8_CONNECTIONS_FILE'], 'w'); json.dump(config, f, indent=2); f.close(); print(f'Connection configuration created at {os.environ[\"LUMINA_V8_CONNECTIONS_FILE\"]}')"

if not exist "%LUMINA_V8_CONNECTIONS_FILE%" (
    echo Failed to create connections configuration file.
    echo Attempting fallback method...
    call :generate_connections_file_fallback
)

if not exist "%LUMINA_V8_CONNECTIONS_FILE%" (
    echo ERROR: Failed to create connections configuration file after fallback.
    pause
    exit /b 1
)

:: Create a symbolic link to the latest connection file for easy access
if exist "%CD%\data\v8\connections\latest_connections.json" del "%CD%\data\v8\connections\latest_connections.json"
copy "%LUMINA_V8_CONNECTIONS_FILE%" "%CD%\data\v8\connections\latest_connections.json" > nul
set "LUMINA_V8_CONNECTIONS_FILE_DISPLAY=%LUMINA_V8_CONNECTIONS_FILE:..\=%"
echo Using connections file: %LUMINA_V8_CONNECTIONS_FILE_DISPLAY%

goto :menu

:check_port_available
set PORT=%1
set COMPONENT=%2
netstat -an | findstr ":%PORT% " > nul
if %ERRORLEVEL% equ 0 (
    echo ERROR: Port %PORT% for %COMPONENT% is already in use.
    set /a ERROR_COUNT+=1
)
goto :eof

:generate_connections_file_fallback
echo Using fallback method to create connections file...
if not exist "data\v8\connections" mkdir data\v8\connections
(
echo {
echo   "system_key": "%LUMINA_V8_SHARED_KEY%",
echo   "components": {
echo     "spatial_temple": {"port": %LUMINA_TEMPLE_PORT%, "host": "localhost"},
echo     "root_connection": {"port": %LUMINA_ROOT_PORT%, "host": "localhost"},
echo     "auto_growth": {"port": %LUMINA_GROWTH_PORT%, "host": "localhost"},
echo     "seed_dispersal": {"port": %LUMINA_SEED_PORT%, "host": "localhost"},
echo     "temple_bridge": {"port": %LUMINA_BRIDGE_PORT%, "host": "localhost"},
echo     "saveconcept": {"port": %LUMINA_CONCEPT_PORT%, "host": "localhost"}
echo   },
echo   "connections": {
echo     "spatial_temple": ["temple_bridge", "root_connection"],
echo     "temple_bridge": ["spatial_temple", "seed_dispersal", "saveconcept"],
echo     "seed_dispersal": ["temple_bridge", "auto_growth"],
echo     "auto_growth": ["seed_dispersal", "root_connection"],
echo     "root_connection": ["spatial_temple", "auto_growth"],
echo     "saveconcept": ["temple_bridge"]
echo   },
echo   "generated_at": "%TIMESTAMP%"
echo }
) > "%LUMINA_V8_CONNECTIONS_FILE%"
echo Connection configuration created using fallback method at %LUMINA_V8_CONNECTIONS_FILE%
goto :eof

:start_spatial_temple
echo Starting Spatial Temple...
start "Spatial Temple" cmd /c "python src\v8\spatial_temple_visualization.py --port=%LUMINA_TEMPLE_PORT% --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% 2> logs\v8\spatial_temple_%TIMESTAMP%.log && pause"
goto :eof

:start_seed_dispersal
echo Starting Seed Dispersal System...
start "Seed Dispersal" cmd /c "python src\v8\seed_dispersal_system.py --port=%LUMINA_SEED_PORT% --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% 2> logs\v8\seed_dispersal_%TIMESTAMP%.log && pause"
goto :eof

:start_temple_bridge
echo Starting Temple to Seed Bridge...
start "Temple Bridge" cmd /c "python src\v8\temple_to_seed_bridge.py --port=%LUMINA_BRIDGE_PORT% --temple-port=%LUMINA_TEMPLE_PORT% --seed-port=%LUMINA_SEED_PORT% --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% 2> logs\v8\temple_bridge_%TIMESTAMP%.log && pause"
goto :eof

:start_auto_growth
echo Starting Auto Seed Growth...
start "Auto Seed Growth" cmd /c "python src\v8\auto_seed_growth.py --port=%LUMINA_GROWTH_PORT% --seed-port=%LUMINA_SEED_PORT% --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% 2> logs\v8\auto_growth_%TIMESTAMP%.log && pause"
goto :eof

:start_root_connection
echo Starting Root Connection System...
start "Root Connection" cmd /c "python src\v8\root_connection_system.py --port=%LUMINA_ROOT_PORT% --temple-port=%LUMINA_TEMPLE_PORT% --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% 2> logs\v8\root_connection_%TIMESTAMP%.log && pause"
goto :eof

:start_saveconcept
echo Starting SaveConcept System...
start "SaveConcept" cmd /c "python src\v8\saveconcept.py --port=%LUMINA_CONCEPT_PORT% --temple-bridge-port=%LUMINA_BRIDGE_PORT% --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% 2> logs\v8\saveconcept_%TIMESTAMP%.log && pause"
goto :eof

:run_component_tests
echo Running component tests...
python src\v8\health_check_server.py --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" 2> logs\v8\component_tests_%TIMESTAMP%.log
pause
goto :eof

:demo_data_generator
echo Generating demo data...
start "Demo Data Generator" cmd /c "python src\v8\demo_data_generator.py && pause"
goto :eof

:start_connection_monitor
echo Starting System Connection Monitor...
start "Connection Monitor" cmd /c "python src\v8\monitor_connections.py --connections-file="%LUMINA_V8_CONNECTIONS_FILE%" --system-key=%LUMINA_V8_SHARED_KEY% --gui 2> logs\v8\monitor_%TIMESTAMP%.log && pause"
goto :eof

:list_started_components
echo.
echo Currently Running Components:
echo ---------------------------
for %%C in (Spatial Temple, Seed Dispersal, Temple Bridge, Auto Seed Growth, Root Connection, SaveConcept, Connection Monitor) do (
    tasklist /fi "windowtitle eq %%C" | findstr /i "python.exe" > nul
    if not errorlevel 1 (
        echo [RUNNING] %%C
    ) else (
        echo [STOPPED] %%C
    )
)
echo.
goto :eof

:start_complete_system
echo Starting Complete v8 Seed System...
echo.
echo Step 1/7: Starting Connection Monitor...
call :start_connection_monitor
timeout /t 3 > nul

echo Step 2/7: Starting Spatial Temple...
call :start_spatial_temple
timeout /t 4 > nul

echo Step 3/7: Starting Root Connection System...
call :start_root_connection
timeout /t 4 > nul

echo Step 4/7: Starting Auto Growth Engine...
call :start_auto_growth
timeout /t 4 > nul

echo Step 5/7: Starting Seed Dispersal System...
call :start_seed_dispersal
timeout /t 4 > nul

echo Step 6/7: Starting Temple-to-Seed Bridge...
call :start_temple_bridge
timeout /t 4 > nul

echo Step 7/7: Starting SaveConcept System...
call :start_saveconcept

echo.
echo All components started! The systems should now be communicating with each other.
echo Use the Connection Monitor to verify connectivity.
echo Log files are saved in logs\v8\ directory for troubleshooting.

call :list_started_components
goto :eof

:check_requirements
echo Checking system requirements...
set REQUIREMENTS_MET=true

:: Check Python version (needs 3.8+)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Python 3.8+ is required. Current version:
    python --version
    set REQUIREMENTS_MET=false
) else (
    echo [OK] Python version:
    python --version
)

:: Check PySide6
python -c "import PySide6" > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [FAIL] PySide6 is not installed
    set REQUIREMENTS_MET=false
) else (
    echo [OK] PySide6 is installed
)

:: Check required packages
python -c "import requests" > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [FAIL] requests package is not installed
    set REQUIREMENTS_MET=false
) else (
    echo [OK] requests package is installed
)

:: Check disk space
for /f "tokens=3" %%a in ('dir /-c /w . ^| findstr /c:"bytes free"') do set FREE_SPACE=%%a
if %FREE_SPACE% LSS 100000000 (
    echo [WARN] Low disk space: %FREE_SPACE% bytes free
) else (
    echo [OK] Disk space: %FREE_SPACE% bytes free
)

:: Summary
if "%REQUIREMENTS_MET%"=="false" (
    echo.
    echo Some requirements are not met. The system might not work correctly.
) else (
    echo.
    echo All system requirements are satisfied.
)
echo.
pause
goto :eof

:stop_all_components
echo Stopping all LUMINA v8 components...
taskkill /fi "windowtitle eq Spatial Temple" /f
taskkill /fi "windowtitle eq Seed Dispersal" /f
taskkill /fi "windowtitle eq Temple Bridge" /f
taskkill /fi "windowtitle eq Auto Seed Growth" /f
taskkill /fi "windowtitle eq Root Connection" /f
taskkill /fi "windowtitle eq SaveConcept" /f
taskkill /fi "windowtitle eq Connection Monitor" /f
echo All components stopped.
goto :eof

:menu
cls
echo.
echo LUMINA V8 Seed System
echo ====================
echo.
echo [1] Start Complete Seed System
echo [2] Start Spatial Temple Only
echo [3] Start Seed Dispersal Only
echo [4] Start Temple-to-Seed Bridge Only
echo [5] Start Auto Growth Engine Only
echo [6] Start Root Connection Only
echo [7] Start SaveConcept System Only
echo [8] Run Component Tests
echo [9] Generate Demo Data
echo [C] Start Connection Monitor
echo [S] Show Running Components
echo [T] Stop All Components
echo [R] Check System Requirements
echo [Q] Quit
echo.

set /p choice=Enter your choice: 

if "%choice%"=="1" call :start_complete_system
if "%choice%"=="2" call :start_spatial_temple
if "%choice%"=="3" call :start_seed_dispersal
if "%choice%"=="4" call :start_temple_bridge
if "%choice%"=="5" call :start_auto_growth
if "%choice%"=="6" call :start_root_connection
if "%choice%"=="7" call :start_saveconcept
if "%choice%"=="8" call :run_component_tests
if "%choice%"=="9" call :demo_data_generator
if /i "%choice%"=="c" call :start_connection_monitor
if /i "%choice%"=="s" call :list_started_components
if /i "%choice%"=="t" call :stop_all_components
if /i "%choice%"=="r" call :check_requirements
if /i "%choice%"=="q" exit /b 0

:: Wait for user input before returning to menu
if /i not "%choice%"=="s" (
    if not "%choice%"=="1" (
        if not "%choice%"=="2" (
            if not "%choice%"=="3" (
                if not "%choice%"=="4" (
                    if not "%choice%"=="5" (
                        if not "%choice%"=="6" (
                            if not "%choice%"=="7" (
                                if not "%choice%"=="9" (
                                    if /i not "%choice%"=="c" (
                                        pause
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

:: Return to menu after function completes
goto :menu 