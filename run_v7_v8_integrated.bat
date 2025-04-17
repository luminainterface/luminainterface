@echo off
echo Starting LUMINA Integrated V7+V8 System...

:: Create necessary directories if they don't exist
if not exist "data" mkdir data
if not exist "data\neural" mkdir data\neural
if not exist "data\memory" mkdir data\memory
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "logs" mkdir logs
if not exist "data\seed" mkdir data\seed
if not exist "data\dream" mkdir data\dream
if not exist "data\autowiki" mkdir data\autowiki
if not exist "data\consciousness" mkdir data\consciousness
if not exist "data\breath" mkdir data\breath
if not exist "data\v7.5" mkdir data\v7.5
if not exist "data\conversations" mkdir data\conversations
if not exist "data\v8" mkdir data\v8
if not exist "data\v8\temple" mkdir data\v8\temple
if not exist "data\v8\knowledge" mkdir data\v8\knowledge
if not exist "data\v8\metrics" mkdir data\v8\metrics
if not exist "data\backups" mkdir data\backups

:: Set Python path to include the project root for proper imports
set PYTHONPATH=%CD%

:: Set Python command variable for consistency
set PYTHON_CMD=python

:: Set environment variables
set DASHBOARD_PORT=5679
set METRICS_DB_PATH=data\neural_metrics.db
set V7_CONNECTION_PORT=5678
set V7_DOCS_PATH=%CD%\docs
set TEMPLATE_PLUGINS_DIRS=plugins;src\v7\plugins;src\plugins;src\visualization\plugins
set GUI_FRAMEWORK=PySide6

:: V7 specific environment variables
set ENABLE_NODE_CONSCIOUSNESS=true
set ENABLE_AUTOWIKI=true
set ENABLE_DREAM_MODE=true
set ENABLE_BREATH_DETECTION=true
set ENABLE_MONDAY_INTEGRATION=true
set ENABLE_ONSITE_MEMORY=true
set V7_KNOWLEDGE_PATH=data\consciousness\knowledge
set V7_DREAM_ARCHIVE_PATH=data\dream\archives
set BREATH_PATTERN_CONFIG=configs\breath_patterns.json
set MEMORY_STORAGE_PATH=data\onsite_memory

:: V7.5 specific environment variables
set V7_5_ENABLED=true
set V7_5_INTERLINK=true
set V7_5_PORT=5680

:: V8 specific environment variables
set V8_KNOWLEDGE_DB_PATH=data\v8\knowledge\knowledge_db.sqlite
set V8_METRICS_DB_PATH=data\v8\metrics\metrics_db.sqlite
set V8_HEALTH_CHECK_PORT=8765
set V8_V7_BRIDGE_PORT=8766
set MISTRAL_API_KEY=nLKZEpq29OihnaArxV7s6KtzsNEiky2A

:: Check if Python is installed
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check for required packages
echo Checking dependencies...
%PYTHON_CMD% -c "import sqlite3, uuid, argparse, threading, json" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install argparse
)

:: Check for PySide6 installation
%PYTHON_CMD% -c "import PySide6" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo PySide6 is not installed. This is required for the V7 Holographic Interface.
    echo Installing PySide6...
    pip install PySide6 pyqtgraph matplotlib numpy pandas
    if %errorlevel% neq 0 (
        echo Failed to install PySide6. Please install it manually with:
        echo pip install PySide6 pyqtgraph matplotlib numpy pandas
        pause
        exit /b 1
    )
    echo PySide6 installed successfully.
)

:: Check for Flask installation (needed for v8 health checks)
%PYTHON_CMD% -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Flask is not installed. This is required for the V8 Health Monitoring.
    echo Installing Flask...
    pip install flask
    if %errorlevel% neq 0 (
        echo Failed to install Flask. Please install it manually with:
        echo pip install flask
        pause
        exit /b 1
    )
    echo Flask installed successfully.
)

:: Check if v7_v8_bridge module exists, create if not
if not exist "src\bridge" mkdir src\bridge
echo Creating V7-V8 bridge directory...
echo >src\bridge\__init__.py

:: Create bridge module if it doesn't exist
if not exist "src\bridge\v7_v8_bridge.py" (
    echo Creating V7-V8 bridge module...
    echo #!/usr/bin/env python3 > src\bridge\v7_v8_bridge.py
    echo """V7-V8 Integration Bridge""" >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo import os >> src\bridge\v7_v8_bridge.py
    echo import sys >> src\bridge\v7_v8_bridge.py
    echo import json >> src\bridge\v7_v8_bridge.py
    echo import time >> src\bridge\v7_v8_bridge.py
    echo import logging >> src\bridge\v7_v8_bridge.py
    echo import threading >> src\bridge\v7_v8_bridge.py
    echo import argparse >> src\bridge\v7_v8_bridge.py
    echo import requests >> src\bridge\v7_v8_bridge.py
    echo from datetime import datetime >> src\bridge\v7_v8_bridge.py
    echo from pathlib import Path >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo # Add parent directory to path for imports >> src\bridge\v7_v8_bridge.py
    echo project_root = str(Path(__file__).parent.parent.parent.absolute()) >> src\bridge\v7_v8_bridge.py
    echo if project_root not in sys.path: >> src\bridge\v7_v8_bridge.py
    echo     sys.path.append(project_root) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo # Setup logging >> src\bridge\v7_v8_bridge.py
    echo logging.basicConfig( >> src\bridge\v7_v8_bridge.py
    echo     level=logging.INFO, >> src\bridge\v7_v8_bridge.py
    echo     format='%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s', >> src\bridge\v7_v8_bridge.py
    echo     handlers=[ >> src\bridge\v7_v8_bridge.py
    echo         logging.FileHandler(f"logs/v7_v8_bridge_%%(datetime.now().strftime('%%Y%%m%%d')).log"), >> src\bridge\v7_v8_bridge.py
    echo         logging.StreamHandler() >> src\bridge\v7_v8_bridge.py
    echo     ] >> src\bridge\v7_v8_bridge.py
    echo ) >> src\bridge\v7_v8_bridge.py
    echo logger = logging.getLogger("v7_v8_bridge") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo class V7V8Bridge: >> src\bridge\v7_v8_bridge.py
    echo     """Bridge between V7 Holographic System and V8 Knowledge CI/CD System""" >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def __init__(self, v8_health_port=8765, v7_connection_port=5678): >> src\bridge\v7_v8_bridge.py
    echo         """Initialize the bridge with ports for both systems""" >> src\bridge\v7_v8_bridge.py
    echo         self.v8_health_url = f"http://localhost:{v8_health_port}/health" >> src\bridge\v7_v8_bridge.py
    echo         self.v8_metrics_url = f"http://localhost:{v8_health_port}/metrics" >> src\bridge\v7_v8_bridge.py
    echo         self.v7_connection_port = v7_connection_port >> src\bridge\v7_v8_bridge.py
    echo         self.running = False >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread = None >> src\bridge\v7_v8_bridge.py
    echo         self.v7_seed = None >> src\bridge\v7_v8_bridge.py
    echo         self.last_sync = None >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         # Try to import v7 seed system >> src\bridge\v7_v8_bridge.py
    echo         try: >> src\bridge\v7_v8_bridge.py
    echo             from src.seed import get_neural_seed >> src\bridge\v7_v8_bridge.py
    echo             self.v7_seed = get_neural_seed() >> src\bridge\v7_v8_bridge.py
    echo             logger.info("Successfully connected to V7 Neural Seed system") >> src\bridge\v7_v8_bridge.py
    echo         except ImportError: >> src\bridge\v7_v8_bridge.py
    echo             logger.warning("Cannot import V7 Neural Seed system") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         logger.info("V7-V8 Bridge initialized") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def start(self): >> src\bridge\v7_v8_bridge.py
    echo         """Start the bridge process""" >> src\bridge\v7_v8_bridge.py
    echo         if self.running: >> src\bridge\v7_v8_bridge.py
    echo             logger.info("Bridge is already running") >> src\bridge\v7_v8_bridge.py
    echo             return >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         self.running = True >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread = threading.Thread(target=self._bridge_loop) >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread.daemon = True >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread.start() >> src\bridge\v7_v8_bridge.py
    echo         logger.info("V7-V8 Bridge started") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def stop(self): >> src\bridge\v7_v8_bridge.py
    echo         """Stop the bridge process""" >> src\bridge\v7_v8_bridge.py
    echo         self.running = False >> src\bridge\v7_v8_bridge.py
    echo         if self.bridge_thread: >> src\bridge\v7_v8_bridge.py
    echo             self.bridge_thread.join(timeout=2.0) >> src\bridge\v7_v8_bridge.py
    echo         logger.info("V7-V8 Bridge stopped") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def _bridge_loop(self): >> src\bridge\v7_v8_bridge.py
    echo         """Main bridge loop to transfer data between V7 and V8""" >> src\bridge\v7_v8_bridge.py
    echo         sync_interval = 15  # Sync every 15 seconds >> src\bridge\v7_v8_bridge.py
    echo         while self.running: >> src\bridge\v7_v8_bridge.py
    echo             try: >> src\bridge\v7_v8_bridge.py
    echo                 # Check V8 health >> src\bridge\v7_v8_bridge.py
    echo                 v8_health = self._check_v8_health() >> src\bridge\v7_v8_bridge.py
    echo                 if v8_health and v8_health.get("status") != "critical": >> src\bridge\v7_v8_bridge.py
    echo                     # Transfer knowledge from V7 to V8 >> src\bridge\v7_v8_bridge.py
    echo                     self._transfer_v7_to_v8() >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo                     # Transfer knowledge from V8 to V7 >> src\bridge\v7_v8_bridge.py
    echo                     self._transfer_v8_to_v7() >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo                     self.last_sync = datetime.now() >> src\bridge\v7_v8_bridge.py
    echo                     logger.info(f"V7-V8 synchronization completed at {self.last_sync.isoformat()}") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo                 logger.error(f"Error in bridge loop: {e}") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             # Wait for next sync >> src\bridge\v7_v8_bridge.py
    echo             time.sleep(sync_interval) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def _check_v8_health(self): >> src\bridge\v7_v8_bridge.py
    echo         """Check the health of the V8 system""" >> src\bridge\v7_v8_bridge.py
    echo         try: >> src\bridge\v7_v8_bridge.py
    echo             response = requests.get(self.v8_health_url, timeout=5) >> src\bridge\v7_v8_bridge.py
    echo             if response.status_code == 200: >> src\bridge\v7_v8_bridge.py
    echo                 return response.json() >> src\bridge\v7_v8_bridge.py
    echo             else: >> src\bridge\v7_v8_bridge.py
    echo                 logger.warning(f"V8 health check returned status code {response.status_code}") >> src\bridge\v7_v8_bridge.py
    echo                 return None >> src\bridge\v7_v8_bridge.py
    echo         except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo             logger.error(f"Error checking V8 health: {e}") >> src\bridge\v7_v8_bridge.py
    echo             return None >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def _transfer_v7_to_v8(self): >> src\bridge\v7_v8_bridge.py
    echo         """Transfer knowledge from V7 to V8""" >> src\bridge\v7_v8_bridge.py
    echo         if not self.v7_seed: >> src\bridge\v7_v8_bridge.py
    echo             logger.warning("Cannot transfer from V7 to V8: V7 seed not available") >> src\bridge\v7_v8_bridge.py
    echo             return >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         try: >> src\bridge\v7_v8_bridge.py
    echo             # Get knowledge from V7 seed system >> src\bridge\v7_v8_bridge.py
    echo             seed_dict = getattr(self.v7_seed, "dictionary", {}) >> src\bridge\v7_v8_bridge.py
    echo             seed_count = len(seed_dict) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             # For now, just log what we would transfer >> src\bridge\v7_v8_bridge.py
    echo             logger.info(f"Would transfer {seed_count} concepts from V7 to V8") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             # In a real implementation, you would: >> src\bridge\v7_v8_bridge.py
    echo             # 1. Convert V7 seed concepts to V8 concept format >> src\bridge\v7_v8_bridge.py
    echo             # 2. Use V8 API to add these concepts to the knowledge database >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo             logger.error(f"Error transferring from V7 to V8: {e}") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def _transfer_v8_to_v7(self): >> src\bridge\v7_v8_bridge.py
    echo         """Transfer knowledge from V8 to V7""" >> src\bridge\v7_v8_bridge.py
    echo         if not self.v7_seed: >> src\bridge\v7_v8_bridge.py
    echo             logger.warning("Cannot transfer from V8 to V7: V7 seed not available") >> src\bridge\v7_v8_bridge.py
    echo             return >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         try: >> src\bridge\v7_v8_bridge.py
    echo             # Get metrics from V8 >> src\bridge\v7_v8_bridge.py
    echo             response = requests.get(self.v8_metrics_url, timeout=5) >> src\bridge\v7_v8_bridge.py
    echo             if response.status_code != 200: >> src\bridge\v7_v8_bridge.py
    echo                 logger.warning(f"V8 metrics returned status code {response.status_code}") >> src\bridge\v7_v8_bridge.py
    echo                 return >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             v8_metrics = response.json() >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             # For now, just log what we would transfer >> src\bridge\v7_v8_bridge.py
    echo             if v8_metrics and isinstance(v8_metrics, list) and len(v8_metrics) > 0: >> src\bridge\v7_v8_bridge.py
    echo                 latest_metrics = v8_metrics[0] >> src\bridge\v7_v8_bridge.py
    echo                 logger.info(f"V8 has {latest_metrics.get('total_concepts', 0)} concepts to potentially transfer to V7") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             # In a real implementation, you would: >> src\bridge\v7_v8_bridge.py
    echo             # 1. Get concepts from V8 knowledge database >> src\bridge\v7_v8_bridge.py
    echo             # 2. Convert to V7 seed format >> src\bridge\v7_v8_bridge.py
    echo             # 3. Add to V7 seed dictionary >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo             logger.error(f"Error transferring from V8 to V7: {e}") >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def get_status(self): >> src\bridge\v7_v8_bridge.py
    echo         """Get the current status of the bridge""" >> src\bridge\v7_v8_bridge.py
    echo         return { >> src\bridge\v7_v8_bridge.py
    echo             "running": self.running, >> src\bridge\v7_v8_bridge.py
    echo             "last_sync": self.last_sync.isoformat() if self.last_sync else None, >> src\bridge\v7_v8_bridge.py
    echo             "v7_connected": self.v7_seed is not None, >> src\bridge\v7_v8_bridge.py
    echo             "v8_health": self._check_v8_health() >> src\bridge\v7_v8_bridge.py
    echo         } >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo def main(): >> src\bridge\v7_v8_bridge.py
    echo     """Main function to run the V7-V8 bridge""" >> src\bridge\v7_v8_bridge.py
    echo     parser = argparse.ArgumentParser(description="V7-V8 Integration Bridge") >> src\bridge\v7_v8_bridge.py
    echo     parser.add_argument("--v8-port", type=int, default=8765, help="V8 health check port") >> src\bridge\v7_v8_bridge.py
    echo     parser.add_argument("--v7-port", type=int, default=5678, help="V7 connection port") >> src\bridge\v7_v8_bridge.py
    echo     args = parser.parse_args() >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     bridge = V7V8Bridge(v8_health_port=args.v8_port, v7_connection_port=args.v7_port) >> src\bridge\v7_v8_bridge.py
    echo     bridge.start() >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     try: >> src\bridge\v7_v8_bridge.py
    echo         logger.info("V7-V8 Bridge running. Press Ctrl+C to stop.") >> src\bridge\v7_v8_bridge.py
    echo         while True: >> src\bridge\v7_v8_bridge.py
    echo             time.sleep(10) >> src\bridge\v7_v8_bridge.py
    echo     except KeyboardInterrupt: >> src\bridge\v7_v8_bridge.py
    echo         logger.info("Keyboard interrupt received, stopping bridge...") >> src\bridge\v7_v8_bridge.py
    echo     finally: >> src\bridge\v7_v8_bridge.py
    echo         bridge.stop() >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo if __name__ == "__main__": >> src\bridge\v7_v8_bridge.py
    echo     main() >> src\bridge\v7_v8_bridge.py
)

:: Check for the requests library (needed for the bridge)
%PYTHON_CMD% -c "import requests" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Requests is not installed. This is required for the V7-V8 Bridge.
    echo Installing Requests...
    pip install requests
    if %errorlevel% neq 0 (
        echo Failed to install Requests. Please install it manually with:
        echo pip install requests
        pause
        exit /b 1
    )
    echo Requests installed successfully.
)

:: Main Integration Menu
:MAIN_MENU
cls
echo.
echo LUMINA INTEGRATED V7+V8 SYSTEM
echo ------------------------------
echo [1] Start Full Integrated System (V7 Holographic + V8 Knowledge CI/CD)
echo [2] Start V7 Holographic System Only
echo [3] Start V8 Knowledge CI/CD System Only
echo [4] Start Integration Bridge Only
echo [5] View System Status
echo [6] Exit
echo.

set /p choice="Enter your choice: "

if "%choice%"=="1" goto START_INTEGRATED
if "%choice%"=="2" goto START_V7
if "%choice%"=="3" goto START_V8
if "%choice%"=="4" goto START_BRIDGE
if "%choice%"=="5" goto VIEW_STATUS
if "%choice%"=="6" goto EXIT_SYSTEM

echo Invalid choice. Please try again.
goto MAIN_MENU

:START_INTEGRATED
cls
echo.
echo Starting Full Integrated System (V7 Holographic + V8 Knowledge CI/CD)...
echo.

:: Start the V8 Knowledge CI/CD system
echo Starting V8 Knowledge CI/CD System...
start "LUMINA V8 Knowledge CI/CD" cmd /c "%PYTHON_CMD% -m src.v8.knowledge_ci_cd_integrated --health-check-port=%V8_HEALTH_CHECK_PORT%"
echo V8 Knowledge CI/CD system started on port %V8_HEALTH_CHECK_PORT%
echo Wait for V8 system to initialize...
timeout /t 5 /nobreak >nul

:: Start Neural Seed System
echo Starting Neural Seed System...
start "LUMINA Neural Seed System" %PYTHON_CMD% -m src.seed --background --growth-rate=medium
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Neural Seed System.
    echo Please check that the src/seed.py module exists and has no errors.
    pause
    goto MAIN_MENU
)

:: Start the V7-V8 Integration Bridge
echo Starting V7-V8 Integration Bridge...
start "LUMINA V7-V8 Bridge" %PYTHON_CMD% -m src.bridge.v7_v8_bridge --v8-port=%V8_HEALTH_CHECK_PORT% --v7-port=%V7_CONNECTION_PORT%
echo V7-V8 Integration Bridge started

:: Start the holographic frontend
echo Starting V7 Holographic Frontend...
start "LUMINA V7 Holographic Interface" %PYTHON_CMD% -m src.v7.ui.holographic_frontend --gui-framework %GUI_FRAMEWORK% --mock
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Holographic Interface.
    echo Please check that PySide6 is installed and src/v7/ui/holographic_frontend.py exists.
    pause
    goto MAIN_MENU
)

:: Wait for V7 system to initialize
timeout /t 3 >nul

:: Start V7.5 Chat Interface with integration
echo Starting V7.5 Chat Interface with V8 integration...
start "LUMINA V7.5 Chat Interface" %PYTHON_CMD% -m src.v7_5.lumina_frontend --interlink --port %V7_5_PORT% --mock
if %errorlevel% neq 0 (
    echo WARNING: Failed to start V7.5 Chat Interface.
    echo Continuing with limited functionality...
    timeout /t 2 >nul
)

:: Start the dashboard
echo Starting Dashboard with V8 integration...
%PYTHON_CMD% -c "import src.visualization.run_qt_dashboard" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Dashboard visualization module not found.
    echo The visualization module will not be started.
    echo Please create the visualization module or install it properly.
    pause
    goto MAIN_MENU
)

if "%GUI_FRAMEWORK%"=="PySide6" (
    start "LUMINA Dashboard" %PYTHON_CMD% -m src.visualization.run_qt_dashboard --v7-port %V7_CONNECTION_PORT% --v8-port %V8_HEALTH_CHECK_PORT% --db-path %METRICS_DB_PATH% --gui-framework PySide6 --mock
) else (
    start "LUMINA Dashboard" %PYTHON_CMD% -m src.visualization.run_qt_dashboard --v7-port %V7_CONNECTION_PORT% --v8-port %V8_HEALTH_CHECK_PORT% --db-path %METRICS_DB_PATH% --mock
)

echo Full Integrated System Started!
echo.
echo Press any key to return to the main menu (systems will continue running)...
pause >nul
goto MAIN_MENU

:START_V7
cls
echo.
echo Starting V7 Holographic System Only...
echo.

:: Call the original V7 batch file
call run_v7_holographic.bat
goto MAIN_MENU

:START_V8
cls
echo.
echo Starting V8 Knowledge CI/CD System Only...
echo.

:: Call the V8 batch file
call run_knowledge_ci_cd_integrated.bat
goto MAIN_MENU

:START_BRIDGE
cls
echo.
echo Starting V7-V8 Integration Bridge Only...
echo.

:: Check if both systems are running
echo Checking if V7 and V8 systems are running...
%PYTHON_CMD% -c "import requests; requests.get('http://localhost:%V8_HEALTH_CHECK_PORT%/health', timeout=2)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: V8 Knowledge CI/CD system does not appear to be running.
    echo Please start the V8 system first.
    pause
    goto MAIN_MENU
)

:: Start the bridge
echo Starting V7-V8 Integration Bridge...
%PYTHON_CMD% -m src.bridge.v7_v8_bridge --v8-port=%V8_HEALTH_CHECK_PORT% --v7-port=%V7_CONNECTION_PORT%
goto MAIN_MENU

:VIEW_STATUS
cls
echo.
echo Checking System Status...
echo.

:: Check V8 Knowledge CI/CD System
echo V8 Knowledge CI/CD System:
%PYTHON_CMD% -c "import requests; response = requests.get('http://localhost:%V8_HEALTH_CHECK_PORT%/health', timeout=2); print(f'Status: {response.json()[\"status\"]}'); print(f'Components: {response.json()[\"components\"]}');" 2>nul
if %errorlevel% neq 0 (
    echo Status: Not running or unavailable
)
echo.

:: Check V7 Holographic System (using simple process check)
echo V7 Holographic System:
tasklist /FI "WINDOWTITLE eq *Holographic Interface*" >nul 2>&1
if %errorlevel% equ 0 (
    echo Status: Running
) else (
    echo Status: Not running or unavailable
)
echo.

:: Check Integration Bridge (using simple process check)
echo V7-V8 Integration Bridge:
tasklist /FI "WINDOWTITLE eq *V7-V8 Bridge*" >nul 2>&1
if %errorlevel% equ 0 (
    echo Status: Running
) else (
    echo Status: Not running or unavailable
)
echo.

echo Press any key to return to the main menu...
pause >nul
goto MAIN_MENU

:EXIT_SYSTEM
cls
echo Shutting down LUMINA Integrated System...

:: Attempt to stop the V7-V8 bridge
taskkill /F /FI "WINDOWTITLE eq *V7-V8 Bridge*" >nul 2>&1

:: Attempt to stop the V8 Knowledge CI/CD system
taskkill /F /FI "WINDOWTITLE eq *LUMINA V8 Knowledge CI/CD*" >nul 2>&1

:: Stop all running Python processes (this will stop V7 components too)
taskkill /F /IM python.exe >nul 2>&1

echo Thank you for using LUMINA Integrated V7+V8 System.
timeout /t 2 >nul
exit 