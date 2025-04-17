@echo off
echo Starting LUMINA Integrated V7-V8 Bridge System...

:: Create necessary directories if they don't exist
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "data\bridge" mkdir data\bridge
if not exist "logs\bridge" mkdir logs\bridge

:: Set Python path to include the project root for proper imports
set PYTHONPATH=%CD%

:: Set Python command variable for consistency
set PYTHON_CMD=python

:: Set environment variables
set DASHBOARD_PORT=5679
set V7_CONNECTION_PORT=5678
set V8_HEALTH_CHECK_PORT=8080
set V8_METRICS_ENDPOINT=http://localhost:8080/metrics

:: Check if Python is installed
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check for required modules
echo Checking for required modules...
%PYTHON_CMD% -c "import sys; import os; missing=False;
modules=['src.v7', 'src.v8.knowledge_ci_cd_integrated'];
for m in modules:
    try: __import__(m); print(f'✓ {m} found')
    except ImportError as e: print(f'✗ {m} not found: {e}'); missing=True
if missing: sys.exit(1)
"

if %errorlevel% neq 0 (
    echo Required modules are missing. Please check the errors above.
    pause
    exit /b 1
)

:: Create bridge module if it doesn't exist
if not exist "src\bridge" mkdir src\bridge
if not exist "src\bridge\__init__.py" (
    echo # Bridge module between v7 and v8 systems > src\bridge\__init__.py
)

:: Create v7_v8_bridge.py if it doesn't exist
if not exist "src\bridge\v7_v8_bridge.py" (
    echo Creating bridge module...
    echo import os, sys, time, threading, logging, json, requests > src\bridge\v7_v8_bridge.py
    echo from datetime import datetime >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo # Configure logging >> src\bridge\v7_v8_bridge.py
    echo logging.basicConfig^( >> src\bridge\v7_v8_bridge.py
    echo     level=logging.INFO, >> src\bridge\v7_v8_bridge.py
    echo     format='%%^(asctime^)s - %%^(name^)s - %%^(levelname^)s - %%^(message^)s', >> src\bridge\v7_v8_bridge.py
    echo     handlers=[ >> src\bridge\v7_v8_bridge.py
    echo         logging.FileHandler^("logs/bridge/v7_v8_bridge.log"^), >> src\bridge\v7_v8_bridge.py
    echo         logging.StreamHandler^(^) >> src\bridge\v7_v8_bridge.py
    echo     ] >> src\bridge\v7_v8_bridge.py
    echo ^) >> src\bridge\v7_v8_bridge.py
    echo logger = logging.getLogger^("V7-V8-Bridge"^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo class V7V8Bridge: >> src\bridge\v7_v8_bridge.py
    echo     """Bridge class to connect v7 holographic system with v8 knowledge CI/CD system""" >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def __init__^(self, v8_health_endpoint='http://localhost:8080/health', >> src\bridge\v7_v8_bridge.py
    echo                v8_metrics_endpoint='http://localhost:8080/metrics', >> src\bridge\v7_v8_bridge.py
    echo                v7_port=5678^): >> src\bridge\v7_v8_bridge.py
    echo         self.v8_health_endpoint = v8_health_endpoint >> src\bridge\v7_v8_bridge.py
    echo         self.v8_metrics_endpoint = v8_metrics_endpoint >> src\bridge\v7_v8_bridge.py
    echo         self.v7_port = v7_port >> src\bridge\v7_v8_bridge.py
    echo         self.running = False >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread = None >> src\bridge\v7_v8_bridge.py
    echo         self.metrics_cache = {} >> src\bridge\v7_v8_bridge.py
    echo         logger.info^("V7-V8 Bridge initialized"^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def check_v8_health^(self^): >> src\bridge\v7_v8_bridge.py
    echo         """Check health status of v8 knowledge system""" >> src\bridge\v7_v8_bridge.py
    echo         try: >> src\bridge\v7_v8_bridge.py
    echo             response = requests.get^(self.v8_health_endpoint, timeout=5^) >> src\bridge\v7_v8_bridge.py
    echo             if response.status_code == 200: >> src\bridge\v7_v8_bridge.py
    echo                 health_data = response.json^(^) >> src\bridge\v7_v8_bridge.py
    echo                 logger.debug^(f"V8 health check: {health_data}"^) >> src\bridge\v7_v8_bridge.py
    echo                 return health_data >> src\bridge\v7_v8_bridge.py
    echo             else: >> src\bridge\v7_v8_bridge.py
    echo                 logger.warning^(f"V8 health check failed with status code {response.status_code}"^) >> src\bridge\v7_v8_bridge.py
    echo                 return {"status": "unavailable", "error": f"Status code: {response.status_code}"} >> src\bridge\v7_v8_bridge.py
    echo         except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo             logger.error^(f"Error checking V8 health: {str^(e^)}"^) >> src\bridge\v7_v8_bridge.py
    echo             return {"status": "error", "error": str^(e^)} >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def get_v8_metrics^(self^): >> src\bridge\v7_v8_bridge.py
    echo         """Get metrics from v8 knowledge system""" >> src\bridge\v7_v8_bridge.py
    echo         try: >> src\bridge\v7_v8_bridge.py
    echo             response = requests.get^(self.v8_metrics_endpoint, timeout=5^) >> src\bridge\v7_v8_bridge.py
    echo             if response.status_code == 200: >> src\bridge\v7_v8_bridge.py
    echo                 metrics_data = response.json^(^) >> src\bridge\v7_v8_bridge.py
    echo                 logger.debug^(f"V8 metrics retrieved: {len^(metrics_data^)} entries"^) >> src\bridge\v7_v8_bridge.py
    echo                 self.metrics_cache = metrics_data >> src\bridge\v7_v8_bridge.py
    echo                 return metrics_data >> src\bridge\v7_v8_bridge.py
    echo             else: >> src\bridge\v7_v8_bridge.py
    echo                 logger.warning^(f"V8 metrics retrieval failed with status code {response.status_code}"^) >> src\bridge\v7_v8_bridge.py
    echo                 return {} >> src\bridge\v7_v8_bridge.py
    echo         except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo             logger.error^(f"Error getting V8 metrics: {str^(e^)}"^) >> src\bridge\v7_v8_bridge.py
    echo             return {} >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def _bridge_loop^(self^): >> src\bridge\v7_v8_bridge.py
    echo         """Main bridge loop that runs in a thread""" >> src\bridge\v7_v8_bridge.py
    echo         logger.info^("Bridge loop started"^) >> src\bridge\v7_v8_bridge.py
    echo         while self.running: >> src\bridge\v7_v8_bridge.py
    echo             try: >> src\bridge\v7_v8_bridge.py
    echo                 # Check V8 health >> src\bridge\v7_v8_bridge.py
    echo                 health = self.check_v8_health^(^) >> src\bridge\v7_v8_bridge.py
    echo                 # Get V8 metrics if healthy >> src\bridge\v7_v8_bridge.py
    echo                 if health.get^("status"^) in ["ok", "degraded"]: >> src\bridge\v7_v8_bridge.py
    echo                     metrics = self.get_v8_metrics^(^) >> src\bridge\v7_v8_bridge.py
    echo                     # TODO: Connect to V7 dashboard to visualize metrics >> src\bridge\v7_v8_bridge.py
    echo                     # TODO: Connect to V7 system to feed knowledge >> src\bridge\v7_v8_bridge.py
    echo             except Exception as e: >> src\bridge\v7_v8_bridge.py
    echo                 logger.error^(f"Error in bridge loop: {str^(e^)}"^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo             # Sleep for 10 seconds before next check >> src\bridge\v7_v8_bridge.py
    echo             time.sleep^(10^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def start^(self^): >> src\bridge\v7_v8_bridge.py
    echo         """Start the bridge""" >> src\bridge\v7_v8_bridge.py
    echo         if self.running: >> src\bridge\v7_v8_bridge.py
    echo             logger.warning^("Bridge is already running"^) >> src\bridge\v7_v8_bridge.py
    echo             return >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         self.running = True >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread = threading.Thread^(target=self._bridge_loop^) >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread.daemon = True >> src\bridge\v7_v8_bridge.py
    echo         self.bridge_thread.start^(^) >> src\bridge\v7_v8_bridge.py
    echo         logger.info^("Bridge started"^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     def stop^(self^): >> src\bridge\v7_v8_bridge.py
    echo         """Stop the bridge""" >> src\bridge\v7_v8_bridge.py
    echo         if not self.running: >> src\bridge\v7_v8_bridge.py
    echo             logger.warning^("Bridge is not running"^) >> src\bridge\v7_v8_bridge.py
    echo             return >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo         self.running = False >> src\bridge\v7_v8_bridge.py
    echo         if self.bridge_thread: >> src\bridge\v7_v8_bridge.py
    echo             self.bridge_thread.join^(timeout=5^) >> src\bridge\v7_v8_bridge.py
    echo         logger.info^("Bridge stopped"^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo def main^(^): >> src\bridge\v7_v8_bridge.py
    echo     """Main function to run the bridge""" >> src\bridge\v7_v8_bridge.py
    echo     import argparse >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     parser = argparse.ArgumentParser^(description='V7-V8 Bridge'^ >> src\bridge\v7_v8_bridge.py
    echo     parser.add_argument^('--v8-health-endpoint', default='http://localhost:8080/health', >> src\bridge\v7_v8_bridge.py
    echo                       help='V8 health check endpoint'^ >> src\bridge\v7_v8_bridge.py
    echo     parser.add_argument^('--v8-metrics-endpoint', default='http://localhost:8080/metrics', >> src\bridge\v7_v8_bridge.py
    echo                       help='V8 metrics endpoint'^ >> src\bridge\v7_v8_bridge.py
    echo     parser.add_argument^('--v7-port', type=int, default=5678, >> src\bridge\v7_v8_bridge.py
    echo                       help='V7 connection port'^ >> src\bridge\v7_v8_bridge.py
    echo     args = parser.parse_args^(^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     # Create and start bridge >> src\bridge\v7_v8_bridge.py
    echo     bridge = V7V8Bridge^( >> src\bridge\v7_v8_bridge.py
    echo         v8_health_endpoint=args.v8_health_endpoint, >> src\bridge\v7_v8_bridge.py
    echo         v8_metrics_endpoint=args.v8_metrics_endpoint, >> src\bridge\v7_v8_bridge.py
    echo         v7_port=args.v7_port >> src\bridge\v7_v8_bridge.py
    echo     ^ >> src\bridge\v7_v8_bridge.py
    echo     bridge.start^(^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo     try: >> src\bridge\v7_v8_bridge.py
    echo         while True: >> src\bridge\v7_v8_bridge.py
    echo             time.sleep^(1^) >> src\bridge\v7_v8_bridge.py
    echo     except KeyboardInterrupt: >> src\bridge\v7_v8_bridge.py
    echo         logger.info^("Keyboard interrupt received, stopping bridge..."^ >> src\bridge\v7_v8_bridge.py
    echo     finally: >> src\bridge\v7_v8_bridge.py
    echo         bridge.stop^(^) >> src\bridge\v7_v8_bridge.py
    echo. >> src\bridge\v7_v8_bridge.py
    echo if __name__ == "__main__": >> src\bridge\v7_v8_bridge.py
    echo     main^(^) >> src\bridge\v7_v8_bridge.py
)

:: Create launcher script
if exist "start_integrated_system.py" del start_integrated_system.py
echo import os, sys, subprocess, time, signal, argparse > start_integrated_system.py
echo from threading import Thread >> start_integrated_system.py
echo. >> start_integrated_system.py
echo def log(msg): >> start_integrated_system.py
echo     print(f"[LAUNCHER] {msg}") >> start_integrated_system.py
echo. >> start_integrated_system.py
echo def parse_args(): >> start_integrated_system.py
echo     parser = argparse.ArgumentParser(description='Start Integrated V7-V8 System') >> start_integrated_system.py
echo     parser.add_argument('--v8-only', action='store_true', help='Only start the v8 system') >> start_integrated_system.py
echo     parser.add_argument('--v7-only', action='store_true', help='Only start the v7 system') >> start_integrated_system.py
echo     parser.add_argument('--health-check-port', type=int, default=8080, help='Health check port for v8 system') >> start_integrated_system.py
echo     return parser.parse_args() >> start_integrated_system.py
echo. >> start_integrated_system.py
echo def start_v8_system(health_check_port): >> start_integrated_system.py
echo     log("Starting V8 Knowledge CI/CD System...") >> start_integrated_system.py
echo     v8_cmd = ["python", "-m", "src.v8.knowledge_ci_cd_integrated", "--health-check-port", str(health_check_port)] >> start_integrated_system.py
echo     return subprocess.Popen(v8_cmd) >> start_integrated_system.py
echo. >> start_integrated_system.py
echo def start_v7_system(): >> start_integrated_system.py
echo     log("Starting V7 Holographic System...") >> start_integrated_system.py
echo     v7_cmd = ["python", "-m", "src.v7.ui.holographic_frontend", "--mock"] >> start_integrated_system.py
echo     return subprocess.Popen(v7_cmd) >> start_integrated_system.py
echo. >> start_integrated_system.py
echo def start_bridge(health_check_port): >> start_integrated_system.py
echo     log("Starting V7-V8 Bridge...") >> start_integrated_system.py
echo     bridge_cmd = ["python", "-m", "src.bridge.v7_v8_bridge", >> start_integrated_system.py
echo                  "--v8-health-endpoint", f"http://localhost:{health_check_port}/health", >> start_integrated_system.py
echo                  "--v8-metrics-endpoint", f"http://localhost:{health_check_port}/metrics"] >> start_integrated_system.py
echo     return subprocess.Popen(bridge_cmd) >> start_integrated_system.py
echo. >> start_integrated_system.py
echo def main(): >> start_integrated_system.py
echo     args = parse_args() >> start_integrated_system.py
echo     processes = [] >> start_integrated_system.py
echo. >> start_integrated_system.py
echo     try: >> start_integrated_system.py
echo         # Start V8 Knowledge CI/CD System if not v7-only >> start_integrated_system.py
echo         if not args.v7_only: >> start_integrated_system.py
echo             v8_process = start_v8_system(args.health_check_port) >> start_integrated_system.py
echo             processes.append(("V8 System", v8_process)) >> start_integrated_system.py
echo             time.sleep(3)  # Wait for V8 to initialize >> start_integrated_system.py
echo. >> start_integrated_system.py
echo         # Start V7 Holographic System if not v8-only >> start_integrated_system.py
echo         if not args.v8_only: >> start_integrated_system.py
echo             v7_process = start_v7_system() >> start_integrated_system.py
echo             processes.append(("V7 System", v7_process)) >> start_integrated_system.py
echo             time.sleep(2)  # Wait for V7 to initialize >> start_integrated_system.py
echo. >> start_integrated_system.py
echo         # Start Bridge if both systems are running >> start_integrated_system.py
echo         if not args.v7_only and not args.v8_only: >> start_integrated_system.py
echo             bridge_process = start_bridge(args.health_check_port) >> start_integrated_system.py
echo             processes.append(("Bridge", bridge_process)) >> start_integrated_system.py
echo. >> start_integrated_system.py
echo         log("All components started successfully") >> start_integrated_system.py
echo         log("Press Ctrl+C to shut down all components") >> start_integrated_system.py
echo. >> start_integrated_system.py
echo         # Keep the script running until interrupted >> start_integrated_system.py
echo         while True: >> start_integrated_system.py
echo             time.sleep(1) >> start_integrated_system.py
echo. >> start_integrated_system.py
echo     except KeyboardInterrupt: >> start_integrated_system.py
echo         log("Shutting down all components...") >> start_integrated_system.py
echo     finally: >> start_integrated_system.py
echo         # Terminate all processes >> start_integrated_system.py
echo         for name, process in processes: >> start_integrated_system.py
echo             if process.poll() is None:  # If still running >> start_integrated_system.py
echo                 log(f"Stopping {name}...") >> start_integrated_system.py
echo                 process.terminate() >> start_integrated_system.py
echo                 process.wait(timeout=5) >> start_integrated_system.py
echo         log("All components have been stopped") >> start_integrated_system.py
echo. >> start_integrated_system.py
echo if __name__ == "__main__": >> start_integrated_system.py
echo     main() >> start_integrated_system.py

:: Main Menu
:MAIN_MENU
cls
echo.
echo LUMINA V7-V8 INTEGRATED SYSTEM
echo ---------------------------------
echo [1] Start Complete Integrated System (V7+V8+Bridge)
echo [2] Start V8 Knowledge CI/CD System Only
echo [3] Start V7 Holographic System Only
echo [4] Start Bridge Component Only
echo [5] Exit
echo.

set /p choice="Enter your choice: "

if "%choice%"=="1" goto START_INTEGRATED
if "%choice%"=="2" goto START_V8
if "%choice%"=="3" goto START_V7
if "%choice%"=="4" goto START_BRIDGE
if "%choice%"=="5" goto EXIT_SYSTEM

echo Invalid choice. Please try again.
goto MAIN_MENU

:START_INTEGRATED
cls
echo.
echo Starting Complete Integrated System (V7+V8+Bridge)...
echo.
%PYTHON_CMD% start_integrated_system.py
echo.
echo All components have been stopped. Returning to menu...
timeout /t 2 >nul
goto MAIN_MENU

:START_V8
cls
echo.
echo Starting V8 Knowledge CI/CD System Only...
echo.
%PYTHON_CMD% start_integrated_system.py --v8-only
echo.
echo V8 System has been stopped. Returning to menu...
timeout /t 2 >nul
goto MAIN_MENU

:START_V7
cls
echo.
echo Starting V7 Holographic System Only...
echo.
%PYTHON_CMD% start_integrated_system.py --v7-only
echo.
echo V7 System has been stopped. Returning to menu...
timeout /t 2 >nul
goto MAIN_MENU

:START_BRIDGE
cls
echo.
echo Starting Bridge Component Only...
echo.
%PYTHON_CMD% -m src.bridge.v7_v8_bridge
echo.
echo Bridge has been stopped. Returning to menu...
timeout /t 2 >nul
goto MAIN_MENU

:EXIT_SYSTEM
cls
echo Shutting down LUMINA V7-V8 Integrated System...
taskkill /F /IM python.exe >nul 2>&1
echo Thank you for using LUMINA.
timeout /t 2 >nul
exit 