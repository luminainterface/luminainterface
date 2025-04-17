@echo off
setlocal enabledelayedexpansion

:: Lumina v8 - Complete Knowledge Cycle System CI/CD
:: This batch file runs the complete knowledge cycle system and ensures
:: proper component setup, validation, and execution.

echo -----------------------------------------------------
echo Lumina v8 - Complete Knowledge Cycle CI/CD Pipeline
echo -----------------------------------------------------
echo.

:: Create a log file with timestamp
set TIMESTAMP=%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOG_FILE=logs\pipeline_%TIMESTAMP%.log
echo Running CI/CD pipeline with log: %LOG_FILE%

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs
echo [%date% %time%] Starting CI/CD pipeline > %LOG_FILE%

:: === ENVIRONMENT VALIDATION ===
call :log "Validating environment..."

:: Ensure Python is in the path
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    call :log "ERROR: Python not found in PATH. Please ensure Python is installed and in your PATH."
    goto :error
)

:: Get Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
call :log "Detected %PYTHON_VERSION%"

:: Ensure required directories exist
if not exist logs mkdir logs
if not exist data mkdir data
if not exist data\concepts mkdir data\concepts
if not exist data\roots mkdir data\roots
if not exist test_results mkdir test_results
if not exist data\conversations mkdir data\conversations
call :log "Directory structure validated"

:: Check for Mistral API key in environment
if defined MISTRAL_API_KEY (
    call :log "Mistral API key found in environment"
) else (
    :: Check for .env file
    if exist .env (
        call :log "Found .env file, checking for Mistral API key"
        findstr /C:"MISTRAL_API_KEY" .env >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            call :log "Mistral API key found in .env file"
        ) else (
            call :log "WARNING: Mistral API key not found in .env file. Chat functionality may use mock mode."
        )
    ) else (
        call :log "WARNING: No .env file found. Chat functionality will use mock mode."
    )
)

:: === DEPENDENCY VALIDATION ===
call :log "Checking dependencies..."

:: Create a requirements check script
echo import sys > check_deps.py
echo deps = ["PyQt5", "PySide6", "numpy", "matplotlib", "dotenv", "mistralai"] >> check_deps.py
echo missing = [] >> check_deps.py
echo for dep in deps: >> check_deps.py
echo     try: >> check_deps.py
echo         __import__(dep) >> check_deps.py
echo         print(f"{dep}: OK") >> check_deps.py
echo     except ImportError: >> check_deps.py
echo         missing.append(dep) >> check_deps.py
echo         print(f"{dep}: MISSING") >> check_deps.py
echo if missing: >> check_deps.py
echo     print(f"Missing dependencies: {', '.join(missing)}") >> check_deps.py
echo     sys.exit(1) >> check_deps.py
echo else: >> check_deps.py
echo     print("All dependencies satisfied") >> check_deps.py
echo     sys.exit(0) >> check_deps.py

:: Run dependency check
call :log "Checking Python dependencies..."
python check_deps.py > check_deps_output.txt 2>&1
if %ERRORLEVEL% neq 0 (
    call :log "Missing dependencies detected. Installing required packages..."
    pip install -r requirements.txt >> %LOG_FILE% 2>&1
    if %ERRORLEVEL% neq 0 (
        call :log "ERROR: Failed to install required packages. See log for details."
        goto :error
    )
    call :log "Dependencies installed successfully"
) else (
    call :log "All dependencies satisfied"
)
del check_deps.py

:: === GPU & 3D CAPABILITY CHECK ===
call :log "Checking system capability for 3D visualization..."
python check_gpu.py > test_results\gpu_check_%TIMESTAMP%.txt 2>&1
python check_qt3d.py > test_results\qt3d_check_%TIMESTAMP%.txt 2>&1
call :log "Capability checks complete. Results in test_results directory."

:: === COMPONENT TESTING ===
call :log "Running component tests..."

:: Test the spatial temple in headless mode
python -c "from src.v8.spatial_temple_mapper import SpatialTempleMapper; mapper = SpatialTempleMapper(); print('SpatialTempleMapper test: SUCCESS')" > test_results\mapper_test_%TIMESTAMP%.txt 2>&1
if %ERRORLEVEL% neq 0 (
    call :log "ERROR: SpatialTempleMapper test failed"
    goto :error
) else (
    call :log "SpatialTempleMapper test passed"
)

:: === MAIN EXECUTION ===
echo.
echo CI/CD Pipeline Complete. System validation successful.
echo.
echo Choose your startup configuration:
echo  [1] Start all components in integrated mode
echo  [2] Start root connection system only
echo  [3] Start complete cycle with visualization
echo  [4] Run full test suite only (no component startup)
echo  [5] Start Lumina v7.5 Chat Interface
echo.
echo  [0] Exit
echo.

set /p choice="Enter your choice (0-5): "

if "%choice%"=="0" goto end
if "%choice%"=="1" goto start_integrated
if "%choice%"=="2" goto start_root_only
if "%choice%"=="3" goto start_with_visualization
if "%choice%"=="4" goto run_tests
if "%choice%"=="5" goto start_chat
goto invalid_choice

:start_integrated
call :log "Starting the complete integrated knowledge cycle..."
python src\v8\root_connection_system.py >> %LOG_FILE% 2>&1
goto end

:start_root_only
call :log "Starting the root connection system only..."
python src\v8\root_connection_system.py --root-only >> %LOG_FILE% 2>&1
goto end

:start_with_visualization
call :log "Starting the complete knowledge cycle with visualization..."
echo This will launch multiple components to show the complete cycle.
echo.
echo Press any key to continue...
pause >nul

:: Start components in separate windows with proper error handling
call :log "Launching Root Connection System..."
start "Root Connection System" cmd /c "python src\v8\root_connection_system.py || (echo Root Connection System Error & pause)"
timeout /t 2 >nul

call :log "Launching Spatial Temple..."
start "Spatial Temple" cmd /c "python src\v8\run_spatial_temple.py --theme temple --nodes 70 --mode 2d || (echo Spatial Temple Error & pause)"
timeout /t 2 >nul

call :log "Launching Temple-to-Seed Bridge..."
start "Temple-to-Seed Bridge" cmd /c "python src\v8\temple_to_seed_bridge.py || (echo Temple-to-Seed Bridge Error & pause)"
goto end

:run_tests
call :log "Running full test suite..."
python src\tests\run_all_tests.py > test_results\full_test_%TIMESTAMP%.txt 2>&1
if %ERRORLEVEL% neq 0 (
    call :log "ERROR: Full test suite failed. See test results for details."
    goto :error
) else (
    call :log "Full test suite passed successfully"
)
goto end

:start_chat
call :log "Starting Lumina v7.5 Chat Interface..."
echo Starting Lumina v7.5 Integrated Chat Interface...
echo.
echo This component provides an advanced chat interface with conversation memory
echo and integration with the neural network system.
echo.
echo Press any key to continue...
pause >nul

:: Ensure the v7.5 directory exists
if not exist src\v7.5 (
    call :log "ERROR: src\v7.5 directory not found."
    goto :error
)

:: Ensure the frontend file exists
if not exist src\v7.5\lumina_frontend.py (
    call :log "ERROR: src\v7.5\lumina_frontend.py not found."
    goto :error
)

:: Start the chat interface
call :log "Launching Lumina v7.5 Chat Interface..."
python src\v7.5\lumina_frontend.py || (
    call :log "ERROR: Failed to start Lumina v7.5 Chat Interface."
    goto :error
)
goto end

:invalid_choice
call :log "Invalid choice. Please run the script again and select a valid option."
goto error

:error
call :log "Error running Lumina v8 Complete Knowledge Cycle CI/CD Pipeline."
echo Failed with errors. See log file: %LOG_FILE%
exit /b 1

:log
echo [%date% %time%] %~1
echo [%date% %time%] %~1 >> %LOG_FILE%
goto :EOF

:end
call :log "CI/CD Pipeline completed successfully"
echo.
echo Thank you for using the Lumina v8 Complete Knowledge Cycle System.
echo Log file: %LOG_FILE%
endlocal 