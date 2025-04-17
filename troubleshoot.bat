@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo LUMINA System Troubleshooting Tool
echo ===========================================
echo.

:: Check Python version
echo Checking Python version...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    goto :end
) else (
    echo [OK] Python is installed
)

:: Check Python version compatibility
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo Detected Python version: !pyver!
if "!pyver:~0,3!" lss "3.8" (
    echo [WARNING] Python version !pyver! may not be compatible
    echo Recommended: Python 3.8 or higher
)

:: Check required directories
echo.
echo Checking required directories...
set dirs=assets logs src
for %%d in (%dirs%) do (
    if not exist "%%d" (
        echo [WARNING] Directory "%%d" does not exist
        echo Creating directory...
        mkdir "%%d"
    ) else (
        echo [OK] Directory "%%d" exists
    )
)

:: Check required files
echo.
echo Checking required files...
set files=lumina_client.py src\central_node_monitor.py
for %%f in (%files%) do (
    if not exist "%%f" (
        echo [ERROR] Required file "%%f" not found
        set missing_files=1
    ) else (
        echo [OK] File "%%f" exists
    )
)

:: Check Python dependencies
echo.
echo Checking Python dependencies...
python -c "import PySide6, numpy, vispy" > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Some Python dependencies may be missing
    echo Installing required packages...
    pip install PySide6 numpy vispy
) else (
    echo [OK] Required Python packages are installed
)

:: Check system resources
echo.
echo Checking system resources...
wmic cpu get name | findstr /v "Name" > nul
if %errorlevel% equ 0 (
    echo [OK] CPU information available
) else (
    echo [WARNING] Could not retrieve CPU information
)

wmic memorychip get capacity | findstr /v "Capacity" > nul
if %errorlevel% equ 0 (
    echo [OK] Memory information available
) else (
    echo [WARNING] Could not retrieve memory information
)

:: Check GPU support
echo.
echo Checking GPU support...
python -c "import vispy; print(vispy.gl.get_default_config())" > nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] GPU support detected
) else (
    echo [WARNING] GPU support may be limited
)

:: Check environment variables
echo.
echo Checking environment variables...
if "%PYTHONPATH%"=="" (
    echo [INFO] PYTHONPATH is not set
) else (
    echo [OK] PYTHONPATH is set
)

:: Generate summary
echo.
echo ===========================================
echo Troubleshooting Summary
echo ===========================================
if defined missing_files (
    echo [CRITICAL] Missing required files
    echo Please ensure all required files are present
) else (
    echo [OK] All required files present
)

echo.
echo Recommendations:
echo 1. Ensure Python 3.8+ is installed
echo 2. Run 'pip install -r requirements.txt' if dependencies are missing
echo 3. Check system resources meet minimum requirements
echo 4. Verify GPU drivers are up to date if using GPU acceleration
echo.

:end
echo Troubleshooting complete. Press any key to exit...
pause > nul 