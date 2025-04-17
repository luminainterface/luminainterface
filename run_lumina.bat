@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo LUMINA Client Launcher
echo ===========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

:: Check Python dependencies
echo Checking Python dependencies...
python -c "import PySide6, numpy, vispy" > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Some Python dependencies may be missing
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install required packages
        goto :end
    )
)

:: Check for GPU support
echo Checking GPU support...
python -c "import vispy; print(vispy.gl.get_default_config())" > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] GPU support may be limited
    echo The application will run in CPU mode
)

:: Create required directories
echo Checking required directories...
set dirs=assets logs assets\fonts model_versions config database spiderweb
for %%d in (%dirs%) do (
    if not exist "%%d" (
        echo Creating directory: %%d
        mkdir "%%d"
    )
)

:: Check for Consolas font
if not exist "assets\fonts\Consolas.ttf" (
    echo [WARNING] Consolas font not found
    echo Copying Consolas font from Windows...
    copy "C:\Windows\Fonts\consola.ttf" "assets\fonts\Consolas.ttf" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Could not copy Consolas font
        echo Please manually copy C:\Windows\Fonts\consola.ttf to assets\fonts\Consolas.ttf
    )
)

:: Initialize database if needed
if not exist "node_zero.db" (
    echo Initializing database...
    python -c "from database.database_manager import DatabaseManager; DatabaseManager.initialize_database()"
)

:: Launch the client with error handling
echo.
echo Starting LUMINA Client...
echo Press Ctrl+C to exit
echo.

:retry
python launch_lumina.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] LUMINA Client crashed with error code %errorlevel%
    echo.
    echo Troubleshooting steps:
    echo 1. Check if all required files are present
    echo 2. Verify Python dependencies are installed
    echo 3. Check system resources
    echo 4. Look for error messages above
    echo 5. Ensure all required directories exist
    echo.
    set /p retry="Would you like to try again? (y/n): "
    if /i "!retry!"=="y" goto retry
)

:: Exit successfully
exit /b 0

:end
echo.
echo Press any key to exit...
pause > nul 