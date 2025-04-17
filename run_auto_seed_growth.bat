@echo off
setlocal

:: Lumina v8 - Auto Seed Growth System Launcher
:: This batch file runs the automated seed growth system which allows
:: concepts to autonomously discover and connect to external knowledge sources.

echo -----------------------------------------------------
echo Lumina v8 - Auto Seed Growth System
echo -----------------------------------------------------
echo.

:: Ensure Python is in the path
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please ensure Python is installed and in your PATH.
    goto :error
)

:: Ensure required directories exist
if not exist logs mkdir logs

:: Check if PySide6 is installed
python -c "import PySide6" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo PySide6 not found. Installing required packages...
    pip install PySide6
    if %ERRORLEVEL% neq 0 (
        echo Failed to install required packages.
        goto :error
    )
)

echo.
echo Starting Lumina v8 Auto Seed Growth System...
echo This system allows concepts to autonomously discover and grow
echo from external knowledge sources - like mold attaching to mold
echo or seeds growing where they find fertile ground.
echo.
echo Press Ctrl+C at any time to stop the growth process.
echo.

:: Run the auto seed growth system
python src\v8\auto_seed_growth.py

if %ERRORLEVEL% neq 0 goto :error

goto :end

:error
echo.
echo Error running Lumina v8 Auto Seed Growth System.
exit /b 1

:end
echo.
echo Lumina v8 Auto Seed Growth System completed.
endlocal 