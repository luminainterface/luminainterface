@echo off
setlocal

:: Lumina v8 - Temple to Seed Bridge Launcher
:: This batch file runs the Temple to Seed Bridge system which connects
:: the Spatial Temple with the Seed Dispersal System to allow concepts to
:: flow like fruits and seeds through the system.

echo -----------------------------------------------------
echo Lumina v8 - Temple to Seed Bridge
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
echo Starting Lumina v8 Temple to Seed Bridge...
echo This interface allows concepts to flow from the Spatial Temple
echo to other systems like a tree spreading its fruits.
echo.

:: Run the temple to seed bridge
python src\v8\temple_to_seed_bridge.py

if %ERRORLEVEL% neq 0 goto :error

goto :end

:error
echo.
echo Error running Lumina v8 Temple to Seed Bridge.
exit /b 1

:end
echo.
echo Lumina v8 Temple to Seed Bridge session completed.
endlocal 