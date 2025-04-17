@echo off
setlocal

:: Lumina v8 - Knowledge Tree System Launcher
:: This batch file runs the complete integrated system with the Spatial Temple,
:: Temple-to-Seed Bridge, and Auto Growth systems working together to spread
:: knowledge like a tree spreading seeds.

echo -----------------------------------------------------
echo Lumina v8 - Knowledge Tree System
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
if not exist data mkdir data
if not exist data\concepts mkdir data\concepts
if not exist data\roots mkdir data\roots

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
echo Welcome to the Lumina v8 Knowledge Tree System
echo.
echo This integrated system implements a biological tree-like paradigm
echo for knowledge growth and dispersal, where:
echo.
echo 1. The Spatial Temple forms the "trunk" and main branches
echo 2. The Temple-to-Seed Bridge manages "fruits" that contain knowledge seeds
echo 3. The Auto Growth system allows seeds to autonomously spread to new areas
echo 4. The Root Connection system completes the cycle by returning nutrients
echo    from fruits back to the roots
echo.
echo Choose which component to start:
echo  [1] Complete Integrated System (without root connection)
echo  [2] Complete Knowledge Cycle (with bidirectional flow)
echo  [3] Spatial Temple Visualization only
echo  [4] Temple-to-Seed Bridge only
echo  [5] Auto Seed Growth System only
echo  [6] Root Connection System only
echo.
echo  [0] Exit
echo.

set /p choice="Enter your choice (1-6, or 0 to exit): "

if "%choice%"=="0" goto end
if "%choice%"=="1" goto start_integrated
if "%choice%"=="2" goto start_complete_cycle
if "%choice%"=="3" goto start_temple
if "%choice%"=="4" goto start_bridge
if "%choice%"=="5" goto start_growth
if "%choice%"=="6" goto start_roots
goto invalid_choice

:start_integrated
echo.
echo Starting the Integrated Knowledge Tree System...
echo.
echo The system will automatically open the Temple-to-Seed Bridge interface
echo which includes the Auto Growth functionality.
echo.
echo Press any key to continue...
pause >nul
start python src\v8\temple_to_seed_bridge.py
timeout /t 2 >nul
start python src\v8\run_spatial_temple.py --theme temple --node-count 70
goto end

:start_complete_cycle
echo.
echo Starting the Complete Knowledge Cycle System...
echo This system adds bidirectional flow, connecting fruits back to roots.
echo.
echo Press any key to continue...
pause >nul
start "Root Connection System" cmd /c "python src\v8\root_connection_system.py"
timeout /t 2 >nul
start "Spatial Temple" cmd /c "python src\v8\run_spatial_temple.py --theme temple --node-count 70"
timeout /t 2 >nul
start "Temple-to-Seed Bridge" cmd /c "python src\v8\temple_to_seed_bridge.py"
goto end

:start_temple
echo.
echo Starting the Spatial Temple Visualization...
echo.
python src\v8\run_spatial_temple.py --theme temple --node-count 70
goto end

:start_bridge
echo.
echo Starting the Temple-to-Seed Bridge...
echo.
python src\v8\temple_to_seed_bridge.py
goto end

:start_growth
echo.
echo Starting the Auto Seed Growth System...
echo.
python src\v8\auto_seed_growth.py
goto end

:start_roots
echo.
echo Starting the Root Connection System...
echo This system enables knowledge to flow from fruits back to roots.
echo.
python src\v8\root_connection_system.py
goto end

:invalid_choice
echo.
echo Invalid choice. Please run the script again and select a valid option.
goto error

:error
echo.
echo Error running Lumina v8 Knowledge Tree System.
exit /b 1

:end
echo.
echo Thank you for using the Lumina v8 Knowledge Tree System.
endlocal 