@echo off
setlocal

:: Lumina v8 Spatial Temple Launcher
:: This batch file runs the Lumina v8 Spatial Temple system

echo -----------------------------------------------------
echo Lumina v8 - Spatial Temple
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
    pip install PySide6 PyOpenGL PyOpenGL_accelerate
    if %ERRORLEVEL% neq 0 (
        echo Failed to install required packages.
        goto :error
    )
)

:: Check for Qt3D modules (optional)
echo Checking for Qt3D modules...
python -c "from PySide6 import Qt3DCore" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Qt3D modules not found. 3D visualization will use fallback mode.
    echo Full 3D experience requires PySide6 with Qt3D modules.
) else (
    echo Qt3D modules detected - full 3D visualization available.
)

echo.
echo Starting Lumina v8 Spatial Temple...
echo.

:: Run the spatial temple visualization with demo data and themed layout
python src\v8\run_spatial_temple.py --demo --theme temple --node-count 70

if %ERRORLEVEL% neq 0 goto :error

goto :end

:error
echo.
echo Error running Lumina v8 Spatial Temple.
exit /b 1

:end
echo.
echo Lumina v8 Spatial Temple session completed.
endlocal 