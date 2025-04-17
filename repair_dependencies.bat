@echo off
echo ===================================
echo   LUMINA V7 Dependency Repair Tool
echo ===================================
echo.

setlocal enabledelayedexpansion

:: Check for Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python and make sure it's in your PATH
    echo.
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

:: Clean environment option
echo Would you like to clean the existing environment before installing dependencies?
echo This will remove and reinstall PySide6 and other core packages.
echo.
set /p clean_env="Clean environment? (y/n): "

if /i "%clean_env%"=="y" (
    echo.
    echo Cleaning environment...
    python -m pip uninstall -y PySide6 PyQt5 PyQt6 numpy pandas matplotlib
    echo Environment cleaned.
    echo.
)

:: Update pip
echo Updating pip...
python -m pip install --upgrade pip
echo.

:: Install core dependencies
echo Installing core dependencies...
python -m pip install PySide6>=6.6.0 --upgrade
python -m pip install numpy>=1.26.0 --upgrade
echo.

:: Install from requirements file
echo Installing from requirements.txt...
if exist requirements.txt (
    python -m pip install -r requirements.txt --upgrade
) else (
    echo requirements.txt not found, installing minimal dependencies...
    python -m pip install pandas matplotlib scipy requests websockets
)
echo.

:: Check installation
echo Checking installation...
python -c "import sys; print(f'Python {sys.version}')"
python -c "import PySide6; print(f'PySide6 {PySide6.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PySide6 installation failed.
    echo Please try running:
    echo   python -m pip install PySide6 --force-reinstall
    echo.
) else (
    echo PySide6 installed successfully.
)

echo.
echo ===================================
echo   Repair completed
echo ===================================
echo.
echo Now you can run launch_v7.bat to start the application.
echo.
pause 