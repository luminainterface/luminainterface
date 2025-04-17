@echo off
echo LUMINA V7.5 - Dependencies Installation
echo =======================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8 or newer.
    echo Visit https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found. Installing required packages...
echo.

:: Create requirements.txt if it doesn't exist
if not exist "requirements.txt" (
    echo Creating requirements.txt file...
    echo PySide6>=6.4.0 > requirements.txt
    echo numpy>=1.21.0 >> requirements.txt
    echo matplotlib>=3.5.0 >> requirements.txt
    echo pandas>=1.3.0 >> requirements.txt
    echo pyqtgraph>=0.12.0 >> requirements.txt
    echo python-dotenv>=0.19.0 >> requirements.txt
    echo requests>=2.26.0 >> requirements.txt
    echo SQLAlchemy>=1.4.0 >> requirements.txt
    echo scipy>=1.7.0 >> requirements.txt
    echo scikit-learn>=1.0.0 >> requirements.txt
    echo tqdm>=4.62.0 >> requirements.txt
    echo pillow>=8.3.0 >> requirements.txt
    echo psutil>=5.8.0 >> requirements.txt
    echo pytest>=6.2.5 >> requirements.txt
)

:: Install packages
echo Installing packages from requirements.txt...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install some packages.
    echo You may need to run this script with administrator privileges or use:
    echo     pip install --user -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo All dependencies successfully installed!
echo.

:: Create necessary directories
echo Creating necessary directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "data\neural" mkdir data\neural
if not exist "data\memory" mkdir data\memory
if not exist "data\onsite_memory" mkdir data\onsite_memory
if not exist "data\seed" mkdir data\seed
if not exist "data\dream" mkdir data\dream
if not exist "data\autowiki" mkdir data\autowiki
if not exist "data\consciousness" mkdir data\consciousness
if not exist "data\breath" mkdir data\breath
if not exist "data\v7.5" mkdir data\v7.5
if not exist "data\conversations" mkdir data\conversations
if not exist "data\db" mkdir data\db
if not exist "logs\db" mkdir logs\db
if not exist "logs\chat" mkdir logs\chat
if not exist "logs\monitor" mkdir logs\monitor

echo.
echo Setup complete. You can now run the LUMINA V7.5 system by executing:
echo     run_v7_holographic.bat
echo.
pause 