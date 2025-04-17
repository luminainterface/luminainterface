@echo off
echo Installing LUMINA V7 Dashboard Dependencies...

:: Create Python virtual environment if it doesn't exist
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        echo Please ensure Python 3.8+ is installed and in your PATH.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install required dependencies
echo Installing core dependencies...
pip install --upgrade pip
pip install PyQt5 PySide6 pyqtgraph matplotlib numpy pandas

:: Install system monitoring dependencies
echo Installing system monitoring tools...
pip install psutil GPUtil

:: Install visualization dependencies
echo Installing visualization libraries...
pip install seaborn plotly

:: Install database dependencies
echo Installing database libraries...
pip install sqlite3-api

:: Install API and communication dependencies
echo Installing API and communication libraries...
pip install flask flask-cors flask-socketio requests websockets

:: Install security dependencies
echo Installing security libraries...
pip install bcrypt PyJWT cryptography

:: Create necessary directories
echo Creating necessary directories...
if not exist data mkdir data
if not exist data\neural_metrics mkdir data\neural_metrics
if not exist data\backups mkdir data\backups
if not exist logs mkdir logs
if not exist config mkdir config
if not exist output\exports mkdir output\exports

echo Installation complete!
echo.
echo To run the dashboard, use: run_qt_dashboard.bat
echo.
pause 