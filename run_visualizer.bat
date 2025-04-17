@echo off
echo Starting Neural Network Visualizer with System Grower...

REM Install required packages
echo Installing required packages...
pip install psutil GPUtil screeninfo PyQt5

REM Start the system grower backend
echo Starting system grower backend...
start /B python src/backend/system_grower.py

REM Wait for backend to initialize
timeout /t 2 /nobreak > nul

REM Start the visualizer
echo Starting visualizer...
python src/frontend/ui/visualizer_launcher.py

REM Cleanup
echo Stopping system grower...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq system_grower.py"

echo Visualizer stopped.
pause 