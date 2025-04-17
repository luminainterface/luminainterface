@echo off
echo Starting Lumina Neural Network Visualization Tools...
echo.

:: Set up Python environment
set PYTHONPATH=%~dp0
cd %~dp0

:: Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and try again.
    pause
    exit /b 1
)

:: Create necessary directories if they don't exist
if not exist logs mkdir logs
if not exist data mkdir data
if not exist output mkdir output
if not exist output\visualizations mkdir output\visualizations

:: Check dashboard requirements first
echo Checking visualization requirements...
python src/visualization/check_dashboard_requirements.py --verbose
if %ERRORLEVEL% neq 0 (
    echo Requirements check failed.
    echo Please install the required libraries and try again.
    pause
    exit /b 1
)

:: Run the recursive pattern visualizer (non-blocking)
echo Generating pattern visualizations in background...
start "Pattern Visualizer" /B python -c "import sys; sys.path.append('.'); from src.visualization.recursive_pattern_visualizer import RecursivePatternVisualizer; visualizer = RecursivePatternVisualizer(); visualizer.load_patterns(); visualizer.visualize_pattern_distribution(); visualizer.visualize_pattern_network(); visualizer.visualize_recursive_depth(); visualizer.generate_report(); print('Pattern visualizations complete.')"

:: Brief pause to allow visualizer to start
timeout /t 2 /nobreak > nul

:: Start the visualization dashboard
echo Starting dashboard...
python src/visualization/create_dashboard.py

:: If the dashboard exits, open the visualization folder
echo.
echo Dashboard has stopped. Opening visualization results...
start "" "output\visualizations"

:: Exit
echo.
echo All visualization tools completed.
pause 