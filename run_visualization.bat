@echo off
echo Starting Recursive Pattern Visualizer...
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

:: Run the recursive pattern visualizer
echo Generating pattern visualizations...
python -c "import sys; sys.path.append('.'); from src.visualization.recursive_pattern_visualizer import RecursivePatternVisualizer; visualizer = RecursivePatternVisualizer(); visualizer.load_patterns(); visualizer.visualize_pattern_distribution(); visualizer.visualize_pattern_network(); visualizer.visualize_recursive_depth(); visualizer.generate_report(); print('Pattern visualizations complete.')"

:: Check if visualization was successful
if %ERRORLEVEL% neq 0 (
    echo Visualization generation failed.
    echo Check logs for more information.
    pause
    exit /b 1
)

:: Open the visualization output directory
echo Opening visualization directory...
start "" "output\visualizations"

:: Exit
echo.
echo Visualization process complete.
pause 