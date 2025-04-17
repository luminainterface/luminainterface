@echo off
echo Starting LUMINA V7 Language Dashboard Integration...

:: Set environment variables
set PYTHONPATH=%CD%;%CD%\src
set DB_PATH=data/neural_metrics.db
set LLM_WEIGHT=0.7
set NN_WEIGHT=0.6
set GUI_FRAMEWORK=PySide6

:: Create necessary directories
if not exist data mkdir data
if not exist logs mkdir logs

:: Optional: Check for dependencies
echo Checking for required dependencies...
python -c "import sys; print(f'Python {sys.version}')"
python -c "import PySide6; print(f'PySide6 {PySide6.__version__}')" 2>nul || echo PySide6 not found, attempting to use PyQt5
python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')" 2>nul || echo Matplotlib not found, visualizations may be limited
python -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>nul || echo NumPy not found, some features may not work

:: Run the integrated dashboard
echo Launching Language Dashboard Integration...
python src/run_language_dashboard_integration.py --db-path=%DB_PATH% --llm-weight=%LLM_WEIGHT% --nn-weight=%NN_WEIGHT% --gui-framework=%GUI_FRAMEWORK%

echo Language Dashboard Integration has been closed.
pause 