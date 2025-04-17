@echo on
echo DIAGNOSTIC: Starting diagnostics...

echo DIAGNOSTIC: Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

echo DIAGNOSTIC: Checking directory structure...
if not exist "data" (
    echo Creating data directory
    mkdir data
)
if not exist "src" (
    echo ERROR: src directory missing
    pause
    exit /b 1
)

echo DIAGNOSTIC: Checking for critical modules...
python -c "import sys; print('Python path:', sys.path)" 
echo.

echo DIAGNOSTIC: Checking for v7.5 module...
python -c "import sys; sys.path.append('.')"
python -c "try:
    import src.v7_5
    print('v7_5 found!')
except ImportError as e:
    print('Import error: {}'.format(e))"
echo.

echo DIAGNOSTIC: Checking for holographic_frontend module...
python -c "try:
    import src.v7.ui.holographic_frontend
    print('holographic_frontend found!')
except ImportError as e:
    print('Import error: {}'.format(e))"
echo.

echo DIAGNOSTIC: Checking for dashboard module...
python -c "try:
    import src.dashboard.run_dashboard
    print('dashboard found!')
except ImportError as e:
    print('Import error: {}'.format(e))"
echo.

echo DIAGNOSTIC: Checking for seed module...
python -c "try:
    import src.seed
    print('seed found!')
except ImportError as e:
    print('Import error: {}'.format(e))"
echo.

echo DIAGNOSTIC: Checking for PySide6...
python -c "try:
    import PySide6
    print('PySide6 found!')
except ImportError as e:
    print('Import error: {}'.format(e))"
echo.

echo DIAGNOSTIC: Checking directory structure...
python -c "import os; print('Directory listing:')" 
dir src /b
echo.

echo DIAGNOSTIC: All checks completed
echo Please check the output above for any errors
pause 