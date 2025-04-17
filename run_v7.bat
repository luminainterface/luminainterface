@echo off
setlocal enabledelayedexpansion

echo ========================================
echo LUMINA V5 DASHBOARD - DEBUG MODE
echo ========================================
echo.

:: Set environment variables with error checking
set LUMINA_ROOT=%~dp0
echo Current directory: %LUMINA_ROOT%

:: Check Python installation
echo Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found in PATH
    echo Try using the full path to your Python executable
    set /p PYTHON_PATH="Enter full path to python.exe: "
) else (
    set PYTHON_PATH=python
    echo Python found: !PYTHON_PATH!
)

:: Verify Python runs
echo Testing Python execution...
%PYTHON_PATH% --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run Python
    goto :error
)

:: Check critical directories
echo Checking required directories...
set MISSING_DIRS=0
if not exist "%LUMINA_ROOT%src" (
    echo MISSING: %LUMINA_ROOT%src
    set /a MISSING_DIRS+=1
)
if not exist "%LUMINA_ROOT%data" (
    echo MISSING: %LUMINA_ROOT%data
    mkdir "%LUMINA_ROOT%data"
    echo Created: %LUMINA_ROOT%data
)
if not exist "%LUMINA_ROOT%logs" (
    echo MISSING: %LUMINA_ROOT%logs
    mkdir "%LUMINA_ROOT%logs"
    echo Created: %LUMINA_ROOT%logs
)

if %MISSING_DIRS% GTR 0 (
    echo WARNING: %MISSING_DIRS% required directories are missing
)

:: Check for required Python packages
echo Checking required Python packages...
echo This may take a moment...
%PYTHON_PATH% -c "import sys; packages=['PySide6', 'numpy', 'matplotlib']; missing=[p for p in packages if p not in sys.modules and __import__(p, fromlist=['']) is None]; print('Missing packages: ' + ', '.join(missing) if missing else 'All packages installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to check Python packages
    echo Attempting to install essential packages...
    %PYTHON_PATH% -m pip install PySide6 numpy matplotlib
)

:: Look for critical files
echo Checking for critical Python files...
set MISSING_FILES=0
set DASHBOARD_PATH=

:: Try different possible locations for the dashboard script
if exist "%LUMINA_ROOT%src\v5\visualization\dashboard.py" (
    set DASHBOARD_PATH=%LUMINA_ROOT%src\v5\visualization\dashboard.py
    echo Found dashboard at: !DASHBOARD_PATH!
) else if exist "%LUMINA_ROOT%src\visualization\dashboard.py" (
    set DASHBOARD_PATH=%LUMINA_ROOT%src\visualization\dashboard.py
    echo Found dashboard at: !DASHBOARD_PATH!
) else if exist "%LUMINA_ROOT%src\v7\gui\main.py" (
    set DASHBOARD_PATH=%LUMINA_ROOT%src\v7\gui\main.py
    echo Found V7 GUI at: !DASHBOARD_PATH!
) else (
    echo MISSING: Dashboard Python script not found
    set /a MISSING_FILES+=1
)

:: Check for PySide6 template (from additional data)
if exist "%LUMINA_ROOT%src\v7\plugin_template.py" (
    echo Found PySide6 template: %LUMINA_ROOT%src\v7\plugin_template.py
) else (
    echo Note: PySide6 template not found. This might be needed.
)

if %MISSING_FILES% GTR 0 (
    echo WARNING: %MISSING_FILES% required files are missing
    echo Creating a minimal dashboard script...
    
    :: Create minimal directory structure
    if not exist "%LUMINA_ROOT%src\visualization" mkdir "%LUMINA_ROOT%src\visualization"
    
    :: Create a minimal dashboard script
    (
        echo import sys
        echo import os
        echo try:
        echo     from PySide6 import QtCore, QtWidgets, QtGui
        echo     print^("PySide6 imported successfully"^)
        echo except ImportError:
        echo     print^("ERROR: PySide6 not found. Install with: pip install PySide6"^)
        echo     sys.exit^(1^)
        echo.
        echo try:
        echo     import numpy as np
        echo     import matplotlib
        echo     matplotlib.use^('Qt5Agg'^)
        echo     from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        echo     from matplotlib.figure import Figure
        echo     print^("Visualization libraries imported successfully"^)
        echo except ImportError as e:
        echo     print^(f"ERROR: Missing visualization libraries. {e}"^)
        echo     print^("Install with: pip install matplotlib numpy"^)
        echo     sys.exit^(1^)
        echo.
        echo class LuminaV5Dashboard^(QtWidgets.QMainWindow^):
        echo     def __init__^(self^):
        echo         super^(^).__init__^(^)
        echo         self.setWindowTitle^("V5 Fractal Echo Visualization"^)
        echo         self.resize^(1200, 800^)
        echo         
        echo         # Create central widget and layout
        echo         central_widget = QtWidgets.QWidget^(^)
        echo         self.setCentralWidget^(central_widget^)
        echo         main_layout = QtWidgets.QVBoxLayout^(central_widget^)
        echo         
        echo         # Status information
        echo         status_label = QtWidgets.QLabel^("DEBUG MODE: Minimal Dashboard"^)
        echo         status_label.setStyleSheet^("color: red; font-weight: bold;"^)
        echo         main_layout.addWidget^(status_label^)
        echo         
        echo         # Info text
        echo         info = QtWidgets.QTextEdit^(^)
        echo         info.setReadOnly^(True^)
        echo         info.append^("Lumina V5 Dashboard Debug Version\n"^)
        echo         info.append^("----------------------------------------\n"^)
        echo         info.append^("This is a minimal dashboard created by the debugging script.\n"^)
        echo         info.append^("If you're seeing this, it means the normal dashboard couldn't be found.\n\n"^)
        echo         info.append^("Debugging Information:\n"^)
        echo         info.append^(f"Python version: {sys.version}\n"^)
        echo         info.append^(f"Working directory: {os.getcwd^(^)}\n"^)
        echo         info.append^(f"PySide6 version: {QtCore.__version__}\n"^)
        echo         main_layout.addWidget^(info^)
        echo         
        echo         # Add a button to exit
        echo         exit_button = QtWidgets.QPushButton^("Exit"^)
        echo         exit_button.clicked.connect^(self.close^)
        echo         main_layout.addWidget^(exit_button^)
        echo.
        echo if __name__ == "__main__":
        echo     app = QtWidgets.QApplication^(sys.argv^)
        echo     window = LuminaV5Dashboard^(^)
        echo     window.show^(^)
        echo     sys.exit^(app.exec_^(^)^)
    ) > "%LUMINA_ROOT%src\visualization\dashboard.py"
    
    set DASHBOARD_PATH=%LUMINA_ROOT%src\visualization\dashboard.py
    echo Created minimal dashboard at: !DASHBOARD_PATH!
)

:: Try to launch the dashboard
echo.
echo Attempting to launch dashboard...
echo Dashboard path: %DASHBOARD_PATH%
echo.
echo Command: %PYTHON_PATH% "%DASHBOARD_PATH%"
echo.
echo If the dashboard doesn't appear, check for error messages below:
echo ----------------------------------------
%PYTHON_PATH% "%DASHBOARD_PATH%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ----------------------------------------
    echo ERROR: Failed to launch dashboard
    echo.
    echo TROUBLESHOOTING:
    echo 1. Make sure PySide6 is installed: pip install PySide6
    echo 2. Check if numpy and matplotlib are installed: pip install numpy matplotlib
    echo 3. Try running the script directly: %PYTHON_PATH% "%DASHBOARD_PATH%"
    echo 4. Look for error messages above
    goto :error
)

goto :end

:error
echo.
echo ERROR: Dashboard failed to start.
echo Creating error log...
echo Error occurred during dashboard startup > "%LUMINA_ROOT%logs\dashboard_error.log"
echo Time: %date% %time% >> "%LUMINA_ROOT%logs\dashboard_error.log"
echo Working directory: %cd% >> "%LUMINA_ROOT%logs\dashboard_error.log"
echo Python path: %PYTHON_PATH% >> "%LUMINA_ROOT%logs\dashboard_error.log"
echo Dashboard path: %DASHBOARD_PATH% >> "%LUMINA_ROOT%logs\dashboard_error.log"
echo.
echo Error log saved to: %LUMINA_ROOT%logs\dashboard_error.log
echo.
echo Press any key to exit...
pause > nul
exit /b 1

:end
echo.
echo Debug complete. If dashboard didn't appear, check errors above.
echo.
echo Press any key to exit...
pause > nul