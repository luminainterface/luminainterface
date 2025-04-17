@echo off
echo Starting LUMINA V7.0.0.3 Unified System with Neural Integration...

:: Set environment variables
set PYTHONPATH=%CD%;%CD%\src
set TEMPLATE_PLUGINS_ENABLED=true
set TEMPLATE_PLUGINS_DIRS=plugins;src\v7\plugins;src\plugins;src\visualization\plugins
set TEMPLATE_AUTO_LOAD_PLUGINS=mistral_neural_chat_plugin.py;consciousness_system_plugin.py;auto_wiki_plugin.py;dream_mode_plugin.py;breath_detection_plugin.py
set TEMPLATE_TITLE=LUMINA V7.0.0.3 Unified System
set TEMPLATE_ICON=icons/neural_icon.png
set V7_DASHBOARD_PORT=5678
set V7_DREAM_MODE_ENABLED=true
set V7_BREATH_DETECTION=true
set V7_MEMORY_SYSTEM=advanced
set V7_NEURAL_WEIGHT=0.6
set V7_LLM_WEIGHT=0.7
set V7_TEMPLATE_PATH=src\v7\template_ui
set V7_MOCK_MODE=true
set V7_GUI_FRAMEWORK=PySide6
set V7_DASHBOARD_PANELS=true

:: Create necessary directories
if not exist data mkdir data
if not exist data\chat_memory mkdir data\chat_memory
if not exist data\consciousness_network mkdir data\consciousness_network
if not exist data\auto_wiki mkdir data\auto_wiki
if not exist data\conversation_nodes mkdir data\conversation_nodes
if not exist data\dream_archive mkdir data\dream_archive
if not exist data\onsite_memory mkdir data\onsite_memory
if not exist plugins mkdir plugins
if not exist config mkdir config
if not exist logs mkdir logs
if not exist scripts mkdir scripts
if not exist src\v7\plugins mkdir src\v7\plugins
if not exist src\plugins mkdir src\plugins
if not exist src\visualization\plugins mkdir src\visualization\plugins
if not exist data\neural_metrics mkdir data\neural_metrics

:: Check for Mistral API key
set MISTRAL_API_KEY_FILE=config\mistral_api_key.txt
set MISTRAL_API_KEY=
if exist %MISTRAL_API_KEY_FILE% (
    for /f "tokens=*" %%a in (%MISTRAL_API_KEY_FILE%) do set MISTRAL_API_KEY=%%a
    echo Found Mistral API key.
    set TEMPLATE_MISTRAL_API_KEY=%MISTRAL_API_KEY%
) else (
    echo No Mistral API key file found at %MISTRAL_API_KEY_FILE%
    echo Will run in mock mode for Mistral integration.
)

:MAIN_MENU
:: Process conversation nodes if file exists
if exist "conversation with monday.md" (
    echo Processing conversation nodes from monday.md...
    python scripts\process_conversation.py "conversation with monday.md" "data\conversation_nodes\monday_nodes.json"
    if errorlevel 1 (
        echo Warning: Failed to process conversation nodes. Continuing without them.
    ) else (
        echo Successfully processed conversation nodes.
    )
) else (
    echo No conversation file found to process.
)

:: Ask what mode to run in
echo.
echo LUMINA V7.0.0.3 - What would you like to run?
echo 1. Full LUMINA V7 Unified System
echo 2. Dashboard Only (Consciousness Network and AutoWiki)
echo 3. Neural-Language Integration Demo
echo 4. System Configuration
echo 5. Qt Dashboard Panels (NEW)
echo 6. Exit
echo.
choice /C 123456 /M "Select an option"
if errorlevel 6 goto EXIT_PROGRAM
if errorlevel 5 goto RunQtDashboard
if errorlevel 4 goto SystemConfig
if errorlevel 3 goto NeuralLanguageDemo
if errorlevel 2 goto RunDashboardOnly
if errorlevel 1 goto RunFullSystem

:SystemConfig
echo.
echo LUMINA V7.0.0.3 System Configuration
echo.
echo Current settings:
echo - Dream Mode: %V7_DREAM_MODE_ENABLED%
echo - Breath Detection: %V7_BREATH_DETECTION%
echo - Memory System: %V7_MEMORY_SYSTEM%
echo - Neural Weight: %V7_NEURAL_WEIGHT%
echo - LLM Weight: %V7_LLM_WEIGHT%
echo - Mock Mode: %V7_MOCK_MODE%
echo - GUI Framework: %V7_GUI_FRAMEWORK%
echo.
echo 1. Toggle Dream Mode
echo 2. Toggle Breath Detection
echo 3. Change Memory System
echo 4. Adjust Neural/LLM Weights
echo 5. Toggle Mock Mode
echo 6. Switch GUI Framework (PyQt5/PySide6)
echo 7. Return to Main Menu
echo.
choice /C 1234567 /M "Select an option"
if errorlevel 7 goto MAIN_MENU
if errorlevel 6 goto SwitchGUIFramework
if errorlevel 5 goto ToggleMockMode
if errorlevel 4 goto AdjustWeights
if errorlevel 3 goto ChangeMemory
if errorlevel 2 goto ToggleBreath
if errorlevel 1 goto ToggleDream

:ToggleMockMode
if "%V7_MOCK_MODE%"=="true" (
    set V7_MOCK_MODE=false
) else (
    set V7_MOCK_MODE=true
)
goto SystemConfig

:ToggleDream
if "%V7_DREAM_MODE_ENABLED%"=="true" (
    set V7_DREAM_MODE_ENABLED=false
) else (
    set V7_DREAM_MODE_ENABLED=true
)
goto SystemConfig

:ToggleBreath
if "%V7_BREATH_DETECTION%"=="true" (
    set V7_BREATH_DETECTION=false
) else (
    set V7_BREATH_DETECTION=true
)
goto SystemConfig

:ChangeMemory
echo.
echo Select Memory System:
echo 1. Basic
echo 2. Advanced (with onsite memory)
echo 3. Full (with autowiki integration)
echo.
choice /C 123 /M "Select memory system"
if errorlevel 3 set V7_MEMORY_SYSTEM=full
if errorlevel 2 set V7_MEMORY_SYSTEM=advanced
if errorlevel 1 set V7_MEMORY_SYSTEM=basic
goto SystemConfig

:AdjustWeights
echo.
echo Current Neural Weight: %V7_NEURAL_WEIGHT%
echo Current LLM Weight: %V7_LLM_WEIGHT%
echo.
echo Select weight preset:
echo 1. Balanced (NN: 0.5, LLM: 0.5)
echo 2. Neural Focus (NN: 0.7, LLM: 0.3)
echo 3. Language Focus (NN: 0.3, LLM: 0.7)
echo 4. Deep Thought (NN: 0.9, LLM: 0.1)
echo 5. Clear Expression (NN: 0.1, LLM: 0.9)
echo.
choice /C 12345 /M "Select weight preset"
if errorlevel 5 (set V7_NEURAL_WEIGHT=0.1 & set V7_LLM_WEIGHT=0.9)
if errorlevel 4 (set V7_NEURAL_WEIGHT=0.9 & set V7_LLM_WEIGHT=0.1)
if errorlevel 3 (set V7_NEURAL_WEIGHT=0.3 & set V7_LLM_WEIGHT=0.7)
if errorlevel 2 (set V7_NEURAL_WEIGHT=0.7 & set V7_LLM_WEIGHT=0.3)
if errorlevel 1 (set V7_NEURAL_WEIGHT=0.5 & set V7_LLM_WEIGHT=0.5)
goto SystemConfig

:SwitchGUIFramework
if "%V7_GUI_FRAMEWORK%"=="PySide6" (
    set V7_GUI_FRAMEWORK=PyQt5
) else (
    set V7_GUI_FRAMEWORK=PySide6
)
goto SystemConfig

:NeuralLanguageDemo
echo.
echo Starting Neural-Language Integration Demo...

:: Run the simplified demo that should work regardless of missing components
echo Launching Simple Mistral Demo...
python src\v7\run_enhanced_mistral_demo.py --mock --nn-weight %V7_NEURAL_WEIGHT% --llm-weight %V7_LLM_WEIGHT% --interactive

echo Neural-Language Demo has been closed.
goto MAIN_MENU

:RunDashboardOnly
echo.
echo Starting V7 Consciousness Network Dashboard...

:: Run the simplified dashboard that should work regardless of missing plugins
echo Launching Simple Dashboard...
if exist run_v7_template_ui_with_plugins.bat (
    call run_v7_template_ui_with_plugins.bat --dashboard-only
) else (
    echo Error: Template UI launcher not found.
    pause
)

echo Dashboard has been closed.
goto MAIN_MENU

:RunFullSystem
:: Run the template UI with defaults that should work on any installation
echo.
echo Launching Unified System with Memory System: %V7_MEMORY_SYSTEM%...

if exist run_v7_template_ui_with_plugins.bat (
    call run_v7_template_ui_with_plugins.bat --mock=%V7_MOCK_MODE% --memory-system=%V7_MEMORY_SYSTEM% --neural-weight=%V7_NEURAL_WEIGHT% --llm-weight=%V7_LLM_WEIGHT%
) else (
    echo Template UI batch file not found. Using fallback method...
    python -m src.v7.template_ui.run_template_ui --mock=%V7_MOCK_MODE% --memory-system=%V7_MEMORY_SYSTEM% --neural-weight=%V7_NEURAL_WEIGHT% --llm-weight=%V7_LLM_WEIGHT%
    
    if errorlevel 1 (
        echo Fallback method failed. Trying last resort method...
        python -c "import os, sys; sys.path.append('src'); from v7.ui import run_simple_ui; run_simple_ui.main()"
    )
)

echo LUMINA V7.0.0.3 Unified System shut down.
goto MAIN_MENU

:RunQtDashboard
echo.
echo Starting LUMINA V7 Qt Dashboard Panels...

:: Check for dependencies and install them if missing
echo Checking dashboard dependencies...
if "%V7_GUI_FRAMEWORK%"=="PySide6" (
    python src/visualization/check_dashboard_requirements.py --install --gui-framework PySide6
) else (
    python src/visualization/check_dashboard_requirements.py --install
)
if errorlevel 1 (
    echo Failed to install required dependencies.
    echo Please install them manually:
    if "%V7_GUI_FRAMEWORK%"=="PySide6" (
        echo pip install PySide6 pyqtgraph matplotlib numpy pandas
    ) else (
        echo pip install PyQt5 pyqtgraph matplotlib numpy pandas
    )
    pause
    goto MAIN_MENU
)

:: Launch the Qt Dashboard
echo Launching Qt Dashboard Panels...
if "%V7_GUI_FRAMEWORK%"=="PySide6" (
    start /B /WAIT python src/visualization/run_qt_dashboard.py --v7-port %V7_DASHBOARD_PORT% --db-path data/neural_metrics.db --gui-framework PySide6 --mock=%V7_MOCK_MODE% --nn-weight=%V7_NEURAL_WEIGHT% --llm-weight=%V7_LLM_WEIGHT%
) else (
    start /B /WAIT python src/visualization/run_qt_dashboard.py --v7-port %V7_DASHBOARD_PORT% --db-path data/neural_metrics.db --mock=%V7_MOCK_MODE% --nn-weight=%V7_NEURAL_WEIGHT% --llm-weight=%V7_LLM_WEIGHT%
)

echo Qt Dashboard has been closed.
goto MAIN_MENU

:EXIT_PROGRAM
echo Exiting LUMINA V7.0.0.3 System...
exit /b 0

:End
pause 