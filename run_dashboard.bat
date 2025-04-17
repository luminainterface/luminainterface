@echo off
echo Starting V7 Consciousness Network Dashboard...

REM Create necessary directories
if not exist "data" mkdir data
if not exist "data\consciousness_network" mkdir data\consciousness_network
if not exist "data\auto_wiki" mkdir data\auto_wiki
if not exist "data\conversation_nodes" mkdir data\conversation_nodes

REM Process conversation nodes from monday.md
echo Processing conversation nodes from monday.md...
python scripts\process_conversation.py "conversation with monday.md" "data\conversation_nodes\monday_nodes.json"

REM Start the plugins in background mode
echo Initializing Consciousness Network Plugin...
start "Consciousness Network" python scripts\run_consciousness_plugin.py --integration monday_nodes --mock_mode true

echo Initializing AutoWiki Plugin...
start "AutoWiki" python scripts\run_autowiki_plugin.py --seed_from monday_nodes --mock_mode true

REM Launch the dashboard UI
echo Launching Dashboard UI...
python scripts\dashboard_ui.py

echo Dashboard is now running. Close this window to shut down all components.
pause 