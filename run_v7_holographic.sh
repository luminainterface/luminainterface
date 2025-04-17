#!/bin/bash
echo "Starting LUMINA V7 Unified System..."

# Check if installation is complete, if not, run installation
if [ ! -d "data" ]; then
    echo "First-time setup detected. Running installation..."
    if [ -f "install_requirements.sh" ]; then
        chmod +x install_requirements.sh
        ./install_requirements.sh
        if [ $? -ne 0 ]; then
            echo "Installation failed. Please run install_requirements.sh manually."
            exit 1
        fi
    else
        echo "Installation script not found. Please run install_requirements.sh manually."
        exit 1
    fi
fi

# Create necessary directories if they don't exist
mkdir -p data
mkdir -p data/neural
mkdir -p data/memory
mkdir -p data/onsite_memory
mkdir -p logs
mkdir -p data/seed
mkdir -p data/dream
mkdir -p data/autowiki
mkdir -p data/consciousness
mkdir -p data/breath
mkdir -p data/v7.5
mkdir -p data/conversations
mkdir -p data/db
mkdir -p logs/db
mkdir -p logs/chat
mkdir -p logs/monitor

# IMPORTANT: Create missing dashboard directory
mkdir -p src/dashboard
if [ ! -f "src/dashboard/__init__.py" ]; then
    touch src/dashboard/__init__.py
fi
if [ ! -f "src/dashboard/run_dashboard.py" ]; then
    cat > src/dashboard/run_dashboard.py << EOF
#!/usr/bin/env python
import sys
import os
print("Starting simple dashboard placeholder...")
print("This is a placeholder for the dashboard module.")
print("The actual dashboard module is missing.")
print("Press Ctrl+C to exit")
while True: pass
EOF
    chmod +x src/dashboard/run_dashboard.py
fi
if [ ! -f "src/dashboard/dashboard_v7_bridge.py" ]; then
    cat > src/dashboard/dashboard_v7_bridge.py << EOF
#!/usr/bin/env python
import sys
import os
print("Neural Seed Bridge placeholder started.")
print("This is a placeholder for the dashboard bridge module.")
if __name__ == "__main__":
    print("Bridge is running in placeholder mode.")
    print("Press Ctrl+C to exit")
    while True: pass
EOF
    chmod +x src/dashboard/dashboard_v7_bridge.py
fi

# Set Python path to include the project root for proper imports
export PYTHONPATH=$(pwd)

# Set Python command variable for consistency
PYTHON_CMD="python3"

# Set environment variables
export DASHBOARD_PORT=5679
export METRICS_DB_PATH="data/neural_metrics.db"
export V7_CONNECTION_PORT=5678
export V7_DOCS_PATH="$PWD/docs"
export TEMPLATE_PLUGINS_DIRS="plugins:src/v7/plugins:src/plugins:src/visualization/plugins"
export GUI_FRAMEWORK="PySide6"

# V7 specific environment variables
export ENABLE_NODE_CONSCIOUSNESS=true
export ENABLE_AUTOWIKI=true
export ENABLE_DREAM_MODE=true
export ENABLE_BREATH_DETECTION=true
export ENABLE_MONDAY_INTEGRATION=true
export ENABLE_ONSITE_MEMORY=true
export V7_KNOWLEDGE_PATH="data/consciousness/knowledge"
export V7_DREAM_ARCHIVE_PATH="data/dream/archives"
export BREATH_PATTERN_CONFIG="configs/breath_patterns.json"
export MEMORY_STORAGE_PATH="data/onsite_memory"

# V7.5 specific environment variables
export V7_5_ENABLED=true
export V7_5_INTERLINK=true
export V7_5_PORT=5680
export MISTRAL_API_KEY="nLKZEpq29OihnaArxV7s6KtzsNEiky2A"

# Check if Python is installed
command -v $PYTHON_CMD >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Python is not installed or not in the PATH."
    echo "Please install Python and try again."
    exit 1
fi

# Check for PySide6 installation
$PYTHON_CMD -c "import PySide6" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo
    echo "PySide6 is not installed. This is required for the V7 Holographic Interface."
    echo "Installing PySide6..."
    pip3 install PySide6 pyqtgraph matplotlib numpy pandas
    if [ $? -ne 0 ]; then
        echo "Failed to install PySide6. Please install it manually with:"
        echo "pip3 install PySide6 pyqtgraph matplotlib numpy pandas"
        read -p "Press Enter to continue..."
    else
        echo "PySide6 installed successfully."
    fi
fi

# Detect which module path to use
V7_5_MODULE_PATH="src.v7_5"
V7_5_MODULE_TYPE="underscore"

# Check for the v7.5 module with dot
$PYTHON_CMD -c "import src.v7.5" 2>/dev/null
if [ $? -eq 0 ]; then
    V7_5_MODULE_PATH="src.v7.5"
    V7_5_MODULE_TYPE="dot"
    echo "Using v7.5 module with dot notation (src.v7.5)"
else
    echo "Using v7.5 module with underscore notation (src.v7_5)"
fi

# Function to display menu
display_menu() {
    clear
    echo
    echo "LUMINA V7 SYSTEM"
    echo "----------------------"
    echo "[1] Start Complete Holographic System (Holo+Dash+Seed)"
    echo "[2] Start Dashboard Panels"
    echo "[3] Start Unified System (Holographic + Dashboard)"
    echo "[4] Start Neural Seed Dashboard"
    echo "[5] Start V7.5 Chat Interface"
    echo "[6] Start V7.5 System Monitor"
    echo "[7] Start Database Connector"
    echo "[8] View Documentation"
    echo "[9] Run System Diagnostics"
    echo "[0] Exit"
    echo
}

# Function to start the complete system
start_complete_system() {
    clear
    echo
    echo "Starting Complete LUMINA V7.5 Holographic System..."
    echo

    # Create a master launcher script
    cat > launcher.py << EOF
import os, sys, subprocess, time, signal
from threading import Thread

# Log function
def log(msg):
    print(f"[LAUNCHER] {msg}")

v7_5_module_path = "${V7_5_MODULE_PATH}"
log(f"Using v7.5 module path: {v7_5_module_path}")

# Define processes to launch
processes = [
    {"name": "Neural Seed", "cmd": ["${PYTHON_CMD}", "-m", "src.seed", "--background", "--growth-rate=medium", "--mock"]},
    {"name": "Holographic UI", "cmd": ["${PYTHON_CMD}", "run_holographic_frontend.py", "--mock"]},
    {"name": "Chat Interface", "cmd": ["${PYTHON_CMD}", "-m", v7_5_module_path + ".lumina_frontend", "--interlink", "--port", "${V7_5_PORT}", "--mock"]},
    {"name": "System Monitor", "cmd": ["${PYTHON_CMD}", "-m", v7_5_module_path + ".system_monitor", "--mock"]},
]

# Store process objects
running_processes = []

# Start processes
for p in processes:
    try:
        log(f"Starting {p['name']}...")
        proc = subprocess.Popen(p['cmd'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running_processes.append({"name": p['name'], "process": proc})
        log(f"{p['name']} started with PID {proc.pid}")
        time.sleep(1) # Brief pause between starts
    except Exception as e:
        log(f"Failed to start {p['name']}: {e}")

# Monitor processes
log("All components started successfully")
log("Press Ctrl+C to shut down all components")

try:
    # Keep running until interrupted
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    log("Shutting down all components...")
    # Terminate all processes
    for p in running_processes:
        log(f"Stopping {p['name']}...")
        if p['process'].poll() is None: # If still running
            p['process'].terminate()
    log("All components have been stopped")
EOF

    # Run the launcher
    echo "Starting all components with unified launcher..."
    $PYTHON_CMD launcher.py
    echo
    echo "All components have been stopped. Returning to menu..."
    sleep 2
    rm launcher.py
}

# Function to start the dashboard
start_dashboard() {
    clear
    echo
    echo "Starting LUMINA V7 Dashboard Panels..."

    $PYTHON_CMD -c "import src.visualization.run_qt_dashboard" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Dashboard visualization module not found."
        echo "The visualization module will not be started."
        echo "Starting simple dashboard instead..."
        $PYTHON_CMD -m src.dashboard.run_dashboard
        echo "Press Enter to exit to menu..."
        read
        return
    fi

    if [ "$GUI_FRAMEWORK" == "PySide6" ]; then
        $PYTHON_CMD -m src.visualization.run_qt_dashboard --v7-port $V7_CONNECTION_PORT --db-path $METRICS_DB_PATH --gui-framework PySide6 --mock
    else
        $PYTHON_CMD -m src.visualization.run_qt_dashboard --v7-port $V7_CONNECTION_PORT --db-path $METRICS_DB_PATH --mock
    fi
    echo "Dashboard has been closed."
}

# Function to start the neural seed dashboard
start_neural_seed() {
    clear
    echo
    echo "Starting LUMINA V7 Neural Seed Dashboard..."
    echo "This dashboard integrates the self-growing neural seed system with the LUMINA V7 infrastructure."

    # Create and run a Python script that starts the seed in the background
    cat > seed_starter.py << EOF
import sys, os, subprocess, time
from threading import Thread

# Start seed in background
seed_proc = subprocess.Popen(["${PYTHON_CMD}", "-c", "from src.seed import get_neural_seed; seed = get_neural_seed(); seed.start_growth(); import time; time.sleep(3600)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Neural Seed system initialized and growing")
# Start a thread to clean up when the main script exits
def cleanup():
    time.sleep(0.5) # Wait a moment for the dashboard to start
    try:
        input("Press Enter to stop the Neural Seed and return to menu...")
    except: pass
    if seed_proc.poll() is None:
        seed_proc.terminate()
Thread(target=cleanup, daemon=True).start()
EOF

    # Run the seed starter script
    $PYTHON_CMD seed_starter.py

    # Try to run the bridge
    $PYTHON_CMD -m src.dashboard.dashboard_v7_bridge --seed-only --db $METRICS_DB_PATH

    echo "Neural Seed Dashboard has been closed."
    rm seed_starter.py
}

# Function to start the unified system
start_unified_system() {
    clear
    echo
    echo "Starting LUMINA V7 Unified System (Holographic + Dashboard)..."

    # Create a Python script to manage the holographic frontend
    cat > unified_launcher.py << EOF
import os, sys, subprocess, time, signal
from threading import Thread

# Start the holographic frontend in the background
holo_cmd = ["${PYTHON_CMD}", "run_holographic_frontend.py", "--gui-framework", "${GUI_FRAMEWORK}", "--mock"]

print("Starting Holographic Frontend...")
holo_proc = subprocess.Popen(holo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if holo_proc.poll() is not None:
    print("ERROR: Failed to start Holographic Interface")
    sys.exit(1)

print("Holographic Interface started successfully")

# Start a thread to clean up when the main script exits
def cleanup():
    time.sleep(1) # Wait a moment for the dashboard to start
    try:
        input("Press Enter to stop all components and return to menu...")
    except: pass
    if holo_proc.poll() is None:
        holo_proc.terminate()
Thread(target=cleanup, daemon=True).start()
EOF

    # Run the unified launcher script
    $PYTHON_CMD unified_launcher.py

    $PYTHON_CMD -c "import src.visualization.run_qt_dashboard" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Dashboard visualization module not found."
        echo "The visualization module will not be started."
        echo "Starting simple dashboard instead..."
        $PYTHON_CMD -m src.dashboard.run_dashboard
        echo "Press Enter to exit to menu..."
        read
        return
    fi

    # Start the dashboard
    echo "Starting Dashboard..."
    if [ "$GUI_FRAMEWORK" == "PySide6" ]; then
        $PYTHON_CMD -m src.visualization.run_qt_dashboard --v7-port $V7_CONNECTION_PORT --db-path $METRICS_DB_PATH --gui-framework PySide6 --mock
    else
        $PYTHON_CMD -m src.visualization.run_qt_dashboard --v7-port $V7_CONNECTION_PORT --db-path $METRICS_DB_PATH --mock
    fi

    echo "Unified System has been closed."
    rm unified_launcher.py
}

# Function to start the chat interface
start_chat_interface() {
    clear
    echo "Starting V7.5 Chat Interface..."
    echo

    mkdir -p logs/chat
    mkdir -p data/chat

    echo "Checking v7.5 chat module..."

    # First try v7.5 directory
    $PYTHON_CMD -c "import src.v7.5.lumina_frontend" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using module from src/v7.5/"
        $PYTHON_CMD -m src.v7.5.lumina_frontend --mock
        echo "Chat interface closed."
        return
    fi

    # Then try v7_5 directory
    $PYTHON_CMD -c "import src.v7_5.lumina_frontend" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using module from src/v7_5/"
        $PYTHON_CMD -m src.v7_5.lumina_frontend --mock
        echo "Chat interface closed."
        return
    fi

    echo "ERROR: v7.5 chat module not found in either src/v7.5/ or src/v7_5/ directories."
    echo "Make sure either src/v7.5/lumina_frontend.py or src/v7_5/lumina_frontend.py exists and is correctly formatted."
    read -p "Press Enter to continue..."
}

# Function to start the system monitor
start_system_monitor() {
    clear
    echo "Starting V7.5 System Monitor..."
    echo

    mkdir -p logs/monitor
    mkdir -p data/monitor

    echo "Checking v7.5 monitor module..."

    # First try v7.5 directory
    $PYTHON_CMD -c "import src.v7.5.system_monitor" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using module from src/v7.5/"
        $PYTHON_CMD -m src.v7.5.system_monitor --mock
        echo "System monitor closed."
        return
    fi

    # Then try v7_5 directory
    $PYTHON_CMD -c "import src.v7_5.system_monitor" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using module from src/v7_5/"
        $PYTHON_CMD -m src.v7_5.system_monitor --mock
        echo "System monitor closed."
        return
    fi

    echo "ERROR: v7.5 monitor module not found in either src/v7.5/ or src/v7_5/ directories."
    echo "Make sure either src/v7.5/system_monitor.py or src/v7_5/system_monitor.py exists and is correctly formatted."
    read -p "Press Enter to continue..."
}

# Function to setup database connector
setup_database_connector() {
    clear
    echo
    echo "Setting up Database Connector..."
    echo

    mkdir -p data/db
    mkdir -p logs/db

    # First try v7.5 directory
    $PYTHON_CMD -c "import src.v7.5.database_connector" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using module from src/v7.5/"
        
        # Create a Python script to manage the database connector
        cat > db_launcher.py << EOF
import os, sys, subprocess, time
print("Starting Database Connector...")
db_cmd = ["${PYTHON_CMD}", "-m", "src.v7.5.database_connector", "--sync", "--mock"]
db_proc = subprocess.Popen(db_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Database Connector started in background mode")
print("The connector will maintain synchronization between local and remote databases")
try:
    input("\nPress Enter to stop the Database Connector and return to menu...")
except: pass
if db_proc.poll() is None:
    db_proc.terminate()
print("Database Connector stopped")
EOF
        
        # Run the database launcher script
        $PYTHON_CMD db_launcher.py
        rm db_launcher.py
        return
    fi

    # Then try v7_5 directory
    $PYTHON_CMD -c "import src.v7_5.database_connector" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Using module from src/v7_5/"
        
        # Create a Python script to manage the database connector
        cat > db_launcher.py << EOF
import os, sys, subprocess, time
print("Starting Database Connector...")
db_cmd = ["${PYTHON_CMD}", "-m", "src.v7_5.database_connector", "--sync", "--mock"]
db_proc = subprocess.Popen(db_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Database Connector started in background mode")
print("The connector will maintain synchronization between local and remote databases")
try:
    input("\nPress Enter to stop the Database Connector and return to menu...")
except: pass
if db_proc.poll() is None:
    db_proc.terminate()
print("Database Connector stopped")
EOF
        
        # Run the database launcher script
        $PYTHON_CMD db_launcher.py
        rm db_launcher.py
        return
    fi

    echo "Error: Database connector module not found in either src/v7.5/ or src/v7_5/ directories."
    echo "Please ensure it's installed correctly."
    read -p "Press Enter to return to the main menu..."
}

# Function to view documentation
view_docs() {
    clear
    echo "Opening documentation..."

    # Show documentation menu
    echo "LUMINA Documentation:"
    echo "[1] MASTERreadme.md - System Architecture Overview"
    echo "[2] v7readme.md - V7 Neural Consciousness System"
    echo "[3] panelsreadme.md - Dashboard Panels Guide"
    echo "[4] SETUP_INSTRUCTIONS.md - Setup Instructions"
    echo "[5] FIXES_SUMMARY.md - Fixes and Improvements"
    echo "[0] Return to main menu"
    echo

    read -p "Enter document number: " doc_choice

    if [ "$doc_choice" == "1" ]; then
        if [ -f "MASTERreadme.md" ]; then
            less MASTERreadme.md
        else
            echo "MASTERreadme.md not found."
        fi
    elif [ "$doc_choice" == "2" ]; then
        if [ -f "docs/v7readme.md" ]; then
            less docs/v7readme.md
        elif [ -f "v7readme.md" ]; then
            less v7readme.md
        else
            echo "v7readme.md not found."
        fi
    elif [ "$doc_choice" == "3" ]; then
        if [ -f "docs/panelsreadme.md" ]; then
            less docs/panelsreadme.md
        elif [ -f "panelsreadme.md" ]; then
            less panelsreadme.md
        else
            echo "panelsreadme.md not found."
        fi
    elif [ "$doc_choice" == "4" ]; then
        if [ -f "SETUP_INSTRUCTIONS.md" ]; then
            less SETUP_INSTRUCTIONS.md
        else
            echo "SETUP_INSTRUCTIONS.md not found."
        fi
    elif [ "$doc_choice" == "5" ]; then
        if [ -f "FIXES_SUMMARY.md" ]; then
            less FIXES_SUMMARY.md
        else
            echo "FIXES_SUMMARY.md not found."
        fi
    elif [ "$doc_choice" == "0" ]; then
        return
    else
        echo "Invalid choice."
    fi

    echo
    read -p "Press Enter to return to documentation menu..."
    view_docs
}

# Function to run diagnostics
run_diagnostics() {
    clear
    echo
    echo "Running System Diagnostics..."
    echo

    # Run the component test script
    $PYTHON_CMD component_test.py

    echo
    read -p "Diagnostics complete. Press Enter to return to the main menu..."
}

# Main menu loop
while true; do
    display_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1) start_complete_system ;;
        2) start_dashboard ;;
        3) start_unified_system ;;
        4) start_neural_seed ;;
        5) start_chat_interface ;;
        6) start_system_monitor ;;
        7) setup_database_connector ;;
        8) view_docs ;;
        9) run_diagnostics ;;
        0) 
            clear
            echo "Shutting down LUMINA V7.5 System..."
            pkill -f python
            echo "Thank you for using LUMINA."
            sleep 2
            exit
            ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
done 