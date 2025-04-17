import os, sys, subprocess, time, signal
from threading import Thread

# Log function
def log(msg):
    print(f"[LAUNCHER] {msg}")

v7_5_module_path = "src.v7_5"
log(f"Using v7.5 module path: {v7_5_module_path}")

# Define processes to launch
processes = [
    {"name": "Neural Seed", "cmd": ["python3", "-m", "src.seed", "--background", "--growth-rate=medium", "--mock"]},
    {"name": "Holographic UI", "cmd": ["python3", "run_holographic_frontend.py", "--mock"]},
    {"name": "Chat Interface", "cmd": ["python3", "-m", v7_5_module_path + ".lumina_frontend", "--interlink", "--port", "5680", "--mock"]},
    {"name": "System Monitor", "cmd": ["python3", "-m", v7_5_module_path + ".system_monitor", "--mock"]},
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
