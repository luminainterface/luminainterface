import os, sys, subprocess, time, signal 
from threading import Thread 
 
# Start the holographic frontend in the background 
holo_cmd = ["python", "run_holographic_frontend.py", "--gui-framework", "PySide6", "--mock"] 
 
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
