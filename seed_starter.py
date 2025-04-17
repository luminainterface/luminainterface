import sys, os, subprocess, time 
from threading import Thread 
 
# Start seed in background 
seed_proc = subprocess.Popen(["python", "-c", "from src.seed import get_neural_seed; seed = get_neural_seed(); seed.start_growth(); import time; time.sleep(3600)"], 
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
