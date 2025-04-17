#!/usr/bin/env python3
"""
Test script for connection monitor
"""

import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_monitor")

def main():
    """Run the connection monitor with various configurations"""
    
    script_path = os.path.join("src", "v8", "monitor_connections.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return 1
        
    logger.info(f"Using script: {script_path}")
    
    # Find connections file
    connections_dir = os.path.join("data", "v8", "connections")
    if not os.path.exists(connections_dir):
        logger.error(f"Connections directory not found: {connections_dir}")
        return 1
        
    latest_connections = os.path.join(connections_dir, "latest_connections.json")
    if os.path.exists(latest_connections):
        connections_file = latest_connections
    else:
        # Find any JSON file
        json_files = [f for f in os.listdir(connections_dir) if f.endswith('.json')]
        if not json_files:
            logger.error(f"No JSON files found in {connections_dir}")
            return 1
        connections_file = os.path.join(connections_dir, json_files[0])
    
    logger.info(f"Using connections file: {connections_file}")
    
    # Run the monitor in console mode
    cmd = [
        sys.executable,  # Python executable
        script_path,
        "--connections", connections_file,
        "--continuous"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        # Run the process, capturing standard error
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        # Wait for process to complete
        exit_code = process.wait()
        logger.info(f"Process exited with code: {exit_code}")
        
        # Print any errors
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"Error output:\n{stderr}")
            
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        process.terminate()
        return 1
    except Exception as e:
        logger.error(f"Error running monitor: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 