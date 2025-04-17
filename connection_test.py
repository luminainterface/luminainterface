import sys
import os
import requests
import json
import traceback
from datetime import datetime

def test_v8_connection():
    print("Testing connection to V8 health check server...")
    try:
        response = requests.get('http://localhost:8765/health', timeout=5)
        print(f"V8 Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=4))
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to V8 health check server. Is it running?")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        return False

def test_v7_connection():
    print("\nTesting connection to V7 system...")
    try:
        response = requests.get('http://localhost:5000/api/v1/system/status', timeout=5)
        print(f"V7 Status Check: {response.status_code}")
        print(json.dumps(response.json(), indent=4))
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to V7 system. Is it running?")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        return False

def check_bridge_status():
    print("\nChecking bridge status...")
    
    # Check if bridge status file exists
    bridge_status_file = os.path.join("logs", "bridge_status.json")
    
    if os.path.exists(bridge_status_file):
        try:
            with open(bridge_status_file, 'r') as f:
                status = json.load(f)
            
            print("Bridge Status:")
            print(json.dumps(status, indent=4))
            
            # Return status of controller
            return status.get("controller_running", False)
        except Exception as e:
            print(f"ERROR: Failed to read bridge status: {str(e)}")
            return False
    else:
        print("ERROR: Bridge status file not found. Is the bridge controller running?")
        return False

def check_bridge_logs():
    print("\nChecking bridge logs...")
    
    log_files = [
        os.path.join("logs", f"bridge_controller_{datetime.now().strftime('%Y%m%d')}.log"),
        os.path.join("logs", f"v7_to_v8_bridge_{datetime.now().strftime('%Y%m%d')}.log"),
        os.path.join("logs", f"v8_v7_knowledge_bridge_{datetime.now().strftime('%Y%m%d')}.log")
    ]
    
    found_logs = False
    
    for log_file in log_files:
        if os.path.exists(log_file):
            found_logs = True
            print(f"\nFound log file: {log_file}")
            try:
                # Print the last 5 lines of the log file
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                print(f"Last {min(5, len(lines))} log entries:")
                for line in lines[-5:]:
                    print(f"  {line.strip()}")
            except Exception as e:
                print(f"ERROR: Failed to read log file: {str(e)}")
    
    if not found_logs:
        print("No bridge log files found for today.")
    
    return found_logs

if __name__ == "__main__":
    v8_status = test_v8_connection()
    v7_status = test_v7_connection()
    bridge_status = check_bridge_status()
    bridge_logs = check_bridge_logs()
    
    print("\nConnection Test Summary:")
    print(f"V8 Connection: {'SUCCESSFUL' if v8_status else 'FAILED'}")
    print(f"V7 Connection: {'SUCCESSFUL' if v7_status else 'FAILED'}")
    print(f"Bridge Status: {'RUNNING' if bridge_status else 'NOT RUNNING'}")
    print(f"Bridge Logs: {'FOUND' if bridge_logs else 'NOT FOUND'}")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not v8_status:
        print("\nTo start V8 Health Check Server:")
        print("  python src/v8/health_check_server.py")
    
    if not bridge_status:
        print("\nTo start V7-V8 Bridges:")
        print("  start_v7_v8_bridges.bat") 