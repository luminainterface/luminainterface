#!/usr/bin/env python
"""
LUMINA System Component Test
----------------------------
A diagnostic tool to test various components of the LUMINA neural system.
"""

import os
import sys
import importlib
import subprocess
import platform
import time
from datetime import datetime

def print_header(title):
    print("\n" + "="*70)
    print(" "*((70-len(title))//2) + title)
    print("="*70)

def print_section(title):
    print("\n" + "-"*70)
    print(title)
    print("-"*70)

def print_result(test_name, result, details=None):
    if result:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"
    
    print(f"{status} | {test_name}")
    
    if details and not result:
        for line in details.split('\n'):
            print(f"       {line}")

def test_directory_structure():
    required_dirs = ['src', 'data', 'logs']
    results = []
    
    for directory in required_dirs:
        exists = os.path.isdir(directory)
        results.append((directory, exists))
        if not exists and directory in ['data', 'logs']:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created missing directory: {directory}")
                results[-1] = (directory, True)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
    
    return results

def test_dependencies():
    dependencies = [
        "PySide6",
        "numpy",
        "matplotlib",
        "pandas"
    ]
    
    results = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            results.append((dep, True))
        except ImportError as e:
            results.append((dep, False, str(e)))
    
    return results

def test_core_modules():
    modules = [
        "src.seed",
        "src.v7.ui.holographic_frontend",
        "src.v7.ui.consciousness_node",
        "src.v7.autowiki",
        "src.v7_5.system_monitor",
        "src.v7_5.lumina_frontend",
    ]
    
    results = []
    
    for module in modules:
        try:
            sys.path.insert(0, os.getcwd())
            importlib.import_module(module)
            results.append((module, True))
        except Exception as e:
            results.append((module, False, str(e)))
    
    return results

def test_component_execution():
    commands = [
        {
            "name": "Neural Seed", 
            "cmd": "python src/seed.py --mock --headless", 
            "timeout": 5
        },
        {
            "name": "Holographic Frontend", 
            "cmd": "python src/v7/ui/holographic_frontend.py --mock --test", 
            "timeout": 5
        },
        {
            "name": "Chat Interface", 
            "cmd": "python src/v7_5/lumina_frontend.py --mock --test", 
            "timeout": 5
        },
        {
            "name": "System Monitor", 
            "cmd": "python src/v7_5/system_monitor.py --mock", 
            "timeout": 5
        }
    ]
    
    results = []
    
    for cmd_info in commands:
        try:
            print(f"Testing: {cmd_info['name']}...")
            start_time = time.time()
            
            # Run with subprocess and kill after timeout
            process = subprocess.Popen(
                cmd_info["cmd"], 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for a short time to see if it crashes immediately
            time.sleep(2)
            
            if process.poll() is None:
                # Still running, which is good
                process.terminate()
                results.append((cmd_info["name"], True))
            else:
                # Process exited
                stdout, stderr = process.communicate()
                exit_code = process.returncode
                
                if exit_code == 0:
                    results.append((cmd_info["name"], True))
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
                    results.append((cmd_info["name"], False, f"Exit code: {exit_code}, Error: {error_msg}"))
                
        except Exception as e:
            results.append((cmd_info["name"], False, str(e)))
    
    return results

def run_diagnostic():
    print_header("LUMINA SYSTEM DIAGNOSTIC REPORT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Test directory structure
    print_section("Directory Structure")
    for directory, exists in test_directory_structure():
        print_result(f"Directory: {directory}", exists)
    
    # Test dependencies
    print_section("Dependencies")
    for dep_result in test_dependencies():
        if len(dep_result) == 2:
            dep, success = dep_result
            print_result(f"Import {dep}", success)
        else:
            dep, success, error = dep_result
            print_result(f"Import {dep}", success, error)
    
    # Test core modules
    print_section("Core Modules")
    for module_result in test_core_modules():
        if len(module_result) == 2:
            module, success = module_result
            print_result(f"Import {module}", success)
        else:
            module, success, error = module_result
            print_result(f"Import {module}", success, error)
    
    # Test component execution
    print_section("Component Execution")
    for component_result in test_component_execution():
        if len(component_result) == 2:
            component, success = component_result
            print_result(f"Execute {component}", success)
        else:
            component, success, error = component_result
            print_result(f"Execute {component}", success, error)
    
    # Provide possible solutions
    print_section("Possible Solutions")
    print("If you encountered any issues:")
    print("1. Ensure all modules are properly installed (pip install -r requirements.txt)")
    print("2. Check path separators (use os.path.join for cross-platform compatibility)")
    print("3. Verify environment variables are set correctly")
    print("4. Fix import errors in the failing modules")
    print("5. Ensure the startup parameters for components are correct")
    
    print_header("END OF DIAGNOSTIC REPORT")

if __name__ == "__main__":
    run_diagnostic() 