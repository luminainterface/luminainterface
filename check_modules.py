#!/usr/bin/env python
import sys
import os

print("Python version:", sys.version)

# Check directory structure
print("\nChecking directory structure...")
if os.path.exists("src"):
    print("src directory exists.")
else:
    print("ERROR: src directory missing!")

# Check required modules
modules_to_check = [
    "src.v7_5",
    "src.v7.ui.holographic_frontend",
    "src.dashboard.run_dashboard", 
    "src.seed",
    "PySide6"
]

print("\nChecking for required modules...")
for module in modules_to_check:
    try:
        __import__(module)
        print(f"✓ {module} is available")
    except ImportError as e:
        print(f"✗ {module} is NOT available: {e}")

# Check critical directories
print("\nChecking v7 directory structure...")
v7_dirs = ["src/v7", "src/v7/ui", "src/v7.5", "src/dashboard"]
for dir_path in v7_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path} exists")
        # Print first few files
        files = os.listdir(dir_path)[:5]
        print(f"  Files: {', '.join(files) if files else 'None'}")
    else:
        print(f"✗ {dir_path} does NOT exist")

print("\nDiagnostic complete.") 