#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'logs',
        'data/memory',
        'data/neural_linguistic',
        'data/v10',
        'assets/fonts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])

def setup_environment():
    """Set up the development environment."""
    print("Setting up environment...")
    create_directories()
    install_dependencies()
    print("Environment setup complete!")

def deploy():
    """Deploy the application."""
    print("Starting deployment...")
    
    # Verify all tests pass
    print("Running tests...")
    subprocess.check_call([sys.executable, "-m", "pytest", "tests/"])
    
    # Create deployment package
    print("Creating deployment package...")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    subprocess.check_call([sys.executable, "setup.py", "sdist", "bdist_wheel"])
    
    print("Deployment package created successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy()
    else:
        setup_environment() 