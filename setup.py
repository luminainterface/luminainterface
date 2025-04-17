from setuptools import setup, find_packages
import os
import sys
import shutil
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def setup_environment():
    """Set up the required environment for Lumina"""
    # Create required directories
    directories = [
        "assets/fonts",
        "data/memory",
        "data/neural_linguistic",
        "data/v10",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Copy Consolas font if it exists in Windows fonts
    windows_font_path = os.path.expandvars("%WINDIR%/Fonts/consola.ttf")
    target_font_path = "assets/fonts/Consolas.ttf"
    
    if os.path.exists(windows_font_path) and not os.path.exists(target_font_path):
        try:
            shutil.copy2(windows_font_path, target_font_path)
            print("Copied Consolas font to assets/fonts")
        except Exception as e:
            print(f"Warning: Could not copy Consolas font: {e}")
            print("Please manually copy Consolas.ttf to assets/fonts directory")

setup(
    name="neural_network_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pytest>=7.3.0',
        'pytest-asyncio>=0.18.0',
        'pytest-mock>=3.14.0',
        'pytest-cov>=6.1.1',
        'aiohttp>=3.8.0',
        'pydantic>=1.8.0',
        'torch>=1.9.0',
        'prometheus_client>=0.11.0',
        'python-json-logger>=2.0.2'
    ],
    python_requires='>=3.8',
    author="Lumina Development Team",
    author_email="lumina@example.com",
    description="Lumina Frontend System - Advanced Neural Network Visualization and Control Interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lumina_frontend",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

if __name__ == "__main__":
    print("Setting up Lumina environment...")
    setup_environment()
    print("\nSetup complete. Please run 'pip install -r requirements.txt' to install dependencies.") 