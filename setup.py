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
    name="lumina_neural_network",
    version='0.9.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "torch>=2.2.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0"
    ],
    python_requires=">=3.10",
    author="Lumina Neural Network Team",
    author_email="team@lumina.ai",
    description="Advanced neural network system with versioned components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lumina-neural-network",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

if __name__ == "__main__":
    print("Setting up Lumina environment...")
    setup_environment()
    print("\nSetup complete. Please run 'pip install -r requirements.txt' to install dependencies.") 