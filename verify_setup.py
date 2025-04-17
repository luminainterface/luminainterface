import os
import sys
from pathlib import Path
import importlib.util

def check_directory(path):
    """Check if directory exists and is accessible."""
    dir_path = Path(path)
    return dir_path.exists() and dir_path.is_dir()

def check_file(path):
    """Check if file exists and is accessible."""
    file_path = Path(path)
    return file_path.exists() and file_path.is_file()

def check_python_module(module_name):
    """Check if a Python module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    print("Verifying CI/CD setup...")
    
    # Check required directories
    required_dirs = [
        "assets/fonts",
        "data/memory",
        "data/neural_linguistic",
        "data/v10",
        "logs",
        "tests",
        "src/tests"
    ]
    
    for dir_path in required_dirs:
        if check_directory(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
    
    # Check required files
    required_files = [
        "requirements.txt",
        "pytest.ini",
        "setup_ci.py",
        "create_test_files.py"
    ]
    
    for file_path in required_files:
        if check_file(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
    
    # Check test files
    versions = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7_5", "v8", "v9", "v10", "v11", "v12"]
    for version in versions:
        test_file = f"src/tests/test_{version}_basic.py"
        if check_file(test_file):
            print(f"✓ Test file exists: {test_file}")
        else:
            print(f"✗ Missing test file: {test_file}")
    
    # Check required Python packages
    required_packages = [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "coverage",
        "mypy",
        "black",
        "flake8",
        "isort",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "python-dotenv",
        "PySide6",
        "cryptography",
        "sqlalchemy",
        "aiosqlite",
        "aiohttp",
        "pydantic",
        "tqdm",
        "colorama",
        "prometheus_client",
        "python-json-logger"
    ]
    
    print("\nChecking required Python packages...")
    for package in required_packages:
        if check_python_module(package):
            print(f"✓ Package installed: {package}")
        else:
            print(f"✗ Package missing: {package}")

if __name__ == "__main__":
    main() 