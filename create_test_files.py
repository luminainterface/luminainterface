import os
from pathlib import Path

def create_test_file(version):
    content = f'''import pytest

def test_{version}_placeholder():
    """Placeholder test for {version}"""
    assert True
'''
    
    file_path = Path(f'src/tests/test_{version}_basic.py')
    file_path.write_text(content)
    print(f"Created test file: {file_path}")

def main():
    # Create tests directory if it doesn't exist
    Path('src/tests').mkdir(parents=True, exist_ok=True)
    
    # Create test files for each version
    versions = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v7_5", "v8", "v9", "v10", "v11", "v12"]
    
    for version in versions:
        create_test_file(version)
        
    print("\nAll test files created successfully!")

if __name__ == "__main__":
    main() 