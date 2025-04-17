import pytest
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def test_dir():
    """Return the directory containing the tests."""
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) 