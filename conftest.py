"""
Pytest configuration file.
"""

import os
import sys
from pathlib import Path
import pytest

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure pytest-asyncio
pytest.ini_options = {
    "asyncio_mode": "strict"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def pytest_configure(config):
    """Configure pytest."""
    config.option.asyncio_mode = "strict" 