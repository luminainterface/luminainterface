#!/usr/bin/env python3
"""
Simplified test script for the Onsite Memory System

This script tests only the most basic functionality of the OnsiteMemory class.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_memory_test")

# Print Python path for debugging
print("Python path:")
for p in sys.path:
    print(f"  - {p}")

def main():
    """Basic test function"""
    print("\nTesting OnsiteMemory basic functionality...")
    
    try:
        # Import the OnsiteMemory class
        from src.v7.onsite_memory import OnsiteMemory
        print("Successfully imported OnsiteMemory")
        
        # Create memory instance
        memory = OnsiteMemory(
            data_dir="data/simple_test",
            memory_file="simple_test.json",
            auto_save=True
        )
        print(f"Created memory instance with file: {memory.memory_file}")
        
        # Add a conversation
        conv_id = memory.add_conversation(
            "Hello, this is a test",
            "Hello! I'm responding to your test message.",
            {"test_metadata": True}
        )
        print(f"Added conversation with ID: {conv_id}")
        
        # Add knowledge
        result = memory.add_knowledge(
            "Test Topic",
            "This is test knowledge content.",
            "Test Source"
        )
        print(f"Added knowledge entry: {result}")
        
        # Get stats
        stats = memory.get_stats()
        print("\nMemory Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save and stop
        memory.save_memory()
        memory.stop()
        print("\nMemory saved and stopped")
        
        print("\nTest completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 