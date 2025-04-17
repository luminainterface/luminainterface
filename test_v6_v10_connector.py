#!/usr/bin/env python3
"""
Test Script for Language Memory V6-V10 Connector

This script tests the language_memory_v6_v10_connector module that connects
the Language Memory System with v6-v10 components.
"""

import logging
import json
from datetime import datetime
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_v6_v10_connector")

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    # Import Language Memory components
    from src.language.language_memory import LanguageMemory
    from src.language_memory_v6_v10_connector import get_language_memory_connector
    
    # Test case texts
    TEST_TEXTS = [
        "The neural network processes data through layers of artificial neurons.",
        "Language memory allows the system to remember and reflect on linguistic patterns.",
        "Consciousness emerges from the complex interactions of neural components.",
        "The system both processes data and reflects on its own processing mechanisms.",
        "All knowledge exists inside the temple of memory, organized spatially."
    ]
    
    def run_tests():
        """Run tests on the v6-v10 connector"""
        logger.info("Testing Language Memory V6-V10 Connector")
        
        # Initialize Language Memory
        logger.info("Initializing Language Memory")
        language_memory = LanguageMemory()
        
        # Create connector
        logger.info("Creating V6-V10 Connector")
        connector = get_language_memory_connector(language_memory)
        
        # Display available components
        logger.info("Available components:")
        for component, status in connector.get_component_status().items():
            logger.info(f"  {component}: {status}")
        
        # Test with all capabilities
        logger.info("\nTesting with all available capabilities")
        for i, text in enumerate(TEST_TEXTS):
            logger.info(f"\nProcessing text {i+1}: {text[:50]}...")
            result = connector.process_text(text)
            logger.info(f"Capabilities used: {result['capabilities_used']}")
        
        # Test with highest available consciousness
        logger.info("\nTesting with highest available consciousness")
        text = "Consciousness emerges from a system's ability to reflect on itself."
        result = connector.process_with_consciousness(text)
        logger.info(f"Process with consciousness result status: {result.get('status')}")
        
        # Test v6 contradiction detection
        logger.info("\nTesting v6 contradiction detection")
        contradiction_text = "All neural networks are deterministic, but some neural networks are not deterministic."
        contradictions = connector.detect_contradictions(contradiction_text)
        logger.info(f"Detected {len(contradictions)} contradictions")
        
        # Show available capabilities
        capabilities = connector.get_available_capabilities()
        logger.info(f"\nAvailable capabilities: {capabilities}")
        
        logger.info("\nAll tests completed successfully!")
    
    if __name__ == "__main__":
        run_tests()

except Exception as e:
    logger.error(f"Error during testing: {str(e)}")
    import traceback
    traceback.print_exc() 