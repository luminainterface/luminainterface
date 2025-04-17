#!/usr/bin/env python3
"""
Test script for the fixed Mistral integration

This script verifies that the fixed Mistral integration works correctly.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TestMistralFixed")

# Import the fixed Mistral integration
try:
    from src.api.mistral_integration_fixed import MistralIntegration
    logger.info("Successfully imported MistralIntegration from fixed module")
except ImportError as e:
    logger.error(f"Failed to import MistralIntegration: {e}")
    logger.error("Make sure you have installed all required packages: pip install mistralai pyside6")
    sys.exit(1)

def test_mistral_fixed():
    """Run tests on the fixed Mistral integration"""
    
    # Check if API key is available
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("MISTRAL_API_KEY environment variable not set")
        logger.warning("Some tests will be skipped")
        logger.warning("Set it by running: export MISTRAL_API_KEY=your_api_key (Linux/Mac) or set MISTRAL_API_KEY=your_api_key (Windows)")
    
    # Create integration instance
    integration = MistralIntegration(api_key=api_key)
    
    # Test 1: Check if integration was initialized
    logger.info("Test 1: Checking initialization...")
    if integration is None:
        logger.error("Failed to create MistralIntegration instance")
        return False
    
    logger.info("MistralIntegration initialized successfully")
    
    # Test 2: Check availability
    logger.info("Test 2: Checking availability...")
    mistral_available = integration.is_available
    processor_available = integration.processor_available
    
    logger.info(f"Mistral available: {mistral_available}")
    logger.info(f"Neural processor available: {processor_available}")
    
    # Test 3: Check system stats
    logger.info("Test 3: Checking system stats...")
    stats = integration.get_system_stats()
    logger.info(f"System stats: {stats}")
    
    # Test 4: Adjust weights
    logger.info("Test 4: Testing weight adjustment...")
    original_weights = {
        "llm": integration.llm_weight,
        "nn": integration.nn_weight
    }
    
    new_weights = integration.adjust_weights(llm_weight=0.6, nn_weight=0.4)
    logger.info(f"Original weights: {original_weights}")
    logger.info(f"New weights: {new_weights}")
    
    if abs(new_weights["llm_weight"] - 0.6) > 0.01 or abs(new_weights["nn_weight"] - 0.4) > 0.01:
        logger.error("Weight adjustment failed")
        return False
    
    # Test 5: Process message if available
    if mistral_available:
        logger.info("Test 5: Testing message processing with Mistral...")
        test_message = "What can you tell me about neural networks?"
        
        try:
            result = integration.process_message(test_message)
            logger.info(f"Input: {result['input']}")
            logger.info(f"LLM Response: {result['llm_response'][:100]}...")
            logger.info(f"Combined Response: {result['combined_response'][:100]}...")
            
            if not result["llm_response"]:
                logger.warning("No LLM response received")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False
    else:
        logger.info("Skipping message processing test (Mistral not available)")
    
    # Overall test result
    success = mistral_available or processor_available
    if success:
        logger.info("Tests completed successfully")
    else:
        logger.warning("Tests completed, but neither Mistral nor processor is available")
    
    return success

if __name__ == "__main__":
    logger.info("Starting Mistral fixed integration tests")
    success = test_mistral_fixed()
    logger.info("Tests finished")
    sys.exit(0 if success else 1) 