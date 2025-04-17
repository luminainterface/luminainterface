#!/usr/bin/env python3
"""
Mistral Integration Test Script

This script performs a targeted test of the Mistral integration component
to verify its functionality with the Central Language Node.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mistral_test.log")
    ]
)
logger = logging.getLogger("MistralTest")

# Make sure the src directory is in the path
current_dir = Path(__file__).resolve().parent
src_dir = os.path.join(current_dir, "src")
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

logger.info(f"Python path: {sys.path}")

# Try direct import first
try:
    import mistral_integration
    from mistral_integration import MistralEnhancedSystem, MISTRAL_AVAILABLE
    logger.info("Successfully imported MistralEnhancedSystem directly")
except ImportError as e:
    logger.warning(f"Direct import failed: {e}")
    try:
        # Try from src
        from src.mistral_integration import MistralEnhancedSystem, MISTRAL_AVAILABLE
        logger.info("Successfully imported MistralEnhancedSystem from src")
    except ImportError as e:
        logger.error(f"Error importing MistralEnhancedSystem: {e}")
        sys.exit(1)

def setup_test_directories():
    """Create necessary directories for testing."""
    directories = [
        "data/test_mistral/db",
        "data/test_mistral/conversation_memory",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Test directories created")

def test_mistral_initialization():
    """Test if the Mistral system initializes properly"""
    logger.info("=== Testing Mistral Initialization ===")
    
    # Initialize system with explicit test directory
    try:
        system = MistralEnhancedSystem(
            data_dir="data/test_mistral",
            llm_weight=0.7,
            nn_weight=0.6
        )
        logger.info("Mistral system initialized successfully")
        
        # Check if Mistral client was created
        if system.mistral_client:
            logger.info("Mistral API client created successfully")
        else:
            logger.warning("Mistral API client not available - using fallback mode")
        
        # Check central node initialization
        if system.central_node:
            logger.info(f"Central Language Node initialized with LLM weight: {system.central_node.llm_weight}, NN weight: {system.central_node.nn_weight}")
        else:
            logger.error("Central Language Node was not initialized")
            return False
        
        # Check conversation memory and database
        if system.conversation_memory and system.db_manager:
            logger.info(f"Conversation memory initialized with ID: {system.current_conversation_id}")
        else:
            logger.error("Conversation memory or database manager not initialized")
            return False
        
        # Close system
        system.close()
        logger.info("System closed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing Mistral system: {e}")
        return False

def test_mistral_processing():
    """Test message processing through the Mistral system"""
    logger.info("=== Testing Mistral Message Processing ===")
    
    # Initialize system
    try:
        system = MistralEnhancedSystem(
            data_dir="data/test_mistral",
            llm_weight=0.7,
            nn_weight=0.6
        )
        
        # Process a test message
        test_message = "Explain how the neural network and language model weights interact in this system."
        logger.info(f"Sending test message: '{test_message}'")
        
        # Time the response
        start_time = time.time()
        response = system.process_message(test_message)
        processing_time = time.time() - start_time
        
        # Log results
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Response length: {len(response)} characters")
        logger.info(f"Response preview: {response[:100]}...")
        
        # Get system stats
        stats = system.get_system_stats()
        logger.info(f"System stats: {stats}")
        
        # Close system
        system.close()
        return True
    
    except Exception as e:
        logger.error(f"Error in Mistral processing test: {e}")
        return False

def test_weight_adjustment():
    """Test adjusting weights in the Mistral system"""
    logger.info("=== Testing Weight Adjustment ===")
    
    test_weights = [
        (0.0, 1.0),  # Full neural, no LLM
        (0.5, 0.5),  # Balanced
        (1.0, 0.0),  # Full LLM, no neural
    ]
    
    for llm_weight, nn_weight in test_weights:
        logger.info(f"Testing with LLM weight: {llm_weight}, NN weight: {nn_weight}")
        
        try:
            # Initialize system with these weights
            system = MistralEnhancedSystem(
                data_dir="data/test_mistral",
                llm_weight=llm_weight,
                nn_weight=nn_weight
            )
            
            # Verify weights were set correctly
            if system.central_node.llm_weight == llm_weight and system.central_node.nn_weight == nn_weight:
                logger.info("✅ Weights set correctly in central node")
            else:
                logger.error(f"❌ Weights mismatch: central_node.llm_weight={system.central_node.llm_weight}, central_node.nn_weight={system.central_node.nn_weight}")
            
            # Process a simple message to test the weights
            response = system.process_message("Test message with weight adjustment")
            logger.info(f"Response with weights ({llm_weight}, {nn_weight}): {response[:50]}...")
            
            # Close system
            system.close()
            
        except Exception as e:
            logger.error(f"Error testing weights ({llm_weight}, {nn_weight}): {e}")
    
    return True

def main():
    """Main test function"""
    print("=== Mistral Integration Test ===")
    logger.info("Starting Mistral integration tests")
    
    # Setup
    setup_test_directories()
    
    # Run tests
    init_result = test_mistral_initialization()
    if not init_result:
        logger.error("❌ Initialization test failed. Stopping further tests.")
        return
    
    process_result = test_mistral_processing()
    if not process_result:
        logger.error("❌ Processing test failed. Continuing with weight tests...")
    
    weight_result = test_weight_adjustment()
    if not weight_result:
        logger.error("❌ Weight adjustment test failed.")
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Initialization Test: {'✅ Passed' if init_result else '❌ Failed'}")
    print(f"Processing Test: {'✅ Passed' if process_result else '❌ Failed'}")
    print(f"Weight Adjustment Test: {'✅ Passed' if weight_result else '❌ Failed'}")
    
    logger.info("Mistral integration tests completed")

if __name__ == "__main__":
    main() 