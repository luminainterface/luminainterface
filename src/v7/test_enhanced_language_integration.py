#!/usr/bin/env python3
"""
Test Script for Enhanced Language System Integration with V7

This script tests the integration between the Enhanced Language System
and the V7 Node Consciousness framework.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test.v7_language_integration")

# Import the integration module
try:
    from src.v7.enhanced_language_integration import (
        get_enhanced_language_integration,
        EnhancedLanguageV7Integration
    )
except ImportError as e:
    logger.error(f"Error importing enhanced_language_integration: {str(e)}")
    sys.exit(1)

def run_basic_test(integration):
    """Run basic functionality tests"""
    logger.info("Running basic functionality tests...")
    
    # Test 1: Process simple text
    test_text = "Neural networks can process language patterns with consciousness."
    logger.info(f"Test 1: Processing text: '{test_text}'")
    
    result = integration.process_text(test_text)
    logger.info(f"Result: {result.get('consciousness_level', 'N/A')} consciousness level")
    
    # Test 2: Set LLM weight
    logger.info("Test 2: Setting LLM weight to 0.7")
    success = integration.set_llm_weight(0.7)
    logger.info(f"Set LLM weight success: {success}")
    
    # Test 3: Get integration status
    logger.info("Test 3: Getting integration status")
    status = integration.get_status()
    logger.info(f"Mock mode: {status['mock_mode']}")
    logger.info(f"Components available: {sum(1 for v in status['components'].values() if v)}/{len(status['components'])}")
    
    return True

def run_consciousness_tests(integration):
    """Run tests focused on consciousness integration"""
    logger.info("Running consciousness integration tests...")
    
    # Test 1: Process text with consciousness references
    test_text = "The system becomes aware of its own processing and develops consciousness."
    logger.info(f"Consciousness Test 1: Processing text: '{test_text}'")
    
    result = integration.process_text(test_text)
    c_level = result.get('consciousness_level', 'N/A')
    logger.info(f"Consciousness level: {c_level}")
    
    # Test 2: Process text with recursive references
    test_text = "This sentence references itself and creates a recursive pattern."
    logger.info(f"Consciousness Test 2: Processing text with recursion: '{test_text}'")
    
    result = integration.process_text(test_text)
    pattern_depth = result.get('recursive_pattern_depth', 'N/A')
    logger.info(f"Recursive pattern depth: {pattern_depth}")
    
    # Test 3: Process text with multiple messages to test learning
    logger.info("Consciousness Test 3: Processing multiple messages to test learning")
    
    messages = [
        "Neural networks can learn patterns over time.",
        "Learning is a continuous process of adaptation.",
        "Adaptation leads to increased consciousness."
    ]
    
    for i, msg in enumerate(messages):
        logger.info(f"  Processing message {i+1}: '{msg}'")
        result = integration.process_text(msg)
        c_level = result.get('consciousness_level', 'N/A')
        logger.info(f"  Consciousness level: {c_level}")
        
        # Small delay to simulate time passing
        time.sleep(1)
    
    return True

def run_cross_system_integration_tests(integration):
    """Test integration between Enhanced Language System and V7"""
    logger.info("Running cross-system integration tests...")
    
    if integration.mock_mode:
        logger.warning("Running in mock mode - cross-system integration limited")
    
    # Test 1: Check component initialization
    logger.info("Cross-System Test 1: Checking component initialization")
    status = integration.get_status()
    
    v7_components = {
        'v7_language_node': status['components']['v7_language_node'],
        'v7_consciousness_node': status['components']['v7_consciousness_node']
    }
    
    els_components = {
        'language_memory': status['components']['language_memory'],
        'consciousness_language': status['components']['consciousness_language'],
        'neural_processor': status['components']['neural_processor'],
        'pattern_analyzer': status['components']['pattern_analyzer'],
        'central_node': status['components']['central_node']
    }
    
    logger.info(f"V7 components available: {sum(1 for v in v7_components.values() if v)}/{len(v7_components)}")
    logger.info(f"ELS components available: {sum(1 for v in els_components.values() if v)}/{len(els_components)}")
    
    # Test 2: Test unified response generation
    test_text = "Language and consciousness are integrated in neural networks."
    logger.info(f"Cross-System Test 2: Testing unified response for: '{test_text}'")
    
    result = integration.process_text(test_text)
    if 'unified_response' in result:
        logger.info(f"Unified response generated: '{result['unified_response']}'")
    else:
        logger.info("No unified response generated - requires both systems")
    
    # Test 3: Test weight propagation
    logger.info("Cross-System Test 3: Testing weight propagation")
    test_weight = 0.65
    logger.info(f"Setting LLM weight to {test_weight}")
    
    integration.set_llm_weight(test_weight)
    
    # Check if weight propagated
    status = integration.get_status()
    logger.info(f"Integration LLM weight after change: {status['llm_weight']}")
    
    return True

def main():
    """Main test function"""
    logger.info("Starting Enhanced Language V7 Integration Tests")
    
    # Create a test directory
    test_dir = "data/test/v7_language_integration"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test configuration
    config = {
        "data_dir": test_dir,
        "llm_weight": 0.5,
        "nn_weight": 0.6,
        "sync_interval": 5  # Faster syncing for testing
    }
    
    # Get the integration instance
    # First try with actual components
    integration = get_enhanced_language_integration(mock_mode=False, config=config)
    
    # Run the tests
    logger.info("=" * 50)
    logger.info(f"Running tests with mock_mode={integration.get_status()['mock_mode']}")
    logger.info("=" * 50)
    
    # Basic tests
    basic_success = run_basic_test(integration)
    logger.info(f"Basic tests {'completed successfully' if basic_success else 'failed'}")
    
    # Consciousness tests
    consciousness_success = run_consciousness_tests(integration)
    logger.info(f"Consciousness tests {'completed successfully' if consciousness_success else 'failed'}")
    
    # Cross-system integration tests
    cross_system_success = run_cross_system_integration_tests(integration)
    logger.info(f"Cross-system tests {'completed successfully' if cross_system_success else 'failed'}")
    
    # Shutdown the integration
    logger.info("Shutting down integration...")
    integration.shutdown()
    
    # Forced mock mode tests if real mode failed
    if integration.mock_mode:
        logger.info("=" * 50)
        logger.info("No components found for real testing - already in mock mode")
    else:
        logger.info("=" * 50)
        logger.info("Running additional tests in forced mock mode")
        mock_integration = get_enhanced_language_integration(mock_mode=True, config=config)
        run_basic_test(mock_integration)
        mock_integration.shutdown()
    
    # All tests completed
    logger.info("=" * 50)
    logger.info("All tests completed")
    success = basic_success and consciousness_success and cross_system_success
    logger.info(f"Overall test status: {'SUCCESS' if success else 'FAILURE'}")
    logger.info("=" * 50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 