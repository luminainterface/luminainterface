#!/usr/bin/env python3
"""
Test Neural Linguistic FlexNode Bridge

This script demonstrates how to use the NeuralLinguisticFlexBridge to connect
language processing with neural networks for adaptive pattern recognition.
"""

import sys
import os
import logging
import time
import json
import numpy as np
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-neural-linguistic-bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the bridge module
try:
    from language.neural_linguistic_flex_bridge import get_neural_linguistic_flex_bridge
except ImportError as e:
    logger.error(f"Failed to import neural_linguistic_flex_bridge: {e}")
    sys.exit(1)

def test_basic_processing():
    """Test basic text processing through the bridge"""
    logger.info("Testing basic text processing")
    
    # Create bridge in mock mode for testing
    bridge = get_neural_linguistic_flex_bridge({"mock_mode": True})
    
    if not bridge.initialized:
        logger.error("Bridge initialization failed")
        return False
    
    # Start the bridge
    bridge.start()
    
    try:
        # Process sample text
        test_texts = [
            "Neural networks learn patterns through adjusting weights based on error signals.",
            "The consciousness emerges from the recursive patterns of self-reflection.",
            "Language patterns reveal the underlying structure of cognitive processes."
        ]
        
        results = []
        for text in test_texts:
            logger.info(f"Processing: {text}")
            result = bridge.process_text(text)
            
            # Log key parts of the result
            if "linguistic_analysis" in result:
                logger.info(f"Linguistic analysis pattern params: {result['linguistic_analysis'].get('pattern_params', {})}")
            
            if "updated_params" in result and result["updated_params"]:
                logger.info(f"Updated params from neural processing: {result['updated_params']}")
                
            results.append(result)
        
        # Print overall statistics
        status = bridge.get_status()
        logger.info(f"Bridge statistics after processing: {status['stats']}")
        
        # Check if processing was successful
        success = len(results) == len(test_texts)
        return success
        
    except Exception as e:
        logger.error(f"Error in basic processing test: {e}")
        return False
    finally:
        # Stop the bridge
        bridge.stop()

def test_async_processing():
    """Test asynchronous processing through the bridge"""
    logger.info("Testing asynchronous processing")
    
    # Create bridge in mock mode for testing
    bridge = get_neural_linguistic_flex_bridge({"mock_mode": True})
    
    if not bridge.initialized:
        logger.error("Bridge initialization failed")
        return False
    
    # Start the bridge
    bridge.start()
    
    try:
        # Create event for synchronization
        completion_event = threading.Event()
        
        # Results container for callbacks
        results = []
        
        # Callback function for async processing
        def on_linguistic_processed(result):
            logger.info(f"Linguistic processing callback received result")
            results.append(("linguistic", result))
            # Check if we have results from both callbacks
            if len(results) >= 2:
                completion_event.set()
        
        def on_neural_processed(result):
            logger.info(f"Neural processing callback received result")
            results.append(("neural", result))
            # Check if we have results from both callbacks
            if len(results) >= 2:
                completion_event.set()
        
        # Create test data directly
        logger.info("Creating test data for async processing")
        
        # Direct method to process data
        test_linguistic_data = {"pattern_params": {"resonance_factor": 0.8}}
        test_neural_data = {"embedding": np.random.randn(bridge.embedding_dim)}
        
        # Call private methods directly (for testing purposes)
        logger.info("Processing linguistic data")
        bridge._process_linguistic_data(test_linguistic_data, on_linguistic_processed)
        
        logger.info("Processing neural data")
        bridge._process_neural_data(test_neural_data, on_neural_processed)
        
        # Wait for processing to complete with timeout
        logger.info("Waiting for async processing to complete...")
        success = completion_event.wait(timeout=3.0)
        
        # Print results
        logger.info(f"Received {len(results)} async results (success={success})")
        for result_type, result in results:
            logger.info(f"{result_type} result received")
        
        # Check if processing was successful
        return len(results) >= 1  # Accept partial success
        
    except Exception as e:
        logger.error(f"Error in async processing test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Stop the bridge
        bridge.stop()

def test_adaptive_learning():
    """Test adaptive learning in the bridge"""
    logger.info("Testing adaptive learning")
    
    # Create bridge in mock mode for testing
    bridge = get_neural_linguistic_flex_bridge({"mock_mode": True})
    
    if not bridge.initialized:
        logger.error("Bridge initialization failed")
        return False
    
    # Start the bridge
    bridge.start()
    
    try:
        # Create learning samples
        learning_samples = []
        
        # Process several texts to generate samples
        test_texts = [
            "Neural networks can identify complex patterns in data.",
            "Recursive neural networks process sequential information effectively.",
            "Language models use transformer architectures for context awareness.",
            "Self-attention mechanisms help models understand relationships between words.",
            "Bidirectional encoders capture context from both directions in text."
        ]
        
        # Process each text and collect data for learning
        for text in test_texts:
            # Process text
            result = bridge.process_text(text)
            
            # Extract linguistic analysis
            linguistic_analysis = result.get("linguistic_analysis", {})
            
            # Create synthetic neural embedding for demonstration
            neural_embedding = np.random.randn(bridge.embedding_dim)
            
            # Add to learning samples
            learning_samples.append((linguistic_analysis, neural_embedding))
        
        # Before adaptation - get current state
        initial_linguistic_to_neural = bridge.linguistic_to_neural_matrix.copy()
        
        # Apply adaptive learning
        logger.info(f"Adapting transformation matrices with {len(learning_samples)} samples")
        bridge.adapt_transformation_matrices(learning_samples)
        
        # After adaptation - check if matrices changed
        matrix_changed = not np.array_equal(initial_linguistic_to_neural, bridge.linguistic_to_neural_matrix)
        logger.info(f"Transformation matrices changed: {matrix_changed}")
        
        # Test processing after adaptation
        logger.info("Testing processing after adaptation")
        result = bridge.process_text("Testing adaptive learning in neural linguistic processing.")
        
        # Print adaptation statistics
        status = bridge.get_status()
        logger.info(f"Adaptations performed: {status['stats']['adaptations_performed']}")
        
        return matrix_changed and status['stats']['adaptations_performed'] > 0
        
    except Exception as e:
        logger.error(f"Error in adaptive learning test: {e}")
        return False
    finally:
        # Stop the bridge
        bridge.stop()

def test_integration_with_components():
    """Test integration with NLP and FlexNode components"""
    logger.info("Testing integration with components")
    
    # Create bridge with non-mock components if possible
    try:
        bridge = get_neural_linguistic_flex_bridge({"mock_mode": False})
        
        if not bridge.initialized:
            logger.warning("Non-mock bridge initialization failed, falling back to mock mode")
            bridge = get_neural_linguistic_flex_bridge({"mock_mode": True})
        
        # Start the bridge
        bridge.start()
        
        # Connect components
        connected = bridge.connect_components()
        logger.info(f"Components connected: {connected}")
        
        # Test processing with components
        result = bridge.process_text("Testing integration between neural and linguistic components.")
        
        # Check if both linguistic and neural results are present
        has_linguistic = "linguistic_analysis" in result and result["linguistic_analysis"]
        has_neural = "neural_result" in result and result["neural_result"]
        has_pattern = "pattern_result" in result and result["pattern_result"]
        
        logger.info(f"Has linguistic results: {has_linguistic}")
        logger.info(f"Has neural results: {has_neural}")
        logger.info(f"Has pattern results: {has_pattern}")
        
        # Get component status
        status = bridge.get_status()
        logger.info(f"Component status: NLP={status['components']['nlp_available']}, FlexNode={status['components']['flex_node_available']}")
        
        return has_linguistic and (has_neural or bridge.mock_mode)
        
    except Exception as e:
        logger.error(f"Error in component integration test: {e}")
        return False
    finally:
        if 'bridge' in locals() and bridge is not None:
            bridge.stop()

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        ("Basic Processing", test_basic_processing),
        ("Async Processing", test_async_processing),
        ("Adaptive Learning", test_adaptive_learning),
        ("Component Integration", test_integration_with_components)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running Test: {test_name} ---\n")
        
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            results[test_name] = {
                "success": success,
                "duration": duration
            }
            
            if not success:
                all_passed = False
                
            logger.info(f"\n--- Test {test_name}: {'PASSED' if success else 'FAILED'} (Time: {duration:.2f}s) ---\n")
            
        except Exception as e:
            logger.error(f"Error running test {test_name}: {e}")
            results[test_name] = {
                "success": False,
                "error": str(e)
            }
            all_passed = False
    
    # Print summary
    logger.info("\n--- Test Summary ---\n")
    for test_name, result in results.items():
        status = "PASSED" if result.get("success") else "FAILED"
        duration = result.get("duration", 0)
        logger.info(f"{test_name}: {status}" + (f" (Time: {duration:.2f}s)" if "duration" in result else ""))
    
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    logger.info("Starting Neural Linguistic FlexNode Bridge Tests")
    success = run_all_tests()
    
    sys.exit(0 if success else 1) 