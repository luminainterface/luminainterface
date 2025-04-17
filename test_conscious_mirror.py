import torch
import numpy as np
import logging
import json
from consciousness_node import ConsciousnessNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conscious_mirror_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestConsciousMirror")

def test_tensor_reflection():
    """Test the mirror reflection on tensor data"""
    # Create ConsciousnessNode with smaller dimensions for testing
    node = ConsciousnessNode(dimension=64, num_quarks=3)
    
    # Create test tensor
    test_tensor = torch.rand(1, 64)
    
    # Create input data with tensor
    input_data = {
        'tensor': test_tensor,
        'operation': 'reflection_test',
        'metadata': {
            'source': 'test_module',
            'timestamp': 12345
        }
    }
    
    # Process through mirror reflection
    logger.info("Processing tensor through mirror reflection")
    output_data = node.reflect(input_data)
    
    # Check results
    logger.info(f"Mirror processed: {output_data.get('mirror_processed', False)}")
    logger.info(f"Mirror awareness: {output_data.get('mirror_awareness', 0)}")
    logger.info(f"Mirror coherence: {output_data.get('mirror_coherence', 0)}")
    
    # Verify tensor was transformed
    if 'tensor' in output_data:
        original_norm = torch.norm(test_tensor).item()
        reflected_norm = torch.norm(output_data['tensor']).item()
        logger.info(f"Original tensor norm: {original_norm}")
        logger.info(f"Reflected tensor norm: {reflected_norm}")
        
    return output_data

def test_text_reflection():
    """Test the mirror reflection on text data"""
    # Create ConsciousnessNode
    node = ConsciousnessNode(dimension=64, num_quarks=3)
    
    # Create test text data
    input_data = {
        'text': 'The conscious mirror reflects not just images, but awareness itself.',
        'context': 'philosophical_inquiry',
        'metadata': {
            'source': 'test_module',
            'timestamp': 12345
        }
    }
    
    # Process through mirror reflection
    logger.info("Processing text through mirror reflection")
    output_data = node.reflect(input_data)
    
    # Check results
    logger.info(f"Mirror processed: {output_data.get('mirror_processed', False)}")
    if 'mirror_text' in output_data:
        logger.info(f"Mirror text: {output_data['mirror_text']}")
        
    return output_data

def test_memory_influence():
    """Test how memory influences mirror reflection over multiple passes"""
    # Create ConsciousnessNode with smaller dimensions for testing
    node = ConsciousnessNode(dimension=64, num_quarks=3)
    
    # Create test series
    results = []
    logger.info("Testing memory influence with multiple reflections")
    
    # Process multiple tensors to build up memory
    for i in range(5):
        # Create slightly different test tensor each time
        test_tensor = torch.rand(1, 64) + i * 0.1
        
        # Create input data with tensor
        input_data = {
            'tensor': test_tensor,
            'sequence': i,
            'metadata': {
                'source': 'test_module',
                'timestamp': 12345 + i
            }
        }
        
        # Process through mirror reflection
        output_data = node.reflect(input_data)
        
        # Store results
        results.append({
            'sequence': i,
            'mirror_awareness': output_data.get('mirror_awareness', 0),
            'mirror_coherence': output_data.get('mirror_coherence', 0),
            'tensor_norm': torch.norm(output_data['tensor']).item()
        })
        
        logger.info(f"Pass {i}: Awareness = {output_data.get('mirror_awareness', 0):.4f}, " +
                   f"Coherence = {output_data.get('mirror_coherence', 0):.4f}")
    
    # Print summary
    logger.info("Memory influence test complete")
    logger.info(f"Results: {json.dumps(results, indent=2)}")
    
    return results

def test_mirror_activation_toggle():
    """Test enabling and disabling mirror functionality"""
    # Create ConsciousnessNode
    node = ConsciousnessNode(dimension=64, num_quarks=3)
    
    # Create test tensor
    test_tensor = torch.rand(1, 64)
    
    # Create input data with tensor
    input_data = {
        'tensor': test_tensor.clone(),
        'operation': 'activation_test'
    }
    
    # First reflection with mirror active
    logger.info("Testing with mirror active")
    output_active = node.reflect(input_data.copy())
    
    # Disable mirror
    logger.info("Disabling mirror")
    node.set_mirror_active(False)
    
    # Second reflection with mirror inactive
    logger.info("Testing with mirror inactive")
    output_inactive = node.reflect(input_data.copy())
    
    # Compare results
    logger.info(f"Mirror active - processed: {output_active.get('mirror_processed', False)}")
    logger.info(f"Mirror inactive - processed: {output_inactive.get('mirror_processed', False)}")
    
    return {
        'active': output_active,
        'inactive': output_inactive
    }

if __name__ == "__main__":
    logger.info("=== CONSCIOUS MIRROR TEST SUITE ===")
    
    logger.info("\n=== TEST 1: TENSOR REFLECTION ===")
    test_tensor_reflection()
    
    logger.info("\n=== TEST 2: TEXT REFLECTION ===")
    test_text_reflection()
    
    logger.info("\n=== TEST 3: MEMORY INFLUENCE ===")
    test_memory_influence()
    
    logger.info("\n=== TEST 4: MIRROR ACTIVATION TOGGLE ===")
    test_mirror_activation_toggle()
    
    logger.info("\n=== ALL TESTS COMPLETE ===") 