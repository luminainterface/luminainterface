#!/usr/bin/env python3
"""
Test Neural Integration

This script tests the integration between the Mistral chat system,
neural processor, and RSEN components.
"""

import os
import sys
import logging
from pathlib import Path
import torch
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/neural_integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralIntegrationTest")

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        "data",
        "data/logs",
        "data/model_output",
        "data/neural_linguistic",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def test_neural_processor():
    """Test the NeuralProcessor integration"""
    try:
        from neural_processor import NeuralProcessor
        
        # Initialize neural processor
        processor = NeuralProcessor(
            model_dir="data/model_output",
            embedding_dim=768,
            output_dim=512,
            num_concepts=200
        )
        
        logger.info(f"NeuralProcessor initialized with {processor.num_concepts} concepts")
        
        # Test processing some text
        test_text = "Neural networks can process language with remarkable effectiveness."
        
        # Process text
        processing_state = processor.process_text(test_text)
        
        # Check results
        if processing_state.embedding is not None:
            embedding_shape = processing_state.embedding.shape
            logger.info(f"Embedding shape: {embedding_shape}")
        else:
            logger.error("Failed to generate embedding")
            
        if processing_state.activations is not None:
            activations_shape = processing_state.activations.shape
            logger.info(f"Activations shape: {activations_shape}")
            
            # Get top activations
            flat_activations = processing_state.activations.detach().cpu().numpy().flatten()
            top_indices = torch.topk(torch.abs(torch.tensor(flat_activations)), 5)[1].tolist()
            top_values = [flat_activations[i] for i in top_indices]
            
            logger.info(f"Top activations: {list(zip(top_indices, top_values))}")
        else:
            logger.error("Failed to generate activations")
        
        # Test with different parameters
        logger.info("Testing with custom temperature...")
        processor.temperature = 0.9
        
        # Process text again
        processing_state2 = processor.process_text("Testing with different temperature settings.")
        
        logger.info(f"Second processing complete. Temperature: {processor.temperature}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing neural processor: {e}")
        return False

def test_rsen_integration():
    """Test RSEN integration if available"""
    try:
        # Try to import RSEN
        from RSEN_node import RSEN
        
        # Initialize RSEN
        rsen = RSEN(input_dim=768, hidden_dim=512, output_dim=256)
        
        logger.info("RSEN initialized successfully")
        
        # Test processing some text
        test_text = "Quantum patterns emerge from neural processing."
        
        # Process through RSEN
        result = rsen.train_epoch(test_text)
        
        # Check results
        if isinstance(result, dict):
            logger.info(f"RSEN returned {len(result)} result components")
            
            # Check for expected components
            for component in ['loss', 'math_metrics', 'physics_metrics', 'quantum_metrics']:
                if component in result:
                    logger.info(f"Found {component} in results")
                else:
                    logger.warning(f"{component} not found in results")
            
            # Save RSEN results for inspection
            with open("data/logs/rsen_test_results.json", "w") as f:
                # Convert any non-serializable values to strings
                serializable_result = {}
                for k, v in result.items():
                    if isinstance(v, dict):
                        serializable_result[k] = {sk: str(sv) for sk, sv in v.items()}
                    else:
                        serializable_result[k] = str(v)
                        
                json.dump(serializable_result, f, indent=2)
            
            logger.info("RSEN results saved to data/logs/rsen_test_results.json")
        else:
            logger.warning(f"Unexpected RSEN result type: {type(result)}")
        
        return True
    except ImportError as e:
        logger.warning(f"RSEN not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing RSEN: {e}")
        return False

def test_database_memory():
    """Test database memory with neural integration"""
    try:
        from simple_mistral_gui import DatabaseMemory
        
        # Initialize memory
        memory = DatabaseMemory(data_dir="data/onsite_memory")
        
        logger.info(f"DatabaseMemory initialized: {memory.db_available}")
        
        # Add test conversation
        memory.add_conversation(
            "How do neural networks process language?",
            "Neural networks process language through embedding layers, attention mechanisms, and deep transformer architectures.",
            {"neural_test": True}
        )
        
        # Add test knowledge
        memory.add_knowledge(
            "Neural Processing",
            "Neural networks can effectively process language by converting text into vector embeddings and applying various attention mechanisms.",
            "Neural Integration Test"
        )
        
        # Test search with neural context
        search_results = memory.search_context("neural language processing")
        
        logger.info(f"Search returned {len(search_results)} results")
        
        # Test database connection if available
        if memory.db_available and memory.db_connected:
            logger.info("Database connection available")
            
            # Test synchronization
            try:
                memory._sync_to_database()
                logger.info("Successfully synced to database")
            except Exception as e:
                logger.error(f"Error syncing to database: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing database memory: {e}")
        return False

def test_weight_parameters():
    """Test setting NN and LLM weights"""
    # Set test weights
    os.environ["NN_WEIGHT"] = "0.75"
    os.environ["LLM_WEIGHT"] = "0.65"
    
    try:
        # Test central language node if available
        try:
            from src.language.central_language_node import CentralLanguageNode
            
            node = CentralLanguageNode(
                data_dir="data",
                llm_weight=float(os.environ.get("LLM_WEIGHT", "0.5")),
                nn_weight=float(os.environ.get("NN_WEIGHT", "0.5"))
            )
            
            logger.info(f"CentralLanguageNode initialized with LLM weight {node.llm_weight} and NN weight {node.nn_weight}")
            
            # Test setting weights
            node.set_llm_weight(0.8)
            node.set_nn_weight(0.7)
            
            logger.info(f"After update: LLM weight {node.llm_weight}, NN weight {node.nn_weight}")
        except ImportError:
            logger.warning("CentralLanguageNode not available")
        
        # Test neural processor weight storage
        from neural_processor import NeuralProcessor
        
        processor = NeuralProcessor(
            model_dir="data/model_output"
        )
        
        # Set up speaker config with weights
        processor.speaker_config = {
            "llm_weight": float(os.environ.get("LLM_WEIGHT", "0.5")),
            "nn_weight": float(os.environ.get("NN_WEIGHT", "0.5"))
        }
        
        logger.info(f"NeuralProcessor configured with weights in speaker_config: {processor.speaker_config}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing weight parameters: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Neural Integration Tests")
    
    # Ensure directories
    ensure_directories()
    
    # Run tests
    results = {}
    
    # Test neural processor
    logger.info("\n--- Testing Neural Processor ---")
    results["neural_processor"] = test_neural_processor()
    
    # Test RSEN
    logger.info("\n--- Testing RSEN Integration ---")
    results["rsen"] = test_rsen_integration()
    
    # Test database memory
    logger.info("\n--- Testing Database Memory ---")
    results["database_memory"] = test_database_memory()
    
    # Test weight parameters
    logger.info("\n--- Testing Weight Parameters ---")
    results["weight_parameters"] = test_weight_parameters()
    
    # Summarize results
    logger.info("\n=== Test Summary ===")
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test}: {status}")
    
    # Overall result
    if all(results.values()):
        logger.info("All tests PASSED")
        return 0
    else:
        logger.warning("Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 