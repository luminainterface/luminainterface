#!/usr/bin/env python3
"""
Mistral Integration Patch

This script creates a patch for the Mistral integration to fix the issue
with NeuralLinguisticProcessor initialization.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("MistralPatch")

# Make sure src directory is in path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

def patch_central_language_node():
    """
    Patches the CentralLanguageNode._initialize_components method to handle 
    the NeuralLinguisticProcessor initialization correctly.
    """
    try:
        # Import the module to patch
        from src.language.central_language_node import CentralLanguageNode
        
        # Store the original method
        original_initialize_components = CentralLanguageNode._initialize_components
        
        # Define the patched method
        def patched_initialize_components(self):
            """Patched version of _initialize_components that handles NeuralLinguisticProcessor correctly."""
            try:
                # Import components using relative imports
                from .language_memory import LanguageMemory
                from .conscious_mirror_language import ConsciousMirrorLanguage
                from .neural_linguistic_processor import NeuralLinguisticProcessor
                from .recursive_pattern_analyzer import RecursivePatternAnalyzer
                
                # Import the new neural linguistic flex bridge
                try:
                    from .neural_linguistic_flex_bridge import get_neural_linguistic_flex_bridge
                    self.neural_flex_bridge = get_neural_linguistic_flex_bridge({
                        "mock_mode": False,  # Try to use real components
                        "embedding_dim": 256,
                        "learning_rate": 0.01,
                        "feedback_alpha": 0.3,
                        "pattern_weight": 0.7
                    })
                    logger.info("Neural Linguistic Flex Bridge initialized successfully")
                    
                    # Start the bridge if initialized successfully
                    if self.neural_flex_bridge.initialized:
                        self.neural_flex_bridge.start()
                        logger.info("Neural Linguistic Flex Bridge started")
                    else:
                        logger.warning("Neural Linguistic Flex Bridge initialization incomplete")
                except ImportError as e:
                    logger.warning(f"Neural Linguistic Flex Bridge not available: {e}")
                    self.neural_flex_bridge = None
                
                # Initialize components with both weights
                self.language_memory = LanguageMemory(
                    data_dir=self.language_memory_dir,
                    llm_weight=self.llm_weight,
                    nn_weight=self.nn_weight
                )
                
                self.conscious_mirror = ConsciousMirrorLanguage(
                    data_dir=self.conscious_mirror_dir,
                    llm_weight=self.llm_weight,
                    nn_weight=self.nn_weight
                )
                
                # *** PATCH: The NeuralLinguisticProcessor only accepts data_dir ***
                self.neural_processor = NeuralLinguisticProcessor(
                    data_dir=self.neural_processor_dir
                )
                # Then set the weights separately
                if hasattr(self.neural_processor, 'set_llm_weight'):
                    self.neural_processor.set_llm_weight(self.llm_weight)
                if hasattr(self.neural_processor, 'set_nn_weight'):
                    self.neural_processor.set_nn_weight(self.nn_weight)
                
                self.pattern_analyzer = RecursivePatternAnalyzer(
                    data_dir=self.recursive_patterns_dir,
                    llm_weight=self.llm_weight,
                    nn_weight=self.nn_weight
                )
                
                logger.info("All components initialized successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import components: {e}")
                raise
            except Exception as e:
                logger.error(f"Error initializing components: {e}")
                raise
        
        # Apply the patch
        CentralLanguageNode._initialize_components = patched_initialize_components
        logger.info("Successfully patched CentralLanguageNode._initialize_components method")
        return True
    except Exception as e:
        logger.error(f"Error patching CentralLanguageNode: {e}")
        return False

def main():
    # Apply the patch
    success = patch_central_language_node()
    
    if success:
        # Try importing MistralEnhancedSystem with the patch
        try:
            from src.mistral_integration import MistralEnhancedSystem
            logger.info("Successfully imported MistralEnhancedSystem with the patch")
            
            # Test creating an instance with the patched code
            system = MistralEnhancedSystem(
                data_dir="data/test_mistral",
                llm_weight=0.7,
                nn_weight=0.6
            )
            
            logger.info("Successfully created MistralEnhancedSystem instance")
            if system.central_node:
                logger.info(f"Central Language Node initialized with LLM weight: {system.central_node.llm_weight}, NN weight: {system.central_node.nn_weight}")
            
            # Test with a simple message
            response = system.process_message("Hello, this is a test message for the patched Mistral integration.")
            logger.info(f"Response: {response}")
            
            # Clean up
            system.close()
            
        except Exception as e:
            logger.error(f"Error testing patched Mistral integration: {e}")
    else:
        logger.error("Failed to apply patch, cannot test Mistral integration")

if __name__ == "__main__":
    main() 