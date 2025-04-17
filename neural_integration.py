#!/usr/bin/env python3
"""
Neural Network Integration Module for Mistral Weighted Chat System

This module integrates the neural network processor with the Mistral weighted chat system,
providing a unified interface for processing text, adjusting weights, and enhancing responses.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/neural_integration.log", mode='a')
    ]
)
logger = logging.getLogger("NeuralIntegration")

# Import our neural network processor
try:
    from neural_network_module import NeuralNetworkProcessor, NeuralProcessingState
    NEURAL_AVAILABLE = True
except ImportError:
    logger.warning("Neural network module not available. Running in limited mode.")
    NEURAL_AVAILABLE = False


class NeuralIntegration:
    """
    Neural Network Integration for Mistral Weighted Chat System
    
    This class integrates the neural network processor with the Mistral chat system,
    allowing for weighted processing of messages and enhanced responses.
    """
    
    def __init__(
        self, 
        model_dir: Optional[str] = None,
        llm_weight: float = 0.7, 
        nn_weight: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the neural integration
        
        Args:
            model_dir: Directory for neural network models and configuration
            llm_weight: Weight for the language model (0.0-1.0)
            nn_weight: Weight for the neural network (0.0-1.0)
            device: Device to use for PyTorch computations ('cpu', 'cuda')
        """
        self.model_dir = model_dir or os.path.join("data", "neural_models")
        self.llm_weight = max(0.0, min(1.0, llm_weight))
        self.nn_weight = max(0.0, min(1.0, nn_weight))
        
        # Initialize neural processor if available
        if NEURAL_AVAILABLE:
            try:
                self.neural_processor = NeuralNetworkProcessor(
                    model_dir=self.model_dir,
                    device=device
                )
                logger.info("Neural network processor initialized successfully")
                self.neural_available = True
                
                # Update the weights in the neural processor
                self.neural_processor.set_nn_weight(self.nn_weight)
                
                # Also update the speaker config
                self.neural_processor.speaker_config["llm_weight"] = self.llm_weight
                self.neural_processor.speaker_config["nn_weight"] = self.nn_weight
                
            except Exception as e:
                logger.error(f"Error initializing neural processor: {e}")
                self.neural_processor = None
                self.neural_available = False
        else:
            logger.warning("Neural network module not available, running with limited functionality")
            self.neural_processor = None
            self.neural_available = False
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "neural_calls": 0,
            "average_neural_score": 0.0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
    
    def adjust_weights(self, llm_weight: Optional[float] = None, nn_weight: Optional[float] = None) -> Dict[str, float]:
        """
        Adjust the weights of the LLM and neural network components
        
        Args:
            llm_weight: New weight for LLM (0-1)
            nn_weight: New weight for neural network (0-1)
            
        Returns:
            Dict with updated weights
        """
        # Update weights if provided
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, llm_weight))
        
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, nn_weight))
            
            # Update neural processor weight if available
            if self.neural_processor:
                self.neural_processor.set_nn_weight(self.nn_weight)
                # Also update the speaker config
                self.neural_processor.speaker_config["llm_weight"] = self.llm_weight
        
        logger.info(f"Weights adjusted - LLM: {self.llm_weight}, NN: {self.nn_weight}")
        
        return {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight
        }
    
    def process_message(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a message using neural network and prepare enhanced information
        
        Args:
            message: Message to process
            context: Optional context from memory or previous messages
            
        Returns:
            Dict with neural processing results and enhancement information
        """
        start_time = time.time()
        self.stats["messages_processed"] += 1
        
        result = {
            "message": message,
            "neural_enhanced": False,
            "neural_score": 0.0,
            "llm_weight": self.llm_weight, 
            "nn_weight": self.nn_weight,
            "processing_time": 0.0
        }
        
        # Skip neural processing if not available
        if not self.neural_available or not self.neural_processor:
            result["neural_enhanced"] = False
            result["processing_time"] = time.time() - start_time
            return result
        
        # Process with neural network
        try:
            self.stats["neural_calls"] += 1
            
            # Get enhanced response information
            neural_result = self.neural_processor.get_enhanced_response(message, context)
            
            # Extract data from neural results
            result.update({
                "neural_enhanced": True,
                "neural_score": neural_result["neural_score"],
                "concept_activations": neural_result.get("concept_activations", {}),
                "token_count": neural_result.get("token_count", 0),
                "top_words": neural_result.get("word_frequencies", {})
            })
            
            # Add context information if available
            if context and "context_neural_score" in neural_result:
                result["context_neural_score"] = neural_result["context_neural_score"]
            
            # Update stats
            curr_time = time.time() - start_time
            self.stats["total_processing_time"] += curr_time
            self.stats["average_processing_time"] = self.stats["total_processing_time"] / self.stats["neural_calls"]
            
            # Update average neural score
            current_avg = self.stats["average_neural_score"]
            new_value = neural_result["neural_score"]
            new_count = self.stats["neural_calls"]
            self.stats["average_neural_score"] = ((current_avg * (new_count - 1)) + new_value) / new_count
            
            result["processing_time"] = curr_time
            
        except Exception as e:
            logger.error(f"Error in neural processing: {e}")
            result["neural_enhanced"] = False
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def enhance_response(self, message: str, response: str, neural_data: Dict[str, Any]) -> str:
        """
        Enhance a response using neural processing data
        
        Args:
            message: Original user message
            response: Original response from LLM
            neural_data: Neural processing data
            
        Returns:
            Enhanced response
        """
        # If neural enhancement isn't available, return original response
        if not neural_data.get("neural_enhanced", False):
            return response
        
        # Get neural insights
        neural_score = neural_data.get("neural_score", 0.0)
        concept_activations = neural_data.get("concept_activations", {})
        
        # Only enhance if neural score is significant
        if neural_score < 0.1 or self.nn_weight < 0.2:
            return response
        
        # Extract top concepts
        top_concepts = list(concept_activations.items())[:3]
        concepts_str = ", ".join([f"{c}: {v:.2f}" for c, v in top_concepts])
        
        # Create neural insight addition based on the nature of the message
        if "?" in message:
            # For questions, add neural insight at the end
            neural_insight = f"\n\n[Neural insight (confidence: {neural_score:.2f}): "
            neural_insight += f"Your question activates these conceptual patterns: {concepts_str}]"
            
            # More subtle integration for lower weights
            if self.nn_weight < 0.4:
                return response
            elif self.nn_weight < 0.6:
                return response + f"\n\n[Neural analysis confidence: {neural_score:.2f}]"
            else:
                return response + neural_insight
                
        elif any(cmd in message.lower() for cmd in ["explain", "describe", "elaborate", "tell me about"]):
            # For explanation requests, integrate more neural influence
            if self.nn_weight > 0.6:
                neural_intro = f"[Neural analysis (weight: {self.nn_weight:.1f})] "
                return neural_intro + response
            else:
                return response
        
        # Default case - just return the original response
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the neural integration
        
        Returns:
            Dict with status information
        """
        status = {
            "neural_available": self.neural_available,
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "stats": self.stats
        }
        
        # Add neural processor information if available
        if self.neural_processor:
            status["neural_processor"] = {
                "model_dir": self.neural_processor.model_dir,
                "temperature": self.neural_processor.temperature,
                "vocab_size": self.neural_processor.vocab_size,
                "vocab_entries": len(self.neural_processor.word_to_idx) if hasattr(self.neural_processor, "word_to_idx") else 0
            }
            
            # Add device information if using PyTorch
            if hasattr(self.neural_processor, "device"):
                status["neural_processor"]["device"] = str(self.neural_processor.device)
        
        return status


# For testing
if __name__ == "__main__":
    # Create neural integration
    integration = NeuralIntegration(
        llm_weight=0.7,
        nn_weight=0.3
    )
    
    # Check status
    status = integration.get_status()
    print("Neural Integration Status:")
    print(json.dumps(status, indent=2))
    
    # Test message processing
    test_message = "What is the relationship between neural networks and language models?"
    print(f"\nProcessing message: '{test_message}'")
    
    # Process message
    result = integration.process_message(test_message)
    print("\nProcessing Result:")
    print(json.dumps(result, indent=2))
    
    # Test weight adjustment
    print("\nAdjusting weights (LLM: 0.4, NN: 0.6)")
    new_weights = integration.adjust_weights(llm_weight=0.4, nn_weight=0.6)
    print(f"New weights: LLM={new_weights['llm_weight']}, NN={new_weights['nn_weight']}")
    
    # Process message with new weights
    result = integration.process_message(test_message)
    print("\nProcessing Result with New Weights:")
    print(json.dumps(result, indent=2))
    
    # Test response enhancement
    sample_response = "Neural networks and language models are complementary technologies. Language models focus on understanding and generating text, while neural networks provide the computational architecture that enables modern language models to learn patterns and relationships in data."
    
    enhanced = integration.enhance_response(test_message, sample_response, result)
    print("\nEnhanced Response:")
    print(enhanced) 