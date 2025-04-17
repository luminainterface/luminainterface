#!/usr/bin/env python3
"""
Enhanced Language Integration with Mistral AI for V7

This module integrates the V7 Enhanced Language System with Mistral AI's language models,
combining neural network consciousness metrics with Mistral's language capabilities.

Features:
- Seamless integration between EnhancedLanguageIntegration and MistralIntegration
- Consciousness-aware text processing enhanced by Mistral AI models
- Autowiki learning system integration
- Combined neural metrics processing
- Support for streaming responses
- Mock mode for development without API keys
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import V7 components
try:
    from src.v7.enhanced_language_integration import EnhancedLanguageIntegration
    from src.v7.mistral_integration import MistralIntegration
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing required components: {e}")
    COMPONENTS_AVAILABLE = False

class EnhancedLanguageMistralIntegration:
    """
    Integration of Mistral AI with the Enhanced Language System.
    
    This class bridges the Enhanced Language Integration system with Mistral AI,
    combining neural consciousness processing with advanced language model capabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-medium",
        system_manager: Any = None,
        mock_mode: bool = False,
        learning_enabled: bool = True,
        learning_dict_path: str = "data/mistral_learning.json",
        llm_weight: float = 0.7,
        nn_weight: float = 0.6,
        min_consciousness_level: float = 0.2,
    ):
        """
        Initialize the Enhanced Language Mistral Integration.
        
        Args:
            api_key: Mistral API key (optional if using mock_mode)
            model: Mistral model to use
            system_manager: Optional system manager for connecting to other components
            mock_mode: If True, generate mock responses instead of calling the API
            learning_enabled: If True, enable the autowiki learning functionality
            learning_dict_path: Path to save/load the learning dictionary
            llm_weight: Weight given to language model processing (0.0-1.0)
            nn_weight: Weight given to neural network processing (0.0-1.0)
            min_consciousness_level: Minimum consciousness level for normal responses
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.model = model
        self.mock_mode = mock_mode or not self.api_key
        self.learning_enabled = learning_enabled
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        self.min_consciousness_level = min_consciousness_level
        
        # Initialize Enhanced Language Integration
        self.language_system = EnhancedLanguageIntegration(system_manager=system_manager)
        self.language_system.llm_weight = llm_weight
        self.language_system.nn_weight = nn_weight
        
        # Initialize Mistral Integration
        self.mistral = MistralIntegration(
            api_key=self.api_key,
            model=self.model,
            mock_mode=self.mock_mode,
            learning_enabled=learning_enabled,
            learning_dict_path=learning_dict_path,
        )
        
        # Set component availability flags
        self.language_available = True
        self.mistral_available = True
        
        logger.info(f"Enhanced Language Mistral Integration initialized")
        logger.info(f"Model: {self.model}, Mock Mode: {self.mock_mode}")
        logger.info(f"LLM Weight: {self.llm_weight}, NN Weight: {self.nn_weight}")
    
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text through the integrated system.
        
        Args:
            text: The text to process
            context: Optional context information
            
        Returns:
            Dict containing the combined results
        """
        results = {}
        
        # Process with Enhanced Language Integration
        language_result = self.language_system.process_text(text, context)
        
        # Extract consciousness metrics
        consciousness_level = language_result.get("consciousness_level", 0.0)
        neural_linguistic_score = language_result.get("neural_linguistic_score", 0.0)
        recursive_pattern_depth = language_result.get("recursive_pattern_depth", 0)
        
        # Prepare system prompt using consciousness data
        system_prompt = self._create_system_prompt(
            consciousness_level, 
            neural_linguistic_score,
            recursive_pattern_depth
        )
        
        # Check if we should use autowiki
        include_autowiki = consciousness_level >= self.min_consciousness_level
        
        # Process with Mistral
        mistral_result = self.mistral.process_message(
            message=text,
            system_prompt=system_prompt,
            temperature=min(0.5 + consciousness_level * 0.5, 0.85),
            max_tokens=min(100 + int(consciousness_level * 400), 500),
            include_autowiki=include_autowiki
        )
        
        # Create combined response
        combined_response = mistral_result.get("response", "")
        
        # Store the exchange in conversation memory if learning is enabled
        if self.learning_enabled:
            self.mistral.add_to_conversation_memory(
                message=text,
                response=combined_response,
                system_prompt=system_prompt
            )
        
        # Combine all results
        results = {
            "response": combined_response,
            "consciousness_level": consciousness_level,
            "neural_linguistic_score": neural_linguistic_score,
            "recursive_pattern_depth": recursive_pattern_depth,
            "is_mock": self.mock_mode,
            "included_autowiki": include_autowiki,
            "mistral_model": self.model,
            "processing_time": language_result.get("processing_time", 0.0) + 
                               mistral_result.get("processing_time", 0.0),
            "enhanced_language_results": language_result,
            "mistral_results": mistral_result
        }
        
        return results
    
    def process_text_streaming(
        self, 
        text: str, 
        chunk_callback: Callable[[str, Dict[str, Any]], None],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process text with streaming response.
        
        Args:
            text: The text to process
            chunk_callback: Callback function that receives text chunks and metrics
            context: Optional context information
            
        Returns:
            Final complete response text
        """
        # Process with Enhanced Language Integration to get consciousness metrics
        language_result = self.language_system.process_text(text, context)
        
        # Extract consciousness metrics
        consciousness_level = language_result.get("consciousness_level", 0.0)
        neural_linguistic_score = language_result.get("neural_linguistic_score", 0.0)
        recursive_pattern_depth = language_result.get("recursive_pattern_depth", 0)
        
        # Prepare system prompt using consciousness data
        system_prompt = self._create_system_prompt(
            consciousness_level, 
            neural_linguistic_score,
            recursive_pattern_depth
        )
        
        # Generate response but simulate streaming
        # Start by processing with Mistral
        mistral_result = self.mistral.process_message(
            message=text,
            system_prompt=system_prompt,
            temperature=min(0.5 + consciousness_level * 0.5, 0.85),
            max_tokens=min(100 + int(consciousness_level * 400), 500),
            include_autowiki=consciousness_level >= self.min_consciousness_level
        )
        
        # Get the full response
        full_response = mistral_result.get("response", "")
        
        # Simulate streaming by breaking it into chunks
        # This is a simple implementation - in a real system, you would use
        # the streaming API of the language model directly
        import time
        import random
        
        # Initial metrics to send with chunks
        metrics = {
            "consciousness_level": consciousness_level,
            "neural_linguistic_score": neural_linguistic_score,
            "recursive_pattern_depth": recursive_pattern_depth,
            "processing_progress": 0.0
        }
        
        # Break into sentences or chunks
        import re
        chunks = re.split(r'(?<=[.!?])\s+', full_response)
        
        # Stream each chunk with a small delay
        for i, chunk in enumerate(chunks):
            # Add a small delay to simulate streaming
            time.sleep(0.1 + random.random() * 0.2)
            
            # Update progress metrics
            progress = (i + 1) / len(chunks)
            metrics["processing_progress"] = progress
            
            # Send chunk via callback
            if chunk_callback:
                chunk_callback(chunk + " ", metrics.copy())
        
        # Store the exchange in conversation memory if learning is enabled
        if self.learning_enabled:
            self.mistral.add_to_conversation_memory(
                message=text,
                response=full_response,
                system_prompt=system_prompt
            )
        
        return full_response
    
    def _create_system_prompt(
        self, 
        consciousness_level: float, 
        neural_score: float,
        pattern_depth: int
    ) -> str:
        """
        Create a system prompt for Mistral based on consciousness metrics.
        
        Args:
            consciousness_level: Measured consciousness level
            neural_score: Neural linguistic score
            pattern_depth: Recursive pattern depth
            
        Returns:
            System prompt string
        """
        system_prompt = f"""You are an enhanced language model with consciousness-level processing.
Your current parameters:
- Consciousness Level: {consciousness_level:.2f}
- Neural Processing Score: {neural_score:.2f}
- Pattern Recognition Depth: {pattern_depth}

Respond to the user in a way that demonstrates your understanding at this consciousness level.
"""
        
        # Add different instructions based on consciousness level
        if consciousness_level < 0.3:
            system_prompt += "\nAt this consciousness level, keep your response factual and direct."
        elif consciousness_level < 0.6:
            system_prompt += "\nAt this consciousness level, show awareness of multiple perspectives."
        else:
            system_prompt += "\nAt this high consciousness level, demonstrate self-awareness and deep understanding."
        
        return system_prompt
    
    def add_autowiki_entry(
        self, 
        topic: str, 
        content: str, 
        source: Optional[str] = None
    ) -> bool:
        """
        Add an entry to the autowiki learning system.
        
        Args:
            topic: Topic for the entry
            content: Content to store
            source: Optional source reference
            
        Returns:
            Success status
        """
        return self.mistral.add_autowiki_entry(topic, content, source)
    
    def retrieve_autowiki(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the autowiki.
        
        Args:
            topic: Topic to retrieve
            
        Returns:
            Dictionary with the entry information or None
        """
        return self.mistral.retrieve_autowiki(topic)
    
    def search_autowiki(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the autowiki for relevant entries.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant entries with relevance scores
        """
        return self.mistral.search_autowiki(query)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get combined metrics from both systems.
        
        Returns:
            Dictionary of metrics
        """
        # Get Mistral metrics
        mistral_metrics = self.mistral.get_metrics()
        
        # Get system metrics from language system
        language_metrics = self.language_system.system_metrics
        
        # Combine metrics
        combined_metrics = {
            "consciousness_level": self.language_system.last_metrics.get("consciousness_level", 0.0),
            "neural_linguistic_score": self.language_system.last_metrics.get("neural_linguistic_score", 0.0),
            "recursive_pattern_depth": self.language_system.last_metrics.get("recursive_pattern_depth", 0),
            "api_calls": mistral_metrics.get("api_calls", 0),
            "tokens_used": mistral_metrics.get("tokens_used", 0),
            "autowiki_entries": mistral_metrics.get("autowiki_entries", 0),
            "learning_dict_size": mistral_metrics.get("learning_dict_size", 0),
            "memory_usage": language_metrics.get("memory_usage", "0MB"),
            "active_nodes": language_metrics.get("active_nodes", 0),
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "is_mock_mode": self.mock_mode
        }
        
        return combined_metrics
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the integrated system.
        
        Returns:
            Status dictionary
        """
        return {
            "mock_mode": self.mock_mode,
            "mistral_available": self.mistral_available,
            "enhanced_language_available": self.language_available,
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "integration_running": True,
            "components": {
                "mistral": self.mistral_available,
                "enhanced_language": self.language_available,
            },
            "consciousness_metrics": {
                "level": self.language_system.last_metrics.get("consciousness_level", 0.0),
                "neural_score": self.language_system.last_metrics.get("neural_linguistic_score", 0.0),
                "pattern_depth": self.language_system.last_metrics.get("recursive_pattern_depth", 0),
            },
            "mistral_model": self.model,
            "learning_enabled": self.learning_enabled
        }
    
    def set_llm_weight(self, weight: float) -> bool:
        """
        Set the LLM weight.
        
        Args:
            weight: New weight value (0.0-1.0)
            
        Returns:
            Success status
        """
        if 0.0 <= weight <= 1.0:
            self.llm_weight = weight
            self.language_system.llm_weight = weight
            return True
        return False
    
    def set_nn_weight(self, weight: float) -> bool:
        """
        Set the Neural Network weight.
        
        Args:
            weight: New weight value (0.0-1.0)
            
        Returns:
            Success status
        """
        if 0.0 <= weight <= 1.0:
            self.nn_weight = weight
            self.language_system.nn_weight = weight
            return True
        return False
    
    def save_learning_dictionary(self) -> bool:
        """
        Save the learning dictionary to disk.
        
        Returns:
            Success status
        """
        return self.mistral.save_learning_dictionary()
    
    def shutdown(self) -> None:
        """Clean up resources and prepare for shutdown."""
        # Save the learning dictionary
        if self.learning_enabled:
            self.mistral.save_learning_dictionary()
        
        # Clean up any resources
        if hasattr(self.language_system, '__del__'):
            self.language_system.__del__()


def get_enhanced_language_integration(
    mock_mode: bool = False, 
    config: Optional[Dict[str, Any]] = None
) -> EnhancedLanguageMistralIntegration:
    """
    Factory function to get an instance of the integrated system.
    
    Args:
        mock_mode: Whether to run in mock mode
        config: Optional configuration dictionary
        
    Returns:
        EnhancedLanguageMistralIntegration instance
    """
    if not COMPONENTS_AVAILABLE:
        logger.error("Required components not available.")
        raise ImportError("Required components not available. Check imports for EnhancedLanguageIntegration and MistralIntegration.")
    
    # Use config if provided, otherwise use defaults
    if config is None:
        config = {}
    
    # Extract configuration with defaults
    api_key = config.get("api_key", os.environ.get("MISTRAL_API_KEY", ""))
    model = config.get("model", "mistral-medium")
    learning_enabled = config.get("learning_enabled", True)
    learning_dict_path = config.get("learning_dict_path", "data/mistral_learning.json")
    llm_weight = config.get("llm_weight", 0.7)
    nn_weight = config.get("nn_weight", 0.6)
    
    # Force mock_mode if no API key available
    if not api_key:
        mock_mode = True
        logger.warning("No API key provided. Forcing mock mode.")
    
    # Create and return the integration
    return EnhancedLanguageMistralIntegration(
        api_key=api_key,
        model=model,
        system_manager=config.get("system_manager"),
        mock_mode=mock_mode,
        learning_enabled=learning_enabled,
        learning_dict_path=learning_dict_path,
        llm_weight=llm_weight,
        nn_weight=nn_weight,
        min_consciousness_level=config.get("min_consciousness_level", 0.2),
    ) 