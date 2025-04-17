#!/usr/bin/env python3
"""
Mistral Integration Module (Fixed)

This module provides integration with the Mistral AI API
for advanced language processing capabilities.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MistralIntegration")

# Import Mistral client
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
    logger.info("Mistral AI client imported successfully")
except ImportError:
    logger.warning("Mistral AI client not found. Install with: pip install mistralai")
    MISTRAL_AVAILABLE = False

# Import language processor
try:
    from src.language.neural_linguistic_processor import NeuralLinguisticProcessor
    PROCESSOR_AVAILABLE = True
    logger.info("Neural Linguistic Processor imported successfully")
except ImportError as e:
    logger.warning(f"Neural Linguistic Processor not available: {e}")
    PROCESSOR_AVAILABLE = False


class MistralIntegration:
    """
    Mistral AI Integration for neural network language processing
    
    This class integrates Mistral's API with the neural linguistic processor
    to provide enhanced language understanding capabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        llm_weight: float = 0.7,
        nn_weight: float = 0.3,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the Mistral Integration
        
        Args:
            api_key: Mistral API key (uses MISTRAL_API_KEY env var if not provided)
            model: Mistral model to use
            llm_weight: Weight to give to LLM responses (0-1)
            nn_weight: Weight to give to neural network processing (0-1)
            temperature: Controls randomness of output (0-1)
            top_p: Nucleus sampling parameter (0-1)
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")
            
        self.model = model
        self.llm_weight = max(0.0, min(1.0, llm_weight))
        self.nn_weight = max(0.0, min(1.0, nn_weight))
        self.temperature = max(0.0, min(1.0, temperature))
        self.top_p = max(0.0, min(1.0, top_p))
        
        # Connection state tracking
        self._connection_tested = False
        self._last_connection_test = 0
        self._connection_retest_interval = 60  # seconds
        
        # Check weights sum to 1.0
        total_weight = self.llm_weight + self.nn_weight
        if total_weight != 1.0:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.llm_weight /= total_weight
            self.nn_weight /= total_weight
        
        # Initialize Mistral client
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral AI client not found. Install with: pip install mistralai")
            
        try:
            self.client = MistralClient(api_key=self.api_key)
            logger.info(f"Mistral AI client initialized with model {self.model}")
            # Test connection
            self._test_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            raise RuntimeError(f"Failed to initialize Mistral client: {e}")
        
        # Initialize neural linguistic processor if available
        self.processor = None
        if PROCESSOR_AVAILABLE:
            try:
                # Create the processor without data_dir parameter
                self.processor = NeuralLinguisticProcessor()
                logger.info("Neural Linguistic Processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Neural Linguistic Processor: {e}")
                # Don't raise here as the neural processor is optional
    
    def _test_connection(self) -> bool:
        """
        Test the connection to Mistral API
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.client:
            return False
            
        # If we've tested recently, use cached result
        now = time.time()
        if self._connection_tested and (now - self._last_connection_test) < self._connection_retest_interval:
            return self.client is not None
            
        # Try a simple API call
        try:
            # Use a very small request to test connection
            # Note: Only using supported parameters
            result = self.client.chat(
                model=self.model,
                messages=[ChatMessage(role="user", content="Test connection")],
                temperature=0.1,
                max_tokens=5
                # top_k parameter removed as it's not supported
            )
            self._connection_tested = True
            self._last_connection_test = now
            logger.info(f"Mistral API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Mistral API connection test failed: {e}")
            self._connection_tested = False
            self._last_connection_test = now
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if Mistral integration is available and initialized"""
        # Re-test connection periodically
        now = time.time()
        if (now - self._last_connection_test) > self._connection_retest_interval:
            return self._test_connection()
        return self.client is not None and self._connection_tested
    
    @property
    def processor_available(self) -> bool:
        """Check if neural linguistic processor is available"""
        return self.processor is not None
    
    def process_message(self, messages: List[Dict[str, str]]) -> str:
        """
        Process a message through both Mistral LLM and neural processing
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Processed response from both systems, combined according to weights
        """
        if not self._is_ready():
            logger.warning("Mistral integration not ready, attempting to initialize")
            
            if not self._test_connection():
                raise RuntimeError("Unable to connect to Mistral API")
                
        logger.info(f"Processing message with {len(messages)} entries")
        
        # Get response from Mistral
        llm_response = ""
        try:
            chat_response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            llm_response = chat_response.choices[0].message.content
            logger.debug(f"Mistral response: {llm_response[:100]}...")
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            
        # Get response from neural processor
        nn_response = ""
        if self.processor is not None and self.nn_weight > 0:
            try:
                # Process the last message's content through the neural processor
                nn_response = self.processor.process_text(messages[-1]['content'])
                logger.debug(f"Neural processor response: {nn_response[:100]}...")
            except Exception as e:
                logger.error(f"Error with neural processor: {e}")
                
        # Combine responses according to weights
        if llm_response and nn_response:
            # Use weighted average approach
            combined = self._combine_responses(llm_response, nn_response)
            return combined
        elif llm_response:
            return llm_response
        elif nn_response:
            return nn_response
        else:
            return "Sorry, I was not able to process your request at this time."
    
    def _combine_responses(self, llm_response: str, nn_response: Any) -> str:
        """
        Combine LLM and neural network responses based on weights
        
        Args:
            llm_response: Response from Mistral LLM
            nn_response: Response from neural processor
            
        Returns:
            Combined response as string
        """
        # Extract text from neural response if it's a dict
        nn_text = nn_response
        if isinstance(nn_text, dict) and "text" in nn_text:
            nn_text = nn_text["text"]
        elif not isinstance(nn_text, str):
            nn_text = str(nn_text)
        
        # If weights heavily favor one side, use that response
        if self.llm_weight > 0.9:
            return llm_response
        if self.nn_weight > 0.9:
            return nn_text
        
        # Split responses into sentences
        llm_sentences = re.split(r'(?<=[.!?])\s+', llm_response.strip())
        nn_sentences = re.split(r'(?<=[.!?])\s+', nn_text.strip())
        
        # Combine sentences with weighted vocabulary
        combined_sentences = []
        max_sentences = max(len(llm_sentences), len(nn_sentences))
        
        for i in range(max_sentences):
            if i < len(llm_sentences) and i < len(nn_sentences):
                # Both have sentences at this position - blend them
                llm_sentence = llm_sentences[i]
                nn_sentence = nn_sentences[i]
                
                # Split into words
                llm_words = llm_sentence.split()
                nn_words = nn_sentence.split()
                
                # Create blended sentence
                blended_words = []
                max_words = max(len(llm_words), len(nn_words))
                
                for j in range(max_words):
                    if j < len(llm_words) and j < len(nn_words):
                        # Both have words at this position - choose based on weights
                        if self.llm_weight > self.nn_weight:
                            blended_words.append(llm_words[j])
                        else:
                            blended_words.append(nn_words[j])
                    elif j < len(llm_words):
                        # Only LLM has words remaining
                        blended_words.append(llm_words[j])
                    else:
                        # Only neural has words remaining
                        blended_words.append(nn_words[j])
                
                # Join words back into sentence
                blended_sentence = ' '.join(blended_words)
                combined_sentences.append(blended_sentence)
                
            elif i < len(llm_sentences):
                # Only LLM has a sentence
                combined_sentences.append(llm_sentences[i])
            else:
                # Only neural has a sentence
                combined_sentences.append(nn_sentences[i])
        
        # Join sentences back together
        combined = ' '.join(combined_sentences)
        
        # Add attribution if weights are balanced
        if 0.4 <= self.llm_weight <= 0.6:
            return f"{combined}\n\n(Response combines insights from both LLM and neural analysis)"
        
        return combined
    
    def _is_ready(self) -> bool:
        """
        Check if the integration is ready to process messages
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.client is not None and self._connection_tested
    
    def adjust_weights(self, llm_weight: Optional[float] = None, nn_weight: Optional[float] = None,
                       temperature: Optional[float] = None, top_p: Optional[float] = None) -> Dict[str, float]:
        """
        Adjust the weights and parameters of LLM and neural network components
        
        Args:
            llm_weight: New weight for LLM (0-1)
            nn_weight: New weight for neural networks (0-1)
            temperature: Controls randomness of output (0-1)
            top_p: Nucleus sampling parameter (0-1)
            
        Returns:
            Dict with updated weights and parameters
        """
        # Update weights if provided
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, llm_weight))
        
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, nn_weight))
        
        # Update LLM parameters if provided
        if temperature is not None:
            self.temperature = max(0.0, min(1.0, temperature))
            
        if top_p is not None:
            self.top_p = max(0.0, min(1.0, top_p))
        
        # Normalize weights to sum to 1.0
        total_weight = self.llm_weight + self.nn_weight
        if total_weight != 1.0:
            self.llm_weight /= total_weight
            self.nn_weight /= total_weight
        
        return {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics and information
        
        Returns:
            Dict with system information
        """
        return {
            "mistral_available": self.is_available,
            "processor_available": self.processor_available,
            "model": self.model if self.is_available else None,
            "weights": {
                "llm": self.llm_weight,
                "neural_network": self.nn_weight
            },
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p
            },
            "connection_status": {
                "connected": self.is_available,
                "last_tested": self._last_connection_test,
                "api_key_provided": bool(self.api_key)
            }
        }

def test_mistral_integration():
    """
    Test function for the MistralIntegration class
    """
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        return
        
    try:
        integration = MistralIntegration(
            api_key=api_key,
            model="mistral-tiny",
            llm_weight=0.7,
            nn_weight=0.3,
            temperature=0.7,
            top_p=0.9
        )
        
        # Test connection
        is_connected = integration.is_available
        print(f"Connection test: {'success' if is_connected else 'failed'}")
        
        # Test processing a message
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about neural networks."}
        ]
        
        try:
            response = integration.process_message(test_messages)
            print("\nResponse:")
            print(response[:200] + "..." if len(response) > 200 else response)
        except Exception as e:
            print(f"Error during test: {e}")
            
        # Test weight adjustment
        print("\nAdjusting weights:")
        new_weights = integration.adjust_weights(llm_weight=0.8, nn_weight=0.2)
        print(f"New weights: {new_weights}")
        
        # Get stats
        print("\nSystem stats:")
        stats = integration.get_system_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error initializing integration: {e}")

if __name__ == "__main__":
    test_mistral_integration() 