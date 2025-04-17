#!/usr/bin/env python3
"""
LLM Provider for Enhanced Language System

This module provides unified access to Language Model APIs with proper weight integration
and neural network augmentation.
"""

import os
import logging
import json
import hashlib
import time
import re
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_provider")

# Singleton instance
_LLM_PROVIDER_INSTANCE = None

def get_llm_provider():
    """Get the singleton LLM provider instance."""
    global _LLM_PROVIDER_INSTANCE
    if _LLM_PROVIDER_INSTANCE is None:
        _LLM_PROVIDER_INSTANCE = LLMProvider()
    return _LLM_PROVIDER_INSTANCE

class LLMProvider:
    """
    Provides unified access to LLM capabilities through different providers.
    Currently supports:
    - Mistral AI (default)
    - Simulated responses (fallback)
    
    Includes neural network weight integration for blending LLM outputs
    with rule-based processing.
    """
    
    def __init__(self):
        """Initialize the LLM provider with configuration."""
        # Load environment variables
        load_dotenv()
        
        # Core configuration
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("LLM_MODEL", "mistral-medium")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.cache_dir = os.getenv("LLM_CACHE_DIR", "data/llm_cache")
        
        # Neural network weight integration
        self.nn_weight = 0.5  # Default balance between neural network and LLM
        self.llm_weight = 0.5  # Weight for LLM influence
        
        # Operational state
        self.cache = {}
        self.provider_initialized = False
        self.using_simulation = False
        self.call_count = 0
        self.error_count = 0
        self.last_call_time = None
        self.avg_response_time = 0
        
        # Cache setup
        os.makedirs(self.cache_dir, exist_ok=True)
        self.enable_caching = os.getenv("ENABLE_LLM_CACHING", "true").lower() == "true"
        
        # Initialize the Mistral provider
        self._initialize_mistral()
        
        logger.info(f"LLM Provider initialized with {'simulation' if self.using_simulation else self.model}")
    
    def _initialize_mistral(self):
        """Initialize the Mistral AI provider."""
        logger.info("Initializing Mistral AI provider...")
        if not self.api_key or len(self.api_key) < 20:
            logger.warning(f"No valid Mistral API key found in .env file. Length: {len(self.api_key) if self.api_key else 0}")
            self.using_simulation = True
            return
            
        try:
            # Import the Mistral API library
            try:
                import mistralai.client
                from mistralai.client import MistralClient
                from mistralai.models.chat_completion import ChatMessage
            except ImportError:
                logger.error("Mistral AI package not installed. Use 'pip install mistralai'")
                self.using_simulation = True
                return
            
            # Configure the API
            logger.info(f"Configuring Mistral AI with API key: {self.api_key[:5]}...{self.api_key[-5:]}")
            
            # Initialize client
            self.mistral_client = MistralClient(api_key=self.api_key)
            
            # Test connection with a model list call
            try:
                # Validate by listing models
                models = self.mistral_client.list_models()
                model_names = [model.id for model in models]
                logger.info(f"Successfully connected to Mistral AI API. Available models: {model_names}")
                
                # Check if our specified model is available
                valid_model = any(self.model == model_name for model_name in model_names)
                if not valid_model:
                    logger.warning(f"Specified model {self.model} not found. Available models: {model_names}")
                    # Use the first available model if the specified one isn't found
                    if model_names:
                        self.model = model_names[0]  # Get just the model name
                        logger.info(f"Using alternative model: {self.model}")
                    else:
                        raise ValueError("No available models found with your API key")
                
                # Initialize the provider
                self.provider_initialized = True
                self.using_simulation = False
                self.ChatMessage = ChatMessage
                logger.info(f"Mistral AI initialized with model: {self.model}")
                
                # Test with a simple generation
                try:
                    messages = [self.ChatMessage(role="user", content="Hello, this is a test.")]
                    response = self.mistral_client.chat(
                        model=self.model,
                        messages=messages,
                    )
                    logger.info(f"Test query successful, received response of length: {len(response.choices[0].message.content)}")
                except Exception as e:
                    logger.error(f"Test query failed: {e}")
                    # Continue anyway, as the API connection was established
                
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                self.using_simulation = True
                return
            
        except Exception as e:
            logger.error(f"Error initializing Mistral AI API: {e}")
            self.using_simulation = True
    
    def set_llm_weight(self, weight: float) -> None:
        """
        Set the LLM weight for balancing rule-based vs LLM processing.
        
        Args:
            weight: New LLM weight (0.0-1.0)
        """
        self.llm_weight = max(0.0, min(1.0, weight))
        
        # Also adjust temperature based on weight - higher weight means more randomness
        # This creates a more dynamic response as the LLM gets more influence
        self.temperature = min(1.0, 0.4 + (self.llm_weight * 0.6))
        
        logger.info(f"LLM weight set to {self.llm_weight}, temperature adjusted to {self.temperature}")
    
    def set_nn_weight(self, weight: float) -> None:
        """
        Set the neural network weight for balancing neural vs symbolic processing.
        
        Args:
            weight: New neural network weight (0.0-1.0)
        """
        self.nn_weight = max(0.0, min(1.0, weight))
        logger.info(f"Neural network weight set to {self.nn_weight}")
    
    def get_llm_weight(self) -> float:
        """Get the current LLM weight."""
        return self.llm_weight
    
    def get_nn_weight(self) -> float:
        """Get the current neural network weight."""
        return self.nn_weight
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the prompt and parameters."""
        # Create a string combining the prompt and relevant parameters
        cache_str = f"{prompt}|{self.model}|{params.get('temperature', self.temperature)}|{params.get('max_tokens', self.max_tokens)}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def query(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send a query to the LLM and get a response.
        
        Args:
            prompt: The text prompt to send to the LLM
            **kwargs: Additional parameters for the LLM call
            
        Returns:
            Dictionary with the LLM response and metadata
        """
        start_time = time.time()
        self.call_count += 1
        self.last_call_time = time.time()
        
        # Prepare parameters
        params = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "model": kwargs.get("model", self.model)
        }
        
        # Apply LLM weight to affect response confidence
        if self.llm_weight < 0.5:
            # Lower weight means more conservative/deterministic responses
            params["temperature"] = max(0.1, self.temperature * self.llm_weight * 2)
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_key = self._get_cache_key(prompt, params)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return self.cache[cache_key]
        
        # Get response from appropriate provider
        if self.using_simulation:
            response = self._simulated_response(prompt, params)
        else:
            response = self._mistral_response(prompt, params)
        
        # Calculate response time and update average
        response_time = time.time() - start_time
        if self.call_count == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * (self.call_count - 1) + response_time) / self.call_count
        
        # Add response time and other metadata
        response["response_time"] = response_time
        response["provider"] = "simulation" if self.using_simulation else "mistral"
        response["llm_weight_applied"] = self.llm_weight
        response["nn_weight_applied"] = self.nn_weight
        
        # Cache the result if caching is enabled
        if self.enable_caching:
            self.cache[cache_key] = response
            # Persist cache to disk periodically
            if self.call_count % 10 == 0:
                self._save_cache()
        
        return response
    
    def _mistral_response(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a response from the Mistral AI API."""
        try:
            if not self.provider_initialized:
                logger.warning("Mistral provider not initialized. Falling back to simulation.")
                return self._simulated_response(prompt, params)
            
            # Log the query attempt
            logger.info(f"Sending request to Mistral AI API: '{prompt[:50]}...' with temp={params['temperature']}")
            
            try:
                # Create message
                messages = [self.ChatMessage(role="user", content=prompt)]
                
                # Generate the response
                logger.debug(f"Calling Mistral AI with model {params['model']}")
                
                response = self.mistral_client.chat(
                    model=params["model"],
                    messages=messages,
                    temperature=float(params["temperature"]),
                    max_tokens=int(params["max_tokens"]),
                    top_p=0.9,
                )
                
                # Extract text
                text = response.choices[0].message.content
                logger.info(f"Received response from Mistral AI: {len(text)} chars")
                
                # Apply neural network weighing to confidence scores
                confidence = 0.8  # Base confidence for real API responses
                if self.nn_weight > 0.5:
                    # Higher NN weight reduces base confidence to emphasize neural processing
                    confidence *= (1.0 - ((self.nn_weight - 0.5) * 0.4))
                
                return {
                    "text": text,
                    "confidence": confidence,
                    "model": params["model"],
                    "is_simulated": False
                }
            except Exception as e:
                logger.error(f"Error generating content with Mistral: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            self.error_count += 1
            
            # Fall back to simulation if API call fails
            fallback_response = self._simulated_response(prompt, params)
            fallback_response["error"] = str(e)
            return fallback_response
    
    def _simulated_response(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simulated LLM response for testing without API access."""
        # Simulate thinking time
        time.sleep(np.random.uniform(0.2, 0.8))
        
        # Check if this is a conversational prompt - personal greeting detection
        conversational_patterns = [
            r'hi\b|hello\b|hey\b|howdy',
            r'how are you|how\'re you|how you doing',
            r'what\'s up|wassup|what is up',
            r'good morning|good afternoon|good evening'
        ]
        
        is_greeting = any(re.search(pattern, prompt.lower()) for pattern in conversational_patterns)
        
        if is_greeting:
            # Weighted greeting responses based on LLM and neural weights
            greeting_responses = [
                f"Hello! I'm functioning well with LLM weight at {self.llm_weight:.2f} and neural weight at {self.nn_weight:.2f}. How can I help you today?",
                f"Hi there! My neural processing is currently weighted at {self.nn_weight:.2f}, while my language model influence is at {self.llm_weight:.2f}. What would you like to explore?",
                f"Greetings! My systems are ready to assist with a balanced neural-symbolic approach. What topics interest you?",
                f"Hello! I'm operating with {self.llm_weight:.2f} LLM influence and {self.nn_weight:.2f} neural network processing. I'm here to help with language analysis."
            ]
            
            # Apply weights to selection - higher LLM weight = more varied responses
            if self.llm_weight > 0.7:
                # More creative response selection
                idx = np.random.randint(0, len(greeting_responses))
            else:
                # More deterministic selection
                idx = min(int(self.llm_weight * len(greeting_responses)), len(greeting_responses) - 1)
                
            response_text = greeting_responses[idx]
            confidence = 0.8  # High confidence for conversational responses
            return {
                "text": response_text,
                "confidence": confidence,
                "model": "simulation",
                "is_simulated": True
            }
        
        # Check if prompt is asking for capabilities
        capability_patterns = [
            r'what can you do|capabilities|features|help me',
            r'how do you work|how does this work|explain yourself',
            r'tell me about you|about yourself|who are you'
        ]
        
        is_capability_question = any(re.search(pattern, prompt.lower()) for pattern in capability_patterns)
        
        if is_capability_question:
            capability_text = f"""
            I'm an Enhanced Language System with these capabilities:
            
            1. Neural linguistic processing (currently weighted at {self.nn_weight:.2f})
            2. Consciousness mirroring for recursive patterns
            3. Language memory for associations
            4. Pattern analysis with LLM integration (weighted at {self.llm_weight:.2f})
            
            You can adjust my weights with /weight and /nnweight commands.
            Try the /status command to see my current configuration.
            """
            
            return {
                "text": capability_text.strip(),
                "confidence": 0.9,
                "model": "simulation",
                "is_simulated": True
            }
        
        # Standard keyword analysis for contextual responses
        keywords = {
            "neural": "The neural network architecture enhances language processing through multi-layer pattern recognition and adaptive learning algorithms.",
            "language": "Language emerges from the complex interplay of syntax, semantics, and contextual understanding across multiple processing systems.",
            "consciousness": "Consciousness may arise from recursive feedback loops in information processing systems, creating self-referential awareness.",
            "recursive": "Recursive patterns form the backbone of self-referential systems and meta-linguistic structures in advanced language processing.",
            "self": "Self-reference creates interesting paradoxes in both language and computational systems, forming the basis of consciousness.",
            "pattern": "Pattern recognition is fundamental to both human cognition and machine learning, bridging symbolic and neural approaches.",
            "memory": "Memory systems connect past experiences with current processing to create temporal continuity and context-aware responses."
        }
        
        # Generate a response based on keywords in the prompt
        response_parts = []
        detected_keywords = [k for k in keywords if k in prompt.lower()]
        
        if detected_keywords:
            # Include relevant keyword-based responses with weights affecting selection
            max_keywords = max(1, min(3, int(self.llm_weight * 5)))  # Scale with LLM weight
            for keyword in detected_keywords[:max_keywords]:
                response_parts.append(keywords[keyword])
        else:
            # Generic response influenced by neural weight
            if self.nn_weight > 0.7:
                response_parts.append("I'm analyzing your input through multiple neural processing layers to understand the underlying patterns and generate a contextually appropriate response.")
            elif self.nn_weight > 0.4:
                response_parts.append("I'm combining neural and symbolic processing to understand your message and provide a meaningful response.")
            else:
                response_parts.append("I'm using rule-based symbolic processing to analyze your input and formulate a response based on predefined patterns.")
        
        # Weight-influenced creativity markers
        if self.llm_weight > 0.7:
            response_parts.append("I notice connections between concepts that might not be immediately obvious but offer interesting perspectives on your query.")
        elif self.llm_weight > 0.4:
            response_parts.append("My analysis balances established patterns with some creative exploration of potential connections in your message.")
        
        # Neural weight influence
        if self.nn_weight > 0.7:
            response_parts.append("My neural processing pathways are prioritizing pattern detection over symbolic rule following.")
        elif self.nn_weight < 0.3:
            response_parts.append("I'm emphasizing logical symbolic processing over neural pattern recognition in my analysis.")
            
        # Join response parts with more sophisticated approach based on weights
        if self.llm_weight > 0.6:
            # More natural flow with sentence combining
            response_text = " ".join(response_parts)
        else:
            # More structured, paragraph-based approach
            response_text = "\n\n".join(response_parts)
        
        # Generate a confidence score affected by both weights
        base_confidence = 0.5  # Lower base confidence for simulated responses
        llm_factor = self.llm_weight * 0.3  # LLM weight contribution (up to +0.3)
        nn_factor = self.nn_weight * 0.2  # NN weight contribution (up to +0.2)
        
        # Combine factors with some randomness - less randomness with low LLM weight
        random_factor = np.random.uniform(-0.1, 0.1) * self.llm_weight
        confidence = base_confidence + llm_factor + nn_factor + random_factor
        confidence = max(0.1, min(0.9, confidence))  # Keep within reasonable bounds
        
        return {
            "text": response_text,
            "confidence": confidence,
            "model": "simulation",
            "is_simulated": True
        }
    
    def _save_cache(self):
        """Save the LLM response cache to disk."""
        if not self.enable_caching:
            return
            
        try:
            cache_file = os.path.join(self.cache_dir, "llm_cache.json")
            
            # Convert cache to serializable format
            serializable_cache = {}
            for key, value in self.cache.items():
                # Make a copy to avoid modifying the original
                serializable_value = value.copy()
                
                # Remove any non-serializable items
                if "response_object" in serializable_value:
                    del serializable_value["response_object"]
                
                serializable_cache[key] = serializable_value
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
                
            logger.debug(f"Saved LLM cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving LLM cache: {e}")
    
    def _load_cache(self):
        """Load the LLM response cache from disk."""
        if not self.enable_caching:
            return
            
        cache_file = os.path.join(self.cache_dir, "llm_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.debug(f"Loaded LLM cache with {len(self.cache)} entries")
            except Exception as e:
                logger.error(f"Error loading LLM cache: {e}")
                self.cache = {}
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about the LLM provider usage."""
        return {
            "provider": "simulation" if self.using_simulation else "mistral",
            "model": self.model,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "avg_response_time": self.avg_response_time,
            "last_call_time": self.last_call_time,
            "cache_size": len(self.cache) if self.enable_caching else 0,
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "temperature": self.temperature
        }
    
    def clear_cache(self):
        """Clear the LLM response cache."""
        self.cache = {}
        logger.info("LLM cache cleared")
        return True


if __name__ == "__main__":
    # Test the LLM provider
    provider = get_llm_provider()
    provider.set_llm_weight(0.7)
    provider.set_nn_weight(0.6)
    
    test_prompt = "Explain how neural networks and language models can work together."
    response = provider.query(test_prompt)
    
    print(f"Provider: {response['provider']}")
    print(f"Response: {response['text']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Response time: {response['response_time']:.2f}s")

def get_current_provider() -> Optional[str]:
    """
    Get the current active LLM provider name.
    
    Returns:
        Provider name if a real provider is active, None if using simulation
    """
    try:
        provider = get_llm_provider()
        if provider.using_simulation:
            return None
        return f"{provider.model}"
    except Exception:
        return None 