#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mistral-Language System Bridge

This module provides integration between the Enhanced Language System
and the Mistral PySide6 application, allowing the Mistral chat to
benefit from the language system's capabilities.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add parent directory to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logger = logging.getLogger("mistral_language_bridge")

# Import language system components
try:
    from src.language.central_language_node import CentralLanguageNode
    from src.language.pyside6_integration import EnhancedLanguageIntegration
    LANGUAGE_SYSTEM_AVAILABLE = True
except ImportError:
    LANGUAGE_SYSTEM_AVAILABLE = False
    logger.warning("Enhanced Language System components not available")

# Import Mistral integration (try different paths)
try:
    from src.v7.ui.mistral_pyside_integration import MistralPySideIntegration
    MISTRAL_INTEGRATION_AVAILABLE = True
except ImportError:
    try:
        from src.v7.mistral_pyside_integration import MistralPySideIntegration
        MISTRAL_INTEGRATION_AVAILABLE = True
    except ImportError:
        MISTRAL_INTEGRATION_AVAILABLE = False
        logger.warning("Mistral integration not available")


class MistralLanguageBridge:
    """
    Bridge between Mistral and Enhanced Language System
    
    This class connects the Mistral chat interface with the Enhanced Language
    System, allowing messages to be processed through the language system
    before being sent to Mistral.
    """
    
    def __init__(
        self,
        mistral_integration: Optional[Any] = None,
        language_integration: Optional[Any] = None,
        data_dir: str = "data",
        llm_weight: float = 0.7,
        nn_weight: float = 0.3,
        config: Dict[str, Any] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the Mistral-Language bridge
        
        Args:
            mistral_integration: Existing MistralPySideIntegration instance
            language_integration: Existing EnhancedLanguageIntegration instance
            data_dir: Directory for data storage
            llm_weight: Initial LLM weight (0.0-1.0)
            nn_weight: Initial NN weight (0.0-1.0)
            config: Additional configuration options
            mock_mode: Whether to use mock mode when creating integrations
        """
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        self.config = config or {}
        self.mock_mode = mock_mode
        
        # Set up integrations
        self.mistral_integration = mistral_integration
        self.language_integration = language_integration
        
        # Initialize integrations if not provided
        if self.language_integration is None:
            self._initialize_language_integration()
        
        if self.mistral_integration is None and MISTRAL_INTEGRATION_AVAILABLE:
            self._initialize_mistral_integration()
            
        logger.info("Mistral-Language Bridge initialized")
        
    def _initialize_language_integration(self):
        """Initialize the language integration"""
        if not LANGUAGE_SYSTEM_AVAILABLE:
            logger.warning("Language system not available. Using mock mode.")
            self.mock_mode = True
        
        try:
            self.language_integration = EnhancedLanguageIntegration(
                data_dir=self.data_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight,
                config=self.config,
                mock_mode=self.mock_mode
            )
            logger.info("Language integration initialized")
        except Exception as e:
            logger.error(f"Error initializing language integration: {e}")
            self.language_integration = None
            
    def _initialize_mistral_integration(self):
        """Initialize the Mistral integration"""
        # This depends on the exact API of the MistralPySideIntegration
        # Modify as needed based on the actual implementation
        try:
            # This assumes you have access to the socket manager and API key
            # Replace with the actual initialization method
            from src.v7.ui.v7_socket_manager import V7SocketManager
            socket_manager = V7SocketManager()
            
            # Get API key from environment or config
            api_key = os.environ.get("MISTRAL_API_KEY", "")
            if not api_key and "mistral_api_key" in self.config:
                api_key = self.config["mistral_api_key"]
                
            # Initialize Mistral integration
            self.mistral_integration = MistralPySideIntegration(socket_manager, api_key)
            logger.info("Mistral integration initialized")
        except Exception as e:
            logger.error(f"Error initializing Mistral integration: {e}")
            self.mistral_integration = None
            
    def process_message(self, message: str, enhance_with_language: bool = True) -> Dict[str, Any]:
        """
        Process a message through the language system and/or Mistral
        
        Args:
            message: Message to process
            enhance_with_language: Whether to enhance with language processing
            
        Returns:
            Dict with processed message and results
        """
        if not message:
            return {"error": "Empty message"}
            
        # Initialize result dict
        result = {
            "original_message": message,
            "enhanced_message": message,
            "language_processing": None,
            "mistral_response": None
        }
        
        # Enhance with language system if requested and available
        if enhance_with_language and self.language_integration:
            try:
                # Process through language system synchronously
                language_result = self.language_integration.process_text(
                    message, component="central", async_mode=False
                )
                
                # Store language processing results
                result["language_processing"] = language_result
                
                # Enhancement logic based on language processing
                # This is where you can modify the message based on language analysis
                if language_result:
                    # Get consciousness level if available
                    if "consciousness_level" in language_result:
                        result["consciousness_level"] = language_result["consciousness_level"]
                        
                    # Add memory associations if available
                    if "memory_associations" in language_result and language_result["memory_associations"]:
                        associations = ", ".join(language_result["memory_associations"])
                        result["enhanced_message"] = f"{message}\n\nAssociations: {associations}"
                        
                    # Add recursive pattern depth if available
                    if "recursive_pattern_depth" in language_result and language_result["recursive_pattern_depth"] > 0:
                        result["recursive_depth"] = language_result["recursive_pattern_depth"]
            except Exception as e:
                logger.error(f"Error in language processing: {e}")
                result["language_error"] = str(e)
        
        # Process through Mistral if available
        if self.mistral_integration:
            try:
                # Use the enhanced message if available
                message_to_process = result.get("enhanced_message", message)
                
                # Create additional parameters based on language processing
                params = {}
                if "consciousness_level" in result:
                    # Example: Adjust temperature based on consciousness level
                    consciousness = result["consciousness_level"]
                    if consciousness > 0.8:
                        params["temperature"] = 0.9  # More creative
                    elif consciousness < 0.3:
                        params["temperature"] = 0.2  # More focused
                
                # Call Mistral integration
                # This needs to be adapted to the actual API of MistralPySideIntegration
                mistral_response = self.mistral_integration.process_message_sync(
                    message_to_process, **params
                )
                
                # Store Mistral response
                result["mistral_response"] = mistral_response
            except Exception as e:
                logger.error(f"Error in Mistral processing: {e}")
                result["mistral_error"] = str(e)
        
        return result
    
    def process_message_async(self, message: str, enhance_with_language: bool = True) -> str:
        """
        Process a message asynchronously through the language system and Mistral
        
        Args:
            message: Message to process
            enhance_with_language: Whether to enhance with language processing
            
        Returns:
            Request ID for the asynchronous processing
        """
        if not message:
            return None
            
        # Enhance with language system if requested and available
        enhanced_message = message
        if enhance_with_language and self.language_integration:
            try:
                # Process through language system synchronously
                language_result = self.language_integration.process_text(
                    message, component="central", async_mode=False
                )
                
                # Enhancement logic based on language processing
                if language_result:
                    # Similar enhancement logic as in process_message
                    if "memory_associations" in language_result and language_result["memory_associations"]:
                        associations = ", ".join(language_result["memory_associations"])
                        enhanced_message = f"{message}\n\nAssociations: {associations}"
            except Exception as e:
                logger.error(f"Error in language processing: {e}")
        
        # Process through Mistral asynchronously if available
        if self.mistral_integration:
            try:
                # Use the enhanced message if it was modified
                request_id = self.mistral_integration.process_message_async(enhanced_message)
                return request_id
            except Exception as e:
                logger.error(f"Error in Mistral async processing: {e}")
                return None
        else:
            return None
    
    def set_llm_weight(self, weight: float) -> bool:
        """Set the LLM weight"""
        if not 0.0 <= weight <= 1.0:
            logger.warning(f"Invalid LLM weight {weight}. Must be between 0.0 and 1.0.")
            return False
            
        # Set in language integration
        if self.language_integration:
            self.language_integration.set_llm_weight(weight)
            
        # Update internal value
        self.llm_weight = weight
        
        # Set in Mistral integration if it has this method
        if self.mistral_integration and hasattr(self.mistral_integration, "set_llm_weight"):
            self.mistral_integration.set_llm_weight(weight)
            
        return True
    
    def set_nn_weight(self, weight: float) -> bool:
        """Set the neural network weight"""
        if not 0.0 <= weight <= 1.0:
            logger.warning(f"Invalid NN weight {weight}. Must be between 0.0 and 1.0.")
            return False
            
        # Set in language integration
        if self.language_integration:
            self.language_integration.set_nn_weight(weight)
            
        # Update internal value
        self.nn_weight = weight
        
        # Set in Mistral integration if it has this method
        if self.mistral_integration and hasattr(self.mistral_integration, "set_nn_weight"):
            self.mistral_integration.set_nn_weight(weight)
            
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the bridge and its components"""
        status = {
            "language_available": self.language_integration is not None,
            "mistral_available": self.mistral_integration is not None,
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "mock_mode": self.mock_mode
        }
        
        # Add language integration status if available
        if self.language_integration:
            try:
                if hasattr(self.language_integration, "get_status"):
                    status["language_status"] = self.language_integration.get_status()
                else:
                    status["language_status"] = "available"
            except:
                status["language_status"] = "error"
        
        # Add Mistral integration status if available
        if self.mistral_integration:
            try:
                if hasattr(self.mistral_integration, "get_status"):
                    status["mistral_status"] = self.mistral_integration.get_status()
                else:
                    status["mistral_status"] = "available"
            except:
                status["mistral_status"] = "error"
        
        return status


def get_mistral_language_bridge(
    mistral_integration=None,
    language_integration=None,
    data_dir="data",
    llm_weight=0.7,
    nn_weight=0.3,
    config=None,
    mock_mode=False
) -> MistralLanguageBridge:
    """
    Get a Mistral-Language bridge instance
    
    This is a convenience function for getting a pre-configured
    instance of the MistralLanguageBridge class.
    
    Args:
        mistral_integration: Existing MistralPySideIntegration instance
        language_integration: Existing EnhancedLanguageIntegration instance
        data_dir: Directory for data storage
        llm_weight: Initial LLM weight (0.0-1.0)
        nn_weight: Initial NN weight (0.0-1.0)
        config: Additional configuration options
        mock_mode: Whether to use mock mode when creating integrations
        
    Returns:
        MistralLanguageBridge instance
    """
    return MistralLanguageBridge(
        mistral_integration=mistral_integration,
        language_integration=language_integration,
        data_dir=data_dir,
        llm_weight=llm_weight,
        nn_weight=nn_weight,
        config=config,
        mock_mode=mock_mode
    ) 