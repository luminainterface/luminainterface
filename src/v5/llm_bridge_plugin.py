"""
LLM Bridge Plugin for V5 Visualization System

This module provides integration between the V5 visualization system and LLM capabilities.
It connects the FrontendSocketManager with Language Memory API and provides LLM
response processing.
"""

import os
import sys
import json
import uuid
import time
import logging
import threading
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required components
from .node_socket import NodeSocket

# Try to import OpenAI module for LLM support
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    logger.warning("OpenAI SDK not available. Using simulated LLM responses instead.")
    HAS_OPENAI = False

# Try to import Anthropic module for Claude support
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    logger.warning("Anthropic SDK not available. Claude models will be simulated.")
    HAS_ANTHROPIC = False

# Try to import Memory API
try:
    from src.memory_api import MemoryAPI
    HAS_MEMORY_API = True
except ImportError:
    logger.warning("Memory API not available. Running without memory enhancement.")
    HAS_MEMORY_API = False


class LLMBridgePlugin:
    """
    Plugin that provides LLM integration for the V5 visualization system.
    
    This plugin connects to the FrontendSocketManager and provides LLM capabilities
    to the V5 visualization interface, with optional memory enhancement.
    """
    
    def __init__(self):
        """Initialize the LLM Bridge Plugin"""
        self.plugin_id = "llm_bridge"
        self.plugin_name = "LLM Bridge"
        self.socket = NodeSocket(self.plugin_id, "v5_plugin")
        
        # Initialize memory API if available
        self.memory_api = None
        if HAS_MEMORY_API:
            try:
                self.memory_api = MemoryAPI()
                logger.info("Memory API initialized for LLM enhancement")
            except Exception as e:
                logger.error(f"Failed to initialize Memory API: {str(e)}")
        
        # Initialize LLM configuration
        self.llm_config = {
            "provider": os.environ.get("LLM_PROVIDER", "openai"),
            "model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
            "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1000")),
            "api_key": os.environ.get("LLM_API_KEY", ""),
            "memory_mode": os.environ.get("LLM_MEMORY_MODE", "combined")
        }
        
        # Register message handlers
        self.socket.register_message_handler("llm_request", self._handle_llm_request)
        self.socket.register_message_handler("update_config", self._handle_update_config)
        self.socket.register_message_handler("get_config", self._handle_get_config)
        self.socket.register_message_handler("get_memory_stats", self._handle_get_memory_stats)
        
        logger.info(f"LLM Bridge Plugin initialized with provider: {self.llm_config['provider']}")
        
    def get_socket_descriptor(self):
        """
        Get the socket descriptor for this plugin.
        
        Returns:
            Dictionary containing plugin socket information
        """
        return {
            "plugin_id": self.plugin_id,
            "plugin_name": self.plugin_name,
            "plugin_type": "v5_plugin",
            "subscription_mode": "request-response",
            "ui_components": ["llm_panel", "chat_panel", "memory_panel"],
            "capabilities": {
                "llm_integration": True,
                "memory_enhanced": HAS_MEMORY_API,
                "providers": {
                    "openai": HAS_OPENAI,
                    "anthropic": HAS_ANTHROPIC
                },
                "memory_modes": ["none", "contextual", "synthesized", "combined"]
            }
        }
        
    def _handle_llm_request(self, message):
        """
        Handle an LLM request from the UI.
        
        Args:
            message: Message containing the LLM request
            
        Returns:
            LLM response data
        """
        logger.info(f"Received LLM request: {message.get('content', {}).get('text', '')[:30]}...")
        
        # Extract request data
        content = message.get("content", {})
        prompt = content.get("text", "")
        system_message = content.get("system_message", "You are a helpful assistant.")
        conversation_history = content.get("conversation_history", [])
        session_id = content.get("session_id", str(uuid.uuid4()))
        
        # Start processing in a separate thread to avoid blocking
        threading.Thread(
            target=self._process_llm_request,
            args=(message, prompt, system_message, conversation_history, session_id)
        ).start()
        
        # Return immediate acknowledgment
        return {
            "status": "processing",
            "request_id": message.get("request_id"),
            "timestamp": datetime.now().isoformat()
        }
        
    def _process_llm_request(self, original_message, prompt, system_message, conversation_history, session_id):
        """
        Process an LLM request in a separate thread.
        
        Args:
            original_message: The original request message
            prompt: The prompt text
            system_message: System message for the LLM
            conversation_history: Previous conversation turns
            session_id: Session identifier
        """
        request_id = original_message.get("request_id")
        
        try:
            # Track processing time
            start_time = time.time()
            
            # Store user message in memory if available
            if self.memory_api:
                store_result = self.memory_api.store_conversation(
                    message=prompt,
                    metadata={
                        "session_id": session_id,
                        "role": "user",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                memory_id = store_result.get("memory_id")
            else:
                memory_id = None
                
            # Enhance with memory if available and enabled
            enhanced_context = ""
            if self.memory_api and self.llm_config["memory_mode"] != "none":
                try:
                    enhanced = self.memory_api.enhance_message_with_memory(
                        message=prompt,
                        enhance_mode=self.llm_config["memory_mode"]
                    )
                    
                    if enhanced["status"] == "success":
                        enhanced_context = enhanced.get("enhanced_context", "")
                except Exception as e:
                    logger.error(f"Error enhancing message with memory: {str(e)}")
            
            # Prepare system message with memory context
            if enhanced_context:
                system_message_with_memory = (
                    f"{system_message}\n\n"
                    f"Relevant memory context:\n{enhanced_context}\n\n"
                    f"Use this memory context to provide a more informed response."
                )
            else:
                system_message_with_memory = system_message
                
            # Get LLM response
            llm_response, usage_stats = self._get_llm_response(
                prompt=prompt,
                system_message=system_message_with_memory,
                conversation_history=conversation_history
            )
            
            # Store assistant response in memory if available
            if self.memory_api:
                response_store_result = self.memory_api.store_conversation(
                    message=llm_response,
                    metadata={
                        "session_id": session_id,
                        "role": "assistant",
                        "timestamp": datetime.now().isoformat(),
                        "in_response_to": memory_id,
                        "usage_stats": usage_stats
                    }
                )
                
            # Prepare response
            processing_time = time.time() - start_time
            
            response = {
                "status": "success",
                "request_id": request_id,
                "response": llm_response,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "memory_enhanced": bool(enhanced_context),
                "usage_stats": usage_stats
            }
            
            # Send response back through socket
            self.socket.send_response(request_id, response)
            
        except Exception as e:
            logger.error(f"Error processing LLM request: {str(e)}")
            
            # Send error response
            self.socket.send_response(request_id, {
                "status": "error",
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
    def _get_llm_response(self, prompt, system_message, conversation_history):
        """
        Get a response from the configured LLM.
        
        Args:
            prompt: The prompt text
            system_message: System message for the LLM
            conversation_history: Previous conversation turns
            
        Returns:
            Tuple of (response text, usage statistics)
        """
        provider = self.llm_config["provider"]
        model = self.llm_config["model"]
        temperature = self.llm_config["temperature"]
        max_tokens = self.llm_config["max_tokens"]
        
        # Convert conversation history to messages format
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for turn in conversation_history:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", "")
            })
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Get response based on provider
        if provider == "openai" and HAS_OPENAI:
            try:
                # Initialize client with API key
                client = openai.OpenAI(api_key=self.llm_config["api_key"])
                
                # Get response
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Prepare usage stats
                usage_stats = {
                    "provider": "openai",
                    "model": model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                return response_text, usage_stats
                
            except Exception as e:
                logger.error(f"Error getting OpenAI response: {str(e)}")
                return self._simulate_llm_response(prompt, system_message), {"provider": "simulated"}
                
        elif provider == "anthropic" and HAS_ANTHROPIC:
            try:
                # Initialize client with API key
                client = anthropic.Anthropic(api_key=self.llm_config["api_key"])
                
                # Convert messages to Anthropic format
                anthropic_prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"
                
                # Get response
                response = client.completions.create(
                    model=model,
                    prompt=anthropic_prompt,
                    temperature=temperature,
                    max_tokens_to_sample=max_tokens,
                )
                
                # Extract response text
                response_text = response.completion
                
                # Prepare usage stats
                usage_stats = {
                    "provider": "anthropic",
                    "model": model,
                }
                
                return response_text, usage_stats
                
            except Exception as e:
                logger.error(f"Error getting Anthropic response: {str(e)}")
                return self._simulate_llm_response(prompt, system_message), {"provider": "simulated"}
                
        else:
            # Fall back to simulated response
            return self._simulate_llm_response(prompt, system_message), {"provider": "simulated"}
            
    def _simulate_llm_response(self, prompt, system_message):
        """
        Simulate an LLM response for when real LLM is not available.
        
        Args:
            prompt: The prompt text
            system_message: System message
            
        Returns:
            Simulated response text
        """
        logger.info(f"Simulating LLM response for: {prompt[:30]}...")
        
        # Simple response templates
        templates = [
            "Based on your question about {topic}, I think {response}",
            "I understand you're asking about {topic}. {response}",
            "Regarding {topic}, the answer is {response}",
            "Let me think about {topic}... {response}"
        ]
        
        # Extract a topic from the prompt
        words = prompt.split()
        if len(words) > 3:
            topic = " ".join(words[1:4])
        else:
            topic = prompt
            
        # Generate a simple response
        import random
        
        simple_responses = [
            "This is a simulated response as the LLM is not available.",
            "I would need more information to provide a complete answer.",
            "That's an interesting question to explore further.",
            "There are multiple perspectives to consider here."
        ]
        
        response = random.choice(templates).format(
            topic=topic,
            response=random.choice(simple_responses)
        )
        
        # Add a note about simulation
        response += "\n\n[Note: This is a simulated response because the configured LLM provider is not available.]"
        
        # Add memory enhancement simulation if in system message
        if "memory context" in system_message:
            response += "\n\n[The response would normally be enhanced with your memory context.]"
            
        return response
        
    def _handle_update_config(self, message):
        """
        Handle a request to update the LLM configuration.
        
        Args:
            message: Message containing the configuration update
            
        Returns:
            Updated configuration
        """
        content = message.get("content", {})
        
        # Update configuration
        for key, value in content.items():
            if key in self.llm_config:
                self.llm_config[key] = value
                
        logger.info(f"Updated LLM configuration: {self.llm_config}")
        
        return {
            "status": "success",
            "config": self.llm_config
        }
        
    def _handle_get_config(self, message):
        """
        Handle a request to get the current LLM configuration.
        
        Args:
            message: The request message
            
        Returns:
            Current configuration
        """
        return {
            "status": "success",
            "config": self.llm_config
        }
        
    def _handle_get_memory_stats(self, message):
        """
        Handle a request to get memory statistics.
        
        Args:
            message: The request message
            
        Returns:
            Memory statistics
        """
        if not self.memory_api:
            return {
                "status": "error",
                "error": "Memory API not available"
            }
            
        try:
            stats = self.memory_api.get_memory_stats()
            return {
                "status": "success",
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


# Plugin factory function for automatic discovery
def create_plugin():
    """Create and return the LLM Bridge Plugin instance"""
    return LLMBridgePlugin()


if __name__ == "__main__":
    # For testing as standalone module
    plugin = LLMBridgePlugin()
    print(f"Plugin descriptor: {json.dumps(plugin.get_socket_descriptor(), indent=2)}")
    print("Plugin initialized. Press Ctrl+C to exit.")
    
    try:
        # Keep the plugin running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Plugin terminated by user.") 