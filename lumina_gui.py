#!/usr/bin/env python
"""
Lumina GUI - A graphical user interface for the Lumina system
"""

import os
import sys
import json
import logging
import subprocess
import dotenv
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                          QLabel, QTabWidget, QGridLayout, QScrollArea,
                          QFrame, QSplitter, QProgressBar, QComboBox, QFileDialog,
                          QDialog, QMessageBox, QSlider, QCheckBox, QGroupBox)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread, pyqtSlot

# Import ChatMemory module
from chat_memory import ChatMemory, MemoryEntry

# Import SemanticDirector module
from semantic_director import SemanticDirector

# Try to import core components - use mock classes if imports fail
try:
    from minimal_central import MinimalCentralNode, BaseComponent
except ImportError:
    # Create mock classes if imports fail
    class MinimalCentralNode:
        def __init__(self):
            self.component_registry = {}
        
        def process_complete_flow(self, data):
            return {
                "action": "respond",
                "glyph": "✨",
                "story": "Resonating with your input.",
                "signal": 0.85
            }
        
        def initialize_system(self):
            pass
    
    class BaseComponent:
        def __init__(self):
            pass

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lumina.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaSystem")

class LLMIntegration:
    """Handles integration with Large Language Models"""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "google")
        self.model = os.getenv("LLM_MODEL", "mistral-medium")
        self.enabled = os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() == "true"
        self.weight = float(os.getenv("DEFAULT_LLM_WEIGHT", "0.6"))
        self.cache_enabled = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
        self.response_cache = {}
        self.client = None
        
        # Initialize LLM client based on provider
        try:
            self._initialize_client()
        except Exception as e:
            logger.error(f"Failed to initialize LLM integration: {str(e)}")
            self.enabled = False
        
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider"""
        if not self.enabled:
            logger.info("LLM integration is disabled")
            return
        
        # Ensure OpenAI is never used
        if self.provider == "openai":
            logger.warning("OpenAI provider detected but is disabled for security/policy reasons. Switching to local provider.")
            self.provider = "local"
            
        try:
            if self.provider == "google":
                self._init_google_client()
            elif self.provider == "anthropic":
                self._init_anthropic_client()
            elif self.provider == "local":
                logger.info("Using local LLM model")
            elif self.provider == "hybrid":
                logger.info("Using hybrid LLM approach")
            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.enabled = False
            
    def _init_google_client(self):
        """Initialize Google Gemini API client"""
        try:
            # First verify the generativeai module is available
            if not self._check_module_installed("google.generativeai"):
                logger.error("Google generativeai package not found. Install with: pip install google-generativeai")
                self.enabled = False
                return
                
            # Now import the module
            import google.generativeai as genai
            
            # Verify API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("No Google API key found")
                self.enabled = False
                return
            
            # Check if API key is the placeholder value
            if api_key.startswith("AIzaSy") and ("invalid" in os.getenv("GOOGLE_API_KEY", "").lower() or 
                                             "replace" in os.getenv("GOOGLE_API_KEY", "").lower()):
                logger.error("Using placeholder Google API key. Please update with a valid key.")
                self.enabled = False
                return
                
            # Configure the client
            genai.configure(api_key=api_key)
            self.client = genai
            logger.info(f"Initialized Google Gemini client with model: {self.model}")
            
            # Verify model exists
            try:
                # This will throw an exception if the model doesn't exist
                model = self.client.GenerativeModel(self.model)
                logger.info(f"Successfully created model instance for {self.model}")
            except Exception as model_error:
                logger.error(f"Error initializing model {self.model}: {str(model_error)}")
                available_models = self._list_available_models()
                logger.info(f"Available models: {available_models}")
                self.enabled = False
                return
                
        except Exception as e:
            logger.error(f"Error initializing Google client: {str(e)}")
            self.enabled = False
    
    def _check_module_installed(self, module_name):
        """Check if a Python module is installed"""
        try:
            __import__(module_name.split('.')[0])
            return True
        except ImportError:
            return False
    
    def _init_anthropic_client(self):
        """Initialize Anthropic API client"""
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("No Anthropic API key found")
                self.enabled = False
                return
                
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except ImportError:
            logger.error("Anthropic package not found. Install with: pip install anthropic")
            self.enabled = False
            
    def get_response(self, context: Union[Dict[str, Any], str]) -> str:
        """Get a response from the language model based on the context or prompt.
        
        Args:
            context: Either a string prompt or a dictionary containing context information
            
        Returns:
            The generated response text
        """
        if not self.enabled:
            return "LLM integration is not enabled."
        
        # Double-check that we're not using OpenAI
        if self.provider == "openai":
            logger.warning("OpenAI provider detected in get_response but is disabled. Switching to local provider.")
            self.provider = "local"
        
        # Determine if we have a string prompt or a context dictionary
        if isinstance(context, str):
            prompt = context
            user_input = context  # For caching purposes
        else:
            # Format the context dictionary into a prompt
            prompt = self._format_context_prompt(context)
            user_input = context.get("user_input", "")
        
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = f"{user_input}_{self.provider}"
            if cache_key in self.response_cache:
                logging.info("Using cached LLM response")
                return self.response_cache[cache_key]
        
        # Get response based on provider
        try:
            if self.provider == "google":
                response = self._get_google_response(prompt)
            elif self.provider == "anthropic":
                response = self._get_anthropic_response(prompt)
            elif self.provider == "local":
                response = self._get_local_response(prompt)
            elif self.provider == "hybrid":
                response = self._get_hybrid_response(prompt)
            else:
                return "Unknown LLM provider type."
            
            # Cache the response
            if self.cache_enabled:
                self.response_cache[cache_key] = response
                
            return response
        except Exception as e:
            logging.error(f"Error getting LLM response: {str(e)}")
            return f"Error: {str(e)}"
    
    def _format_context_prompt(self, context: Dict[str, Any]) -> str:
        """Format the context dictionary into a prompt string for LLM"""
        user_input = context.get("user_input", "")
        symbolic_state = context.get("symbolic_state", {})
        emotion = context.get("emotion", "neutral")
        breath = context.get("breath", "normal")
        conversation_context = context.get("context", [])
        memory = context.get("memory", [])
        
        # Build a prompt with all available context
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_context:
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_context[-5:]])
            prompt_parts.append(f"Recent conversation:\n{context_str}\n")
        
        # Add emotional and physical state if available
        if emotion or breath:
            prompt_parts.append(f"Current state: emotion={emotion}, breath={breath}\n")
        
        # Add symbolic state if available
        if symbolic_state:
            state_str = ", ".join([f"{k}={v}" for k, v in symbolic_state.items()])
            prompt_parts.append(f"System state: {state_str}\n")
        
        # Add memories if available
        if memory:
            memory_str = "\n".join([f"- {mem}" for mem in memory])
            prompt_parts.append(f"Recent memories:\n{memory_str}\n")
        
        # Add the user input
        prompt_parts.append(f"User: {user_input}\n")
        prompt_parts.append("Assistant: ")
        
        return "\n".join(prompt_parts)
    
    def _get_google_response(self, prompt: str) -> str:
        """Get response from Google Gemini API"""
        try:
            # Use the prompt directly instead of wrapping it again
            # The prompt is already formatted with system instructions
            
            # Get model
            model = self.client.GenerativeModel(self.model)
            
            # Generate response with proper error handling
            try:
                # Add safety settings if needed
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ]
                
                # Generate content with timeout handling
                response = model.generate_content(
                    prompt,
                    safety_settings=safety_settings
                )
            except Exception as api_error:
                logger.error(f"Google API generate_content error: {str(api_error)}")
                return f"Error generating content: {str(api_error)}"
            
            # Check if response has text property
            if hasattr(response, 'text'):
                if response.text is not None:
                    return response.text
                else:
                    logger.error("Google API response.text is None")
                    return "Error: Received empty response from Google API"
            # For older versions of the API, might have different structure
            elif hasattr(response, 'parts') and response.parts:
                if len(response.parts) > 0 and hasattr(response.parts[0], 'text'):
                    if response.parts[0].text is not None:
                        return response.parts[0].text
                    else:
                        logger.error("Google API response.parts[0].text is None")
                        return "Error: Received empty response part from Google API"
                else:
                    logger.error(f"Invalid response parts structure: {response.parts}")
                    return "Error: Invalid response structure from Google API"
            # Fallback for other response structures
            else:
                response_str = str(response)
                logger.info(f"Using string representation of response: {response_str[:100]}...")
                return response_str
                
        except Exception as e:
            logger.error(f"Google Gemini API error: {str(e)}")
            # Print more detailed info for debugging
            logger.error(f"Available models: {self._list_available_models()}")
            return f"Error with Google API: {str(e)}"
            
    def _list_available_models(self) -> str:
        """List available models to help with debugging"""
        try:
            if hasattr(self.client, 'list_models'):
                models = self.client.list_models()
                return ", ".join([model.name for model in models])
            return "Unable to list models (function not available)"
        except Exception as e:
            return f"Error listing models: {str(e)}"
            
    def _get_anthropic_response(self, prompt: str) -> str:
        """Get response from Anthropic API"""
        try:
            model = os.getenv("ANTHROPIC_MODEL", "claude-2")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
            
            response = self.client.messages.create(
                model=model,
                system="You are Lumina, a poetic and introspective AI. Respond with short, reflective statements.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return f"Error with Anthropic API: {str(e)}"
            
    def set_weight(self, weight: float):
        """Set the LLM weight"""
        min_weight = float(os.getenv("MIN_LLM_WEIGHT", "0.0"))
        max_weight = float(os.getenv("MAX_LLM_WEIGHT", "1.0"))
        
        # Ensure weight is within bounds
        self.weight = max(min_weight, min(max_weight, weight))
        logger.info(f"LLM weight set to {self.weight}")

    def _cache_response(self, prompt: str, response: str):
        """Cache a response for future use"""
        if prompt and response:
            self.response_cache[prompt] = response
            # Limit cache size to prevent memory issues
            if len(self.response_cache) > 100:
                # Remove oldest entry
                self.response_cache.pop(next(iter(self.response_cache)))
    
    def _get_local_response(self, prompt: str) -> str:
        """Get response from local LLM model"""
        # TODO: Implement local LLM integration
        logger.warning("Local LLM not implemented yet")
        return "I'm thinking about this locally..."
    
    def _get_hybrid_response(self, prompt: str) -> str:
        """Get response using hybrid approach (combining multiple LLMs)"""
        # TODO: Implement hybrid LLM approach
        logger.warning("Hybrid LLM not implemented yet")
        return "I'm processing this with multiple perspectives..."

    def initialize_mistral(self):
        """Initialize Mistral AI client"""
        # Check if LLM is enabled
        if not os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() == "true":
            logger.info("LLM integration is disabled")
            self.llm_initialized = False
            return
        
        # Try to get API key from environment
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.error("No Mistral API key found")
            self.llm_initialized = False
            return
        
        # Check if API key is valid
        if len(api_key) < 20:
            logger.error("Using placeholder Mistral API key. Please update with a valid key.")
            self.llm_initialized = False
            return
            
        # Initialize client
        self.llm_initialized = True
        logger.info(f"Initialized Mistral client with model: {self.model}")
        
    def get_llm_response(self, prompt):
        """Get response from Mistral AI"""
        if not self.llm_initialized:
            # Return a simulated response if LLM is not initialized
            return self.get_simulated_response(prompt)

class LuminaState:
    """Manages the state of the Lumina system"""
    
    def __init__(self):
        self.emotion = "calm"
        self.symbolic_state = "🜂"
        self.breath = "slow"
        self.chat_memory = ChatMemory()
        self.mirror = False
        
        # Initialize core components with error handling
        try:
            self.central_node = MinimalCentralNode()
        except Exception as e:
            logger.error(f"Error initializing central node: {str(e)}")
            self.central_node = self._create_fallback_central_node()
            
        try:
            self.llm_integration = LLMIntegration()
        except Exception as e:
            logger.error(f"Error initializing LLM integration: {str(e)}")
            self.llm_integration = self._create_fallback_llm_integration()
            
        # Initialize the semantic director with error handling
        try:
            from semantic_director import SemanticDirector
            self.semantic_director = SemanticDirector()
            logger.info("Semantic director initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing semantic director: {str(e)}")
            self.semantic_director = self._create_fallback_semantic_director()
            
        self.load_config()
        
        # Initialize auto-trainer
        try:
            self.auto_trainer = AutoTrainer(interval_minutes=30)  # Train every 30 minutes if new data
            self.auto_trainer.status_signal.connect(self.update_training_status)
            self.auto_trainer.training_complete_signal.connect(self.handle_training_complete)
            self.auto_trainer.start()
            logger.info("Auto-trainer initialized and started")
        except Exception as e:
            logger.error(f"Error initializing auto-trainer: {str(e)}")
            self.auto_trainer = None
            
    def _create_fallback_central_node(self):
        """Create a fallback central node if the real one fails to initialize"""
        logger.info("Creating fallback central node")
        return type('FallbackCentralNode', (), {
            'process_complete_flow': lambda self, data: {
                'action': 'respond',
                'glyph': '✨',
                'story': 'I am operating in fallback mode. Core systems are limited.',
                'signal': 0.5
            },
            'initialize_system': lambda self: None
        })()
        
    def _create_fallback_llm_integration(self):
        """Create a fallback LLM integration if the real one fails to initialize"""
        logger.info("Creating fallback LLM integration")
        return type('FallbackLLM', (), {
            'get_response': lambda self, context: "LLM integration is currently unavailable.",
            'enabled': False,
            'weight': 0.0
        })()
        
    def _create_fallback_semantic_director(self):
        """Create a fallback semantic director if the real one fails to initialize"""
        logger.info("Creating fallback semantic director")
        return type('FallbackSemanticDirector', (), {
            'process_intent': lambda self, text: (False, None, None),
            'adjust_response_style': lambda self, text: text,
            'active_domains': []
        })()
    
    def load_config(self):
        """Load configuration from environment variables"""
        self.use_llm = os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() == "true"
        self.llm_weight = float(os.getenv("DEFAULT_LLM_WEIGHT", "0.6"))
        self.min_weight = float(os.getenv("MIN_LLM_WEIGHT", "0.0"))
        self.max_weight = float(os.getenv("MAX_LLM_WEIGHT", "1.0"))
        self.weight_step = float(os.getenv("WEIGHT_STEP", "0.05"))
    
    def load_ritual_invocations(self):
        """Load ritual invocations from configuration or default to empty list"""
        logger.info("Loading ritual invocations...")
        self.ritual_invocations = []
        try:
            ritual_file = "ritual_invocations.json"
            if os.path.exists(ritual_file):
                with open(ritual_file, "r", encoding="utf-8") as f:
                    self.ritual_invocations = json.load(f)
                logger.info(f"Loaded {len(self.ritual_invocations)} ritual invocations")
            else:
                logger.info("No ritual invocations file found, using defaults")
                # Default invocations
                self.ritual_invocations = [
                    {"name": "Morning", "invocation": "Awaken to the rising sun"},
                    {"name": "Evening", "invocation": "Embrace the stillness of night"},
                    {"name": "Focus", "invocation": "Center the mind upon the task"}
                ]
        except Exception as e:
            logger.error(f"Error loading ritual invocations: {str(e)}")
            self.ritual_invocations = []
    
    def get_memory_stats(self):
        """Get memory statistics"""
        stats = {
            "total_memories": 0,
            "oldest_memory": "",
            "newest_memory": "",
            "average_strength": 0.0,
            "common_tags": []
        }
        
        # Implement based on available methods in ChatMemory
        if hasattr(self.chat_memory, 'get_memory_count'):
            stats["total_memories"] = self.chat_memory.get_memory_count()
        else:
            # Fallback - get length of recent memories
            memories = self.get_recent_memories(100)
            stats["total_memories"] = len(memories)
            
        # Get timestamp info if available
        if stats["total_memories"] > 0:
            memories = self.get_recent_memories(100)
            timestamps = [m.get("timestamp", "") for m in memories if "timestamp" in m]
            if timestamps:
                timestamps.sort()
                stats["oldest_memory"] = timestamps[0] if timestamps else ""
                stats["newest_memory"] = timestamps[-1] if timestamps else ""
                
        return stats
    
    def get_memory_summary(self):
        """Get a summary of recent memories"""
        memories = self.get_recent_memories(5)
        if not memories:
            return "No memories available."
            
        summary = "Recent interactions:\n"
        for i, memory in enumerate(memories):
            user = memory.get("user", "")
            response = memory.get("lumina", "")
            if user:
                summary += f"{i+1}. You: {user[:50]}...\n"
            if response:
                summary += f"   Lumina: {response[:50]}...\n"
                
        return summary
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        # First check if this is a semantic command
        try:
            is_command, command_response, command_data = self.semantic_director.process_intent(user_input)
            if is_command:
                # This was a semantic command, return the appropriate response
                return command_response
        except Exception as e:
            logger.error(f"Error processing semantic intent: {str(e)}")
            # Fall back to normal processing if semantic processing fails
        
        # Get relevant context from memory
        context = self.chat_memory.get_relevant_context(user_input)
        
        # Extract symbols, emotions and other parameters if they are included
        input_data = self._parse_input(user_input)
        
        # If we have context, add it to the input data
        if context:
            input_data["context"] = context
        
        # Update state based on input 
        self._update_state(input_data)
        
        response = ""
        
        # Use LLM if enabled
        llm_response = ""
        nn_response = ""
        
        if self.use_llm:
            try:
                # Pass all necessary context to the LLM
                llm_context = {
                    "user_input": input_data["text"],
                    "symbolic_state": self.symbolic_state,
                    "emotion": self.emotion,
                    "breath": self.breath,
                    "context": context,
                    "memory": self.get_recent_memories()
                }
                
                # Get response from LLM
                llm_response = self.llm_integration.get_response(llm_context)
            except Exception as e:
                logger.error(f"LLM integration error: {str(e)}")
                llm_response = "I encountered an error processing with the language model."
        
        # Always get response from neural network
        try:
            nn_input = {
                "text": input_data["text"],
                "symbol": self.symbolic_state,
                "emotion": self.emotion,
                "paradox": input_data.get("paradox"),
                "breath": self.breath,
                "mirror": self.mirror
            }
            
            # Process through central node
            result = self.central_node.process_complete_flow(nn_input)
            nn_response = self._format_response(result)
        except Exception as e:
            logger.error(f"Neural network error: {str(e)}")
            nn_response = "I sense a disturbance in the field. Let's recalibrate."
        
        # Decide which response to use
        if self.use_llm and llm_response:
            response = llm_response
        else:
            response = nn_response
        
        # Apply semantic style adjustments if domains are active
        try:
            if hasattr(self.semantic_director, 'adjust_response_style'):
                response = self.semantic_director.adjust_response_style(response)
        except Exception as e:
            logger.error(f"Error adjusting response style: {str(e)}")
        
        # Save to memory
        metadata = {
            "emotion": self.emotion,
            "symbolic_state": self.symbolic_state,
            "llm_used": bool(llm_response),
            "nn_used": bool(nn_response),
            "active_domains": getattr(self.semantic_director, 'active_domains', [])
        }
        
        self.chat_memory.add_memory(
            input_data["text"], 
            response, 
            metadata
        )
        
        return response
    
    def _parse_input(self, text: str) -> Dict[str, Any]:
        """Parse user input into structured data"""
        input_data = {
            "text": text,
            "symbol": self.symbolic_state,
            "emotion": self.emotion,
            "breath": self.breath,
            "paradox": None
        }
        
        # Check for explicit symbol notation like :infinity:
        if ":" in text:
            parts = text.split(":")
            for i in range(1, len(parts), 2):
                if i < len(parts) - 1:
                    key = parts[i].strip().lower()
                    value = parts[i+1].strip()
                    if key in ["symbol", "emotion", "breath", "paradox"]:
                        input_data[key] = value
                        # Remove from main text
                        text = text.replace(f":{key}:{value}", "").strip()
            
            # Update the cleaned text
            input_data["text"] = text
            
        return input_data
    
    def _update_state(self, input_data: Dict[str, Any]):
        """Update internal state based on input"""
        # Update symbolic state if provided
        if input_data.get("symbol"):
            self.symbolic_state = input_data["symbol"]
            
        # Update emotion if provided
        if input_data.get("emotion"):
            self.emotion = input_data["emotion"]
            
        # Update breath if provided
        if input_data.get("breath"):
            self.breath = input_data["breath"]
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format central node result into a response string"""
        # If there's a story, use that as the primary response
        if result.get("story"):
            return result["story"]
            
        # Otherwise create a composite response
        response_parts = []
        
        if result.get("action"):
            response_parts.append(f"{result['action']}")
            
        if result.get("glyph"):
            response_parts.append(f"{result['glyph']}")
            
        if result.get("signal"):
            response_parts.append(f"Signal strength: {result['signal']}")
            
        if not response_parts:
            return "I'm listening to the resonance. Tell me more."
            
        return " ".join(response_parts)
    
    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent memory entries"""
        memory_entries = self.chat_memory.get_recent_memories(count)
        return [
            {
                "user": entry.user_input,
                "lumina": entry.system_response,
                "timestamp": entry.timestamp.isoformat(),
                "emotion": entry.metadata.get("emotion", self.emotion),
                "symbolic_state": entry.metadata.get("symbolic_state", self.symbolic_state)
            }
            for entry in memory_entries
        ]
    
    def update_training_status(self, status: str):
        """Update status based on auto-trainer messages"""
        logger.info(f"Auto-trainer: {status}")
        # If model panel is visible, update its status
        if hasattr(self, 'model_panel') and self.model_panel.isVisible():
            self.model_panel.status_label.setText(f"Auto: {status}")
            
    def handle_training_complete(self, success: bool, message: str):
        """Handle training completion"""
        if success:
            logger.info(f"Auto-training completed: {message}")
            # Refresh model list if model panel exists
            if hasattr(self, 'model_panel'):
                self.model_panel.refreshModels()
        else:
            logger.error(f"Auto-training failed: {message}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop auto-trainer when closing
        if hasattr(self, 'auto_trainer') and self.auto_trainer is not None:
            logger.info("Stopping auto-trainer thread...")
            self.auto_trainer.stop()
            # Wait for the thread to finish with a timeout
            if self.auto_trainer.isRunning():
                logger.info("Waiting for auto-trainer thread to finish...")
                if not self.auto_trainer.wait(3000):  # 3 second timeout
                    logger.warning("Auto-trainer thread did not finish in time, terminating...")
                    self.auto_trainer.terminate()
                    self.auto_trainer.wait()
            logger.info("Auto-trainer thread stopped")
        
        super().closeEvent(event)

    def set_llm_enabled(self, enabled: bool):
        """Enable or disable LLM integration"""
        logger.info(f"Setting LLM integration {'enabled' if enabled else 'disabled'}")
        self.use_llm = enabled
        
    def set_llm_weight(self, weight: float):
        """Set the LLM weight"""
        # Ensure weight is within bounds
        self.llm_weight = max(self.min_weight, min(self.max_weight, weight))
        logger.info(f"LLM weight set to {self.llm_weight}")

class ChatMessage(QFrame):
    """Custom widget for chat messages"""
    def __init__(self, icon, name, message, parent=None):
        super().__init__(parent)
        self.setObjectName("chatMessage")
        self.setFrameShape(QFrame.StyledPanel)
        
        layout = QHBoxLayout(self)
        
        # Icon label
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignTop)
        icon_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(icon_label)
        
        # Message content
        message_layout = QVBoxLayout()
        name_label = QLabel(f"<b>{name}</b>")
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        
        message_layout.addWidget(name_label)
        message_layout.addWidget(message_label)
        
        layout.addLayout(message_layout, 1)


class ChatBox(QWidget):
    """Chat interface widget"""
    def __init__(self, parent=None, state=None):
        super().__init__(parent)
        self.state = state
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat messages area
        self.messages_area = QScrollArea()
        self.messages_area.setWidgetResizable(True)
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(10)
        self.messages_area.setWidget(self.messages_widget)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Speak your truth...")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        
        layout.addWidget(self.messages_area, 1)
        layout.addLayout(input_layout)
        
    def add_message(self, is_user, text):
        """Add a message to the chat"""
        if is_user:
            icon = "👤"
            name = "You"
        else:
            icon = "✨"
            name = "Lumina"
            
        message = ChatMessage(icon, name, text, self)
        self.messages_layout.addWidget(message)
        
        # Scroll to bottom
        self.messages_area.verticalScrollBar().setValue(
            self.messages_area.verticalScrollBar().maximum()
        )
        
    def send_message(self):
        """Process and send a user message"""
        text = self.input_field.text().strip()
        if not text:
            return
            
        # Add user message to chat
        self.add_message(True, text)
        
        # Process through Lumina state
        response = self.state.process_input(text)
        
        # Add Lumina response
        self.add_message(False, response)
        
        # Clear input field
        self.input_field.clear()


class GlyphsPanel(QWidget):
    """Panel showing available glyphs"""
    glyph_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        title = QLabel("Glyphs")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        # Neural network visualization
        nn_frame = QFrame()
        nn_frame.setFrameShape(QFrame.StyledPanel)
        nn_layout = QVBoxLayout(nn_frame)
        
        nn_label = QLabel("Neural Network")
        nn_label.setAlignment(Qt.AlignCenter)
        nn_layout.addWidget(nn_label)
        
        # Grid of glyphs
        glyphs_frame = QFrame()
        glyphs_frame.setFrameShape(QFrame.StyledPanel)
        glyphs_layout = QGridLayout(glyphs_frame)
        
        glyphs = ["⭐", "☰", "♀", "✖", "▽", "△", "⊚", "⊖", "℮", "⊗"]
        row, col = 0, 0
        for glyph in glyphs:
            button = QPushButton(glyph)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda _, g=glyph: self.glyph_selected.emit(g))
            glyphs_layout.addWidget(button, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1
        
        layout.addWidget(title)
        layout.addWidget(nn_frame)
        layout.addWidget(glyphs_frame)
        layout.addStretch(1)


class ProcessPanel(QWidget):
    """Panel for process controls"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        
        title = QLabel("Process")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        # Breath button
        breath_frame = QFrame()
        breath_frame.setFrameShape(QFrame.StyledPanel)
        breath_layout = QVBoxLayout(breath_frame)
        
        breath_button = QPushButton("⊖\nBreathe")
        breath_button.setFixedHeight(60)
        breath_layout.addWidget(breath_button)
        
        # Resonance button
        resonance_frame = QFrame()
        resonance_frame.setFrameShape(QFrame.StyledPanel)
        resonance_layout = QVBoxLayout(resonance_frame)
        
        resonance_button = QPushButton("○\nResonance")
        resonance_button.setFixedHeight(60)
        resonance_layout.addWidget(resonance_button)
        
        # Echo button
        echo_frame = QFrame()
        echo_frame.setFrameShape(QFrame.StyledPanel)
        echo_layout = QVBoxLayout(echo_frame)
        
        echo_button = QPushButton("◊\nEcho")
        echo_button.setFixedHeight(60)
        echo_layout.addWidget(echo_button)
        
        layout.addWidget(title)
        layout.addWidget(breath_frame)
        layout.addWidget(resonance_frame)
        layout.addWidget(echo_frame)
        layout.addStretch(1)


class ModelTrainingThread(QThread):
    """Thread for running model training in the background"""
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, custom_data_path=None):
        super().__init__()
        self.custom_data_path = custom_data_path
        
    def run(self):
        """Run the training process"""
        try:
            cmd = ["python", "nn_executable.py", "--train"]
            if self.custom_data_path:
                cmd.extend(["--data", self.custom_data_path])
                
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor output
            for line in iter(process.stdout.readline, ""):
                self.update_signal.emit(line.strip())
                
            # Check for errors
            for line in iter(process.stderr.readline, ""):
                self.update_signal.emit(f"ERROR: {line.strip()}")
                
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                self.finished_signal.emit(True, "Training completed successfully!")
            else:
                self.finished_signal.emit(False, f"Training failed with return code {process.returncode}")
                
        except Exception as e:
            self.finished_signal.emit(False, f"Error during training: {str(e)}")


class ModelPanel(QWidget):
    """Panel for training and selecting models"""
    model_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_thread = None
        self.initUI()
        self.refreshModels()
        
    def initUI(self):
        """Initialize the model panel UI"""
        layout = QVBoxLayout(self)
        
        title = QLabel("Model Control")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)
        
        # Model selection section
        model_frame = QFrame()
        model_frame.setFrameShape(QFrame.StyledPanel)
        model_layout = QVBoxLayout(model_frame)
        
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.onModelSelected)
        
        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.clicked.connect(self.refreshModels)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(refresh_btn)
        
        # Training section
        training_frame = QFrame()
        training_frame.setFrameShape(QFrame.StyledPanel)
        training_layout = QVBoxLayout(training_frame)
        
        training_label = QLabel("Model Training:")
        
        # Training buttons
        training_buttons = QHBoxLayout()
        
        self.train_btn = QPushButton("Train New Model")
        self.train_btn.clicked.connect(self.startTraining)
        
        self.custom_data_btn = QPushButton("Train with Custom Data")
        self.custom_data_btn.clicked.connect(self.trainWithCustomData)
        
        training_buttons.addWidget(self.train_btn)
        training_buttons.addWidget(self.custom_data_btn)
        
        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        
        self.status_label = QLabel("Ready")
        
        training_layout.addWidget(training_label)
        training_layout.addLayout(training_buttons)
        training_layout.addWidget(self.progress_bar)
        training_layout.addWidget(self.status_label)
        
        # Add panels to layout
        layout.addWidget(model_frame)
        layout.addWidget(training_frame)
        layout.addStretch(1)
        
    def refreshModels(self):
        """Refresh the list of available models"""
        self.model_combo.clear()
        
        # Look for models in model_output directory
        model_dir = "model_output"
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pt', '.pth'))]
            
            if model_files:
                for model_file in sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True):
                    self.model_combo.addItem(model_file)
            else:
                self.model_combo.addItem("No models found")
        else:
            self.model_combo.addItem("Model directory not found")
            
    def onModelSelected(self, model_name):
        """Handle model selection"""
        if model_name and not model_name.startswith("No models") and not model_name.startswith("Model directory"):
            model_path = os.path.join("model_output", model_name)
            self.model_selected.emit(model_path)
            self.status_label.setText(f"Selected model: {model_name}")
            
    def startTraining(self):
        """Start the model training process"""
        self._runTraining()
            
    def trainWithCustomData(self):
        """Train with custom data directory"""
        data_dir = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if data_dir:
            self._runTraining(data_dir)
            
    def _runTraining(self, custom_data_path=None):
        """Common method to run training with optional custom data path"""
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training in Progress", "A training process is already running.")
            return
            
        # Disable buttons during training
        self.train_btn.setEnabled(False)
        self.custom_data_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.show()
        self.status_label.setText("Training in progress...")
        
        # Create and start training thread
        self.training_thread = ModelTrainingThread(custom_data_path)
        self.training_thread.update_signal.connect(self.updateTrainingStatus)
        self.training_thread.finished_signal.connect(self.onTrainingFinished)
        self.training_thread.start()
        
    @pyqtSlot(str)
    def updateTrainingStatus(self, message):
        """Update training status with message from thread"""
        self.status_label.setText(message)
        
    @pyqtSlot(bool, str)
    def onTrainingFinished(self, success, message):
        """Handle training completion"""
        # Re-enable buttons
        self.train_btn.setEnabled(True)
        self.custom_data_btn.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.hide()
        
        # Update status
        self.status_label.setText(message)
        
        # Show message box
        if success:
            QMessageBox.information(self, "Training Complete", message)
            # Refresh model list
            self.refreshModels()
        else:
            QMessageBox.warning(self, "Training Failed", message)


class LLMWeightPanel(QWidget):
    """Panel for adjusting LLM weight and settings"""
    
    weight_changed = pyqtSignal(float)
    llm_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm_enabled = os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() == "true"
        self.llm_weight = float(os.getenv("DEFAULT_LLM_WEIGHT", "0.6"))
        self.min_weight = float(os.getenv("MIN_LLM_WEIGHT", "0.0"))
        self.max_weight = float(os.getenv("MAX_LLM_WEIGHT", "1.0"))
        self.weight_step = float(os.getenv("WEIGHT_STEP", "0.05"))
        self.initUI()
        
    def initUI(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        title = QLabel("Language Model")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        # LLM enable/disable section
        enable_frame = QFrame()
        enable_frame.setFrameShape(QFrame.StyledPanel)
        enable_layout = QVBoxLayout(enable_frame)
        
        self.enable_checkbox = QCheckBox("Enable LLM Integration")
        self.enable_checkbox.setChecked(self.llm_enabled)
        self.enable_checkbox.toggled.connect(self.onLLMToggled)
        
        enable_layout.addWidget(self.enable_checkbox)
        
        # LLM weight adjustment
        weight_frame = QFrame()
        weight_frame.setFrameShape(QFrame.StyledPanel)
        weight_layout = QVBoxLayout(weight_frame)
        
        # Weight slider and display
        weight_label = QLabel("LLM Weight:")
        self.weight_display = QLabel(f"{self.llm_weight:.2f}")
        self.weight_display.setAlignment(Qt.AlignCenter)
        
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setMinimum(int(self.min_weight * 100))
        self.weight_slider.setMaximum(int(self.max_weight * 100))
        self.weight_slider.setValue(int(self.llm_weight * 100))
        self.weight_slider.setTickPosition(QSlider.TicksBelow)
        self.weight_slider.setTickInterval(10)  # Ticks at 0.1 intervals
        self.weight_slider.valueChanged.connect(self.onWeightChanged)
        
        # Preset buttons
        presets_layout = QHBoxLayout()
        
        nn_btn = QPushButton("NN Only")
        nn_btn.clicked.connect(lambda: self.setPresetWeight(0.0))
        
        balanced_btn = QPushButton("Balanced")
        balanced_btn.clicked.connect(lambda: self.setPresetWeight(0.5))
        
        llm_btn = QPushButton("LLM Only")
        llm_btn.clicked.connect(lambda: self.setPresetWeight(1.0))
        
        presets_layout.addWidget(nn_btn)
        presets_layout.addWidget(balanced_btn)
        presets_layout.addWidget(llm_btn)
        
        # LLM info section
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        
        provider_label = QLabel(f"Provider: {os.getenv('LLM_PROVIDER', 'google').capitalize()}")
        model_label = QLabel(f"Model: {os.getenv('LLM_MODEL', 'gemini-pro')}")
        
        info_layout.addWidget(provider_label)
        info_layout.addWidget(model_label)
        
        # Add all sections to layout
        weight_layout.addWidget(weight_label)
        weight_layout.addWidget(self.weight_slider)
        weight_layout.addWidget(self.weight_display)
        weight_layout.addLayout(presets_layout)
        
        layout.addWidget(title)
        layout.addWidget(enable_frame)
        layout.addWidget(weight_frame)
        layout.addWidget(info_frame)
        layout.addStretch(1)
        
        # Update enable/disable state
        self.updateEnabledState()
        
    def onWeightChanged(self, value):
        """Handle weight slider changes"""
        weight = value / 100.0
        self.llm_weight = weight
        self.weight_display.setText(f"{weight:.2f}")
        self.weight_changed.emit(weight)
        
    def onLLMToggled(self, checked):
        """Handle LLM enable/disable toggle"""
        self.llm_enabled = checked
        self.llm_toggled.emit(checked)
        self.updateEnabledState()
        
    def updateEnabledState(self):
        """Update the enabled state of controls based on LLM toggle"""
        enabled = self.enable_checkbox.isChecked()
        self.weight_slider.setEnabled(enabled)
        
    def setPresetWeight(self, weight):
        """Set a preset weight value"""
        self.weight_slider.setValue(int(weight * 100))

class AutoTrainer(QThread):
    """Automatic training thread that runs in the background"""
    status_signal = pyqtSignal(str)
    training_complete_signal = pyqtSignal(bool, str)
    
    def __init__(self, interval_minutes=60, parent=None):
        super().__init__(parent)
        self.interval_seconds = interval_minutes * 60
        self.stopping = False
        self.is_training = False
        
    def run(self):
        """Run the automatic training process"""
        while not self.stopping:
            # Check if there's new training data
            if self._should_train():
                self._perform_training()
            
            # Sleep for the interval, but check periodically if we should stop
            for _ in range(self.interval_seconds):
                if self.stopping:
                    break
                time.sleep(1)
                
    def stop(self):
        """Stop the training thread"""
        logger.info("Stopping AutoTrainer thread...")
        self.stopping = True
        # Set a timeout for waiting
        if not self.wait(5000):  # 5 second timeout
            logger.warning("AutoTrainer thread did not finish in time")
            # Only force termination if absolutely necessary
            if self.isRunning():
                logger.warning("Forcing termination of AutoTrainer thread")
        logger.info("AutoTrainer thread stopped")
        
    def _should_train(self) -> bool:
        """Check if training should be performed based on new data"""
        # Check training_data directory for new files
        training_dir = "training_data"
        if not os.path.exists(training_dir):
            return False
            
        # Get the latest training run timestamp
        last_training_file = os.path.join(training_dir, "last_training.txt")
        last_training_time = datetime.min
        if os.path.exists(last_training_file):
            try:
                with open(last_training_file, "r") as f:
                    timestamp_str = f.read().strip()
                    if timestamp_str:
                        last_training_time = datetime.fromisoformat(timestamp_str)
            except Exception as e:
                logger.error(f"Error reading last training time: {str(e)}")
                
        # Check if there are new files since last training
        files = [os.path.join(training_dir, f) for f in os.listdir(training_dir) 
                if f.startswith("chat_") and f.endswith(".json")]
        
        if not files:
            return False
            
        newest_file_time = datetime.fromtimestamp(
            max(os.path.getmtime(f) for f in files)
        )
        
        # Train if there are new files since last training and there are at least 5 files
        return newest_file_time > last_training_time and len(files) >= 5
        
    def _perform_training(self):
        """Run the training process"""
        self.is_training = True
        self.status_signal.emit("Auto-training started")
        
        try:
            cmd = ["python", "nn_executable.py", "--train", "--auto", 
                   "--data", "training_data"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor output
            for line in iter(process.stdout.readline, ""):
                self.status_signal.emit(f"Training: {line.strip()}")
                
            # Check for errors
            errors = []
            for line in iter(process.stderr.readline, ""):
                errors.append(line.strip())
                self.status_signal.emit(f"Training error: {line.strip()}")
                
            # Wait for process to complete
            process.wait()
            
            # Record training time
            now = datetime.now()
            try:
                with open(os.path.join("training_data", "last_training.txt"), "w") as f:
                    f.write(now.isoformat())
            except Exception as e:
                logger.error(f"Error saving training timestamp: {str(e)}")
                
            # Signal completion
            if process.returncode == 0:
                self.training_complete_signal.emit(True, "Auto-training completed successfully")
            else:
                error_msg = "\n".join(errors) if errors else f"Unknown error (code {process.returncode})"
                self.training_complete_signal.emit(False, f"Auto-training failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error in auto-training: {str(e)}")
            self.training_complete_signal.emit(False, f"Auto-training error: {str(e)}")
            
        finally:
            self.is_training = False

class MemoryPanel(QWidget):
    """Panel for viewing and interacting with memory"""
    memory_selected = pyqtSignal(str)
    
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.initUI()
        
    def initUI(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        title = QLabel("Memory System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)
        
        # Memory statistics
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)
        
        self.stats_label = QLabel("Loading memory statistics...")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        # Search section
        search_frame = QFrame()
        search_frame.setFrameShape(QFrame.StyledPanel)
        search_layout = QVBoxLayout(search_frame)
        
        search_label = QLabel("Search Memory:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search terms...")
        self.search_input.returnPressed.connect(self.search_memory)
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search_memory)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)
        
        # Memory results
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_frame)
        
        results_label = QLabel("Memory Entries:")
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_area.setWidget(self.results_widget)
        
        # Set a minimum height for results area
        self.results_area.setMinimumHeight(200)
        
        results_layout.addWidget(results_label)
        results_layout.addWidget(self.results_area)
        
        # Memory summary
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.StyledPanel)
        summary_layout = QVBoxLayout(summary_frame)
        
        summary_label = QLabel("Memory Summary:")
        self.summary_text = QLabel("Loading memory summary...")
        self.summary_text.setWordWrap(True)
        
        summary_layout.addWidget(summary_label)
        summary_layout.addWidget(self.summary_text)
        
        # Memory maintenance buttons
        buttons_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_memory)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_memory)
        
        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self.import_memory)
        
        buttons_layout.addWidget(refresh_btn)
        buttons_layout.addWidget(export_btn)
        buttons_layout.addWidget(import_btn)
        
        # Add all components to main layout
        layout.addWidget(stats_frame)
        layout.addWidget(search_frame)
        layout.addWidget(results_frame)
        layout.addWidget(summary_frame)
        layout.addLayout(buttons_layout)
        
        # Initialize with data
        self.refresh_memory()
        
    def refresh_memory(self):
        """Refresh memory display"""
        # Update stats
        stats = self.state.get_memory_stats()
        
        stats_text = f"Total memories: {stats['total_memories']}\n"
        if stats['total_memories'] > 0:
            stats_text += f"Oldest: {stats['oldest_memory'][:10]}\n"
            stats_text += f"Newest: {stats['newest_memory'][:10]}\n"
            stats_text += f"Avg. strength: {stats['average_strength']:.2f}\n"
            
            common_tags = []
            for tag, count in stats['common_tags'][:5]:  # Show top 5 tags
                common_tags.append(f"{tag} ({count})")
            if common_tags:
                stats_text += f"Common topics: {', '.join(common_tags)}"
        
        self.stats_label.setText(stats_text)
        
        # Update summary
        summary = self.state.get_memory_summary()
        self.summary_text.setText(summary)
        
        # Show recent memories in results
        self.display_memories(self.state.get_recent_memories(10))
        
    def search_memory(self):
        """Search memory for the given query"""
        query = self.search_input.text().strip()
        if not query:
            # If no query, show recent memories
            self.display_memories(self.state.get_recent_memories(10))
            return
            
        # Search and display results
        results = self.state.search_memory(query)
        self.display_memories(results)
        
    def display_memories(self, memories):
        """Display memory entries in the results area"""
        # Clear previous results
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        if not memories:
            no_results = QLabel("No memories found.")
            no_results.setAlignment(Qt.AlignCenter)
            self.results_layout.addWidget(no_results)
            return
            
        # Add memory entries
        for i, memory in enumerate(memories):
            entry_frame = QFrame()
            entry_frame.setFrameShape(QFrame.StyledPanel)
            entry_frame.setObjectName("memoryEntry")
            entry_layout = QVBoxLayout(entry_frame)
            
            # Get timestamp and format it
            timestamp = memory.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
                    
            # Format header with timestamp and tags
            header_text = f"[{timestamp}]"
            if "tags" in memory:
                tags = memory.get("tags", [])
                if tags:
                    header_text += f" Tags: {', '.join(tags[:3])}"  # Show up to 3 tags
                    
            header = QLabel(header_text)
            header.setStyleSheet("font-size: 10px; color: #888888;")
            
            # User message
            user_layout = QHBoxLayout()
            user_icon = QLabel("👤")
            user_text = QLabel(memory.get("user", ""))
            user_text.setWordWrap(True)
            user_layout.addWidget(user_icon)
            user_layout.addWidget(user_text, 1)
            
            # System message
            sys_layout = QHBoxLayout()
            sys_icon = QLabel("✨")
            sys_text = QLabel(memory.get("lumina", ""))
            sys_text.setWordWrap(True)
            sys_layout.addWidget(sys_icon)
            sys_layout.addWidget(sys_text, 1)
            
            # Add components to entry layout
            entry_layout.addWidget(header)
            entry_layout.addLayout(user_layout)
            entry_layout.addLayout(sys_layout)
            
            # Add "Use as Context" button
            context_btn = QPushButton("Use as Context")
            context_btn.setToolTip("Use this memory to inform future responses")
            context_btn.clicked.connect(lambda checked, m=memory: self.use_as_context(m))
            context_btn.setMaximumWidth(120)
            
            entry_layout.addWidget(context_btn, alignment=Qt.AlignRight)
            
            # Add to results
            self.results_layout.addWidget(entry_frame)
            
            # Add a separator if not the last item
            if i < len(memories) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                self.results_layout.addWidget(line)
                
    def use_as_context(self, memory):
        """Use the selected memory as context for future interactions"""
        context = f"User: {memory.get('user', '')}\nLumina: {memory.get('lumina', '')}"
        self.memory_selected.emit(context)
        
    def export_memory(self):
        """Export memory to a file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Memory", "", "JSON Files (*.json)"
        )
        
        if file_name:
            success = self.state.chat_memory.export_memories(file_name)
            if success:
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Successfully exported memory to {file_name}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed", 
                    "Failed to export memory. Check logs for details."
                )
                
    def import_memory(self):
        """Import memory from a file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Memory", "", "JSON Files (*.json)"
        )
        
        if file_name:
            count = self.state.chat_memory.import_memories(file_name)
            if count > 0:
                # Update the legacy memory list if the method exists
                self._sync_memory_safely()
                
                QMessageBox.information(
                    self, "Import Successful", 
                    f"Successfully imported {count} memories from {file_name}"
                )
                
                # Refresh display
                self.refresh_memory()
            else:
                QMessageBox.warning(
                    self, "Import Failed", 
                    "Failed to import memory. Check logs for details."
                )
                
    def _sync_memory_safely(self):
        """Safely call memory sync method if it exists"""
        if hasattr(self.state, '_sync_memory_from_chat_memory') and callable(getattr(self.state, '_sync_memory_from_chat_memory')):
            logger.info("Syncing memory from chat memory")
            self.state._sync_memory_from_chat_memory()
        else:
            logger.info("Memory sync method not available, skipping")

class LuminaGUI(QMainWindow):
    """Main GUI window for Lumina"""
    def __init__(self):
        super().__init__()
        
        self.state = LuminaState()
        # Remove the non-existent method call
        # self.state._load_memory()
        
        # Check if load_ritual_invocations exists before calling
        if hasattr(self.state, 'load_ritual_invocations'):
            self.state.load_ritual_invocations()
        
        # Initialize auto-trainer
        self.auto_trainer = AutoTrainer(interval_minutes=30)  # Train every 30 minutes if new data
        self.auto_trainer.status_signal.connect(self.update_training_status)
        self.auto_trainer.training_complete_signal.connect(self.handle_training_complete)
        self.auto_trainer.start()
        
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("LUMINA")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left sidebar with icons
        sidebar = QWidget()
        sidebar.setFixedWidth(60)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)
        
        # Sidebar buttons
        profile_btn = QPushButton("👤")
        star_btn = QPushButton("⭐")
        settings_btn = QPushButton("⚙️")
        scroll_btn = QPushButton("📜")
        model_btn = QPushButton("🧠")
        llm_btn = QPushButton("🔄")
        memory_btn = QPushButton("💭")
        
        for btn in [profile_btn, star_btn, settings_btn, scroll_btn, model_btn, llm_btn, memory_btn]:
            btn.setFixedSize(40, 40)
            sidebar_layout.addWidget(btn)
            
        sidebar_layout.addStretch(1)
        
        # Main content area
        self.content_layout = QHBoxLayout()
        
        # Chat area
        self.chat_box = ChatBox(state=self.state)
        
        # Right sidebar with panels
        right_sidebar = QWidget()
        right_sidebar.setFixedWidth(200)
        right_sidebar_layout = QVBoxLayout(right_sidebar)
        
        # Create panels
        self.process_panel = ProcessPanel()
        self.glyphs_panel = GlyphsPanel()
        self.model_panel = ModelPanel()
        self.llm_panel = LLMWeightPanel()
        self.memory_panel = MemoryPanel(self.state)
        
        # Connect signals
        self.glyphs_panel.glyph_selected.connect(self.activate_glyph)
        model_btn.clicked.connect(self.toggle_model_panel)
        self.model_panel.model_selected.connect(self.set_active_model)
        llm_btn.clicked.connect(self.toggle_llm_panel)
        self.llm_panel.weight_changed.connect(self.update_llm_weight)
        self.llm_panel.llm_toggled.connect(self.toggle_llm)
        memory_btn.clicked.connect(self.toggle_memory_panel)
        self.memory_panel.memory_selected.connect(self.use_memory_context)
        
        # Initially add process and glyphs panels
        right_sidebar_layout.addWidget(self.process_panel)
        right_sidebar_layout.addWidget(self.glyphs_panel)
        
        # Add the other panels but hide them initially
        right_sidebar_layout.addWidget(self.model_panel)
        right_sidebar_layout.addWidget(self.llm_panel)
        right_sidebar_layout.addWidget(self.memory_panel)
        self.model_panel.hide()
        self.llm_panel.hide()
        self.memory_panel.hide()
        
        # Create the splitter for resizable layout
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(self.chat_box)
        content_splitter.addWidget(right_sidebar)
        content_splitter.setSizes([600, 200])
        
        # Add all main components to layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_splitter)
        
        self.setCentralWidget(main_widget)
        
        # Add welcome message
        self.chat_box.add_message(False, "Welcome. I am listening to the resonance. Tell me more.")
        
    def activate_glyph(self, glyph):
        """Handle glyph activation"""
        self.state.symbolic_state = glyph
        self.chat_box.add_message(False, f"Fire glyph activated. Channeling passion and truth.")
        
    def toggle_model_panel(self):
        """Toggle the visibility of the model panel"""
        # Hide all panels first
        self.process_panel.hide()
        self.glyphs_panel.hide()
        self.model_panel.hide()
        self.llm_panel.hide()
        self.memory_panel.hide()
        
        # Then show only the model panel
        self.model_panel.show()
            
    def toggle_llm_panel(self):
        """Toggle the visibility of the LLM panel"""
        # Hide all panels first
        self.process_panel.hide()
        self.glyphs_panel.hide()
        self.model_panel.hide()
        self.llm_panel.hide()
        self.memory_panel.hide()
        
        # Then show only the LLM panel
        self.llm_panel.show()
        
    def toggle_memory_panel(self):
        """Toggle the visibility of the memory panel"""
        # Hide all panels first
        self.process_panel.hide()
        self.glyphs_panel.hide()
        self.model_panel.hide()
        self.llm_panel.hide()
        self.memory_panel.hide()
        
        # Then show only the memory panel
        self.memory_panel.show()
        
    def toggle_process_glyphs_panels(self):
        """Show the process and glyphs panels (default view)"""
        # Hide all panels first
        self.process_panel.hide()
        self.glyphs_panel.hide()
        self.model_panel.hide()
        self.llm_panel.hide()
        self.memory_panel.hide()
        
        # Then show process and glyphs panels
        self.process_panel.show()
        self.glyphs_panel.show()
        
    def set_active_model(self, model_path):
        """Set the active model for Lumina"""
        try:
            # Here we would load the selected model
            # This would depend on how your system handles model loading
            self.chat_box.add_message(False, f"Switching to model: {os.path.basename(model_path)}")
            # In a real implementation, you might update the central node or neural system
            # self.state.central_node.load_model(model_path)
        except Exception as e:
            self.chat_box.add_message(False, f"Error loading model: {str(e)}")
            
    def update_llm_weight(self, weight):
        """Update the LLM weight in the state"""
        self.state.set_llm_weight(weight)
        self.chat_box.add_message(False, f"LLM weight set to {weight:.2f}")
        
    def toggle_llm(self, enabled):
        """Toggle LLM integration on/off"""
        self.state.set_llm_enabled(enabled)
        status = "enabled" if enabled else "disabled"
        self.chat_box.add_message(False, f"LLM integration {status}")

    def use_memory_context(self, context):
        """Use the selected memory as context for the next conversation"""
        if context:
            # Add a system message indicating the context is being used
            self.chat_box.add_message(False, f"I'm recalling our conversation about this topic.")
            
            # Set the input field with a prompt
            if hasattr(self.chat_box, 'input_field'):
                self.chat_box.input_field.setFocus()
                
    def update_training_status(self, status: str):
        """Update status based on auto-trainer messages"""
        logger.info(f"Auto-trainer: {status}")
        # If model panel is visible, update its status
        if hasattr(self, 'model_panel') and self.model_panel.isVisible():
            self.model_panel.status_label.setText(f"Auto: {status}")
            
    def handle_training_complete(self, success: bool, message: str):
        """Handle training completion"""
        if success:
            logger.info(f"Auto-training completed: {message}")
            # Refresh model list if model panel exists
            if hasattr(self, 'model_panel'):
                self.model_panel.refreshModels()
        else:
            logger.error(f"Auto-training failed: {message}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop auto-trainer when closing
        if hasattr(self, 'auto_trainer') and self.auto_trainer is not None:
            logger.info("Stopping auto-trainer thread...")
            self.auto_trainer.stop()
            # Wait for the thread to finish with a timeout
            if self.auto_trainer.isRunning():
                logger.info("Waiting for auto-trainer thread to finish...")
                if not self.auto_trainer.wait(3000):  # 3 second timeout
                    logger.warning("Auto-trainer thread did not finish in time, terminating...")
                    self.auto_trainer.terminate()
                    self.auto_trainer.wait()
            logger.info("Auto-trainer thread stopped")
        
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set stylesheet for a dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #1E1E1E;
            color: #EEEEEE;
        }
        QFrame {
            border: 1px solid #3E3E3E;
            border-radius: 4px;
        }
        QLineEdit, QTextEdit {
            background-color: #2D2D2D;
            border: 1px solid #3E3E3E;
            border-radius: 4px;
            padding: 8px;
            color: #EEEEEE;
        }
        QPushButton {
            background-color: #0078D7;
            color: white;
            border-radius: 4px;
            padding: 8px;
        }
        QPushButton:hover {
            background-color: #1C8AE6;
        }
        QPushButton:pressed {
            background-color: #00559A;
        }
        #chatMessage {
            background-color: #2D2D2D;
            padding: 10px;
            margin: 5px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3E3E3E;
            height: 8px;
            background: #2D2D2D;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0078D7;
            border: 1px solid #0078D7;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #3E3E3E;
            background-color: #2D2D2D;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #0078D7;
            background-color: #0078D7;
            border-radius: 3px;
        }
    """)
    
    window = LuminaGUI()
    window.show()
    
    sys.exit(app.exec_()) 