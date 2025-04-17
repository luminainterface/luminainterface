#!/usr/bin/env python3
"""
LUMINA v7.5 Central Node
Core component that manages the Spiderweb architecture
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from .signal_component import SignalComponent
from .signal_system import SignalBus
from datetime import datetime
import json
from .mistral_integration import MistralIntegration
from .neural_weighting_network import NeuralWeightingNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"central_node_{os.getpid()}.log"))
    ]
)
logger = logging.getLogger("CentralNode")

class CentralNode(SignalComponent):
    """Central node for managing chat interactions and neural state"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-v0.1"):
        super().__init__("central_node", SignalBus())
        self.components: Dict[str, SignalComponent] = {}
        self._initialized = False
        
        # Initialize components
        self.mistral = MistralIntegration(model_path)
        self.neural_network = NeuralWeightingNetwork()
        
        # State management
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, list] = {}
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the central node"""
        try:
            if self._initialized:
                return
                
            # Initialize base class
            await super().initialize()
                
            # Register handlers
            self.register_handler("component_register", self._handle_component_register)
            self.register_handler("component_unregister", self._handle_component_unregister)
            self.register_handler("state_update", self._handle_state_update)
            self.register_handler("error", self._handle_error)
            
            self._initialized = True
            logger.info("Central node initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize central node: {e}")
            raise
            
    def register_component(self, component: SignalComponent) -> bool:
        """Register a component with the central node"""
        if not self._initialized:
            logger.error("Central node not initialized")
            return False
            
        if component.name in self.components:
            logger.warning(f"Component {component.name} already registered")
            return False
            
        self.components[component.name] = component
        logger.info(f"Component registered: {component.name}")
        
        # Broadcast component registration
        asyncio.create_task(self.emit_signal("component_registered", {
            "component": component.name,
            "state": getattr(component, 'state', {})
        }))
        
        return True
        
    def unregister_component(self, component_name: str) -> bool:
        """Unregister a component from the central node"""
        if not self._initialized:
            logger.error("Central node not initialized")
            return False
            
        if component_name not in self.components:
            logger.warning(f"Component {component_name} not registered")
            return False
            
        del self.components[component_name]
        logger.info(f"Component unregistered: {component_name}")
        
        # Broadcast component unregistration
        asyncio.create_task(self.emit_signal("component_unregistered", {
            "component": component_name
        }))
        
        return True
        
    def get_component(self, component_name: str) -> Optional[SignalComponent]:
        """Get a registered component by name"""
        return self.components.get(component_name)
        
    def get_all_components(self) -> Dict[str, SignalComponent]:
        """Get all registered components"""
        return self.components.copy()
        
    async def _handle_component_register(self, data: Any) -> None:
        """Handle component registration"""
        if not isinstance(data, dict) or "component" not in data:
            logger.error("Invalid component registration data")
            return
            
        component = data["component"]
        if not isinstance(component, SignalComponent):
            logger.error("Invalid component type")
            return
            
        self.register_component(component)
        
    async def _handle_component_unregister(self, data: Any) -> None:
        """Handle component unregistration"""
        if not isinstance(data, dict) or "component" not in data:
            logger.error("Invalid component unregistration data")
            return
            
        component_name = data["component"]
        if not isinstance(component_name, str):
            logger.error("Invalid component name type")
            return
            
        self.unregister_component(component_name)
        
    async def _handle_state_update(self, data: Any) -> None:
        """Handle component state updates"""
        if not isinstance(data, dict):
            logger.error("Invalid state update data")
            return
            
        component = data.get("component")
        state = data.get("state")
        
        if not component or not state:
            logger.error("Missing component or state in update")
            return
            
        # Update component state in central node
        if component in self.components:
            comp = self.components[component]
            if hasattr(comp, 'state'):
                comp.state.update(state)
                logger.info(f"Updated state for component {component}")
            
    async def _handle_error(self, data: Any) -> None:
        """Handle error messages"""
        if not isinstance(data, dict):
            logger.error("Invalid error data")
            return
            
        component = data.get("component")
        error = data.get("error")
        
        if not component or not error:
            logger.error("Missing component or error in error message")
            return
            
        logger.error(f"Error in component {component}: {error}")
        
    async def cleanup(self) -> None:
        """Clean up central node resources"""
        # Clean up all components
        for component in list(self.components.values()):
            await component.cleanup()
            
        self.components.clear()
        await super().cleanup()
        logger.info("Central node cleaned up")
        
    def start_conversation(self) -> str:
        """Start a new conversation"""
        conversation_id = self.mistral.start_conversation()
        self.active_conversations[conversation_id] = {
            'start_time': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'message_count': 0
        }
        self.conversation_history[conversation_id] = []
        return conversation_id
        
    def process_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Process a message in a conversation"""
        try:
            if conversation_id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
                
            # Update conversation state
            self.active_conversations[conversation_id]['last_activity'] = datetime.now().isoformat()
            self.active_conversations[conversation_id]['message_count'] += 1
            
            # Get current neural state
            neural_state = self.neural_network.get_neural_state()
            
            # Process message with Mistral
            response = self.mistral.process_message(
                conversation_id,
                message,
                temperature=neural_state['temperature'],
                top_p=neural_state['top_p']
            )
            
            # Update neural state based on interaction
            self.neural_network.update_neural_state(message, response['response'])
            
            # Store in conversation history
            self.conversation_history[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'user_message': message,
                'system_response': response['response'],
                'neural_state': neural_state
            })
            
            return {
                'response': response['response'],
                'conversation_id': conversation_id,
                'neural_state': neural_state
            }
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise
            
    def end_conversation(self, conversation_id: str):
        """End a conversation"""
        try:
            if conversation_id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
                
            # Save conversation history
            self._save_conversation_history(conversation_id)
            
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
        except Exception as e:
            logger.error(f"Failed to end conversation: {str(e)}")
            raise
            
    def get_conversation_history(self, conversation_id: str) -> list:
        """Get conversation history"""
        if conversation_id not in self.conversation_history:
            raise ValueError(f"Conversation {conversation_id} not found")
        return self.conversation_history[conversation_id]
        
    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Get active conversations"""
        return self.active_conversations.copy()
        
    def adjust_parameters(self, params: Dict[str, float]):
        """Adjust model parameters"""
        try:
            self.mistral.adjust_parameters(params)
            logger.info(f"Adjusted parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to adjust parameters: {str(e)}")
            raise
            
    def save_state(self):
        """Save system state"""
        try:
            # Save neural network state
            self.neural_network.save_state("data/neural_network_state.pt")
            
            # Save conversation histories
            for conv_id, history in self.conversation_history.items():
                self._save_conversation_history(conv_id)
                
            logger.info("System state saved")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {str(e)}")
            raise
            
    def load_state(self):
        """Load system state"""
        try:
            # Load neural network state if it exists
            if os.path.exists("data/neural_network_state.pt"):
                self.neural_network.load_state("data/neural_network_state.pt")
                
            # Load conversation histories
            for filename in os.listdir("data"):
                if filename.startswith("conversation_") and filename.endswith(".json"):
                    conv_id = filename.split("_")[1].split(".")[0]
                    with open(f"data/{filename}", "r") as f:
                        self.conversation_history[conv_id] = json.load(f)
                        
            logger.info("System state loaded")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {str(e)}")
            raise
            
    def _save_conversation_history(self, conversation_id: str):
        """Save conversation history to file"""
        try:
            filename = f"data/conversation_{conversation_id}.json"
            with open(filename, "w") as f:
                json.dump(self.conversation_history[conversation_id], f)
            logger.info(f"Saved conversation history to {filename}")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {str(e)}")
            raise 
        logger.info("Central node cleaned up") 
        if not component or not error:
            logger.error("Missing component or error in error message")
            return
            
        logger.error(f"Error in component {component}: {error}")
        
    async def cleanup(self) -> None:
        """Clean up central node resources"""
        # Clean up all components
        for component in list(self.components.values()):
            await component.cleanup()
            
        self.components.clear()
        await super().cleanup()
        logger.info("Central node cleaned up")
        
    def start_conversation(self) -> str:
        """Start a new conversation"""
        conversation_id = self.mistral.start_conversation()
        self.active_conversations[conversation_id] = {
            'start_time': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'message_count': 0
        }
        self.conversation_history[conversation_id] = []
        return conversation_id
        
    def process_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Process a message in a conversation"""
        try:
            if conversation_id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
                
            # Update conversation state
            self.active_conversations[conversation_id]['last_activity'] = datetime.now().isoformat()
            self.active_conversations[conversation_id]['message_count'] += 1
            
            # Get current neural state
            neural_state = self.neural_network.get_neural_state()
            
            # Process message with Mistral
            response = self.mistral.process_message(
                conversation_id,
                message,
                temperature=neural_state['temperature'],
                top_p=neural_state['top_p']
            )
            
            # Update neural state based on interaction
            self.neural_network.update_neural_state(message, response['response'])
            
            # Store in conversation history
            self.conversation_history[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'user_message': message,
                'system_response': response['response'],
                'neural_state': neural_state
            })
            
            return {
                'response': response['response'],
                'conversation_id': conversation_id,
                'neural_state': neural_state
            }
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            raise
            
    def end_conversation(self, conversation_id: str):
        """End a conversation"""
        try:
            if conversation_id not in self.active_conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
                
            # Save conversation history
            self._save_conversation_history(conversation_id)
            
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
        except Exception as e:
            logger.error(f"Failed to end conversation: {str(e)}")
            raise
            
    def get_conversation_history(self, conversation_id: str) -> list:
        """Get conversation history"""
        if conversation_id not in self.conversation_history:
            raise ValueError(f"Conversation {conversation_id} not found")
        return self.conversation_history[conversation_id]
        
    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Get active conversations"""
        return self.active_conversations.copy()
        
    def adjust_parameters(self, params: Dict[str, float]):
        """Adjust model parameters"""
        try:
            self.mistral.adjust_parameters(params)
            logger.info(f"Adjusted parameters: {params}")
        except Exception as e:
            logger.error(f"Failed to adjust parameters: {str(e)}")
            raise
            
    def save_state(self):
        """Save system state"""
        try:
            # Save neural network state
            self.neural_network.save_state("data/neural_network_state.pt")
            
            # Save conversation histories
            for conv_id, history in self.conversation_history.items():
                self._save_conversation_history(conv_id)
                
            logger.info("System state saved")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {str(e)}")
            raise
            
    def load_state(self):
        """Load system state"""
        try:
            # Load neural network state if it exists
            if os.path.exists("data/neural_network_state.pt"):
                self.neural_network.load_state("data/neural_network_state.pt")
                
            # Load conversation histories
            for filename in os.listdir("data"):
                if filename.startswith("conversation_") and filename.endswith(".json"):
                    conv_id = filename.split("_")[1].split(".")[0]
                    with open(f"data/{filename}", "r") as f:
                        self.conversation_history[conv_id] = json.load(f)
                        
            logger.info("System state loaded")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {str(e)}")
            raise
            
    def _save_conversation_history(self, conversation_id: str):
        """Save conversation history to file"""
        try:
            filename = f"data/conversation_{conversation_id}.json"
            with open(filename, "w") as f:
                json.dump(self.conversation_history[conversation_id], f)
            logger.info(f"Saved conversation history to {filename}")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {str(e)}")
            raise 
        logger.info("Central node cleaned up") 