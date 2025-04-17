"""
Mistral API Integration for LUMINA v7.5 with Neural Network and Spiderweb Architecture
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import mistralai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    'model_path': 'models/mistral-local',  # Local model path
    'use_local': True,
    'fallback_mode': True  # Enable fallback mode for testing
}

class NeuralWeightingNetwork(nn.Module):
    def __init__(self, input_size: int = 1024):
        """Initialize neural network for weighting parameters
        
        Args:
            input_size: Size of input embeddings (default 1024 for Mistral)
        """
        super().__init__()
        self.input_size = input_size
        
        # Use device with CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        ).to(self.device)
        
        # Parameter prediction heads
        self.temperature_head = nn.Linear(128, 1).to(self.device)
        self.top_p_head = nn.Linear(128, 1).to(self.device)
        self.top_k_head = nn.Linear(128, 1).to(self.device)
        self.llm_weight_head = nn.Linear(128, 1).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        logger.info(f"Initialized NeuralWeightingNetwork with input size {input_size}")
    
    def forward(self, x: torch.Tensor) -> Tuple[float, float, int, float]:
        """Forward pass to compute parameter weights
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (temperature, top_p, top_k, llm_weight)
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Resize input if needed using adaptive pooling
        if x.shape[-1] != self.input_size:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.input_size).squeeze(1)
        
        # Encode input
        encoded = self.encoder(x)
        
        # Compute parameters with appropriate bounds
        temperature = torch.sigmoid(self.temperature_head(encoded)) * 2.0  # Range [0, 2]
        top_p = torch.sigmoid(self.top_p_head(encoded))  # Range [0, 1]
        top_k = torch.sigmoid(self.top_k_head(encoded)) * 100  # Range [0, 100]
        llm_weight = torch.sigmoid(self.llm_weight_head(encoded))  # Range [0, 1]
        
        return (
            temperature.item(),
            top_p.item(),
            int(top_k.item()),
            llm_weight.item()
        )

class MemoryNode:
    """Memory node for storing conversation and context data"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.creation_time = datetime.now().isoformat()
        self.last_access = self.creation_time
        self.data: Dict[str, Any] = {}
        self.connections: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.neural_state: Dict[str, Any] = {}
        
    def update(self, data: Dict[str, Any]) -> None:
        """Update node data"""
        self.data.update(data)
        self.last_access = datetime.now().isoformat()
        
    def connect(self, node_id: str) -> None:
        """Connect to another node"""
        if node_id not in self.connections:
            self.connections.append(node_id)
            
    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """Set node embeddings"""
        self.embeddings = embeddings
        
    def update_neural_state(self, state: Dict[str, Any]) -> None:
        """Update neural state information"""
        self.neural_state.update(state)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        data = {
            "node_id": self.node_id,
            "creation_time": self.creation_time,
            "last_access": self.last_access,
            "data": self.data,
            "connections": self.connections,
            "metadata": self.metadata,
            "neural_state": self.neural_state
        }
        if self.embeddings is not None:
            data["embeddings"] = self.embeddings.tolist()
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create node from dictionary"""
        node = cls(data["node_id"])
        node.creation_time = data["creation_time"]
        node.last_access = data["last_access"]
        node.data = data["data"]
        node.connections = data["connections"]
        node.metadata = data["metadata"]
        node.neural_state = data.get("neural_state", {})
        if "embeddings" in data:
            node.embeddings = np.array(data["embeddings"])
        return node

class MistralIntegration:
    """Integration with Mistral AI for chat functionality"""
    
    def __init__(self, model_name: str = "mistral-medium"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.memory = {}
        self.conversations = {}
        self._init_model()
        
    def _init_model(self):
        """Initialize the Mistral model and tokenizer"""
        try:
            if MODEL_CONFIG['use_local'] and MODEL_CONFIG['fallback_mode']:
                self.logger.info("Using fallback mode for testing")
                return True
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.logger.info(f"Initialized Mistral model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Mistral model: {str(e)}")
            if MODEL_CONFIG['fallback_mode']:
                self.logger.info("Falling back to test mode")
                return True
            return False
            
    def start_conversation(self) -> str:
        """Start a new conversation"""
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversations[conversation_id] = []
        return conversation_id
        
    def process_message(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process a message through the Mistral model"""
        try:
            # Prepare input for the model
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Generate response
            outputs = self.model.generate(
                inputs,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {str(e)}")
            raise
            
    def create_memory_node(self, data: Dict[str, Any]) -> str:
        """Create a new memory node"""
        node_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory[node_id] = {
            'id': node_id,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        return node_id
        
    def search_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search the memory system"""
        results = []
        for node_id, node_data in self.memory.items():
            # Simple keyword matching for now
            if any(query.get('keywords', []) in str(value) for value in node_data['data'].values()):
                results.append(node_data)
        return results
        
    def adjust_parameters(self, **kwargs):
        """Adjust model parameters"""
        valid_params = ['temperature', 'top_p', 'max_new_tokens']
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
                
    def save_neural_network(self, path: str):
        """Save the neural network state"""
        try:
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Saved neural network state to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save neural network state: {str(e)}")
            raise
            
    def load_neural_network(self, path: str):
        """Load the neural network state"""
        try:
            self.model.load_state_dict(torch.load(path))
            self.logger.info(f"Loaded neural network state from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load neural network state: {str(e)}")
            raise
            
    def save_memory(self):
        """Save the memory system"""
        try:
            with open('memory.json', 'w') as f:
                json.dump(self.memory, f)
            self.logger.info("Saved memory system")
        except Exception as e:
            self.logger.error(f"Failed to save memory system: {str(e)}")
            raise
            
    def load_memory(self):
        """Load the memory system"""
        try:
            if os.path.exists('memory.json'):
                with open('memory.json', 'r') as f:
                    self.memory = json.load(f)
                self.logger.info("Loaded memory system")
        except Exception as e:
            self.logger.error(f"Failed to load memory system: {str(e)}")
            raise

    def update_neural_weights(self, features: torch.Tensor) -> Tuple[float, float, float, float]:
        """Update weights using neural network"""
        self.neural_network.eval()
        with torch.no_grad():
            weights = self.neural_network(features)
            
        # Extract weights
        llm_w, nn_w, temp, top_p = weights.cpu().numpy()
        
        # Update parameters
        self.llm_weight = float(llm_w)
        self.nn_weight = float(nn_w)
        self.temperature = float(temp)
        self.top_p = float(top_p)
        
        return self.llm_weight, self.nn_weight, self.temperature, self.top_p
        
    def train_neural_network(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Train the neural network"""
        self.neural_network.train()
        self.optimizer.zero_grad()
        
        outputs = self.neural_network(features)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text using Mistral API
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embedding vector
        """
        try:
            # Call Mistral API to get embedding
            response = self.client.embeddings(
                model=self.model_name,
                input=text
            )
            
            if not response or not response.data or not response.data[0].embedding:
                raise ValueError("No embedding returned from API")
                
            embedding = np.array(response.data[0].embedding)
            
            # Resize embedding if needed
            if embedding.shape[0] != self.neural_network.input_size:
                logger.info(f"Resizing embedding from {embedding.shape[0]} to {self.neural_network.input_size}")
                embedding = self.resize_embedding(embedding, self.neural_network.input_size)
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing embedding: {str(e)}")
            # Return zero vector of correct size as fallback
            return np.zeros(self.neural_network.input_size)
            
    def resize_embedding(self, embedding: np.ndarray, target_size: int) -> np.ndarray:
        """Resize embedding vector to target size using interpolation
        
        Args:
            embedding: Original embedding vector
            target_size: Desired size
            
        Returns:
            Resized embedding vector
        """
        # Reshape to 2D for interpolation
        orig_shape = embedding.shape
        embedding_2d = embedding.reshape(1, -1)
        
        # Use linear interpolation to resize
        x = np.linspace(0, 1, embedding_2d.shape[1])
        x_new = np.linspace(0, 1, target_size)
        
        resized = np.array([np.interp(x_new, x, embedding_2d[0])])
        
        # Normalize to preserve magnitude
        orig_norm = np.linalg.norm(embedding)
        resized = resized * (orig_norm / np.linalg.norm(resized))
        
        return resized.reshape(target_size)

    def update_neural_state(self, message: str, response: str) -> Dict[str, float]:
        """Update neural state based on message and response"""
        # Compute embeddings
        msg_embedding = self.compute_embedding(message)
        resp_embedding = self.compute_embedding(response)
        
        # Update neural state
        self.neural_state["resonance"] = float(np.dot(msg_embedding, resp_embedding))
        self.neural_state["coherence"] = float(np.mean(resp_embedding))
        self.neural_state["engagement"] = float(np.max(resp_embedding))
        self.neural_state["complexity"] = float(np.std(resp_embedding))
        
        return self.neural_state
        
    def get_conversation_history(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            return []
            
        conv_nodes = [
            node for node in self.memory_nodes.values()
            if node.data.get("type") == "conversation" 
            and node.data.get("conversation_id") == conv_id
        ]
        
        if not conv_nodes:
            return []
            
        conv_node = conv_nodes[0]
        message_nodes = [
            self.memory_nodes[msg_id].data
            for msg_id in conv_node.data.get("messages", [])
            if msg_id in self.memory_nodes
        ]
        
        return message_nodes
        
    def search_memory(self, query: Dict[str, Any]) -> List[MemoryNode]:
        """Search memory nodes matching query"""
        results = []
        for node in self.memory_nodes.values():
            match = True
            for key, value in query.items():
                if key not in node.data or node.data[key] != value:
                    match = False
                    break
            if match:
                results.append(node)
        return results 
    def process_message(self, conversation: List[Dict[str, str]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a message using the Mistral API with neural network integration"""
        try:
            # Start new conversation if needed
            if not self.current_conversation_id:
                self.start_conversation()
                
            # Update conversation history
            self.conversation_history.extend(conversation)
            
            # Get latest message
            latest_message = conversation[-1]["content"]
            
            # Compute message embeddings
            embeddings = self.compute_embedding(latest_message)
            
            # Update neural weights
            features = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            self.update_neural_weights(features)
            
            # Convert conversation to ChatMessage format
            messages = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in conversation
            ]
            
            # Add context and neural state
            if context or self.neural_state:
                system_context = {
                    "context": context or {},
                    "neural_state": self.neural_state,
                    "weights": {
                        "llm": self.llm_weight,
                        "nn": self.nn_weight
                    }
                }
                messages.insert(0, ChatMessage(
                    role="system",
                    content=f"Context: {json.dumps(system_context)}"
                ))
            
            # Call Mistral API
            chat_response = self.client.chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            
            # Extract response
            if chat_response and chat_response.choices:
                response_text = chat_response.choices[0].message.content
                
                # Update neural state
                neural_state = self.update_neural_state(latest_message, response_text)
                
                # Create memory node for response
                response_node_id = self.create_memory_node({
                    "type": "message",
                    "conversation_id": self.current_conversation_id,
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "context": context,
                    "neural_state": neural_state,
                    "parameters": {
                        "llm_weight": self.llm_weight,
                        "nn_weight": self.nn_weight,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k
                    }
                })
                
                # Update conversation node
                conv_nodes = [
                    node for node in self.memory_nodes.values()
                    if node.data.get("type") == "conversation" 
                    and node.data.get("conversation_id") == self.current_conversation_id
                ]
                if conv_nodes:
                    conv_node = conv_nodes[0]
                    conv_node.data["messages"].append(response_node_id)
                    conv_node.update_neural_state(neural_state)
                    self.save_memory()
                
                logger.debug(f"Generated response: {response_text[:100]}...")
                return {
                    "response": response_text,
                    "conversation_id": self.current_conversation_id,
                    "memory_node_id": response_node_id,
                    "timestamp": datetime.now().isoformat(),
                    "neural_state": neural_state,
                    "parameters": {
                        "llm_weight": self.llm_weight,
                        "nn_weight": self.nn_weight,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k
                    }
                }
            else:
                raise ValueError("No response generated from Mistral API")
                
        except Exception as e:
            logger.error(f"Error processing message with Mistral: {str(e)}")
            raise
            
    def adjust_parameters(self, llm_weight: float = None, nn_weight: float = None,
                        temperature: float = None, top_p: float = None, 
                        top_k: int = None, max_tokens: int = None):
        """Adjust all parameters"""
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, llm_weight))
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, nn_weight))
        if temperature is not None:
            self.temperature = max(0.0, min(2.0, temperature))
        if top_p is not None:
            self.top_p = max(0.0, min(1.0, top_p))
        if top_k is not None:
            self.top_k = max(1, min(100, top_k))
        if max_tokens is not None:
            self.max_tokens = max(1, min(4096, max_tokens))
            
        logger.debug(
            f"Adjusted parameters: llm_w={self.llm_weight}, nn_w={self.nn_weight}, "
            f"temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, "
            f"max_tokens={self.max_tokens}"
        )

    def save_neural_network(self, path: str = "data/neural_network.pt"):
        """Save neural network state"""
        torch.save({
            'model_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_neural_network(self, path: str = "data/neural_network.pt"):
        """Load neural network state"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.neural_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def create_memory_node(self, data: Dict[str, Any], connections: List[str] = None) -> str:
        """Create a new memory node"""
        node_id = f"node_{int(time.time() * 1000)}"
        node = MemoryNode(node_id)
        node.update(data)
        if connections:
            for conn in connections:
                node.connect(conn)
                if conn in self.memory_nodes:
                    self.memory_nodes[conn].connect(node_id)
        
        self.memory_nodes[node_id] = node
        self.save_memory()
        return node_id
        
    def get_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a memory node by ID"""
        return self.memory_nodes.get(node_id)
        
    def update_memory_node(self, node_id: str, data: Dict[str, Any]) -> bool:
        """Update a memory node's data"""
        if node_id in self.memory_nodes:
            self.memory_nodes[node_id].update(data)
            self.save_memory()
            return True
        return False
        
    def connect_memory_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Create a bidirectional connection between two memory nodes"""
        if node_id1 in self.memory_nodes and node_id2 in self.memory_nodes:
            self.memory_nodes[node_id1].connect(node_id2)
            self.memory_nodes[node_id2].connect(node_id1)
            self.save_memory()
            return True
        return False
        
    def save_memory(self) -> None:
        """Save memory nodes to disk"""
        memory_data = {
            node_id: node.to_dict() 
            for node_id, node in self.memory_nodes.items()
        }
        with open(self.memory_path / "memory.json", "w") as f:
            json.dump(memory_data, f, indent=2)
            
    def load_memory(self) -> None:
        """Load memory nodes from disk"""
        try:
            if (self.memory_path / "memory.json").exists():
                with open(self.memory_path / "memory.json", "r") as f:
                    memory_data = json.load(f)
                self.memory_nodes = {
                    node_id: MemoryNode.from_dict(node_data)
                    for node_id, node_data in memory_data.items()
                }
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            self.memory_nodes = {}
            
    def start_conversation(self) -> str:
        """Start a new conversation and return its ID"""
        conversation_id = str(uuid.uuid4())
        self.current_conversation_id = conversation_id
        self.memory_nodes[conversation_id] = MemoryNode(
            id=conversation_id,
            content="Conversation start",
            metadata={
                "type": "conversation_start",
                "timestamp": datetime.now().isoformat(),
                "neural_state": self.get_neural_state()
            }
        )
        return conversation_id
        
    def get_conversation_history(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            return []
            
        conv_nodes = [
            node for node in self.memory_nodes.values()
            if node.data.get("type") == "conversation" 
            and node.data.get("conversation_id") == conv_id
        ]
        
        if not conv_nodes:
            return []
            
        conv_node = conv_nodes[0]
        message_nodes = [
            self.memory_nodes[msg_id].data
            for msg_id in conv_node.data.get("messages", [])
            if msg_id in self.memory_nodes
        ]
        
        return message_nodes
        
    def search_memory(self, query: Dict[str, Any]) -> List[MemoryNode]:
        """Search memory nodes matching query"""
        results = []
        for node in self.memory_nodes.values():
            match = True
            for key, value in query.items():
                if key not in node.data or node.data[key] != value:
                    match = False
                    break
            if match:
                results.append(node)
        return results 
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, llm_weight))
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, nn_weight))
        if temperature is not None:
            self.temperature = max(0.0, min(2.0, temperature))
        if top_p is not None:
            self.top_p = max(0.0, min(1.0, top_p))
        if top_k is not None:
            self.top_k = max(1, min(100, top_k))
        if max_tokens is not None:
            self.max_tokens = max(1, min(4096, max_tokens))
            
        logger.debug(
            f"Adjusted parameters: llm_w={self.llm_weight}, nn_w={self.nn_weight}, "
            f"temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, "
            f"max_tokens={self.max_tokens}"
        )

    def save_neural_network(self, path: str = "data/neural_network.pt"):
        """Save neural network state"""
        torch.save({
            'model_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_neural_network(self, path: str = "data/neural_network.pt"):
        """Load neural network state"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.neural_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def create_memory_node(self, data: Dict[str, Any], connections: List[str] = None) -> str:
        """Create a new memory node"""
        node_id = f"node_{int(time.time() * 1000)}"
        node = MemoryNode(node_id)
        node.update(data)
        if connections:
            for conn in connections:
                node.connect(conn)
                if conn in self.memory_nodes:
                    self.memory_nodes[conn].connect(node_id)
        
        self.memory_nodes[node_id] = node
        self.save_memory()
        return node_id
        
    def get_memory_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a memory node by ID"""
        return self.memory_nodes.get(node_id)
        
    def update_memory_node(self, node_id: str, data: Dict[str, Any]) -> bool:
        """Update a memory node's data"""
        if node_id in self.memory_nodes:
            self.memory_nodes[node_id].update(data)
            self.save_memory()
            return True
        return False
        
    def connect_memory_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Create a bidirectional connection between two memory nodes"""
        if node_id1 in self.memory_nodes and node_id2 in self.memory_nodes:
            self.memory_nodes[node_id1].connect(node_id2)
            self.memory_nodes[node_id2].connect(node_id1)
            self.save_memory()
            return True
        return False
        
    def save_memory(self) -> None:
        """Save memory nodes to disk"""
        memory_data = {
            node_id: node.to_dict() 
            for node_id, node in self.memory_nodes.items()
        }
        with open(self.memory_path / "memory.json", "w") as f:
            json.dump(memory_data, f, indent=2)
            
    def load_memory(self) -> None:
        """Load memory nodes from disk"""
        try:
            if (self.memory_path / "memory.json").exists():
                with open(self.memory_path / "memory.json", "r") as f:
                    memory_data = json.load(f)
                self.memory_nodes = {
                    node_id: MemoryNode.from_dict(node_data)
                    for node_id, node_data in memory_data.items()
                }
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            self.memory_nodes = {}
            
    def start_conversation(self) -> str:
        """Start a new conversation and return its ID"""
        conversation_id = str(uuid.uuid4())
        self.current_conversation_id = conversation_id
        self.memory_nodes[conversation_id] = MemoryNode(
            id=conversation_id,
            content="Conversation start",
            metadata={
                "type": "conversation_start",
                "timestamp": datetime.now().isoformat(),
                "neural_state": self.get_neural_state()
            }
        )
        return conversation_id
        
    def get_conversation_history(self, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            return []
            
        conv_nodes = [
            node for node in self.memory_nodes.values()
            if node.data.get("type") == "conversation" 
            and node.data.get("conversation_id") == conv_id
        ]
        
        if not conv_nodes:
            return []
            
        conv_node = conv_nodes[0]
        message_nodes = [
            self.memory_nodes[msg_id].data
            for msg_id in conv_node.data.get("messages", [])
            if msg_id in self.memory_nodes
        ]
        
        return message_nodes
        
    def search_memory(self, query: Dict[str, Any]) -> List[MemoryNode]:
        """Search memory nodes matching query"""
        results = []
        for node in self.memory_nodes.values():
            match = True
            for key, value in query.items():
                if key not in node.data or node.data[key] != value:
                    match = False
                    break
            if match:
                results.append(node)
        return results 