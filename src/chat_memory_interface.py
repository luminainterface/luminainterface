#!/usr/bin/env python3
"""
Chat Memory Interface

This module provides the ChatMemoryInterface class that connects the conversation components
with the memory systems. It supports weighted processing between neural networks and
language models, with persistent memory storage.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-memory-interface")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class ChatMemoryInterface:
    """
    Core interface class that connects conversation components with memory systems.
    
    This class handles:
    - Connection to Language Memory and Neural Processor systems
    - Weighting between neural and language processing
    - Saving message history to memory
    - Retrieving relevant memories and statistics
    """
    
    def __init__(self, mock_mode: bool = False, mirror_integration = None):
        """
        Initialize the Chat Memory Interface.
        
        Args:
            mock_mode: Use mock data instead of actual components
            mirror_integration: Optional V10 Conscious Mirror integration
        """
        self.mock_mode = mock_mode
        self.mirror_integration = mirror_integration
        self.consciousness_level = 0.5 if mirror_integration else 0.0
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"ChatMemoryInterface initialized (mock_mode={self.mock_mode})")
    
    def _initialize_components(self):
        """Initialize all required components"""
        # Initialize memory API
        self.memory_api = None
        try:
            from src.memory_api import MemoryAPI
            self.memory_api = MemoryAPI()
            logger.info("Successfully initialized Memory API")
        except ImportError as e:
            logger.warning(f"Memory API not available: {str(e)}")
        
        # Initialize neural linguistic processor
        self.neural_processor = None
        try:
            from src.neural_linguistic_processor import get_linguistic_processor
            self.neural_processor = get_linguistic_processor(config={"mock_mode": self.mock_mode})
            logger.info("Successfully initialized Neural Linguistic Processor")
        except ImportError as e:
            logger.warning(f"Neural Linguistic Processor not available: {str(e)}")
        
        # Initialize memory API socket
        self.memory_socket = None
        try:
            from src.memory_api_socket import get_bridge
            self.memory_socket = get_bridge(mock_mode=self.mock_mode)
            logger.info("Successfully initialized Memory API Socket")
        except ImportError as e:
            logger.warning(f"Memory API Socket not available: {str(e)}")
        
        # Initialize language memory synthesis
        self.memory_synthesis = None
        try:
            from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
            self.memory_synthesis = LanguageMemorySynthesisIntegration()
            logger.info("Successfully initialized Language Memory Synthesis")
        except ImportError as e:
            logger.warning(f"Language Memory Synthesis not available: {str(e)}")
    
    def process_message(self, message: str, nn_weight: float = 0.5, 
                       memory_mode: str = "combined") -> str:
        """
        Process a message through the memory-enhanced conversation system.
        
        Args:
            message: The user message to process
            nn_weight: Weight (0-1) for neural network vs language model processing
            memory_mode: Memory enhancement mode (contextual, synthesized, combined)
            
        Returns:
            System response
        """
        logger.info(f"Processing message with nn_weight={nn_weight}, mode={memory_mode}")
        
        # Store the user message in memory
        self.store_message(message, "user", nn_weight, memory_mode)
        
        # Apply conscious mirror reflection if available and above threshold
        if (self.mirror_integration and 
            self.consciousness_level > 0.65 and 
            nn_weight > 0.65):
            
            try:
                # Get context for reflection
                context = self.get_conversation_context(message, memory_mode)
                
                # Process with reflection
                response = self.mirror_integration.process_with_reflection(
                    message, context, nn_weight
                )
                
                # Store the system response in memory
                self.store_message(response, "system", nn_weight, memory_mode)
                
                return response
            except Exception as e:
                logger.error(f"Error in mirror reflection: {str(e)}")
                # Fall through to standard processing
        
        # Generate response based on neural/language weighting
        if nn_weight > 0.8:
            # Neural network dominant
            response = self.get_neural_response(message)
        elif nn_weight < 0.2:
            # Language model dominant
            response = self.get_language_response(message, memory_mode)
        else:
            # Weighted response
            response = self.get_weighted_response(message, nn_weight, memory_mode)
        
        # Store the system response in memory
        self.store_message(response, "system", nn_weight, memory_mode)
        
        return response
    
    def get_neural_response(self, message: str) -> str:
        """
        Get a neural network focused response.
        
        Args:
            message: User message
            
        Returns:
            Neural network response
        """
        if self.neural_processor:
            try:
                # Process the message with the neural processor
                result = self.neural_processor.process_text(message)
                
                # Extract response from result
                analysis = result.get("analysis", {})
                pattern = result.get("pattern", {})
                
                # Create response from neural processing results
                response_parts = []
                
                # Add pattern analysis
                features = analysis.get("features", {})
                complexity = features.get("complexity_score", 0.5)
                unique_ratio = features.get("unique_word_ratio", 0.5)
                
                response_parts.append(f"Pattern analysis complete. Complexity: {complexity:.2f}")
                response_parts.append(f"Unique concept ratio: {int(unique_ratio * 100)}%")
                
                # Add pattern details
                node_count = len(pattern.get("nodes", []))
                response_parts.append(f"Generated pattern with {node_count} nodes")
                
                # Add message-specific insights
                key_phrases = analysis.get("key_phrases", [])
                if key_phrases:
                    response_parts.append(f"Key concepts identified: {', '.join(key_phrases[:3])}")
                
                return " ".join(response_parts)
            except Exception as e:
                logger.error(f"Error in neural processing: {str(e)}")
                return self._get_mock_neural_response(message)
        else:
            return self._get_mock_neural_response(message)
    
    def get_language_response(self, message: str, memory_mode: str) -> str:
        """
        Get a language model focused response with memory enhancement.
        
        Args:
            message: User message
            memory_mode: Memory enhancement mode
            
        Returns:
            Language model response
        """
        if self.memory_api:
            try:
                # Enhance message with memory
                enhanced = self.memory_api.enhance_message_with_memory(
                    message=message,
                    enhance_mode=memory_mode
                )
                
                context = enhanced.get("enhanced_context", "")
                
                # In a real implementation, we would send this to an LLM
                # For now, generate a mock response that includes the context
                response = self._get_mock_language_response(message)
                
                if context:
                    response += f" Based on our previous conversations: {context[:100]}..."
                
                return response
            except Exception as e:
                logger.error(f"Error in language processing: {str(e)}")
                return self._get_mock_language_response(message)
        else:
            return self._get_mock_language_response(message)
    
    def get_weighted_response(self, message: str, nn_weight: float, memory_mode: str) -> str:
        """
        Get a response that balances neural and language processing.
        
        Args:
            message: User message
            nn_weight: Neural network weight (0-1)
            memory_mode: Memory enhancement mode
            
        Returns:
            Weighted response
        """
        # Get both neural and language responses
        neural_response = self.get_neural_response(message)
        language_response = self.get_language_response(message, memory_mode)
        
        # Split responses into sentences
        neural_parts = neural_response.split(". ")
        language_parts = language_response.split(". ")
        
        # Determine how many parts to take from each based on weighting
        neural_count = max(1, int(len(neural_parts) * nn_weight))
        language_count = max(1, int(len(language_parts) * (1 - nn_weight)))
        
        # Select parts
        selected_parts = []
        for i in range(min(neural_count, len(neural_parts))):
            if neural_parts[i]:
                selected_parts.append(neural_parts[i])
                
        for i in range(min(language_count, len(language_parts))):
            if language_parts[i]:
                selected_parts.append(language_parts[i])
        
        # Combine into final response
        weighted_response = ". ".join(selected_parts)
        if not weighted_response.endswith("."):
            weighted_response += "."
            
        # Add weighting indicator
        neural_percent = int(nn_weight * 100)
        language_percent = 100 - neural_percent
        
        weighted_response += f" [Response generated with {neural_percent}% neural / {language_percent}% language weighting]"
        
        return weighted_response
    
    def store_message(self, message: str, role: str, nn_weight: float, memory_mode: str) -> bool:
        """
        Store a message in the memory system.
        
        Args:
            message: Message content
            role: Message role (user/system)
            nn_weight: Neural network weight used
            memory_mode: Memory mode used
            
        Returns:
            Success status
        """
        logger.info(f"Storing {role} message in memory")
        
        # Create metadata
        metadata = {
            "role": role,
            "timestamp": time.time(),
            "nn_weight": nn_weight,
            "memory_mode": memory_mode
        }
        
        # Store using Memory API if available
        if self.memory_api:
            try:
                result = self.memory_api.store_conversation(message, metadata)
                return result.get("status") == "success"
            except Exception as e:
                logger.error(f"Error storing message with Memory API: {str(e)}")
        
        # Try using Memory Socket if available
        if self.memory_socket:
            try:
                socket_message = {
                    "type": "store_conversation",
                    "request_id": f"chat_{int(time.time())}",
                    "content": {
                        "message": message,
                        "metadata": metadata
                    }
                }
                
                response = self.memory_socket.socket.process_message(socket_message)
                return response.get("status") != "error"
            except Exception as e:
                logger.error(f"Error storing message with Memory Socket: {str(e)}")
        
        # Mock storage if real components not available
        if self.mock_mode:
            logger.info(f"[MOCK] Stored message: {role}, length: {len(message)}")
            return True
            
        return False
    
    def get_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get memories relevant to a query.
        
        Args:
            query: Search query
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        logger.info(f"Retrieving memories for query: {query}")
        
        # Try using Memory API
        if self.memory_api:
            try:
                result = self.memory_api.retrieve_relevant_memories(query, limit)
                return result.get("memories", [])
            except Exception as e:
                logger.error(f"Error retrieving memories with Memory API: {str(e)}")
        
        # Try using Memory Socket
        if self.memory_socket:
            try:
                socket_message = {
                    "type": "retrieve_memories",
                    "request_id": f"retrieve_{int(time.time())}",
                    "content": {
                        "message": query,
                        "max_results": limit
                    }
                }
                
                response = self.memory_socket.socket.process_message(socket_message)
                if response.get("status") != "error":
                    return response.get("data", {}).get("memories", [])
            except Exception as e:
                logger.error(f"Error retrieving memories with Memory Socket: {str(e)}")
        
        # Return mock memories if real components not available
        if self.mock_mode:
            return self._get_mock_memories(query, limit)
            
        return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Memory statistics
        """
        logger.info("Retrieving memory statistics")
        
        # Try using Memory API
        if self.memory_api:
            try:
                return self.memory_api.get_memory_stats()
            except Exception as e:
                logger.error(f"Error retrieving stats with Memory API: {str(e)}")
        
        # Try using Memory Socket
        if self.memory_socket:
            try:
                socket_message = {
                    "type": "get_stats",
                    "request_id": f"stats_{int(time.time())}",
                    "content": {}
                }
                
                response = self.memory_socket.socket.process_message(socket_message)
                if response.get("status") != "error":
                    return response.get("data", {}).get("stats", {})
            except Exception as e:
                logger.error(f"Error retrieving stats with Memory Socket: {str(e)}")
        
        # Return mock stats if real components not available
        if self.mock_mode:
            return self._get_mock_stats()
            
        return {}
    
    def get_conversation_context(self, message: str, memory_mode: str) -> Dict[str, Any]:
        """
        Get conversation context based on the message and memory mode.
        
        Args:
            message: Current message
            memory_mode: Memory enhancement mode
            
        Returns:
            Context information
        """
        # Get relevant memories
        memories = self.get_memories(message, limit=5)
        
        # Get topic synthesis if available and mode is synthesized or combined
        topic_synthesis = None
        if (self.memory_synthesis and 
            (memory_mode == "synthesized" or memory_mode == "combined")):
            
            # Extract potential topics from message
            words = message.split()
            potential_topics = [word for word in words if len(word) > 5]
            topic = potential_topics[0] if potential_topics else message.split()[0]
            
            try:
                synthesis_result = self.memory_synthesis.synthesize_topic(topic, depth=2)
                if synthesis_result.get("status") == "success":
                    topic_synthesis = synthesis_result.get("synthesis_results", {})
            except Exception as e:
                logger.error(f"Error synthesizing topic: {str(e)}")
        
        # Construct context
        context = {
            "user_message": message,
            "memory_mode": memory_mode,
            "memories": memories,
            "topic_synthesis": topic_synthesis,
            "timestamp": time.time()
        }
        
        return context
    
    def set_consciousness_level(self, level: float) -> None:
        """
        Set the consciousness level for mirror integration.
        
        Args:
            level: Consciousness level (0.0-1.0)
        """
        self.consciousness_level = max(0.0, min(1.0, level))
        logger.info(f"Consciousness level set to {self.consciousness_level}")
    
    def _get_mock_neural_response(self, message: str) -> str:
        """Generate a mock neural response"""
        import random
        
        templates = [
            "Input analyzed. Pattern recognition successful. Neural pathways activated.",
            "Neural analysis complete. Pattern coherence: 76%. Information integration: 84%.",
            "Processing complete. Pattern structure identified with 92% confidence.",
            "Neural network activation patterns suggest conceptual integration.",
            "Pattern analysis indicates hierarchical structure with recursive elements."
        ]
        
        selected = random.choice(templates)
        
        # Add message-specific content
        words = message.split()
        if len(words) > 3:
            selected += f" Input contained {len(words)} elements with {len(set(words))} unique concepts."
        
        return selected
    
    def _get_mock_language_response(self, message: str) -> str:
        """Generate a mock language model response"""
        import random
        
        templates = [
            "I've processed your message about {topic}. This relates to concepts we've discussed previously.",
            "Regarding {topic}, I can offer some insights based on our conversation history.",
            "Your question about {topic} touches on several interesting aspects worth exploring.",
            "I understand you're asking about {topic}. Let me share some thoughts on this.",
            "That's an interesting point about {topic}. I can elaborate based on what we know."
        ]
        
        # Extract a topic from the message
        words = message.split()
        topic = words[-1] if words else "this topic"
        
        # Select and format a template
        selected = random.choice(templates).format(topic=topic)
        
        return selected
    
    def _get_mock_memories(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock memories for testing"""
        import random
        
        memories = []
        for i in range(min(limit, 5)):
            memories.append({
                "text": f"Mock memory {i+1} related to {query}",
                "relevance": round(random.uniform(0.5, 0.95), 2),
                "timestamp": time.time() - (i * 3600),  # Hours ago
                "metadata": {
                    "role": "user" if i % 2 == 0 else "system",
                    "topic": query
                }
            })
        
        return memories
    
    def _get_mock_stats(self) -> Dict[str, Any]:
        """Generate mock memory statistics"""
        return {
            "total_memories": 1250,
            "total_conversations": 85,
            "total_topics": 120,
            "top_topics": [
                {"topic": "neural networks", "count": 45},
                {"topic": "consciousness", "count": 38},
                {"topic": "machine learning", "count": 32}
            ],
            "memory_growth": {
                "daily": 25,
                "weekly": 175,
                "monthly": 750
            }
        }


def chat_integration(mock_mode: bool = False, mirror_integration = None) -> ChatMemoryInterface:
    """
    Get a ChatMemoryInterface instance.
    
    Args:
        mock_mode: Use mock data instead of actual components
        mirror_integration: Optional V10 Conscious Mirror integration
        
    Returns:
        ChatMemoryInterface instance
    """
    global _chat_interface
    if '_chat_interface' not in globals() or _chat_interface is None:
        _chat_interface = ChatMemoryInterface(mock_mode, mirror_integration)
    return _chat_interface


if __name__ == "__main__":
    # Test the chat memory interface
    chat = chat_integration(mock_mode=True)
    
    # Process a few test messages with different weights
    for weight in [0.2, 0.5, 0.8]:
        print(f"\n--- Testing with NN weight: {weight} ---")
        response = chat.process_message(
            "Tell me about neural networks and consciousness",
            nn_weight=weight,
            memory_mode="combined"
        )
        print(f"Response: {response}")
    
    # Get memory stats
    stats = chat.get_memory_stats()
    print("\n--- Memory Stats ---")
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"Total conversations: {stats.get('total_conversations', 0)}")
    
    print("\nChat memory interface test complete!") 