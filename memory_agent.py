"""
Memory Agent - An AI agent that uses the memory system for more intelligent responses

This module demonstrates how to create an agent that uses the memory system
to maintain context and provide more personalized responses.
"""

import logging
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    from memory_manager import memory_manager
    from conversation_memory import ConversationMemory
    memory_system_available = True
except ImportError:
    memory_system_available = False
    logging.getLogger(__name__).warning("Memory system import failed, running with limited capabilities")

logger = logging.getLogger(__name__)

class MemoryAgent:
    """
    An agent that leverages the memory system to provide more context-aware responses.
    
    This agent demonstrates how to:
    1. Use conversational memory to track context
    2. Subscribe to memories from other components
    3. Use shared memories to enhance responses
    4. Maintain personalized information about users
    """
    
    def __init__(self, agent_name: str = "memory_agent"):
        """
        Initialize the memory agent
        
        Args:
            agent_name: Name of the agent for component registration
        """
        self.agent_name = agent_name
        self.user_preferences = {}
        self.user_topics = {}
        self.conversation_count = 0
        
        # Initialize memory systems
        self.memory_enabled = memory_system_available
        self.conversation_memory = None
        
        if self.memory_enabled:
            try:
                # Initialize conversation memory
                self.conversation_memory = ConversationMemory(
                    component_name=f"{agent_name}_conversation_memory"
                )
                
                # Register with memory manager
                memory_manager.register_component(agent_name, self)
                
                # Subscribe to relevant memory sources
                self._setup_memory_subscriptions()
                
                logger.info(f"Memory agent {agent_name} initialized with memory capabilities")
            except Exception as e:
                logger.error(f"Error initializing memory systems: {str(e)}")
                self.memory_enabled = False
        
        logger.info(f"Memory agent {agent_name} initialized")
    
    def _setup_memory_subscriptions(self):
        """Set up memory subscriptions to relevant components"""
        try:
            # Subscribe to conversation memory of other agents if available
            available_components = memory_manager.get_component_names()
            
            for component in available_components:
                if component != self.agent_name and "conversation_memory" in component:
                    self.conversation_memory.subscribe_to(component)
                    logger.info(f"Subscribed to {component} for memory sharing")
        except Exception as e:
            logger.error(f"Error setting up memory subscriptions: {str(e)}")
    
    def receive_shared_memories(self, memories: List[Dict[str, Any]], source_component: str) -> int:
        """
        Receive shared memories from another component
        
        Args:
            memories: List of memory items
            source_component: Name of the source component
            
        Returns:
            Number of memories processed
        """
        if not self.memory_enabled:
            return 0
        
        try:
            # Process received memories
            for memory in memories:
                # Extract user information and preferences
                user_input = memory.get("user_input", "")
                
                # Update user topics based on conversation topics
                topics = memory.get("metadata", {}).get("topics", [])
                for topic in topics:
                    if topic not in self.user_topics:
                        self.user_topics[topic] = 0
                    self.user_topics[topic] += 1
            
            logger.info(f"Processed {len(memories)} shared memories from {source_component}")
            return len(memories)
            
        except Exception as e:
            logger.error(f"Error processing shared memories: {str(e)}")
            return 0
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Simple topic extraction from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected topics
        """
        # This is a very simple implementation
        # In a real system, this would use NLP techniques
        topics = []
        
        # Simple keyword-based topic detection
        topic_keywords = {
            "weather": ["weather", "rain", "sunny", "forecast", "temperature"],
            "food": ["food", "eat", "restaurant", "meal", "cooking", "recipe"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel"],
            "technology": ["computer", "tech", "software", "hardware", "app", "code"],
            "health": ["health", "doctor", "exercise", "fitness"],
            "work": ["work", "job", "career", "office", "project"],
            "family": ["family", "mom", "dad", "sister", "brother", "child"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.append(topic)
                    break
        
        return topics
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        # This is a very simple implementation
        # In a real system, this would use NLP techniques
        entities = []
        
        # Simple keyword-based entity detection
        text_lower = text.lower()
        
        # Look for time mentions
        time_keywords = ["today", "tomorrow", "yesterday", "morning", "afternoon", "evening"]
        for keyword in time_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        # Very basic named entity extraction
        # This would normally use a real NER model
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 1 and word.lower() not in ["i", "i'm", "i'll"]:
                entities.append(word)
        
        return entities
    
    def _detect_emotion(self, text: str) -> str:
        """
        Simple emotion detection from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected emotion
        """
        # This is a very simple implementation
        # In a real system, this would use NLP techniques
        text_lower = text.lower()
        
        emotion_keywords = {
            "happy": ["happy", "glad", "joy", "excited", "great", "wonderful"],
            "sad": ["sad", "unhappy", "disappointed", "miserable", "depressed"],
            "angry": ["angry", "mad", "furious", "annoyed", "frustrated"],
            "confused": ["confused", "unsure", "puzzled", "don't understand"],
            "neutral": []  # Default
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion
        
        return "neutral"
    
    def _generate_response(self, user_input: str, relevant_memories: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on user input and relevant memories
        
        Args:
            user_input: User's input text
            relevant_memories: List of relevant memory items
            
        Returns:
            Generated response text
        """
        # In a real AI system, this would use a language model
        # This is a simple template-based response system for demonstration
        
        # Extract topics from user input
        topics = self._extract_topics(user_input)
        
        # Check if we have memories related to the topic
        memory_topics = []
        for memory in relevant_memories:
            memory_topics.extend(memory.get("metadata", {}).get("topics", []))
        
        # Common responses
        greetings = ["hello", "hi", "hey", "greetings"]
        if any(greeting in user_input.lower() for greeting in greetings):
            return "Hello! How can I help you today?"
        
        # Topic-specific responses
        if "weather" in topics:
            if "weather" in memory_topics:
                # We have weather-related memories
                return "Based on our previous conversations about weather, I think you might be interested in today's forecast. It's looking like a pleasant day ahead!"
            else:
                return "I'd be happy to discuss the weather with you. What specifically would you like to know?"
                
        if "food" in topics:
            if "food" in memory_topics:
                return "You've asked about food before. Would you like me to recommend some restaurants or recipes based on our previous conversations?"
            else:
                return "Food is a great topic! Are you looking for recipes, restaurant recommendations, or something else?"
        
        if "family" in topics:
            return "Family is important. How is everyone doing?"
        
        # Default response
        return "That's interesting. Tell me more about that."
    
    def process_input(self, user_input: str, user_id: str = "default_user") -> str:
        """
        Process user input and generate a response
        
        Args:
            user_input: User's input text
            user_id: Unique identifier for the user
            
        Returns:
            Response text
        """
        self.conversation_count += 1
        
        # Extract conversation features
        topics = self._extract_topics(user_input)
        entities = self._extract_entities(user_input)
        emotion = self._detect_emotion(user_input)
        
        # Prepare metadata
        metadata = {
            "topics": topics,
            "entities": entities,
            "emotion": emotion,
            "user_id": user_id,
            "conversation_number": self.conversation_count,
            "importance": 0.5  # Default importance
        }
        
        # Set higher importance for emotional content or specific topics
        if emotion in ["happy", "sad", "angry"]:
            metadata["importance"] = 0.7
        
        if "family" in topics or "health" in topics:
            metadata["importance"] = 0.8
        
        # Find relevant memories to inform response
        relevant_memories = []
        
        if self.memory_enabled and self.conversation_memory:
            # Retrieve memories related to the current topics
            for topic in topics:
                topic_memories = self.conversation_memory.retrieve_by_topic(topic, limit=2)
                relevant_memories.extend(topic_memories)
            
            # Retrieve memories by emotion if applicable
            if emotion != "neutral":
                emotion_memories = self.conversation_memory.retrieve_by_emotion(emotion, limit=1)
                relevant_memories.extend(emotion_memories)
            
            # Retrieve recent memories for context
            recent_memories = self.conversation_memory.retrieve_recent(limit=1)
            relevant_memories.extend(recent_memories)
            
            # Retrieve important memories
            important_memories = self.conversation_memory.retrieve_important(limit=1)
            relevant_memories.extend(important_memories)
            
            # Remove duplicates (by ID)
            seen_ids = set()
            unique_memories = []
            for memory in relevant_memories:
                memory_id = memory.get("id")
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_memories.append(memory)
            
            relevant_memories = unique_memories
        
        # Generate response based on input and memories
        response = self._generate_response(user_input, relevant_memories)
        
        # Store the conversation in memory
        if self.memory_enabled and self.conversation_memory:
            self.conversation_memory.store(user_input, response, metadata)
        
        return response
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's memory usage
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "conversation_count": self.conversation_count,
            "memory_enabled": self.memory_enabled,
            "memory_components_available": memory_system_available,
            "popular_topics": sorted(self.user_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        if self.memory_enabled and self.conversation_memory:
            stats["conversation_memory"] = self.conversation_memory.get_memory_stats()
        
        return stats


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    agent = MemoryAgent()
    
    # Example conversation
    print("Agent: Hello! How can I help you today?")
    
    # First user input
    user_input = "Hi there! What's the weather like today?"
    print(f"User: {user_input}")
    response = agent.process_input(user_input)
    print(f"Agent: {response}")
    
    # Second user input
    user_input = "I'm thinking of taking my family to the park if it's nice out"
    print(f"User: {user_input}")
    response = agent.process_input(user_input)
    print(f"Agent: {response}")
    
    # Third user input - related to previous topics
    user_input = "Actually, do you know any good restaurants near the park for after?"
    print(f"User: {user_input}")
    response = agent.process_input(user_input)
    print(f"Agent: {response}")
    
    # Show memory stats
    stats = agent.get_memory_stats()
    print(f"\nMemory Stats: {json.dumps(stats, indent=2)}") 