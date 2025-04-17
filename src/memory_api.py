#!/usr/bin/env python3
"""
Memory System API

Provides a clean API interface for integrating the Language Memory System
with external systems like LLM/NN chat interfaces.
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_api")

# Import memory components
from src.conversation_memory import ConversationMemory
from src.english_language_trainer import EnglishLanguageTrainer
from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration


class MemoryAPI:
    """
    API interface for the Language Memory System.
    
    This class provides a clean, stable interface for external systems
    (like LLM/NN chat interfaces) to interact with the memory system.
    """
    
    def __init__(self):
        """Initialize the Memory API"""
        self.memory_system = None
        self.conversation_memory = None
        self.language_trainer = None
        self.initialized = False
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all memory system components"""
        try:
            # Initialize conversation memory
            self.conversation_memory = ConversationMemory()
            
            # Initialize language memory synthesis
            self.memory_system = LanguageMemorySynthesisIntegration()
            
            # Get language trainer from memory system
            if "language_trainer" in self.memory_system.components:
                self.language_trainer = self.memory_system.components["language_trainer"]
            
            self.initialized = True
            logger.info("Memory API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Memory API: {str(e)}")
            self.initialized = False
    
    def store_conversation(self, 
                         message: str, 
                         metadata: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
        """
        Store a conversation message in memory.
        
        Args:
            message: The conversation message to store
            metadata: Optional metadata about the conversation
                     (topic, emotion, keywords, etc.)
        
        Returns:
            Dictionary containing status and the stored memory ID
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory API not initialized"}
        
        try:
            # Ensure metadata exists
            metadata = metadata or {}
            
            # Store conversation in memory
            memory = self.conversation_memory.store(
                content=message,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "memory_id": memory.get("id"),
                "timestamp": memory.get("timestamp")
            }
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def retrieve_relevant_memories(self, 
                                 message: str, 
                                 max_results: int = 5
                                ) -> Dict[str, Any]:
        """
        Retrieve memories relevant to the current conversation message.
        
        This method uses multiple retrieval strategies to find the most
        relevant memories for the given message.
        
        Args:
            message: The conversation message to find memories for
            max_results: Maximum number of results to return
        
        Returns:
            Dictionary containing relevant memories and metadata
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory API not initialized"}
        
        try:
            # Extract potential topics using language trainer
            extracted_topics = self._extract_topics(message)
            
            # Combined results from multiple retrieval methods
            all_memories = []
            
            # Search for the full message text
            text_results = self.conversation_memory.search_text(message)
            all_memories.extend(text_results)
            
            # Search for each extracted topic
            for topic in extracted_topics:
                topic_results = self.conversation_memory.retrieve_by_topic(topic)
                all_memories.extend(topic_results)
            
            # Deduplicate results based on memory ID
            unique_memories = {}
            for memory in all_memories:
                memory_id = memory.get("id")
                if memory_id and memory_id not in unique_memories:
                    unique_memories[memory_id] = memory
            
            # Select the top results
            top_memories = list(unique_memories.values())[:max_results]
            
            return {
                "status": "success",
                "memories": top_memories,
                "count": len(top_memories),
                "extracted_topics": extracted_topics
            }
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract potential topics from text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            List of potential topics
        """
        topics = []
        
        # Simple topic extraction based on keywords
        # In a production system, this would use NLP techniques
        common_topics = [
            "learning", "memory", "language", "consciousness",
            "technology", "ai", "neural networks", "data"
        ]
        
        for topic in common_topics:
            if topic.lower() in text.lower():
                topics.append(topic)
        
        return topics
    
    def synthesize_topic(self, 
                        topic: str, 
                        depth: int = 3
                       ) -> Dict[str, Any]:
        """
        Synthesize memories around a specific topic.
        
        Args:
            topic: The topic to synthesize
            depth: How deep to search for related memories (1-5)
        
        Returns:
            Dictionary containing synthesis results and metadata
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory API not initialized"}
        
        try:
            # Perform synthesis
            synthesis_results = self.memory_system.synthesize_topic(topic, depth)
            
            # Add status field for API consistency
            synthesis_results["status"] = "success"
            
            return synthesis_results
        except Exception as e:
            logger.error(f"Error synthesizing topic: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def enhance_message_with_memory(self, 
                                  message: str, 
                                  enhance_mode: str = "contextual"
                                 ) -> Dict[str, Any]:
        """
        Enhance a message with relevant memories for LLM/NN chat integration.
        
        This method prepares memory context that can be injected into
        an LLM prompt or NN input to enhance responses with memory.
        
        Args:
            message: The message to enhance
            enhance_mode: Enhancement mode:
                         - "contextual": Add relevant memories as context
                         - "synthesized": Add topic synthesis as context
                         - "combined": Both contextual and synthesized
        
        Returns:
            Dictionary with enhanced context and metadata
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory API not initialized"}
        
        try:
            # Get relevant memories
            relevant_result = self.retrieve_relevant_memories(message)
            
            # Initialize enhancement components
            memory_context = ""
            topics_context = ""
            synthesized_context = ""
            mode_contexts = {
                "contextual": [],
                "synthesized": [],
                "topics": []
            }
            
            # Process memory results if successful
            if relevant_result["status"] == "success":
                # Format memories for contextual enhancement
                memories = relevant_result.get("memories", [])
                extracted_topics = relevant_result.get("extracted_topics", [])
                
                if memories and enhance_mode in ["contextual", "combined"]:
                    memory_lines = []
                    for memory in memories:
                        content = memory.get("content", "")
                        timestamp = memory.get("timestamp", "")
                        memory_lines.append(f"[Memory from {timestamp}]: {content}")
                    
                    memory_context = "\n".join(memory_lines)
                    mode_contexts["contextual"] = memory_lines
                
                # Get topic synthesis for extracted topics
                if extracted_topics and enhance_mode in ["synthesized", "combined"]:
                    synthesis_contexts = []
                    for topic in extracted_topics:
                        synthesis = self.synthesize_topic(topic)
                        if synthesis["status"] == "success" and "synthesis_results" in synthesis:
                            memory = synthesis["synthesis_results"]["synthesized_memory"]
                            insights = memory.get("novel_insights", [])
                            if insights:
                                insight_text = "\n".join([f"- {insight}" for insight in insights[:3]])
                                synthesis_text = f"[Synthesis of '{topic}']\nCore understanding: {memory.get('core_understanding', '')}\nInsights:\n{insight_text}"
                                synthesis_contexts.append(synthesis_text)
                    
                    synthesized_context = "\n\n".join(synthesis_contexts)
                    mode_contexts["synthesized"] = synthesis_contexts
                
                # Format the extracted topics
                if extracted_topics:
                    topics_context = f"Relevant topics: {', '.join(extracted_topics)}"
                    mode_contexts["topics"] = extracted_topics
            
            # Combine contexts based on mode
            final_context = ""
            if enhance_mode == "contextual":
                final_context = memory_context
            elif enhance_mode == "synthesized":
                final_context = "\n\n".join([topics_context, synthesized_context])
            elif enhance_mode == "combined":
                final_context = "\n\n".join([memory_context, topics_context, synthesized_context])
            
            # Return the enhanced context
            return {
                "status": "success",
                "enhanced_context": final_context.strip(),
                "mode": enhance_mode,
                "mode_contexts": mode_contexts,
                "original_message": message
            }
        except Exception as e:
            logger.error(f"Error enhancing message with memory: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary of memory statistics
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory API not initialized"}
        
        try:
            stats = self.memory_system.get_stats()
            stats["status"] = "success"
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_training_examples(self, 
                            topic: str, 
                            count: int = 3
                           ) -> Dict[str, Any]:
        """
        Get training examples for a specific topic.
        
        Args:
            topic: The topic to generate examples for
            count: Number of examples to generate
        
        Returns:
            Dictionary containing generated examples
        """
        if not self.initialized or not self.language_trainer:
            return {"status": "error", "error": "Language trainer not initialized"}
        
        try:
            examples = self.language_trainer.generate_training_data(topic, count)
            
            return {
                "status": "success",
                "examples": examples,
                "count": len(examples),
                "topic": topic
            }
        except Exception as e:
            logger.error(f"Error generating training examples: {str(e)}")
            return {"status": "error", "error": str(e)}


# Example usage in a chat integration context
def chat_integration_example():
    """Example of how to use the Memory API with a chat system"""
    # Initialize the Memory API
    memory_api = MemoryAPI()
    
    # Example user message
    user_message = "Tell me about neural networks and how they learn patterns."
    
    # 1. Store the user message in memory
    store_result = memory_api.store_conversation(
        message=user_message,
        metadata={
            "topic": "neural networks",
            "source": "user_chat",
            "keywords": ["neural networks", "learning", "patterns"]
        }
    )
    print(f"Stored message with ID: {store_result.get('memory_id')}")
    
    # 2. Enhance the message with memory context for LLM
    enhanced = memory_api.enhance_message_with_memory(
        message=user_message,
        enhance_mode="combined"
    )
    
    # 3. Build an LLM prompt with the enhanced context
    llm_prompt = f"""
    User message: {user_message}
    
    Relevant memory context:
    {enhanced.get('enhanced_context', '')}
    
    Please respond to the user's message using the memory context if relevant.
    """
    
    print("\nEnhanced LLM Prompt:")
    print(llm_prompt)
    
    # 4. After receiving LLM response, store it in memory
    llm_response = "Neural networks learn patterns through a process called training, where they adjust their internal parameters based on examples."
    
    store_response = memory_api.store_conversation(
        message=llm_response,
        metadata={
            "topic": "neural networks",
            "source": "assistant_chat",
            "keywords": ["neural networks", "training", "parameters", "examples"]
        }
    )
    print(f"\nStored response with ID: {store_response.get('memory_id')}")
    
    # 5. Get memory statistics
    stats = memory_api.get_memory_stats()
    print("\nMemory System Statistics:")
    print(f"- Synthesis count: {stats.get('synthesis_stats', {}).get('synthesis_count', 0)}")
    print(f"- Topics synthesized: {stats.get('synthesis_stats', {}).get('topics_synthesized', [])}")
    print(f"- Memory count: {stats.get('language_memory_stats', {}).get('memory_count', 0)}")


if __name__ == "__main__":
    # Run the example
    chat_integration_example() 