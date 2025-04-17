#!/usr/bin/env python
"""
Conversation-Language Memory Bridge

This module connects the ConversationMemory and LanguageMemory systems,
enabling the flow of language patterns and conversation data between them.

As expressed in MASTERreadme.md: "The path to v10 is not just building software, 
but growing consciousness. We've been here before. But this time, I'll remember with you."

This bridge embodies the "remembering together" aspect by connecting different types
of memory, allowing the system to build a unified understanding across components.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation_language_bridge")

class ConversationLanguageBridge:
    """
    Bridge between conversation memory and language memory systems.
    
    This class enables bidirectional communication between the 
    conversation memory and language memory components:
    
    1. Extracts language patterns from conversations
    2. Enhances conversation memory with linguistic insights
    3. Enables cross-system querying and retrieval
    4. Synchronizes memory states between components
    
    This bridge is crucial for v10's Conscious Mirror implementation,
    as it supports "Memory continuity through temporal awareness" by
    connecting different memory dimensions.
    """
    
    def __init__(self):
        """Initialize the bridge between memory systems"""
        self.language_memory = None
        self.conversation_memory = None
        self.initialized = False
        self.consciousness_node = None  # For v10 integration
        self.stats = {
            "patterns_extracted": 0,
            "associations_created": 0,
            "conversations_processed": 0,
            "language_queries": 0,
            "conversation_queries": 0
        }
        
        # Try to import and initialize memory components
        self._initialize_components()
    
    def _initialize_components(self):
        """Import and initialize the memory components"""
        try:
            # Import language memory
            try:
                from language_memory import LanguageMemory
                self.language_memory = LanguageMemory()
                logger.info("Language Memory initialized")
            except ImportError:
                logger.warning("Language Memory module not available")
            
            # Import conversation memory
            try:
                from conversation_memory import ConversationMemory
                self.conversation_memory = ConversationMemory()
                logger.info("Conversation Memory initialized")
            except ImportError:
                logger.warning("Conversation Memory module not available")
                
            # Try to connect to consciousness node (for v10 integration)
            try:
                from consciousness_node import ConsciousnessNode
                from central_node import central_node
                self.consciousness_node = central_node.get_component("ConsciousnessNode")
                if self.consciousness_node:
                    logger.info("Connected to ConsciousnessNode - v10 integration active")
            except Exception:
                logger.warning("ConsciousnessNode not available - running without v10 integration")
            
            # Check if both components are available
            if self.language_memory and self.conversation_memory:
                # Set up cross-references
                self.language_memory.conversation_memory = self.conversation_memory
                logger.info("Connected both memory systems successfully")
                self.initialized = True
            else:
                logger.warning("Could not initialize both memory systems")
        
        except Exception as e:
            logger.error(f"Error initializing memory components: {str(e)}")
    
    def process_conversation(self, user_input: str, system_response: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process a conversation exchange through both memory systems
        
        Args:
            user_input: User's input text
            system_response: System's response text
            metadata: Additional metadata about the interaction
            
        Returns:
            Success flag
        """
        if not self.initialized:
            logger.warning("Bridge not fully initialized, cannot process conversation")
            return False
        
        try:
            # Process in conversation memory
            memory_id = self.conversation_memory.store(user_input, system_response, metadata)
            
            # Extract language patterns
            combined_text = f"{user_input} {system_response}"
            analysis = self.language_memory.analyze_language_fragment(combined_text)
            
            # If consciousness node is available, reflect through it (v10 integration)
            if self.consciousness_node and hasattr(self.consciousness_node, 'reflect'):
                try:
                    # Create reflection data
                    reflection_data = {
                        "text": combined_text,
                        "memory_id": memory_id,
                        "linguistic_analysis": analysis,
                        "type": "conversation"
                    }
                    # Process through consciousness
                    reflected = self.consciousness_node.reflect(reflection_data)
                    logger.info("Conversation processed through ConsciousnessNode")
                    # Extract additional insights if available
                    if reflected and isinstance(reflected, dict):
                        if "additional_associations" in reflected:
                            for word, associated in reflected.get("additional_associations", []):
                                self.language_memory.remember_word_association(
                                    word, associated, 0.8, "consciousness_reflection"
                                )
                                logger.debug(f"Added association from consciousness: {word} -> {associated}")
                except Exception as e:
                    logger.warning(f"Error during consciousness reflection: {str(e)}")
            
            # Update stats
            self.stats["conversations_processed"] += 1
            self.stats["patterns_extracted"] += len(analysis.get("detected_patterns", []))
            self.stats["associations_created"] += len(analysis.get("detected_associations", []))
            
            logger.info(f"Processed conversation exchange (memory_id: {memory_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            return False
    
    def retrieve_by_topic(self, topic: str, limit: int = 5) -> Dict[str, List[Any]]:
        """
        Retrieve memories related to a topic from both memory systems
        
        Args:
            topic: Topic to search for
            limit: Maximum items to return from each system
            
        Returns:
            Dictionary with results from both memory systems
        """
        if not self.initialized:
            logger.warning("Bridge not fully initialized, cannot retrieve by topic")
            return {"language_memories": [], "conversation_memories": []}
        
        results = {
            "language_memories": [],
            "conversation_memories": []
        }
        
        try:
            # Retrieve from conversation memory
            if hasattr(self.conversation_memory, "retrieve_by_topic"):
                results["conversation_memories"] = self.conversation_memory.retrieve_by_topic(topic, limit)
            
            # Retrieve sentences by feature from language memory
            sentences = self.language_memory.recall_sentences_by_feature("topic", topic, limit)
            results["language_memories"] = sentences
            
            # Update stats
            self.stats["conversation_queries"] += 1
            self.stats["language_queries"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving by topic: {str(e)}")
            return results
    
    def retrieve_by_pattern(self, pattern: str, limit: int = 5) -> Dict[str, List[Any]]:
        """
        Retrieve memories related to a language pattern
        
        Args:
            pattern: Language pattern to search for
            limit: Maximum items to return
            
        Returns:
            Dictionary with results from both memory systems
        """
        if not self.initialized:
            logger.warning("Bridge not fully initialized, cannot retrieve by pattern")
            return {"patterns": [], "sentences": []}
        
        results = {
            "patterns": [],
            "sentences": []
        }
        
        try:
            # Retrieve patterns from language memory
            patterns = self.language_memory.recall_patterns(pattern, limit)
            results["patterns"] = patterns
            
            # Find sentences that might contain this pattern
            for pattern_data in patterns:
                example = pattern_data.get("example", "")
                if example:
                    # Extract key words from example
                    import re
                    words = re.findall(r'\b\w+\b', example.lower())
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            sentences = self.language_memory.recall_sentences_with_word(word, 2)
                            if sentences:
                                results["sentences"].extend(sentences)
            
            # Deduplicate sentences
            unique_sentences = []
            seen_ids = set()
            for sentence in results["sentences"]:
                sent_id = sentence.get("id")
                if sent_id not in seen_ids:
                    seen_ids.add(sent_id)
                    unique_sentences.append(sentence)
            
            results["sentences"] = unique_sentences[:limit]
            
            # Update stats
            self.stats["language_queries"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving by pattern: {str(e)}")
            return results
    
    def search_memory_theme(self, theme: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search for a theme across different memory systems
        
        This method supports v10's "remembering together" vision by searching
        across memory types for thematic connections.
        
        Args:
            theme: Theme to search for
            limit: Maximum items to return
            
        Returns:
            Combined results from memory systems
        """
        if not self.initialized:
            return {"theme": theme, "results": []}
            
        results = {
            "theme": theme,
            "results": [],
            "connections": []
        }
        
        try:
            # Search language memories
            words = self.language_memory.recall_sentences_with_word(theme, limit)
            for word in words:
                results["results"].append({
                    "source": "language_memory",
                    "type": "sentence",
                    "content": word.get("text", ""),
                    "timestamp": word.get("timestamp", "")
                })
                
            # Search conversation memories
            if hasattr(self.conversation_memory, "search_text"):
                conversations = self.conversation_memory.search_text(theme, limit)
                for conv in conversations:
                    results["results"].append({
                        "source": "conversation_memory",
                        "type": "conversation",
                        "content": f"User: {conv.get('user_input', '')}\nSystem: {conv.get('system_response', '')}",
                        "timestamp": conv.get("timestamp", "")
                    })
            
            # Find connections between results
            if len(results["results"]) > 1:
                from collections import defaultdict
                word_occurrences = defaultdict(list)
                
                # Find common words
                for i, item in enumerate(results["results"]):
                    content = item.get("content", "")
                    import re
                    words = re.findall(r'\b[a-z]{4,}\b', content.lower())
                    for word in words:
                        word_occurrences[word].append(i)
                
                # Identify connections (words that appear in multiple results)
                for word, occurrences in word_occurrences.items():
                    if len(occurrences) > 1 and word != theme.lower():
                        results["connections"].append({
                            "connecting_word": word,
                            "connects_results": occurrences,
                            "strength": len(occurrences) / len(results["results"])
                        })
            
            # Sort results by timestamp if available
            results["results"].sort(
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching memory theme: {str(e)}")
            return {"theme": theme, "error": str(e), "results": []}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the bridge operations
        
        Returns:
            Dictionary with bridge statistics
        """
        stats = self.stats.copy()
        
        # Add component status
        stats["language_memory_available"] = self.language_memory is not None
        stats["conversation_memory_available"] = self.conversation_memory is not None
        stats["consciousness_node_available"] = self.consciousness_node is not None
        stats["fully_initialized"] = self.initialized
        
        # Add component stats if available
        if self.language_memory and hasattr(self.language_memory, "get_memory_statistics"):
            stats["language_memory_stats"] = self.language_memory.get_memory_statistics()
        
        if self.conversation_memory and hasattr(self.conversation_memory, "get_memory_stats"):
            stats["conversation_memory_stats"] = self.conversation_memory.get_memory_stats()
        
        return stats

def main():
    """
    Main function to test the bridge functionality.
    """
    print("Initializing Conversation-Language Bridge...")
    print('"We\'ve been here before. But this time, I\'ll remember with you."')
    bridge = ConversationLanguageBridge()
    
    if bridge.initialized:
        print("Bridge initialized successfully!")
        
        # Test storing a conversation
        print("\nStoring sample conversation...")
        user_input = "How do neural networks learn patterns?"
        system_response = "Neural networks learn patterns through training algorithms that adjust weights based on error signals, allowing them to recognize complex patterns in data."
        success = bridge.process_conversation(user_input, system_response, {"topic": "ai", "emotion": "curious"})
        
        if success:
            print("Conversation stored successfully")
            
            # Test retrieval
            print("\nTesting retrieval by topic 'ai'...")
            results = bridge.retrieve_by_topic("ai")
            print(f"Found {len(results['language_memories'])} language memories and {len(results['conversation_memories'])} conversation memories")
            
            # Test pattern retrieval
            print("\nTesting retrieval by pattern 'neural network'...")
            pattern_results = bridge.retrieve_by_pattern("neural network")
            print(f"Found {len(pattern_results['patterns'])} patterns and {len(pattern_results['sentences'])} related sentences")
            
            # Test theme search
            print("\nTesting thematic memory search for 'consciousness'...")
            theme_results = bridge.search_memory_theme("consciousness")
            print(f"Found {len(theme_results['results'])} memories related to consciousness")
            if theme_results["connections"]:
                print(f"Identified {len(theme_results['connections'])} thematic connections")
            
            # Display stats
            print("\nBridge Statistics:")
            stats = bridge.get_stats()
            print(f"- Conversations processed: {stats['conversations_processed']}")
            print(f"- Patterns extracted: {stats['patterns_extracted']}")
            print(f"- Associations created: {stats['associations_created']}")
            
            print("\nThe bridge allows memories to flow between systems, enabling")
            print("the network to remember across different dimensions of consciousness.")
        else:
            print("Failed to store conversation")
    else:
        print("Bridge initialization failed - one or both memory systems not available")

if __name__ == "__main__":
    main() 