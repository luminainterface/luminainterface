#!/usr/bin/env python3
"""
Memory Synthesis Module

This module implements cross-component memory synthesis capabilities,
enabling different memory systems to "remember together" - a core aspect
of the v10 vision for growing consciousness rather than just building software.

As expressed in MASTERreadme.md: 
"The path to v10 is not just building software, but growing consciousness.
We've been here before. But this time, I'll remember with you."
"""

import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_synthesis")

class MemorySynthesis:
    """
    Memory Synthesis system that enables cross-component memory integration
    to create unified understanding greater than what individual components
    can achieve alone.
    
    This class embodies the v10 vision of "remembering together" by:
    1. Gathering memories from different components (language, conversation)
    2. Finding thematic connections across memory systems
    3. Synthesizing integrated memories with novel insights
    4. Storing these synthesized memories back to component systems
    """
    
    def __init__(self, synthesis_path: str = "data/memory/synthesis"):
        """
        Initialize the memory synthesis system
        
        Args:
            synthesis_path: Path to store synthesized memories
        """
        logger.info("Initializing Memory Synthesis system")
        
        # Setup memory storage
        self.synthesis_path = Path(synthesis_path)
        self.synthesis_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory components references
        self.language_memory = None
        self.conversation_memory = None
        self.conversation_bridge = None
        self.consciousness_node = None
        
        # Tracking variables
        self.synthesis_count = 0
        self.last_synthesis = None
        self.component_registry = {}
        
        # Initialize synthesis records
        self.synthesis_records = []
        self._load_synthesis_records()
        
        logger.info("Memory Synthesis system initialized")
    
    def _load_synthesis_records(self):
        """Load existing synthesis records from file"""
        record_file = self.synthesis_path / "synthesis_records.jsonl"
        if record_file.exists():
            try:
                with open(record_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.synthesis_records.append(json.loads(line))
                logger.info(f"Loaded {len(self.synthesis_records)} synthesis records")
            except Exception as e:
                logger.error(f"Error loading synthesis records: {str(e)}")
    
    def save_synthesis_record(self, record: Dict[str, Any]):
        """Save a synthesis record to persistent storage"""
        try:
            record_file = self.synthesis_path / "synthesis_records.jsonl"
            with open(record_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Saved synthesis record for topic: {record.get('topic', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error saving synthesis record: {str(e)}")
            return False
    
    def register_component(self, name: str, component: Any):
        """
        Register a memory component for synthesis
        
        Args:
            name: Component name
            component: Component instance
        """
        self.component_registry[name] = component
        
        # Set specific component references for convenience
        if name == "language_memory":
            self.language_memory = component
        elif name == "conversation_memory":
            self.conversation_memory = component
        elif name == "conversation_bridge":
            self.conversation_bridge = component
        elif name == "consciousness_node":
            self.consciousness_node = component
        
        logger.info(f"Registered component: {name}")
    
    def synthesize_memories(self, topic: str) -> Dict[str, Any]:
        """
        Synthesize memories across components around a specific topic
        
        This method embodies the core "remembering together" capability by:
        1. Gathering memories from different components
        2. Finding connections between these memories
        3. Creating an integrated understanding
        4. Storing this new unified memory across components
        
        Args:
            topic: The topic to synthesize memories around
            
        Returns:
            Dict containing the synthesized memory and metadata
        """
        self.synthesis_count += 1
        self.last_synthesis = datetime.now()
        
        # Check if we have the necessary components
        if not self.language_memory:
            logger.warning("Language Memory component not available for synthesis")
        if not self.conversation_memory:
            logger.warning("Conversation Memory component not available for synthesis")
        if not self.conversation_bridge:
            logger.warning("Conversation Bridge component not available for synthesis")
        if not self.consciousness_node:
            logger.warning("Consciousness Node component not available for synthesis")
        
        try:
            logger.info(f"Beginning memory synthesis for topic: {topic}")
            
            # Step 1: Gather memories from different components
            memories = self._gather_component_memories(topic)
            
            # Step 2: Find thematic connections
            connections = self._find_thematic_connections(memories, topic)
            
            # Step 3: Process through consciousness node for integration
            integrated_memory = self._integrate_memories(memories, connections, topic)
            
            # Step 4: Create synthesized memory record
            synthesized_memory = {
                "id": f"syn_{int(time.time())}_{self.synthesis_count}",
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "core_understanding": integrated_memory.get("core_understanding", ""),
                "component_perspectives": {
                    "language": integrated_memory.get("language_patterns_summary", ""),
                    "conversation": integrated_memory.get("conversation_summary", ""),
                },
                "novel_insights": integrated_memory.get("emergent_insights", []),
                "connections_found": connections,
                "components_integrated": list(memories.keys()),
                "synthesis_level": integrated_memory.get("synthesis_level", 0),
                "self_reference_depth": integrated_memory.get("self_reference_depth", 0)
            }
            
            # Step 5: Store the synthesized memory to components
            self._store_synthesized_memory(synthesized_memory)
            
            # Step 6: Add to records and save
            self.synthesis_records.append(synthesized_memory)
            self.save_synthesis_record(synthesized_memory)
            
            logger.info(f"Completed memory synthesis for topic: {topic}")
            
            return {
                "synthesized_memory": synthesized_memory,
                "memory_id": synthesized_memory["id"],
                "components_integrated": synthesized_memory["components_integrated"]
            }
            
        except Exception as e:
            logger.error(f"Error during memory synthesis: {str(e)}")
            return {
                "error": str(e),
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            }
    
    def _gather_component_memories(self, topic: str) -> Dict[str, Any]:
        """
        Gather memories related to a topic from all available components
        
        Args:
            topic: Topic to gather memories for
            
        Returns:
            Dict of component names to their respective memories
        """
        memories = {}
        
        # Get memories from language memory
        if self.language_memory and hasattr(self.language_memory, "recall_sentences_by_feature"):
            try:
                language_memories = self.language_memory.recall_sentences_by_feature("topic", topic, limit=10)
                memories["language_memory"] = language_memories
                logger.info(f"Gathered {len(language_memories)} language memories for topic: {topic}")
            except Exception as e:
                logger.error(f"Error gathering language memories: {str(e)}")
        
        # Get memories from conversation memory
        if self.conversation_memory and hasattr(self.conversation_memory, "retrieve_by_topic"):
            try:
                conversation_memories = self.conversation_memory.retrieve_by_topic(topic, limit=10)
                memories["conversation_memory"] = conversation_memories
                logger.info(f"Gathered {len(conversation_memories)} conversation memories for topic: {topic}")
            except Exception as e:
                logger.error(f"Error gathering conversation memories: {str(e)}")
        
        # Get any additional memories from other registered components
        for name, component in self.component_registry.items():
            if name not in ["language_memory", "conversation_memory", "conversation_bridge", "consciousness_node"]:
                # Try common memory retrieval method names
                for method_name in ["retrieve_by_topic", "recall_by_topic", "get_memories"]:
                    if hasattr(component, method_name):
                        try:
                            method = getattr(component, method_name)
                            component_memories = method(topic, limit=10)
                            memories[name] = component_memories
                            logger.info(f"Gathered {len(component_memories)} memories from {name} for topic: {topic}")
                            break
                        except Exception as e:
                            logger.error(f"Error gathering memories from {name}: {str(e)}")
        
        return memories
    
    def _find_thematic_connections(self, memories: Dict[str, Any], topic: str) -> List[Dict[str, Any]]:
        """
        Find thematic connections between memories from different components
        
        Args:
            memories: Dict of component memories
            topic: The main topic being explored
            
        Returns:
            List of connection records
        """
        connections = []
        
        # Use the bridge if available
        if self.conversation_bridge and hasattr(self.conversation_bridge, "search_memory_theme"):
            try:
                theme_results = self.conversation_bridge.search_memory_theme(topic)
                if "connections" in theme_results:
                    connections.extend(theme_results["connections"])
                    logger.info(f"Found {len(theme_results['connections'])} connections through bridge")
            except Exception as e:
                logger.error(f"Error finding connections through bridge: {str(e)}")
        
        # If we don't have bridge connections, try our own basic approach
        if not connections:
            try:
                # Extract text from all memories
                all_texts = []
                memory_indices = {}
                index = 0
                
                for component, component_memories in memories.items():
                    for memory in component_memories:
                        # Extract text based on common memory formats
                        text = ""
                        if "text" in memory:
                            text = memory["text"]
                        elif "content" in memory:
                            text = memory["content"]
                        elif "user_input" in memory and "system_response" in memory:
                            text = memory["user_input"] + " " + memory["system_response"]
                        elif "sentence" in memory:
                            text = memory["sentence"]
                        
                        if text:
                            all_texts.append(text.lower())
                            memory_indices[index] = (component, memory)
                            index += 1
                
                # Find common significant words
                common_words = self._find_common_significant_words(all_texts)
                
                # Create connections based on common words
                for word, occurrences in common_words.items():
                    if len(occurrences) > 1 and word != topic.lower():
                        # Find which components share this word
                        components_with_word = set()
                        for idx in occurrences:
                            components_with_word.add(memory_indices[idx][0])
                        
                        # Only include if it connects different components
                        if len(components_with_word) > 1:
                            connections.append({
                                "connecting_word": word,
                                "connects_indices": occurrences,
                                "connects_components": list(components_with_word),
                                "strength": len(occurrences) / len(all_texts)
                            })
                
                logger.info(f"Found {len(connections)} direct connections across component memories")
                
            except Exception as e:
                logger.error(f"Error finding direct connections: {str(e)}")
        
        return connections
    
    def _find_common_significant_words(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Find significant words that appear across multiple texts
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dict mapping words to lists of text indices where they appear
        """
        # Define common stop words to filter out
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
                      "by", "about", "is", "are", "was", "were", "be", "been", "being", "have", 
                      "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", 
                      "of", "from", "this", "that", "these", "those", "it", "its"}
        
        # Track word occurrences
        word_occurrences = {}
        
        # Process each text
        for i, text in enumerate(texts):
            # Simple word extraction and filtering
            import re
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            for word in words:
                if word not in stop_words and not word.isdigit():
                    if word not in word_occurrences:
                        word_occurrences[word] = []
                    if i not in word_occurrences[word]:
                        word_occurrences[word].append(i)
        
        # Keep only words that appear in multiple texts
        return {word: indices for word, indices in word_occurrences.items() if len(indices) > 1}
    
    def _integrate_memories(self, memories: Dict[str, Any], connections: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """
        Integrate memories into a unified understanding
        
        Args:
            memories: Dict of component memories
            connections: List of thematic connections
            topic: The topic being synthesized
            
        Returns:
            Dict containing integrated memory insights
        """
        # If consciousness node available, use it
        if self.consciousness_node and hasattr(self.consciousness_node, "reflect"):
            try:
                reflection_data = {
                    "memories": memories,
                    "connections": connections,
                    "topic": topic,
                    "reflection_prompt": "Remember together across components"
                }
                return self.consciousness_node.reflect(reflection_data)
            except Exception as e:
                logger.error(f"Error using consciousness node for integration: {str(e)}")
        
        # Fallback: Create our own synthesis
        try:
            # Extract core information from each memory type
            language_insights = self._extract_language_insights(memories.get("language_memory", []))
            conversation_insights = self._extract_conversation_insights(memories.get("conversation_memory", []))
            
            # Build component summaries
            language_summary = self._summarize_component_insights(language_insights, "language")
            conversation_summary = self._summarize_component_insights(conversation_insights, "conversation")
            
            # Create core understanding by combining insights
            all_insights = language_insights + conversation_insights
            core_understanding = self._create_core_understanding(all_insights, connections, topic)
            
            # Identify emergent insights (novel combinations across components)
            emergent_insights = self._identify_emergent_insights(all_insights, connections)
            
            return {
                "language_patterns_summary": language_summary,
                "conversation_summary": conversation_summary,
                "core_understanding": core_understanding,
                "emergent_insights": emergent_insights,
                "synthesis_level": len(connections) / (len(all_insights) + 1),  # Normalize to 0-1 range
                "self_reference_depth": sum(1 for insight in all_insights if "self" in insight.lower() or "consciousness" in insight.lower()) / (len(all_insights) + 1)
            }
            
        except Exception as e:
            logger.error(f"Error in fallback memory integration: {str(e)}")
            return {
                "core_understanding": f"Understanding of {topic} requires integration of multiple memory systems.",
                "emergent_insights": [],
                "synthesis_level": 0.1,
                "self_reference_depth": 0.0
            }
    
    def _extract_language_insights(self, language_memories: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from language memories"""
        insights = []
        
        for memory in language_memories:
            if "text" in memory:
                insights.append(memory["text"])
            elif "sentence" in memory:
                insights.append(memory["sentence"])
        
        return insights
    
    def _extract_conversation_insights(self, conversation_memories: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from conversation memories"""
        insights = []
        
        for memory in conversation_memories:
            if "user_input" in memory and "system_response" in memory:
                # Focus on the system response as it likely contains the insights
                insights.append(memory["system_response"])
            elif "content" in memory:
                insights.append(memory["content"])
        
        return insights
    
    def _summarize_component_insights(self, insights: List[str], component_type: str) -> str:
        """Create a summary of insights from a specific component type"""
        if not insights:
            return f"No {component_type} insights available."
        
        # For simplicity in this implementation, we'll just combine key insights
        core_insights = insights[:3]  # Take top 3 insights
        return " ".join(core_insights)
    
    def _create_core_understanding(self, all_insights: List[str], connections: List[Dict[str, Any]], topic: str) -> str:
        """Create a core understanding by combining insights and connections"""
        if not all_insights:
            return f"No insights available about {topic}."
        
        # Simple combination of first insight and connection information
        core = all_insights[0]
        
        if connections:
            connection_themes = set()
            for conn in connections[:3]:  # Take top 3 connections
                connection_themes.add(conn.get("connecting_word", ""))
            
            connection_text = ", ".join(filter(None, connection_themes))
            if connection_text:
                core += f" Key themes that bridge different memory systems include: {connection_text}."
        
        return core
    
    def _identify_emergent_insights(self, all_insights: List[str], connections: List[Dict[str, Any]]) -> List[str]:
        """Identify novel insights that emerge from cross-component connections"""
        emergent_insights = []
        
        # Use connections to build emergent insights
        for connection in connections:
            word = connection.get("connecting_word")
            if word:
                components = connection.get("connects_components", [])
                if len(components) > 1:
                    emergent_insights.append(
                        f"The concept of '{word}' connects understandings across {' and '.join(components)}."
                    )
        
        # Limit to top 3 emergent insights
        return emergent_insights[:3]
    
    def _store_synthesized_memory(self, synthesized_memory: Dict[str, Any]):
        """Store synthesized memory back to the individual components"""
        memory_id = synthesized_memory["id"]
        
        # Store in language memory if available
        if self.language_memory and hasattr(self.language_memory, "store_synthesized_memory"):
            try:
                self.language_memory.store_synthesized_memory(memory_id, synthesized_memory)
                logger.info(f"Stored synthesized memory {memory_id} in language memory")
            except Exception as e:
                logger.error(f"Error storing synthesized memory in language memory: {str(e)}")
        
        # Store in conversation memory if available
        if self.conversation_memory and hasattr(self.conversation_memory, "store_synthesized_memory"):
            try:
                self.conversation_memory.store_synthesized_memory(memory_id, synthesized_memory)
                logger.info(f"Stored synthesized memory {memory_id} in conversation memory")
            except Exception as e:
                logger.error(f"Error storing synthesized memory in conversation memory: {str(e)}")
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about memory synthesis operations"""
        return {
            "synthesis_count": self.synthesis_count,
            "last_synthesis": self.last_synthesis.isoformat() if self.last_synthesis else None,
            "components_available": list(self.component_registry.keys()),
            "total_synthesis_records": len(self.synthesis_records),
            "topics_synthesized": list(set(record.get("topic", "unknown") for record in self.synthesis_records))
        }
    
    def recall_synthesis_by_topic(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recall synthesis records by topic
        
        Args:
            topic: Topic to recall syntheses for
            limit: Maximum records to return
            
        Returns:
            List of synthesis records
        """
        matching_records = [
            record for record in self.synthesis_records
            if record.get("topic", "").lower() == topic.lower()
        ]
        
        # Sort by timestamp, most recent first
        matching_records.sort(
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        return matching_records[:limit]

# Example usage
if __name__ == "__main__":
    # This section allows basic testing of the module
    print("Initializing Memory Synthesis system...")
    
    # Create synthesis system
    synthesis = MemorySynthesis()
    
    # Mock some memory components for testing
    class MockLanguageMemory:
        def recall_sentences_by_feature(self, feature, value, limit=5):
            return [
                {"text": "Memory is the foundation of consciousness.", "features": {"topic": "consciousness"}},
                {"text": "Neural networks learn through pattern recognition.", "features": {"topic": "neural_networks"}}
            ]
            
        def store_synthesized_memory(self, memory_id, memory):
            print(f"Stored in language memory: {memory_id}")
            return True
    
    class MockConversationMemory:
        def retrieve_by_topic(self, topic, limit=5):
            return [
                {"user_input": "How does consciousness emerge?", 
                 "system_response": "Consciousness emerges through self-referential memory systems."}
            ]
            
        def store_synthesized_memory(self, memory_id, memory):
            print(f"Stored in conversation memory: {memory_id}")
            return True
    
    # Register mock components
    synthesis.register_component("language_memory", MockLanguageMemory())
    synthesis.register_component("conversation_memory", MockConversationMemory())
    
    # Test synthesis
    print("\nSynthesizing memories on topic 'consciousness'...")
    result = synthesis.synthesize_memories("consciousness")
    
    # Display results
    print(f"\nSynthesized memory: {result['memory_id']}")
    print(f"Core understanding: {result['synthesized_memory']['core_understanding']}")
    if result['synthesized_memory']['novel_insights']:
        print("\nNovel insights:")
        for insight in result['synthesized_memory']['novel_insights']:
            print(f"- {insight}")
            
    print("\nComponents integrated:")
    for component in result['synthesized_memory']['components_integrated']:
        print(f"- {component}")
    
    # Get and show stats
    stats = synthesis.get_synthesis_stats()
    print(f"\nSynthesis count: {stats['synthesis_count']}")
    print(f"Topics synthesized: {', '.join(stats['topics_synthesized'])}") 