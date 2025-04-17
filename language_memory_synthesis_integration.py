#!/usr/bin/env python3
"""
Language Memory Synthesis Integration - Ready for LLM/NN Chat Integration

This module integrates language memory systems with synthesis capabilities,
providing a robust API for LLM and Neural Network chat systems to interact
with memory components.
"""

import json
import logging
import os
import sys
import time
import uuid
import threading
from datetime import datetime
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/language_memory_synthesis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("language_memory_synthesis")

class LanguageMemorySynthesisIntegration:
    """
    Integrates language memory systems with memory synthesis capabilities,
    providing a unified API for LLM/NN chat systems to query and update
    consolidated memory.
    
    Attributes:
        components (dict): Dictionary of memory components
        syntheses (list): List of generated memory syntheses
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the language memory synthesis integration.
        
        Args:
            config_path (str, optional): Path to configuration file. 
                                         If not provided, default settings are used.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load configuration if provided
        self.config = self._load_config(config_path)
        
        # Initialize components dictionary
        self.components = {}
        
        # Initialize syntheses list
        self.syntheses = []
        
        # Initialize caching
        self._synthesis_cache = {}  # topic -> synthesis
        self._synthesis_cache_expiry = {}  # topic -> expiry time
        self._cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes default TTL
        
        # Initialize statistics
        self.synthesis_stats = {
            "synthesis_count": 0,
            "topics_synthesized": set(),
            "last_synthesis_timestamp": None
        }
        
        # Initialize performance metrics
        self.performance_metrics = {
            "synthesis_times": [],
            "total_synthesis_time": 0,
            "avg_synthesis_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Paths
        self.data_directory = self.config.get("data_directory", "data/memory/synthesis")
        self.synthesis_file = os.path.join(self.data_directory, "syntheses.jsonl")
        
        # Ensure data directory exists
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Load existing syntheses if any
        self._load_syntheses()
        
        logger.info("Language Memory Synthesis Integration initialized")
    
    def _load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        default_config = {
            "cache_ttl": 300,
            "data_directory": "data/memory/synthesis",
            "components": {
                "conversation_memory": {
                    "enabled": True,
                    "class": "ConversationMemory",
                    "module": "conversation_memory"
                },
                "language_trainer": {
                    "enabled": True,
                    "class": "EnglishLanguageTrainer",
                    "module": "english_language_trainer"
                }
            },
            "synthesis": {
                "max_related_topics": 5,
                "max_insights_per_synthesis": 10
            },
            "api": {
                "rate_limit": 20,  # requests per minute
                "max_topic_length": 100,
                "allow_remote_connections": False
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def _initialize_components(self):
        """
        Initialize all memory components based on configuration.
        """
        component_config = self.config.get("components", {})
        
        for component_name, config in component_config.items():
            if not config.get("enabled", True):
                logger.info(f"Component {component_name} is disabled, skipping")
                continue
            
            try:
                module_name = config.get("module")
                class_name = config.get("class")
                
                if not module_name or not class_name:
                    logger.warning(f"Missing module or class name for {component_name}")
                    continue
                
                # Import the module
                module = __import__(module_name, fromlist=[class_name])
                component_class = getattr(module, class_name)
                
                # Initialize the component
                component = component_class()
                
                # Add to components dictionary
                self.components[component_name] = component
                
                logger.info(f"Component {component_name} initialized")
            except Exception as e:
                logger.error(f"Error initializing component {component_name}: {str(e)}")
    
    def _load_syntheses(self):
        """
        Load existing memory syntheses from file.
        """
        if not os.path.exists(self.synthesis_file):
            logger.info(f"No existing syntheses file found at {self.synthesis_file}")
            return
        
        try:
            loaded_syntheses = []
            with open(self.synthesis_file, 'r') as f:
                for line in f:
                    try:
                        synthesis = json.loads(line.strip())
                        loaded_syntheses.append(synthesis)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in syntheses file")
            
            with self._lock:
                self.syntheses = loaded_syntheses
                
                # Update statistics
                self.synthesis_stats["synthesis_count"] = len(loaded_syntheses)
                for synthesis in loaded_syntheses:
                    if "topics" in synthesis:
                        for topic in synthesis["topics"]:
                            self.synthesis_stats["topics_synthesized"].add(topic)
                
                if loaded_syntheses:
                    self.synthesis_stats["last_synthesis_timestamp"] = loaded_syntheses[-1].get("timestamp")
            
            logger.info(f"Loaded {len(loaded_syntheses)} syntheses")
        except Exception as e:
            logger.error(f"Error loading syntheses: {str(e)}")
    
    def _store_synthesis(self, synthesis):
        """
        Store a synthesized memory.
        
        Args:
            synthesis: The synthesized memory to store
        """
        try:
            with self._lock:
                self.syntheses.append(synthesis)
                
                # Update synthesis stats
                self.synthesis_stats["synthesis_count"] += 1
                for topic in synthesis.get("topics", []):
                    self.synthesis_stats["topics_synthesized"].add(topic)
                self.synthesis_stats["last_synthesis_timestamp"] = synthesis.get("timestamp")
                
                # Append to file
                with open(self.synthesis_file, 'a') as f:
                    f.write(json.dumps(synthesis) + '\n')
            
            logger.info(f"Stored synthesis with ID {synthesis.get('id')}")
        except Exception as e:
            logger.error(f"Error storing synthesis: {str(e)}")
    
    def _update_synthesis_cache(self, topic, synthesis):
        """
        Update the synthesis cache with a new synthesis.
        
        Args:
            topic: The topic of the synthesis
            synthesis: The synthesized memory
        """
        with self._lock:
            expiry_time = time.time() + self._cache_ttl
            self._synthesis_cache[topic] = synthesis
            self._synthesis_cache_expiry[topic] = expiry_time
    
    def retrieve_topic_synthesis(self, topic):
        """
        Retrieve the synthesis for a specific topic.
        
        Args:
            topic: The topic to retrieve
        
        Returns:
            Dictionary containing synthesis or None if not found
        """
        with self._lock:
            # Check cache first
            if topic in self._synthesis_cache:
                if self._synthesis_cache_expiry[topic] > time.time():
                    # Cache hit
                    self.performance_metrics["cache_hits"] += 1
                    logger.info(f"Cache hit for topic '{topic}'")
                    return {
                        "synthesis_results": {
                            "synthesized_memory": self._synthesis_cache[topic],
                            "related_topics": self._identify_related_topics(topic, self._synthesis_cache[topic])
                        },
                        "from_cache": True
                    }
                else:
                    # Cache expired
                    del self._synthesis_cache[topic]
                    del self._synthesis_cache_expiry[topic]
            
            # Cache miss
            self.performance_metrics["cache_misses"] += 1
            
            # Search for syntheses that include this topic
            for synthesis in reversed(self.syntheses):  # Most recent first
                if "topics" in synthesis and topic in synthesis["topics"]:
                    return {
                        "synthesis_results": {
                            "synthesized_memory": synthesis,
                            "related_topics": self._identify_related_topics(topic, synthesis)
                        },
                        "from_file": True
                    }
            
            return None
    
    def synthesize_topic(self, topic, depth=2, force_refresh=False):
        """
        Synthesize a topic from all available memory components.
        
        Args:
            topic: The topic to synthesize
            depth: How deep to search for related memories (1-3)
            force_refresh: Whether to force regeneration of synthesis
        
        Returns:
            Dictionary containing synthesis results
        """
        # Validate input
        if not topic or not isinstance(topic, str):
            return {"error": "Invalid topic - must be a non-empty string"}
        
        topic = topic.strip().lower()
        if len(topic) > self.config.get("api", {}).get("max_topic_length", 100):
            return {"error": f"Topic exceeds maximum length of {self.config.get('api', {}).get('max_topic_length', 100)}"}
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached = self.retrieve_topic_synthesis(topic)
            if cached:
                return cached
        
        # Start timing
        start_time = time.time()
        
        logger.info(f"Synthesizing topic: '{topic}' with depth {depth}")
        
        # Initialize response structure
        response = {
            "synthesis_results": None,
            "component_contributions": {},
            "errors": []
        }
        
        try:
            with self._lock:
                # Collect memories from all components
                collected_memories = self._collect_component_memories(topic, depth)
                response["component_contributions"] = collected_memories
                
                # Generate synthesis from collected memories
                synthesis = self._generate_synthesis(topic, collected_memories)
                response["synthesis_results"] = {
                    "synthesized_memory": synthesis,
                    "related_topics": self._identify_related_topics(topic, synthesis)
                }
                
                # Store the synthesis
                self._store_synthesis(synthesis)
                
                # Update cache
                self._update_synthesis_cache(topic, synthesis)
                
                # Update performance metrics
                end_time = time.time()
                synthesis_time = end_time - start_time
                
                self.performance_metrics["synthesis_times"].append(synthesis_time)
                self.performance_metrics["total_synthesis_time"] += synthesis_time
                self.performance_metrics["avg_synthesis_time"] = (
                    self.performance_metrics["total_synthesis_time"] / 
                    len(self.performance_metrics["synthesis_times"])
                )
        except Exception as e:
            error_msg = f"Error generating synthesis: {str(e)}"
            logger.error(error_msg)
            response["errors"].append(error_msg)
        
        return response
    
    def _collect_component_memories(self, topic, depth):
        """
        Collect memories related to a topic from all available components.
        
        Args:
            topic: The topic to search for
            depth: How deep to search for related memories
        
        Returns:
            Dictionary mapping component names to their memory contributions
        """
        component_memories = {}
        
        # Collect from conversation memory
        if "conversation_memory" in self.components:
            try:
                memory_component = self.components["conversation_memory"]
                # Combine memories from topic and keyword search
                topic_memories = memory_component.retrieve_by_topic(topic)
                keyword_memories = memory_component.retrieve_by_keyword(topic)
                text_search_memories = memory_component.search_text(topic)
                
                # Combine unique memories
                all_memories = list({m['id']: m for m in topic_memories + keyword_memories + text_search_memories}.values())
                
                component_memories["conversation_memory"] = {
                    "memories": all_memories,
                    "count": len(all_memories)
                }
                
                logger.info(f"Collected {len(all_memories)} memories from conversation memory")
            except Exception as e:
                logger.error(f"Error collecting conversation memories: {str(e)}")
                component_memories["conversation_memory"] = {"error": str(e)}
        
        # Collect from language trainer
        if "language_trainer" in self.components:
            try:
                language_trainer = self.components["language_trainer"]
                
                # Generate some training examples related to the topic
                training_data = language_trainer.generate_training_data(topic, 5)
                
                component_memories["language_trainer"] = {
                    "generated_examples": training_data,
                    "count": len(training_data)
                }
                
                logger.info(f"Generated {len(training_data)} examples from language trainer")
            except Exception as e:
                logger.error(f"Error collecting language trainer data: {str(e)}")
                component_memories["language_trainer"] = {"error": str(e)}
        
        # Add more component memory collection here
        
        return component_memories
    
    def _generate_synthesis(self, topic, component_memories):
        """
        Generate a synthesized memory from component contributions.
        
        Args:
            topic: The topic being synthesized
            component_memories: Dictionary of memories from each component
        
        Returns:
            A synthesized memory object
        """
        # In a production environment, this might use an LLM or other AI method
        # to create a more sophisticated synthesis
        
        # Extract active components that provided memories
        active_components = []
        for component, data in component_memories.items():
            if "error" not in data:
                active_components.append(component)
        
        # Create a timestamp for the synthesis
        timestamp = datetime.now().isoformat()
        
        # Generate a unique ID for this synthesis
        synthesis_id = str(uuid.uuid4())
        
        # Extract core insights from component memories
        core_insights = []
        
        # Process conversation memories
        if "conversation_memory" in component_memories and "memories" in component_memories["conversation_memory"]:
            conv_memories = component_memories["conversation_memory"]["memories"]
            for memory in conv_memories[:5]:  # Limit to first 5 for simplicity
                if "content" in memory:
                    core_insights.append(f"From conversation: {memory['content']}")
        
        # Process language trainer memories
        if "language_trainer" in component_memories and "generated_examples" in component_memories["language_trainer"]:
            examples = component_memories["language_trainer"]["generated_examples"]
            for example in examples[:3]:  # Limit to first 3 for simplicity
                core_insights.append(f"Language pattern: {example}")
        
        # Create synthesis object
        synthesis = {
            "id": synthesis_id,
            "timestamp": timestamp,
            "topics": [topic],
            "integrated_components": active_components,
            "core_understanding": f"Synthesized understanding of '{topic}' based on memories from {', '.join(active_components)}.",
            "novel_insights": [
                f"Topic '{topic}' appears in {len(active_components)} memory components",
                f"Created synthesis with {len(core_insights)} core insights"
            ] + core_insights[:3],  # Add top 3 insights
            "component_contributions": {
                component: len(data.get("memories", [])) if "memories" in data else 
                          len(data.get("generated_examples", [])) if "generated_examples" in data else 0
                for component, data in component_memories.items() if "error" not in data
            }
        }
        
        logger.info(f"Generated synthesis for topic '{topic}' with ID {synthesis_id}")
        return synthesis
    
    @lru_cache(maxsize=100)
    def _identify_related_topics(self, topic, synthesis):
        """
        Identify topics related to the synthesized memory.
        
        Args:
            topic: The original topic
            synthesis: The synthesized memory
        
        Returns:
            List of related topics
        """
        # In a production environment, this would use more sophisticated
        # semantic analysis or NLP techniques
        
        related_topics = [topic]
        
        # Extract potential related topics from core insights
        if "novel_insights" in synthesis:
            for insight in synthesis["novel_insights"]:
                # Simple approach: look for quoted terms
                if "'" in insight:
                    parts = insight.split("'")
                    for i in range(1, len(parts), 2):
                        if parts[i] != topic and len(parts[i]) > 2:
                            related_topics.append(parts[i])
        
        # Return unique topics limited by config
        max_topics = self.config.get("synthesis", {}).get("max_related_topics", 5)
        return list(set(related_topics))[:max_topics]
    
    def clear_cache(self):
        """Clear the synthesis cache"""
        with self._lock:
            self._synthesis_cache = {}
            self._synthesis_cache_expiry = {}
            logger.info("Synthesis cache cleared")
    
    def get_stats(self):
        """
        Get statistics about memory synthesis operations.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = {
                "synthesis_stats": {
                    "synthesis_count": self.synthesis_stats["synthesis_count"],
                    "topics_synthesized": list(self.synthesis_stats["topics_synthesized"]),
                    "last_synthesis_timestamp": self.synthesis_stats["last_synthesis_timestamp"]
                },
                "language_memory_stats": {},
                "component_stats": {},
                "performance_metrics": {
                    "avg_synthesis_time": round(self.performance_metrics["avg_synthesis_time"], 3),
                    "cache_hits": self.performance_metrics["cache_hits"],
                    "cache_misses": self.performance_metrics["cache_misses"],
                    "total_syntheses": len(self.performance_metrics["synthesis_times"]),
                    "cache_hit_ratio": round(
                        self.performance_metrics["cache_hits"] / 
                        max(1, self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]),
                        2
                    )
                }
            }
            
            # Get language memory stats if available
            if "conversation_memory" in self.components:
                try:
                    memory_stats = self.components["conversation_memory"].get_memory_stats()
                    stats["language_memory_stats"] = {
                        "memory_count": memory_stats.get("total_memories", 0),
                        "topics": memory_stats.get("top_topics", []),
                        "sentence_count": memory_stats.get("total_sentences", 0)
                    }
                except Exception as e:
                    logger.error(f"Error getting language memory stats: {str(e)}")
                    stats["language_memory_stats"] = {"error": str(e)}
            
            # Get component stats
            for component_name, component in self.components.items():
                stats["component_stats"][component_name] = {"active": True}
            
            return stats
    
    def shutdown(self):
        """
        Perform clean shutdown operations.
        
        This ensures all data is properly saved and resources are released.
        """
        logger.info("Shutting down Language Memory Synthesis Integration")
        
        # Ensure any cached data is saved
        with self._lock:
            # Clear caches to avoid memory leaks
            self.clear_cache()
            
            # Additional cleanup if needed
            
            logger.info("Language Memory Synthesis Integration shutdown complete")
    
    # -------------- LLM/NN Chat Integration Methods --------------
    
    def process_chat_message(self, message, user_id=None, session_id=None, metadata=None):
        """
        Process an incoming chat message, store in memory, and return relevant syntheses.
        
        Args:
            message (str): The chat message content
            user_id (str, optional): Unique identifier for the user
            session_id (str, optional): Unique identifier for the chat session
            metadata (dict, optional): Additional metadata about the message
            
        Returns:
            dict: Response with relevant memories and syntheses
        """
        start_time = time.time()
        
        if not message:
            return {"error": "Empty message provided", "status_code": 400}
            
        # Generate a default user_id if none provided
        if not user_id:
            user_id = "anonymous_user"
            
        # Generate a default session_id if none provided
        if not session_id:
            session_id = f"session_{str(uuid.uuid4())[:8]}"
            
        # Initialize metadata if None
        if metadata is None:
            metadata = {}
            
        response = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "message_stored": False,
            "relevant_topics": [],
            "relevant_syntheses": [],
            "processing_time_ms": 0,
            "errors": []
        }
        
        try:
            # Store message in conversation memory if available
            if "conversation_memory" in self.components:
                conversation = {
                    "content": message,
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "source": "chat_interface",
                        **metadata
                    }
                }
                
                self.components["conversation_memory"].store(conversation)
                response["message_stored"] = True
                logger.debug(f"Stored chat message for user {user_id} in session {session_id}")
                
                # Extract potential topics using more sophisticated approach
                potential_topics = self._extract_topics_from_message(message)
                response["relevant_topics"] = potential_topics
                
                # Get syntheses for potential topics (with parallel processing for production)
                syntheses = self._get_syntheses_for_topics(potential_topics)
                response["relevant_syntheses"] = syntheses
        except Exception as e:
            error_msg = f"Error processing chat message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            response["errors"].append(error_msg)
            
        # Calculate processing time
        end_time = time.time()
        response["processing_time_ms"] = round((end_time - start_time) * 1000, 2)
            
        return response
    
    def _extract_topics_from_message(self, message):
        """
        Extract potential topics from a message using simple NLP techniques.
        
        Args:
            message (str): The message to extract topics from
            
        Returns:
            list: Potential topics extracted from the message
        """
        # In production, this would use more sophisticated NLP or topic modeling
        
        # Simple approach: extract potential nouns and noun phrases
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
                     "by", "about", "as", "of", "this", "that", "is", "are", "was", "were", "be", 
                     "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", 
                     "shall", "should", "can", "could", "may", "might", "must"}
        
        # Clean the message and split into words
        cleaned_message = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in message.lower())
        words = [word.strip() for word in cleaned_message.split() if len(word.strip()) > 3 and word.strip().lower() not in stop_words]
        
        # Simple frequency-based approach
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get potential topics based on frequency and length
        potential_topics = sorted(word_freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        
        # Extract up to 5 potential topics
        return [topic for topic, _ in potential_topics[:5]]
    
    def _get_syntheses_for_topics(self, topics):
        """
        Get syntheses for a list of topics with optimized performance.
        
        Args:
            topics (list): List of topics to get syntheses for
            
        Returns:
            list: List of synthesis information for each topic
        """
        results = []
        
        # In production, this could use parallel processing for better performance
        for topic in topics:
            try:
                synthesis = self.retrieve_topic_synthesis(topic)
                if synthesis and "synthesis_results" in synthesis:
                    results.append({
                        "topic": topic,
                        "synthesis": synthesis["synthesis_results"]["synthesized_memory"],
                        "from_cache": synthesis.get("from_cache", False),
                        "related_topics": synthesis["synthesis_results"].get("related_topics", [])
                    })
            except Exception as e:
                logger.warning(f"Error retrieving synthesis for topic '{topic}': {str(e)}")
                # Continue processing other topics
        
        return results
    
    def generate_response_context(self, message, user_id=None, session_id=None, metadata=None):
        """
        Generate context for an LLM to respond to a message based on memory syntheses.
        
        Args:
            message (str): The user's message to respond to
            user_id (str, optional): Unique identifier for the user
            session_id (str, optional): Unique identifier for the chat session
            metadata (dict, optional): Additional metadata about the message
            
        Returns:
            dict: Context information for LLM response generation
        """
        start_time = time.time()
        
        # Process the message to store and get relevant syntheses
        process_result = self.process_chat_message(message, user_id, session_id, metadata)
        
        # Initialize context
        context = {
            "request_id": process_result.get("request_id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "user_id": user_id or "anonymous_user",
            "session_id": session_id,
            "memory_context": [],
            "historical_context": [],
            "processing_time_ms": 0,
            "errors": process_result.get("errors", [])
        }
        
        try:
            # Add synthesis information to context
            for synthesis_info in process_result.get("relevant_syntheses", []):
                synthesis = synthesis_info.get("synthesis", {})
                
                memory_item = {
                    "topic": synthesis_info.get("topic"),
                    "core_understanding": synthesis.get("core_understanding", ""),
                    "insights": synthesis.get("novel_insights", []),
                    "timestamp": synthesis.get("timestamp"),
                    "related_topics": synthesis_info.get("related_topics", [])
                }
                
                context["memory_context"].append(memory_item)
                
            # If no syntheses found but topics identified, suggest creating them
            if not context["memory_context"] and process_result.get("relevant_topics"):
                for topic in process_result.get("relevant_topics")[:2]:  # Limit to top 2 for performance
                    # Force creation of synthesis for this topic
                    new_synthesis = self.synthesize_topic(topic, force_refresh=True)
                    
                    if new_synthesis and "synthesis_results" in new_synthesis:
                        synthesis = new_synthesis["synthesis_results"]["synthesized_memory"]
                        
                        memory_item = {
                            "topic": topic,
                            "core_understanding": synthesis.get("core_understanding", ""),
                            "insights": synthesis.get("novel_insights", []),
                            "timestamp": synthesis.get("timestamp"),
                            "related_topics": new_synthesis["synthesis_results"].get("related_topics", []),
                            "newly_created": True
                        }
                        
                        context["memory_context"].append(memory_item)
            
            # Add recent conversation history for context if available
            if "conversation_memory" in self.components and session_id:
                try:
                    recent_memories = self.components["conversation_memory"].retrieve_recent(limit=5, filters={"session_id": session_id})
                    if recent_memories:
                        # Format memories for the context
                        for memory in recent_memories:
                            if memory.get("content"):
                                context["historical_context"].append({
                                    "content": memory.get("content"),
                                    "timestamp": memory.get("timestamp"),
                                    "metadata": memory.get("metadata", {})
                                })
                except Exception as e:
                    logger.warning(f"Error retrieving conversation history: {str(e)}")
                    
        except Exception as e:
            error_msg = f"Error generating response context: {str(e)}"
            logger.error(error_msg, exc_info=True)
            context["errors"].append(error_msg)
        
        # Calculate processing time
        end_time = time.time()
        context["processing_time_ms"] = round((end_time - start_time) * 1000, 2)
        
        return context
    
    def get_llm_system_prompt(self, user_id=None, session_id=None):
        """
        Generate a system prompt for LLM integration that provides context about memory capabilities.
        
        Args:
            user_id (str, optional): User identifier for personalized prompts
            session_id (str, optional): Session identifier for contextual prompts
            
        Returns:
            str: Formatted system prompt for LLM integration
        """
        # Get component statistics
        stats = self.get_stats()
        
        # Build a system prompt that includes memory capabilities
        system_prompt = (
            "You are an AI assistant with enhanced memory capabilities provided by the Language Memory Synthesis system. "
            "You can recall conversations and synthesize information across multiple interactions. "
            f"Your memory system contains {stats['language_memory_stats'].get('memory_count', 0)} memories "
            f"across {len(stats['synthesis_stats'].get('topics_synthesized', []))} topics. "
            "When responding, incorporate relevant memories naturally without explicitly mentioning the memory system. "
            "Focus on providing helpful, accurate responses that build on previous conversations when relevant."
        )
        
        # Add personalization if user_id is provided
        if user_id and "conversation_memory" in self.components:
            try:
                user_memories = self.components["conversation_memory"].retrieve_recent(
                    limit=5, filters={"user_id": user_id}
                )
                
                if user_memories:
                    system_prompt += (
                        f"\n\nYou have previously interacted with this user. "
                        f"You have {len(user_memories)} recent memories with them. "
                        "Use this context to provide continuity in your responses."
                    )
            except Exception as e:
                logger.warning(f"Error retrieving user history for prompt: {str(e)}")
                
        return system_prompt
        
    def api_health_check(self):
        """
        Verify the operational status of all components.
        Used for health monitoring in production.
        
        Returns:
            dict: Health status information
        """
        health_status = {
            "status": "operational",
            "version": self.config.get("version", "1.0.0"),
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check each component
        for name, component in self.components.items():
            try:
                # Try to access a basic attribute or method of each component
                if hasattr(component, "get_memory_stats"):
                    stats = component.get_memory_stats()
                    health_status["components"][name] = {
                        "status": "operational",
                        "memory_count": stats.get("total_memories", 0)
                    }
                else:
                    # Generic check if component exists
                    health_status["components"][name] = {
                        "status": "operational"
                    }
            except Exception as e:
                # Log and record the failure
                logger.error(f"Component {name} health check failed: {str(e)}")
                health_status["components"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                # If any component is not operational, change overall status
                health_status["status"] = "degraded"
        
        # Check synthesis functionality
        try:
            # Try a simple synthesis operation
            if self._synthesis_cache and len(self._synthesis_cache) > 0:
                # Use a topic from cache to avoid unnecessary processing
                test_topic = next(iter(self._synthesis_cache.keys()))
                health_status["components"]["synthesis"] = {
                    "status": "operational",
                    "cache_entries": len(self._synthesis_cache)
                }
            else:
                health_status["components"]["synthesis"] = {
                    "status": "operational",
                    "cache_entries": 0
                }
        except Exception as e:
            logger.error(f"Synthesis health check failed: {str(e)}")
            health_status["components"]["synthesis"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status

# -------------- External API Interface for LLM Integration --------------

class LanguageMemoryAPI:
    """
    External API interface for the Language Memory System that provides
    simple integration points for LLM and chat systems.
    
    This class serves as a wrapper around the LanguageMemorySynthesisIntegration
    to provide a clean, production-ready API for external services.
    """
    
    def __init__(self, synthesis_system=None, config=None):
        """
        Initialize the Language Memory API with an existing synthesis system
        or create a new one with the provided configuration.
        
        Args:
            synthesis_system (LanguageMemorySynthesisIntegration, optional): An existing synthesis system
            config (dict, optional): Configuration for creating a new synthesis system
        """
        self.start_time = time.time()
        
        if synthesis_system is not None:
            self.system = synthesis_system
            logger.info("LanguageMemoryAPI initialized with existing synthesis system")
        else:
            # Create a new system with default or provided configuration
            if config is None:
                config = {
                    "cache_enabled": True,
                    "cache_size": 100,
                    "persist_path": "data/memory/synthesis_cache.json",
                    "version": "1.0.0"
                }
            
            self.system = LanguageMemorySynthesisIntegration(config)
            logger.info("Created new LanguageMemorySynthesisIntegration for API")
        
        # API versioning and tracking
        self.api_version = "1.0.0"
        self.request_counter = 0
    
    # -------------- Core API Methods --------------
    
    def process_message(self, message, user_id=None, session_id=None, metadata=None):
        """
        Process a message and return memory-enhanced response context.
        Main integration point for LLM and chat systems.
        
        Args:
            message (str): The user message to process
            user_id (str, optional): Unique identifier for the user
            session_id (str, optional): Unique identifier for the chat session
            metadata (dict, optional): Additional metadata about the message
            
        Returns:
            dict: Complete context for LLM response generation
        """
        self.request_counter += 1
        
        # Start timing the request
        start_time = time.time()
        
        # Validate inputs
        if not isinstance(message, str) or not message.strip():
            return {
                "status": "error",
                "error": "Message cannot be empty",
                "status_code": 400,
                "request_id": str(uuid.uuid4())
            }
        
        # Generate response context
        try:
            # Get full response context from the synthesis system
            context = self.system.generate_response_context(
                message, 
                user_id=user_id, 
                session_id=session_id,
                metadata=metadata
            )
            
            # Get system prompt for LLM
            system_prompt = self.system.get_llm_system_prompt(user_id, session_id)
            
            # Create the complete response
            response = {
                "status": "success",
                "request_id": context.get("request_id"),
                "api_version": self.api_version,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "system_prompt": system_prompt,
                "context": context
            }
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "status": "error",
                "error": error_msg,
                "status_code": 500,
                "request_id": str(uuid.uuid4()),
                "api_version": self.api_version,
                "timestamp": datetime.now().isoformat()
            }
    
    def store_external_memory(self, content, metadata=None):
        """
        Store external memory (e.g., from LLM responses) in the memory system.
        
        Args:
            content (str): The content to store
            metadata (dict, optional): Additional metadata about the content
            
        Returns:
            dict: Storage result information
        """
        if metadata is None:
            metadata = {}
        
        try:
            # Create a memory object
            memory = {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source": "external_llm",
                    "is_llm_response": True,
                    **metadata
                }
            }
            
            # Store in conversation memory if available
            if "conversation_memory" in self.system.components:
                self.system.components["conversation_memory"].store(memory)
                
                return {
                    "status": "success",
                    "stored": True,
                    "timestamp": memory["timestamp"]
                }
            else:
                return {
                    "status": "error",
                    "error": "Conversation memory component not available",
                    "stored": False
                }
                
        except Exception as e:
            error_msg = f"Error storing external memory: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "status": "error",
                "error": error_msg,
                "stored": False
            }
    
    def get_topics(self, limit=20):
        """
        Get a list of synthesized topics available in the memory system.
        
        Args:
            limit (int, optional): Maximum number of topics to return
            
        Returns:
            dict: List of topics with metadata
        """
        try:
            stats = self.system.get_stats()
            topics = stats.get("synthesis_stats", {}).get("topics_synthesized", [])
            
            # Sort topics by recency if timestamp info is available
            if topics and isinstance(topics[0], dict) and "timestamp" in topics[0]:
                sorted_topics = sorted(topics, key=lambda x: x.get("timestamp", ""), reverse=True)
                topics_list = sorted_topics[:limit]
            else:
                # If just a list of topic strings, return them directly
                topics_list = topics[:limit]
            
            return {
                "status": "success",
                "count": len(topics_list),
                "topics": topics_list
            }
        
        except Exception as e:
            error_msg = f"Error retrieving topics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "status": "error",
                "error": error_msg,
                "count": 0,
                "topics": []
            }
    
    def get_topic_details(self, topic):
        """
        Get detailed information about a specific topic.
        
        Args:
            topic (str): The topic to retrieve details for
            
        Returns:
            dict: Topic synthesis and related information
        """
        try:
            # Retrieve synthesis for the topic
            synthesis = self.system.retrieve_topic_synthesis(topic)
            
            if synthesis and "synthesis_results" in synthesis:
                return {
                    "status": "success",
                    "topic": topic,
                    "synthesis": synthesis["synthesis_results"]["synthesized_memory"],
                    "related_topics": synthesis["synthesis_results"].get("related_topics", []),
                    "from_cache": synthesis.get("from_cache", False),
                    "timestamp": synthesis["synthesis_results"]["synthesized_memory"].get("timestamp")
                }
            else:
                return {
                    "status": "not_found",
                    "topic": topic,
                    "error": f"No synthesis found for topic '{topic}'"
                }
                
        except Exception as e:
            error_msg = f"Error retrieving topic details: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "status": "error",
                "topic": topic,
                "error": error_msg
            }
    
    def health_check(self):
        """
        Check the health and status of the Language Memory API and system.
        
        Returns:
            dict: Health status information
        """
        # Get system health check
        system_health = self.system.api_health_check()
        
        # Add API-specific information
        api_health = {
            "status": system_health.get("status", "unknown"),
            "api_version": self.api_version,
            "system_version": system_health.get("version", "unknown"),
            "uptime_seconds": int(time.time() - self.start_time),
            "requests_processed": self.request_counter,
            "components": system_health.get("components", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return api_health
    
    def clear_cache(self):
        """
        Clear the synthesis cache.
        
        Returns:
            dict: Result of the cache clearing operation
        """
        try:
            # Clear the cache in the system
            self.system.clear_cache()
            
            return {
                "status": "success",
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error clearing cache: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

# Example usage:
'''
# Create the API interface
memory_api = LanguageMemoryAPI()

# Process a message and get context for LLM
response = memory_api.process_message(
    "Can you tell me about machine learning?",
    user_id="user123",
    session_id="session456"
)

# Get system prompt for LLM
system_prompt = response.get("system_prompt")

# Get memory context to include in LLM context
memory_context = response.get("context", {}).get("memory_context", [])

# After LLM generates a response, store it
memory_api.store_external_memory(
    "Machine learning is a branch of artificial intelligence...",
    metadata={
        "user_id": "user123",
        "session_id": "session456",
        "is_response_to": "Can you tell me about machine learning?"
    }
)
'''

# Basic testing code
if __name__ == "__main__":
    # Integration initialization
    integration = LanguageMemorySynthesisIntegration()
    
    # Test topic synthesis
    results = integration.synthesize_topic("learning")
    
    # Print synthesis results
    print("\nSynthesis Results:")
    synthesis = results["synthesis_results"]["synthesized_memory"]
    print(f"ID: {synthesis['id']}")
    print(f"Topics: {synthesis['topics']}")
    print(f"Core Understanding: {synthesis['core_understanding']}")
    print(f"Novel Insights: {synthesis['novel_insights']}")
    
    # Test cache
    print("\nRunning cached synthesis...")
    cached_results = integration.synthesize_topic("learning")
    print(f"From cache: {cached_results.get('from_cache', False)}")
    
    # Test LLM integration methods
    print("\nTesting LLM Chat Integration:")
    chat_result = integration.process_chat_message("I need to learn about machine learning algorithms")
    print(f"Message stored: {chat_result.get('message_stored')}")
    print(f"Relevant topics: {chat_result.get('relevant_topics')}")
    print(f"Number of relevant syntheses: {len(chat_result.get('relevant_syntheses', []))}")
    
    # Test context generation
    context = integration.generate_response_context("Tell me about neural networks and deep learning")
    print("\nLLM Context:")
    print(f"User message: {context.get('user_message')}")
    print(f"Memory context items: {len(context.get('memory_context', []))}")
    
    # Print statistics
    stats = integration.get_stats()
    print("\nSystem Statistics:")
    print(f"Synthesis Count: {stats['synthesis_stats']['synthesis_count']}")
    print(f"Topics Synthesized: {stats['synthesis_stats']['topics_synthesized']}")
    print(f"Cache Hit Ratio: {stats['performance_metrics']['cache_hit_ratio']}")
    print(f"Average Synthesis Time: {stats['performance_metrics']['avg_synthesis_time']} seconds") 