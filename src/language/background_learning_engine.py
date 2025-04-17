#!/usr/bin/env python3
"""
Background Learning Engine

This module provides the core background learning and self-learning 
capabilities for the neural network project, integrating with both 
the language system and neural network components.
"""

import os
import sys
import time
import json
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from queue import Queue
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import components
try:
    from language.neural_linguistic_processor import NeuralLinguisticProcessor
    from language.language_memory import LanguageMemory
    from language.conscious_mirror_language import ConsciousMirrorLanguage
    from language.conversation_memory import ConversationMemory
    from language.database_manager import DatabaseManager
    from v7.autowiki import AutoWiki
    COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("Some components could not be imported. Running in limited mode.")
    COMPONENTS_AVAILABLE = False

class BackgroundLearningEngine:
    """
    Comprehensive background learning engine that coordinates various 
    learning mechanisms across the system.
    
    Core capabilities:
    1. Pattern extraction and analysis from conversation history
    2. Autonomous knowledge expansion through AutoWiki
    3. Neural network weight optimization based on feedback
    4. Cross-component learning synchronization
    5. Adaptive learning rate based on content complexity
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Background Learning Engine.
        
        Args:
            config: Configuration parameters for the engine
        """
        self.config = config or {}
        
        # Core flags
        self.running = False
        self.is_initialized = False
        
        # Component connections
        self.nlp = None  # NeuralLinguisticProcessor
        self.language_memory = None  # LanguageMemory
        self.conversation_memory = None  # ConversationMemory
        self.conscious_mirror = None  # ConsciousMirrorLanguage
        self.db_manager = None  # DatabaseManager
        self.autowiki = None  # AutoWiki
        
        # Learning parameters
        self.learning_config = {
            "enable_pattern_learning": True,
            "enable_concept_extraction": True,
            "enable_autowiki_learning": True,
            "enable_feedback_learning": True,
            "enable_neural_adaptation": True,
            "learning_interval_minutes": 15,
            "max_patterns_per_cycle": 50,
            "min_pattern_confidence": 0.65,
            "learning_rate": 0.01,
            "adaptation_threshold": 0.7,
            "memory_retention_days": 30
        }
        
        # Update with provided config
        if self.config.get("learning", None):
            self.learning_config.update(self.config.get("learning"))
        
        # Runtime statistics
        self.statistics = {
            "learning_cycles": 0,
            "patterns_extracted": 0,
            "concepts_learned": 0,
            "autowiki_entries": 0,
            "neural_adaptations": 0,
            "start_time": datetime.now().isoformat(),
            "last_cycle": None,
            "avg_cycle_time_ms": 0
        }
        
        # State and memory
        self.learned_patterns = {}
        self.concept_map = {}
        self.learning_queue = Queue()
        self.feedback_history = []
        
        # Background threads
        self.learning_thread = None
        self.pattern_thread = None
        self.autowiki_thread = None
        
        # Initialize and connect components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize and connect to required components"""
        try:
            if not COMPONENTS_AVAILABLE:
                logger.warning("Cannot initialize components: imports failed")
                return
                
            # Initialize NeuralLinguisticProcessor
            try:
                self.nlp = NeuralLinguisticProcessor()
                logger.info("Connected to NeuralLinguisticProcessor")
            except Exception as e:
                logger.error(f"Error initializing NeuralLinguisticProcessor: {e}")
                
            # Initialize LanguageMemory
            try:
                self.language_memory = LanguageMemory()
                logger.info("Connected to LanguageMemory")
            except Exception as e:
                logger.error(f"Error initializing LanguageMemory: {e}")
                
            # Initialize ConversationMemory
            try:
                self.conversation_memory = ConversationMemory()
                logger.info("Connected to ConversationMemory")
            except Exception as e:
                logger.error(f"Error initializing ConversationMemory: {e}")
                
            # Initialize ConsciousMirrorLanguage
            try:
                self.conscious_mirror = ConsciousMirrorLanguage()
                logger.info("Connected to ConsciousMirrorLanguage")
            except Exception as e:
                logger.error(f"Error initializing ConsciousMirrorLanguage: {e}")
                
            # Initialize DatabaseManager
            try:
                self.db_manager = DatabaseManager()
                logger.info("Connected to DatabaseManager")
            except Exception as e:
                logger.error(f"Error initializing DatabaseManager: {e}")
                
            # Initialize AutoWiki
            try:
                self.autowiki = AutoWiki(auto_fetch=False)
                logger.info("Connected to AutoWiki")
            except Exception as e:
                logger.error(f"Error initializing AutoWiki: {e}")
                
            # Mark as initialized
            self.is_initialized = True
            logger.info("Background Learning Engine components initialized")
            
        except Exception as e:
            logger.error(f"Error during component initialization: {e}")

    def start(self):
        """Start the background learning processes"""
        if not self.is_initialized:
            logger.warning("Cannot start: components not initialized")
            return False
            
        if self.running:
            logger.warning("Background learning already running")
            return True
            
        self.running = True
        
        # Start main learning thread
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
        # Start pattern extraction thread
        if self.learning_config["enable_pattern_learning"]:
            self.pattern_thread = threading.Thread(
                target=self._pattern_extraction_loop,
                daemon=True
            )
            self.pattern_thread.start()
            
        # Start autowiki learning thread
        if self.learning_config["enable_autowiki_learning"] and self.autowiki:
            self.autowiki_thread = threading.Thread(
                target=self._autowiki_learning_loop,
                daemon=True
            )
            self.autowiki_thread.start()
            
        logger.info("Background learning processes started")
        return True
        
    def stop(self):
        """Stop the background learning processes"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads to finish
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
            
        if self.pattern_thread and self.pattern_thread.is_alive():
            self.pattern_thread.join(timeout=5.0)
            
        if self.autowiki_thread and self.autowiki_thread.is_alive():
            self.autowiki_thread.join(timeout=5.0)
            
        logger.info("Background learning processes stopped")
        
    def _learning_loop(self):
        """Main learning loop that coordinates all learning activities"""
        while self.running:
            try:
                cycle_start = time.time()
                
                # Perform learning cycle
                self._perform_learning_cycle()
                
                # Calculate cycle time
                cycle_time = (time.time() - cycle_start) * 1000  # ms
                self.statistics["avg_cycle_time_ms"] = (
                    (self.statistics["avg_cycle_time_ms"] * self.statistics["learning_cycles"] + cycle_time) / 
                    (self.statistics["learning_cycles"] + 1)
                )
                
                # Update statistics
                self.statistics["learning_cycles"] += 1
                self.statistics["last_cycle"] = datetime.now().isoformat()
                
                # Save state periodically
                if self.statistics["learning_cycles"] % 10 == 0:
                    self._save_state()
                    
                # Sleep until next cycle
                interval_minutes = self.learning_config["learning_interval_minutes"]
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                time.sleep(60)  # Sleep on error
                
    def _perform_learning_cycle(self):
        """Perform a complete learning cycle"""
        logger.info("Starting learning cycle")
        
        # Process items in learning queue
        self._process_learning_queue()
        
        # Perform neural adaptation
        if self.learning_config["enable_neural_adaptation"]:
            self._adapt_neural_weights()
            
        # Extract and learn concepts
        if self.learning_config["enable_concept_extraction"]:
            self._extract_and_learn_concepts()
            
        # Update learning rate based on content complexity
        self._update_learning_parameters()
        
        logger.info(f"Completed learning cycle #{self.statistics['learning_cycles'] + 1}")
        
    def _pattern_extraction_loop(self):
        """Background loop for extracting patterns from conversation history"""
        while self.running:
            try:
                # Sleep for a bit to avoid overloading the system
                time.sleep(random.uniform(60, 120))
                
                # Skip if conversations not available
                if not self.conversation_memory:
                    continue
                    
                # Get recent conversations
                recent_conversations = self.conversation_memory.get_recent_conversations(
                    max_conversations=5
                )
                
                if not recent_conversations:
                    continue
                    
                # Process each conversation
                for conversation in recent_conversations:
                    self._extract_patterns_from_conversation(conversation)
                    
            except Exception as e:
                logger.error(f"Error in pattern extraction loop: {e}")
                time.sleep(120)  # Sleep on error
                
    def _extract_patterns_from_conversation(self, conversation):
        """Extract patterns from a conversation and add to learning queue"""
        if not self.nlp:
            return
            
        try:
            # Get exchanges from conversation
            exchanges = self.conversation_memory.get_exchanges(conversation["id"])
            
            if not exchanges:
                return
                
            # Combine all text for analysis
            combined_text = " ".join([
                exchange.get("user_input", "") + " " + exchange.get("system_response", "")
                for exchange in exchanges
            ])
            
            # Extract patterns
            patterns = self.nlp.recognize_patterns(combined_text)
            
            # Filter patterns by confidence
            confident_patterns = [
                p for p in patterns 
                if p.get("confidence", 0) >= self.learning_config["min_pattern_confidence"]
            ]
            
            # Add patterns to learning queue
            for pattern in confident_patterns[:self.learning_config["max_patterns_per_cycle"]]:
                self.learning_queue.put({
                    "type": "pattern",
                    "data": pattern,
                    "source": f"conversation_{conversation['id']}",
                    "timestamp": datetime.now().isoformat()
                })
                
            self.statistics["patterns_extracted"] += len(confident_patterns)
            
        except Exception as e:
            logger.error(f"Error extracting patterns from conversation: {e}")
            
    def _autowiki_learning_loop(self):
        """Background loop for AutoWiki learning"""
        if not self.autowiki:
            return
            
        while self.running:
            try:
                # Sleep to avoid overloading the system
                time.sleep(random.uniform(300, 600))  # 5-10 minutes
                
                # Process a few items from the queue
                result = self.autowiki.process_queue(max_items=3)
                
                if result["success"] > 0:
                    # Update statistics
                    self.statistics["autowiki_entries"] += result["success"]
                    
                    # Extract patterns from new autowiki entries
                    for topic in result["topics"]:
                        self._extract_patterns_from_topic(topic)
                
            except Exception as e:
                logger.error(f"Error in autowiki learning loop: {e}")
                time.sleep(300)  # Sleep on error
                
    def _extract_patterns_from_topic(self, topic):
        """Extract patterns from an autowiki topic"""
        if not self.nlp or not self.autowiki:
            return
            
        try:
            # Get topic content
            content = self.autowiki.retrieve_autowiki(topic)
            
            if not content:
                return
                
            # Extract patterns
            patterns = self.nlp.recognize_patterns(content.get("content", ""))
            
            # Filter patterns by confidence
            confident_patterns = [
                p for p in patterns 
                if p.get("confidence", 0) >= self.learning_config["min_pattern_confidence"]
            ]
            
            # Add patterns to learning queue
            for pattern in confident_patterns[:self.learning_config["max_patterns_per_cycle"]]:
                self.learning_queue.put({
                    "type": "pattern",
                    "data": pattern,
                    "source": f"autowiki_{topic}",
                    "timestamp": datetime.now().isoformat()
                })
                
            self.statistics["patterns_extracted"] += len(confident_patterns)
            
        except Exception as e:
            logger.error(f"Error extracting patterns from topic: {e}")
            
    def _process_learning_queue(self):
        """Process items in the learning queue"""
        processed = 0
        
        while not self.learning_queue.empty() and processed < 100:
            try:
                item = self.learning_queue.get()
                
                # Process based on item type
                if item["type"] == "pattern":
                    self._learn_pattern(item["data"], item["source"])
                elif item["type"] == "concept":
                    self._learn_concept(item["data"], item["source"])
                elif item["type"] == "feedback":
                    self._process_feedback(item["data"])
                    
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processing learning queue item: {e}")
                
            finally:
                self.learning_queue.task_done()
                
        if processed > 0:
            logger.info(f"Processed {processed} items from learning queue")
            
    def _learn_pattern(self, pattern, source):
        """Learn a pattern and store it in memory"""
        if not self.language_memory:
            return
            
        try:
            # Generate a pattern ID if not present
            if "id" not in pattern:
                pattern["id"] = f"pattern_{hash(pattern.get('text', ''))}"
                
            # Store in learned patterns
            self.learned_patterns[pattern["id"]] = {
                "pattern": pattern,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "usage_count": 0
            }
            
            # Store word associations in language memory
            if "text" in pattern:
                words = pattern["text"].split()
                for i in range(len(words) - 1):
                    self.language_memory.store_word_association(
                        words[i], words[i+1], pattern.get("confidence", 0.5)
                    )
                    
            # Store in database if available
            if self.db_manager:
                # Add pattern detection to the database
                self.db_manager.add_pattern_detection({
                    "pattern_text": pattern.get("text", ""),
                    "pattern_type": pattern.get("type", "unknown"),
                    "confidence": pattern.get("confidence", 0.5),
                    "source": source
                })
                
            # Update statistics
            self.statistics["concepts_learned"] += 1
            
        except Exception as e:
            logger.error(f"Error learning pattern: {e}")
            
    def _extract_and_learn_concepts(self):
        """Extract and learn concepts from recent conversations"""
        if not self.conversation_memory:
            return
            
        try:
            # Get recent exchanges
            recent_exchanges = self.conversation_memory.get_recent_exchanges(max_results=20)
            
            if not recent_exchanges:
                return
                
            # Extract concepts from exchanges
            concepts = {}
            
            for exchange in recent_exchanges:
                exchange_concepts = exchange.get("concepts", {})
                
                for concept_name, concept_data in exchange_concepts.items():
                    if concept_name not in concepts:
                        concepts[concept_name] = concept_data.copy()
                    else:
                        # Update existing concept with new data
                        existing = concepts[concept_name]
                        existing["importance"] = max(
                            existing["importance"], 
                            concept_data.get("importance", 0)
                        )
                        existing["count"] = existing.get("count", 1) + 1
                        
            # Learn important concepts
            for concept_name, concept_data in concepts.items():
                if concept_data.get("importance", 0) >= 0.6:
                    self._learn_concept(
                        {
                            "name": concept_name,
                            "data": concept_data
                        },
                        "conversation_mining"
                    )
                    
        except Exception as e:
            logger.error(f"Error extracting and learning concepts: {e}")
            
    def _learn_concept(self, concept, source):
        """Learn a concept and store it in concept map"""
        try:
            concept_name = concept.get("name", "")
            
            if not concept_name:
                return
                
            # Store in concept map
            if concept_name not in self.concept_map:
                self.concept_map[concept_name] = {
                    "data": concept.get("data", {}),
                    "sources": [source],
                    "first_seen": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "importance": concept.get("data", {}).get("importance", 0.5),
                    "count": 1
                }
            else:
                # Update existing concept
                existing = self.concept_map[concept_name]
                existing["last_updated"] = datetime.now().isoformat()
                existing["count"] += 1
                
                if source not in existing["sources"]:
                    existing["sources"].append(source)
                    
                # Update importance
                new_importance = concept.get("data", {}).get("importance", 0.5)
                existing["importance"] = max(existing["importance"], new_importance)
                
            # Update statistics
            self.statistics["concepts_learned"] += 1
            
        except Exception as e:
            logger.error(f"Error learning concept: {e}")
            
    def _process_feedback(self, feedback):
        """Process feedback for learning adaptation"""
        try:
            # Add to feedback history
            self.feedback_history.append({
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit history size
            if len(self.feedback_history) > 100:
                self.feedback_history = self.feedback_history[-100:]
                
            # Update conversation memory if available
            if (self.conversation_memory and 
                "exchange_id" in feedback and 
                "value" in feedback):
                    
                self.conversation_memory.learn_from_feedback(
                    feedback["exchange_id"],
                    feedback["value"],
                    feedback.get("type", "explicit")
                )
                
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            
    def _adapt_neural_weights(self):
        """Adapt neural weights based on learning patterns"""
        if not self.nlp:
            return
            
        try:
            # Calculate adaptation factors from recent patterns
            adaptation_factors = self._calculate_adaptation_factors()
            
            if not adaptation_factors:
                return
                
            # Apply adaptations to NLP weight
            if hasattr(self.nlp, "set_neural_weight"):
                current_weight = getattr(self.nlp, "nn_weight", 0.5)
                
                # Calculate new weight
                new_weight = current_weight
                
                if adaptation_factors.get("increase_neural", False):
                    new_weight = min(0.9, current_weight + self.learning_config["learning_rate"])
                    
                elif adaptation_factors.get("decrease_neural", False):
                    new_weight = max(0.1, current_weight - self.learning_config["learning_rate"])
                    
                # Apply if changed
                if new_weight != current_weight:
                    self.nlp.set_neural_weight(new_weight)
                    logger.info(f"Adapted NLP neural weight to {new_weight:.2f}")
                    self.statistics["neural_adaptations"] += 1
                
            # Apply adaptations to LLM weight
            if hasattr(self.nlp, "set_llm_weight"):
                current_weight = getattr(self.nlp, "llm_weight", 0.5)
                
                # Calculate new weight
                new_weight = current_weight
                
                if adaptation_factors.get("increase_llm", False):
                    new_weight = min(0.9, current_weight + self.learning_config["learning_rate"])
                    
                elif adaptation_factors.get("decrease_llm", False):
                    new_weight = max(0.1, current_weight - self.learning_config["learning_rate"])
                    
                # Apply if changed
                if new_weight != current_weight:
                    self.nlp.set_llm_weight(new_weight)
                    logger.info(f"Adapted NLP LLM weight to {new_weight:.2f}")
                    self.statistics["neural_adaptations"] += 1
                    
        except Exception as e:
            logger.error(f"Error adapting neural weights: {e}")
            
    def _calculate_adaptation_factors(self):
        """Calculate adaptation factors based on learning history"""
        if len(self.feedback_history) < 5:
            return {}
            
        # Calculate average feedback value from recent history
        recent_feedback = self.feedback_history[-20:]
        avg_feedback = sum(fb["feedback"].get("value", 0.5) for fb in recent_feedback) / len(recent_feedback)
        
        # Get feedback trend (increasing or decreasing)
        if len(recent_feedback) >= 10:
            first_half = recent_feedback[:len(recent_feedback)//2]
            second_half = recent_feedback[len(recent_feedback)//2:]
            
            first_avg = sum(fb["feedback"].get("value", 0.5) for fb in first_half) / len(first_half)
            second_avg = sum(fb["feedback"].get("value", 0.5) for fb in second_half) / len(second_half)
            
            trend = second_avg - first_avg
        else:
            trend = 0
            
        # Determine adaptation factors
        factors = {}
        
        # Neural weight adaptations
        if avg_feedback > 0.7 and trend >= 0:
            # Responses are good and getting better - increase neural influence
            factors["increase_neural"] = True
        elif avg_feedback < 0.4 and trend <= 0:
            # Responses are poor and getting worse - decrease neural influence
            factors["decrease_neural"] = True
            
        # LLM weight adaptations
        if avg_feedback < 0.5 and trend < 0:
            # Responses are getting worse - increase LLM influence
            factors["increase_llm"] = True
        elif avg_feedback > 0.8 and trend > 0:
            # Responses are excellent and improving - might reduce LLM reliance
            factors["decrease_llm"] = True
            
        return factors
        
    def _update_learning_parameters(self):
        """Update learning parameters based on content complexity"""
        try:
            # Calculate average pattern complexity
            recent_patterns = list(self.learned_patterns.values())[-50:]
            
            if not recent_patterns:
                return
                
            # Calculate average complexity
            complexity_sum = sum(
                p["pattern"].get("complexity", 0.5) 
                for p in recent_patterns 
                if "complexity" in p["pattern"]
            )
            
            if complexity_sum > 0:
                avg_complexity = complexity_sum / len(recent_patterns)
                
                # Adjust learning rate based on complexity
                if avg_complexity > 0.7:
                    # More complex content - slower learning
                    self.learning_config["learning_rate"] = max(
                        0.005, 
                        self.learning_config["learning_rate"] * 0.95
                    )
                elif avg_complexity < 0.3:
                    # Simpler content - faster learning
                    self.learning_config["learning_rate"] = min(
                        0.05, 
                        self.learning_config["learning_rate"] * 1.05
                    )
                    
                logger.info(f"Updated learning rate to {self.learning_config['learning_rate']:.4f} based on complexity")
                
        except Exception as e:
            logger.error(f"Error updating learning parameters: {e}")
            
    def _save_state(self):
        """Save the learning engine state to disk"""
        try:
            # Create state directory
            state_dir = os.path.join("data", "background_learning")
            os.makedirs(state_dir, exist_ok=True)
            
            # Save statistics
            with open(os.path.join(state_dir, "statistics.json"), "w") as f:
                json.dump(self.statistics, f, indent=2)
                
            # Save learned patterns (latest 1000)
            recent_patterns = dict(list(self.learned_patterns.items())[-1000:])
            with open(os.path.join(state_dir, "learned_patterns.json"), "w") as f:
                json.dump(recent_patterns, f, indent=2)
                
            # Save concept map
            with open(os.path.join(state_dir, "concept_map.json"), "w") as f:
                json.dump(self.concept_map, f, indent=2)
                
            logger.info("Saved background learning engine state")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    def _load_state(self):
        """Load the learning engine state from disk"""
        try:
            state_dir = os.path.join("data", "background_learning")
            
            # Load statistics
            stats_path = os.path.join(state_dir, "statistics.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    self.statistics.update(json.load(f))
                    
            # Load learned patterns
            patterns_path = os.path.join(state_dir, "learned_patterns.json")
            if os.path.exists(patterns_path):
                with open(patterns_path, "r") as f:
                    self.learned_patterns = json.load(f)
                    
            # Load concept map
            concepts_path = os.path.join(state_dir, "concept_map.json")
            if os.path.exists(concepts_path):
                with open(concepts_path, "r") as f:
                    self.concept_map = json.load(f)
                    
            logger.info("Loaded background learning engine state")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            
    def add_learning_item(self, item_type: str, data: Any, source: str = "manual") -> bool:
        """
        Add an item to the learning queue
        
        Args:
            item_type: Type of item (pattern, concept, feedback)
            data: The data to learn
            source: Source of the data
            
        Returns:
            bool: Success status
        """
        try:
            self.learning_queue.put({
                "type": item_type,
                "data": data,
                "source": source,
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Error adding learning item: {e}")
            return False
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return self.statistics.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the learning engine"""
        return {
            "running": self.running,
            "initialized": self.is_initialized,
            "learning_rate": self.learning_config["learning_rate"],
            "queue_size": self.learning_queue.qsize(),
            "patterns_count": len(self.learned_patterns),
            "concepts_count": len(self.concept_map),
            "learning_cycles": self.statistics["learning_cycles"],
            "components": {
                "nlp": self.nlp is not None,
                "language_memory": self.language_memory is not None,
                "conversation_memory": self.conversation_memory is not None,
                "conscious_mirror": self.conscious_mirror is not None,
                "db_manager": self.db_manager is not None,
                "autowiki": self.autowiki is not None
            }
        }
        
    def clear_learning_history(self, days_to_keep: int = 7) -> int:
        """
        Clear old learning history
        
        Args:
            days_to_keep: Number of days of history to keep
            
        Returns:
            int: Number of items removed
        """
        try:
            # Calculate cutoff date
            cutoff = datetime.now().timestamp() - (days_to_keep * 86400)
            cutoff_date = datetime.fromtimestamp(cutoff).isoformat()
            
            # Clear old patterns
            original_count = len(self.learned_patterns)
            self.learned_patterns = {
                k: v for k, v in self.learned_patterns.items()
                if v.get("timestamp", "9999") > cutoff_date
            }
            
            removed_count = original_count - len(self.learned_patterns)
            logger.info(f"Cleared {removed_count} patterns from learning history")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error clearing learning history: {e}")
            return 0


def get_background_learning_engine(config: Dict[str, Any] = None) -> BackgroundLearningEngine:
    """
    Get the singleton instance of the background learning engine
    
    Args:
        config: Configuration parameters for the engine
        
    Returns:
        BackgroundLearningEngine: The singleton instance
    """
    # TODO: Implement singleton pattern
    return BackgroundLearningEngine(config) 