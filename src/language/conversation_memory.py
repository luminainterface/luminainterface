#!/usr/bin/env python3
"""
Conversation Memory Module

Provides persistent memory storage for conversations, enabling the system to learn
from interactions and develop contextual awareness over time.
"""

import os
import json
import time
import logging
import uuid
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Import database manager for persistence
from .database_manager import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Manages storage and retrieval of conversation history with learning capabilities.
    
    Features:
    - Stores complete conversation history
    - Tracks key concepts mentioned in conversations
    - Builds semantic associations between topics
    - Develops user preference model
    - Provides context-aware retrieval
    - Learns from interaction patterns
    """
    
    def create_conversation(self):
        """
        Create a new conversation
        
        Returns:
            str: Conversation ID
        """
        # Use the database manager to create a conversation if available
        if hasattr(self, 'db_manager') and self.db_manager:
            return self.db_manager.create_conversation()
        
        # Generate a unique ID
        import uuid
        return str(uuid.uuid4())

    def __init__(self, data_dir: str = "data/conversation_memory"):
        """
        Initialize the conversation memory system.
        
        Args:
            data_dir: Directory to store memory files
        """
        logger.info("Initializing Conversation Memory")
        
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize the database manager
        self.db_manager = DatabaseManager(data_dir=data_dir)
        
        # Create a new conversation
        self.current_conversation_id = self.db_manager.create_conversation(
            title="Conversation " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata={"start_time": datetime.datetime.now().isoformat()}
        )
        
        # Memory structures
        self.exchanges = []
        self.concepts = {}
        self.concept_associations = {}
        self.user_preferences = {}
        
        # Memory stats
        self.total_exchanges = 0
        self.unique_concepts = 0
        self.total_learning_value = 0.0
        
        logger.info(f"Conversation Memory initialized with conversation ID: {self.current_conversation_id}")
    
    def store_exchange(self, user_message: str, system_response: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a conversation exchange in memory
        
        Args:
            user_message: The message from the user
            system_response: The response from the system
            metadata: Additional metadata about the exchange
            
        Returns:
            str: ID of the stored exchange
        """
        if metadata is None:
            metadata = {}
            
        # Generate unique ID for this exchange
        exchange_id = f"exchange_{int(time.time())}_{len(self.exchanges)}"
        
        # Store timestamp
        timestamp = datetime.datetime.now()
        
        # Set default values if not provided
        if "llm_weight" not in metadata:
            metadata["llm_weight"] = 0.5
        if "nn_weight" not in metadata:
            metadata["nn_weight"] = 0.5
        
        # Default learning value (can be updated later with feedback)
        learning_value = metadata.get("learning_value", 0.3)
        
        # Get component scores from metadata if available
        consciousness_level = metadata.get("consciousness_level", 0.0)
        neural_score = metadata.get("neural_score", 0.0)
        pattern_confidence = metadata.get("pattern_confidence", 0.0)
        
        # Store in database
        db_exchange_id = self.db_manager.store_exchange(
            conversation_id=self.current_conversation_id,
            user_message=user_message,
            system_response=system_response,
            llm_weight=metadata.get("llm_weight", 0.5),
            nn_weight=metadata.get("nn_weight", 0.5),
            learning_value=learning_value,
            consciousness_level=consciousness_level,
            neural_score=neural_score,
            pattern_confidence=pattern_confidence,
            metadata=metadata
        )
        
        # Store pattern detections if available
        if "patterns" in metadata and isinstance(metadata["patterns"], list):
            for pattern in metadata["patterns"]:
                if isinstance(pattern, dict) and "type" in pattern and "text" in pattern:
                    self.db_manager.add_pattern_detection(
                        exchange_id=db_exchange_id,
                        pattern_type=pattern["type"],
                        pattern_text=pattern["text"],
                        confidence=pattern.get("confidence", 0.5),
                        detection_method=pattern.get("method", "unknown")
                    )
        
        # Extract concepts from the exchange
        concepts = self._extract_concepts(user_message, system_response)
        
        # Store concepts in database
        for concept, importance in concepts.items():
            self.db_manager.add_concept(
                exchange_id=db_exchange_id,
                concept_text=concept,
                importance=importance
            )
        
        # Update memory structures
        exchange_data = {
            "id": exchange_id,
            "db_id": db_exchange_id,
            "user_message": user_message,
            "system_response": system_response,
            "timestamp": timestamp.isoformat(),
            "concepts": concepts,
            "metadata": metadata,
            "learning_value": learning_value
        }
        
        self.exchanges.append(exchange_data)
        self.total_exchanges += 1
        self.total_learning_value += learning_value
        
        # Update concept memory
        for concept, importance in concepts.items():
            if concept not in self.concepts:
                self.concepts[concept] = {
                    "count": 1,
                    "importance": importance,
                    "first_seen": timestamp.isoformat(),
                    "last_seen": timestamp.isoformat(),
                    "exchanges": [exchange_id]
                }
                self.unique_concepts += 1
            else:
                self.concepts[concept]["count"] += 1
                self.concepts[concept]["importance"] = (
                    self.concepts[concept]["importance"] * 0.7 + importance * 0.3
                )
                self.concepts[concept]["last_seen"] = timestamp.isoformat()
                self.concepts[concept]["exchanges"].append(exchange_id)
        
        # Update concept associations
        for concept1 in concepts:
            if concept1 not in self.concept_associations:
                self.concept_associations[concept1] = {}
                
            for concept2 in concepts:
                if concept1 != concept2:
                    if concept2 not in self.concept_associations[concept1]:
                        self.concept_associations[concept1][concept2] = 1
                    else:
                        self.concept_associations[concept1][concept2] += 1
        
        # Record metrics
        self.db_manager.record_metric(
            metric_name="exchange_count",
            metric_value=self.total_exchanges,
            metric_type="learning"
        )
        
        self.db_manager.record_metric(
            metric_name="concept_count",
            metric_value=self.unique_concepts,
            metric_type="learning"
        )
        
        logger.info(f"Stored exchange in conversation memory with ID: {exchange_id}")
        return exchange_id
    
    def get_context(self, user_query: str, max_results: int = 3) -> List[Dict]:
        """
        Get relevant context based on the user query
        
        Args:
            user_query: The user's query
            max_results: Maximum number of context items to return
            
        Returns:
            List[Dict]: List of relevant exchanges as context
        """
        if not user_query or not self.exchanges:
            return []
        
        # Extract concepts from the query
        query_concepts = self._extract_concepts(user_query, "")
        
        # If no concepts found, search by text similarity
        if not query_concepts:
            # Use the database search functionality
            search_results = self.db_manager.search_exchanges(user_query, limit=max_results)
            return search_results
            
        # Score exchanges based on concept overlap
        scored_exchanges = []
        for exchange in self.exchanges:
            score = 0.0
            exchange_concepts = exchange.get("concepts", {})
            
            # Calculate concept overlap score
            for query_concept, query_importance in query_concepts.items():
                if query_concept in exchange_concepts:
                    # Score based on concept importance in both query and exchange
                    concept_score = query_importance * exchange_concepts[query_concept]
                    score += concept_score
                    
                    # Boost score for more recent exchanges
                    time_factor = 1.0
                    if "timestamp" in exchange:
                        try:
                            exchange_time = datetime.datetime.fromisoformat(exchange["timestamp"])
                            time_diff = (datetime.datetime.now() - exchange_time).total_seconds()
                            # Decay factor for older exchanges
                            time_factor = max(0.5, min(1.0, 1.0 - (time_diff / (24 * 3600 * 7))))
                        except (ValueError, TypeError):
                            pass
                            
                    score *= time_factor
                    
                # Also check for similar concepts through associations
                elif query_concept in self.concept_associations:
                    for related_concept, strength in self.concept_associations[query_concept].items():
                        if related_concept in exchange_concepts:
                            # Lower score for associated concepts
                            assoc_score = query_importance * exchange_concepts[related_concept] * (strength / 10)
                            score += assoc_score
            
            # Factor in the learning value
            learning_boost = exchange.get("learning_value", 0.3)
            score *= (1.0 + learning_boost)
            
            scored_exchanges.append((exchange, score))
        
        # Sort by score and take top results
        scored_exchanges.sort(key=lambda x: x[1], reverse=True)
        top_exchanges = [ex for ex, score in scored_exchanges[:max_results]]
        
        return top_exchanges
    
    def learn_from_feedback(self, exchange_id: str, feedback_value: float, 
                          feedback_type: str = "explicit") -> bool:
        """
        Update memory based on explicit feedback
        
        Args:
            exchange_id: ID of the exchange
            feedback_value: Feedback value (0.0-1.0)
            feedback_type: Type of feedback (explicit, implicit)
            
        Returns:
            bool: Success status
        """
        # Find the exchange
        exchange_index = None
        for i, exchange in enumerate(self.exchanges):
            if exchange["id"] == exchange_id:
                exchange_index = i
                break
                
        if exchange_index is None:
            logger.warning(f"Exchange with ID {exchange_id} not found")
            return False
            
        # Update learning value based on feedback
        current_value = self.exchanges[exchange_index].get("learning_value", 0.3)
        
        # Weight explicit feedback more heavily
        if feedback_type == "explicit":
            # 70% new feedback, 30% current value
            new_value = feedback_value * 0.7 + current_value * 0.3
        else:
            # 30% new feedback, 70% current value
            new_value = feedback_value * 0.3 + current_value * 0.7
            
        self.exchanges[exchange_index]["learning_value"] = new_value
        
        # Update user preferences based on this feedback
        exchange = self.exchanges[exchange_index]
        
        # Extract key concepts that might represent preferences
        concepts = exchange.get("concepts", {})
        for concept, importance in concepts.items():
            # Only consider important concepts
            if importance > 0.6:
                if concept not in self.user_preferences:
                    self.user_preferences[concept] = {
                        "value": feedback_value,
                        "confidence": importance * 0.5,
                        "examples": [exchange_id]
                    }
                else:
                    # Update existing preference
                    self.user_preferences[concept]["value"] = (
                        self.user_preferences[concept]["value"] * 0.7 + feedback_value * 0.3
                    )
                    self.user_preferences[concept]["confidence"] = min(
                        0.95, 
                        self.user_preferences[concept]["confidence"] + 0.05
                    )
                    if exchange_id not in self.user_preferences[concept]["examples"]:
                        self.user_preferences[concept]["examples"].append(exchange_id)
                
                # Update in database
                self.db_manager.update_user_preference(
                    preference_key=concept,
                    preference_value=self.user_preferences[concept]["value"],
                    confidence=self.user_preferences[concept]["confidence"],
                    example_exchange_id=exchange.get("db_id")
                )
        
        # Update total learning value
        self.total_learning_value = sum(ex.get("learning_value", 0.0) for ex in self.exchanges)
        
        # Record feedback metric
        self.db_manager.record_metric(
            metric_name="feedback_received",
            metric_value=feedback_value,
            metric_type="learning",
            details={"exchange_id": exchange_id, "feedback_type": feedback_type}
        )
        
        logger.info(f"Updated learning value for exchange {exchange_id} to {new_value:.3f}")
        return True
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning progress
        
        Returns:
            Dict: Learning statistics
        """
        # Get stats from the database for accurate counts
        db_stats = self.db_manager.get_learning_statistics()
        
        # Calculate learning rate
        if self.total_exchanges > 0:
            avg_learning_value = self.total_learning_value / self.total_exchanges
        else:
            avg_learning_value = 0.0
            
        # Calculate concept density
        if self.total_exchanges > 0:
            concept_density = self.unique_concepts / self.total_exchanges
        else:
            concept_density = 0.0
            
        # Get top concepts by importance
        top_concepts = sorted(
            [(c, details["importance"]) for c, details in self.concepts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Merge with database stats
        stats = {
            "total_exchanges": db_stats.get("total_exchanges", self.total_exchanges),
            "unique_concepts": db_stats.get("total_concepts_extracted", self.unique_concepts),
            "avg_learning_value": db_stats.get("avg_learning_value", avg_learning_value),
            "concept_density": concept_density,
            "top_concepts": [{"concept": c, "importance": i} for c, i in top_concepts],
            "user_preferences": len(self.user_preferences),
            "concept_associations": sum(len(assocs) for assocs in self.concept_associations.values()),
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        return stats
    
    def save_memories(self) -> bool:
        """
        Save all memories to disk.
        
        Returns:
            Success status
        """
        try:
            # Save concepts
            concepts_file = os.path.join(self.data_dir, "concepts.json")
            with open(concepts_file, 'w') as f:
                json.dump(self.concepts, f, indent=2)
                
            # Save concept associations
            associations_file = os.path.join(self.data_dir, "concept_associations.json")
            with open(associations_file, 'w') as f:
                json.dump(self.concept_associations, f, indent=2)
                
            # Save user preferences
            preferences_file = os.path.join(self.data_dir, "user_preferences.json")
            with open(preferences_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
                
            logger.info("Saved conversation memory to disk")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation memory: {e}")
            return False
    
    def _extract_concepts(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Extract key concepts from text
        
        Args:
            text1: First text to analyze
            text2: Second text to analyze
            
        Returns:
            Dict[str, float]: Concepts with importance scores
        """
        # Simple concept extraction for now - just get important words
        # In a real system, this would use NLP techniques like entity extraction,
        # keyword extraction, etc.
        combined_text = (text1 + " " + text2).lower()
        
        # Remove basic punctuation and split by whitespace
        words = combined_text.replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").split()
        
        # Simple stopwords list
        stopwords = {
            "a", "an", "the", "and", "is", "it", "in", "to", "for", "of", "that", "this",
            "was", "with", "be", "are", "on", "at", "by", "as", "i", "you", "we", "they",
            "he", "she", "what", "how", "when", "where", "why", "but", "or", "so", "if"
        }
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 2 and word not in stopwords:
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
                    
        # Calculate importance scores
        total_words = sum(word_counts.values())
        concepts = {}
        
        if total_words > 0:
            for word, count in word_counts.items():
                # Score based on frequency and length of word (simple TF-IDF like approach)
                importance = (count / total_words) * min(1.0, len(word) / 10)
                
                # Only keep somewhat important concepts
                if importance > 0.05:
                    concepts[word] = importance
                    
        return concepts
    
    def close(self):
        """Close memory system and save state"""
        self.save_memories()
        self.db_manager.close()
        logger.info("Conversation memory system closed")


# For testing
if __name__ == "__main__":
    memory = ConversationMemory()
    
    # Test storing exchanges
    memory.store_exchange(
        "What is the difference between neural networks and symbolic AI?",
        "Neural networks learn patterns from data, while symbolic AI uses explicit rules and logic.",
        {"llm_weight": 0.7, "nn_weight": 0.6}
    )
    
    memory.store_exchange(
        "How do neural networks learn?",
        "Neural networks learn by adjusting weights through backpropagation based on training examples.",
        {"llm_weight": 0.7, "nn_weight": 0.6}
    )
    
    # Test context retrieval
    context = memory.get_context("Tell me more about neural networks")
    print(f"Found {len(context)} relevant exchanges for context")
    
    # Print learning stats
    stats = memory.get_learning_stats()
    print(f"Learning progress: {stats['total_exchanges']} exchanges processed")
    print(f"Identified {stats['unique_concepts']} concepts")
    
    # Save memories
    memory.save_memories() 