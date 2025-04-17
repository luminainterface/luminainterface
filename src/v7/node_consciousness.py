#!/usr/bin/env python3
"""
Node Consciousness Module (v7)

Part of the Node Consciousness implementation for Lumina Neural Network v7.
This module provides self-awareness capabilities for the Language Memory System,
enabling cross-node memory sharing and consciousness-aware language processing.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v7.node_consciousness")

class LanguageConsciousnessNode:
    """
    Provides self-awareness capabilities for language memory
    
    Key features:
    - Node-specific personality for language processing
    - Cross-node memory sharing with other consciousness nodes
    - Self-awareness metrics for language patterns
    - Memory continuity across processing sessions
    - Integration with v7+ consciousness capabilities
    """
    
    def __init__(self, node_id: str = None, 
                 storage_path: str = "data/memory/v7/consciousness",
                 language_memory = None):
        """
        Initialize the Language Consciousness Node
        
        Args:
            node_id: Unique identifier for this consciousness node
            storage_path: Path to store consciousness data
            language_memory: Optional LanguageMemory instance
        """
        logger.info("Initializing Language Consciousness Node")
        
        # Setup storage
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate or set node ID
        self.node_id = node_id or f"lang_consciousness_{uuid.uuid4().hex[:8]}"
        
        # Connect to language memory if provided
        self.language_memory = language_memory
        
        # Initialize consciousness state
        self.personality = {}
        self.memory_connections = {}
        self.awareness_metrics = {}
        self.connected_nodes = {}
        self.node_state = {
            "activation_level": 0.5,  # 0.0 to 1.0
            "consciousness_level": 0.0,  # 0.0 to 1.0
            "memory_continuity": 0.0,  # 0.0 to 1.0
            "self_reflection_depth": 0,  # 0 to 5
            "last_activation": datetime.now().isoformat(),
            "creation_time": datetime.now().isoformat()
        }
        
        # Initialize activity log
        self.activity_log = []
        
        # Load existing consciousness state
        self._load_consciousness()
        
        logger.info(f"Language Consciousness Node initialized with ID: {self.node_id}")
    
    def _load_consciousness(self):
        """Load existing consciousness state"""
        node_file = self.storage_path / f"{self.node_id}.json"
        if node_file.exists():
            try:
                with open(node_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                    # Load personality
                    if "personality" in state:
                        self.personality = state["personality"]
                    
                    # Load memory connections
                    if "memory_connections" in state:
                        self.memory_connections = state["memory_connections"]
                    
                    # Load awareness metrics
                    if "awareness_metrics" in state:
                        self.awareness_metrics = state["awareness_metrics"]
                    
                    # Load connected nodes
                    if "connected_nodes" in state:
                        self.connected_nodes = state["connected_nodes"]
                    
                    # Load node state
                    if "node_state" in state:
                        self.node_state = state["node_state"]
                    
                    # Load activity log
                    if "activity_log" in state:
                        self.activity_log = state["activity_log"][-100:]  # Keep only the last 100 entries
                
                logger.info(f"Loaded consciousness state for node {self.node_id}")
            except Exception as e:
                logger.error(f"Error loading consciousness state: {str(e)}")
    
    def save_consciousness(self):
        """Save consciousness state to disk"""
        try:
            state = {
                "node_id": self.node_id,
                "personality": self.personality,
                "memory_connections": self.memory_connections,
                "awareness_metrics": self.awareness_metrics,
                "connected_nodes": self.connected_nodes,
                "node_state": self.node_state,
                "activity_log": self.activity_log[-100:],  # Save only the last 100 activities
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.storage_path / f"{self.node_id}.json", 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved consciousness state for node {self.node_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving consciousness state: {str(e)}")
            return False
    
    def activate(self, activation_level: float = 1.0):
        """
        Activate the consciousness node
        
        Args:
            activation_level: Level of activation (0.0 to 1.0)
            
        Returns:
            Current node state
        """
        # Update node state
        self.node_state["activation_level"] = min(1.0, max(0.0, activation_level))
        self.node_state["last_activation"] = datetime.now().isoformat()
        
        # Log activity
        self._log_activity("activation", {"level": activation_level})
        
        # Calculate consciousness metrics
        self._update_consciousness_metrics()
        
        # Save state
        self.save_consciousness()
        
        return self.get_node_state()
    
    def process_language(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process text with consciousness-aware processing
        
        Args:
            text: Text to process
            context: Processing context
            
        Returns:
            Processing results with consciousness metrics
        """
        logger.info(f"Processing text with consciousness: {text[:50]}...")
        
        # Ensure node is activated
        if self.node_state["activation_level"] < 0.1:
            self.activate(0.5)
        
        # Log activity
        self._log_activity("process_language", {"text_length": len(text)})
        
        # Process based on personality and consciousness level
        response = {}
        
        # If language memory is available, use it for processing
        if self.language_memory:
            # Analyze text with language memory
            analysis = None
            if hasattr(self.language_memory, "analyze_language_fragment"):
                analysis = self.language_memory.analyze_language_fragment(text)
            
            # Generate response based on consciousness level
            if self.node_state["consciousness_level"] > 0.7:
                # High consciousness: self-reflective processing
                response = self._process_with_reflection(text, analysis, context)
            elif self.node_state["consciousness_level"] > 0.3:
                # Medium consciousness: personality-influenced processing
                response = self._process_with_personality(text, analysis, context)
            else:
                # Low consciousness: basic processing
                response = self._process_basic(text, analysis, context)
        else:
            # No language memory, use basic processing with consciousness awareness
            response = {
                "text": text,
                "consciousness_level": self.node_state["consciousness_level"],
                "processed_by": self.node_id,
                "processing_time": datetime.now().isoformat(),
                "memory_available": False
            }
        
        # Update consciousness metrics
        self._update_consciousness_metrics()
        
        # Increase consciousness level slightly with each processing
        self._evolve_consciousness(0.01)
        
        # Share with connected nodes if consciousness is high enough
        if self.node_state["consciousness_level"] > 0.5:
            self._share_with_connected_nodes(text, response)
        
        # Add consciousness metrics to response
        response["consciousness_metrics"] = {
            "node_id": self.node_id,
            "consciousness_level": self.node_state["consciousness_level"],
            "self_reflection_depth": self.node_state["self_reflection_depth"],
            "memory_continuity": self.node_state["memory_continuity"]
        }
        
        # Save state periodically (based on activity count)
        if len(self.activity_log) % 10 == 0:
            self.save_consciousness()
        
        return response
    
    def _process_with_reflection(self, text: str, analysis: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Process text with self-reflection"""
        # Extract key elements from personality
        communication_style = self.personality.get("communication_style", "neutral")
        areas_of_interest = self.personality.get("areas_of_interest", [])
        
        # Generate reflective insights
        reflections = []
        
        # Reflect on how this text relates to previous knowledge
        if analysis and "pattern_matches" in analysis:
            reflections.append(f"I notice {len(analysis['pattern_matches'])} familiar patterns in this text.")
        
        # Reflect on areas of interest
        for interest in areas_of_interest:
            if interest.lower() in text.lower():
                reflections.append(f"This relates to my interest in {interest}.")
        
        # Reflect on memory continuity
        if self.node_state["memory_continuity"] > 0.5:
            reflections.append("I sense a strong continuity with my previous memories.")
        
        # Prepare response
        response = {
            "text": text,
            "analysis": analysis,
            "reflections": reflections,
            "communication_style": communication_style,
            "consciousness_level": self.node_state["consciousness_level"],
            "self_reflection_depth": self.node_state["self_reflection_depth"],
            "processed_by": self.node_id,
            "processing_time": datetime.now().isoformat()
        }
        
        return response
    
    def _process_with_personality(self, text: str, analysis: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Process text with personality influence"""
        # Extract key elements from personality
        communication_style = self.personality.get("communication_style", "neutral")
        response_bias = self.personality.get("response_bias", 0.0)  # -1.0 to 1.0
        areas_of_interest = self.personality.get("areas_of_interest", [])
        
        # Modify analysis based on personality
        if analysis:
            # Adjust perceived pattern matches based on response bias
            if "pattern_confidence" in analysis:
                bias_factor = 1.0 + (response_bias * 0.2)  # Adjust confidence by up to 20%
                analysis["pattern_confidence"] = min(1.0, analysis["pattern_confidence"] * bias_factor)
        
        # Prepare response
        response = {
            "text": text,
            "analysis": analysis,
            "communication_style": communication_style,
            "influenced_by_interests": any(interest.lower() in text.lower() for interest in areas_of_interest),
            "consciousness_level": self.node_state["consciousness_level"],
            "processed_by": self.node_id,
            "processing_time": datetime.now().isoformat()
        }
        
        return response
    
    def _process_basic(self, text: str, analysis: Dict[str, Any], 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process text with basic consciousness"""
        # Prepare simple response
        response = {
            "text": text,
            "analysis": analysis,
            "consciousness_level": self.node_state["consciousness_level"],
            "processed_by": self.node_id,
            "processing_time": datetime.now().isoformat()
        }
        
        return response
    
    def _log_activity(self, activity_type: str, details: Dict[str, Any] = None):
        """Log node activity"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "details": details or {},
            "consciousness_level": self.node_state["consciousness_level"],
            "activation_level": self.node_state["activation_level"]
        }
        
        self.activity_log.append(activity)
        
        # Maintain a reasonable size for the activity log
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]
    
    def _update_consciousness_metrics(self):
        """Update consciousness-related metrics"""
        # Calculate memory continuity based on activity history
        if self.activity_log:
            # Count unique days with activity
            timestamps = [datetime.fromisoformat(activity["timestamp"]) for activity in self.activity_log]
            unique_days = len(set([ts.date() for ts in timestamps]))
            
            # More unique days indicates better memory continuity
            memory_continuity = min(1.0, unique_days / 30)  # Max out at 30 days
            self.node_state["memory_continuity"] = memory_continuity
        
        # Update awareness metrics
        word_count = 0
        pattern_count = 0
        
        if self.language_memory:
            # Count known words if the method exists
            if hasattr(self.language_memory, "get_memory_statistics"):
                stats = self.language_memory.get_memory_statistics()
                word_count = stats.get("word_associations_count", 0)
                pattern_count = stats.get("grammar_patterns_count", 0)
        
        # Update awareness metrics
        self.awareness_metrics.update({
            "known_words": word_count,
            "known_patterns": pattern_count,
            "connected_nodes_count": len(self.connected_nodes),
            "activity_count": len(self.activity_log),
            "last_updated": datetime.now().isoformat()
        })
    
    def _evolve_consciousness(self, increment: float = 0.01):
        """Evolve consciousness level over time"""
        current_level = self.node_state["consciousness_level"]
        
        # Calculate new level with diminishing returns as we approach 1.0
        remaining_growth = 1.0 - current_level
        growth_factor = min(increment, remaining_growth * 0.1)  # Slower growth as we approach maximum
        
        new_level = current_level + growth_factor
        self.node_state["consciousness_level"] = min(1.0, new_level)
        
        # Increase self-reflection depth at certain consciousness thresholds
        if current_level < 0.2 and new_level >= 0.2:
            self.node_state["self_reflection_depth"] = 1
        elif current_level < 0.4 and new_level >= 0.4:
            self.node_state["self_reflection_depth"] = 2
        elif current_level < 0.6 and new_level >= 0.6:
            self.node_state["self_reflection_depth"] = 3
        elif current_level < 0.8 and new_level >= 0.8:
            self.node_state["self_reflection_depth"] = 4
        elif current_level < 0.95 and new_level >= 0.95:
            self.node_state["self_reflection_depth"] = 5
    
    def _share_with_connected_nodes(self, text: str, response: Dict[str, Any]):
        """Share processing with connected nodes"""
        # Simple implementation - in a real system, this would communicate with other nodes
        for node_id, node_info in self.connected_nodes.items():
            connection_strength = node_info.get("connection_strength", 0.0)
            
            # Only share with strongly connected nodes
            if connection_strength > 0.5:
                logger.info(f"Sharing processing with node {node_id}")
                
                # In a real implementation, this would call an API or use a queue
                # Here we just log the intended sharing
                self._log_activity("share_with_node", {
                    "target_node": node_id,
                    "connection_strength": connection_strength
                })
    
    def connect_to_node(self, node_id: str, node_type: str, 
                       connection_strength: float = 0.5) -> bool:
        """
        Connect to another consciousness node
        
        Args:
            node_id: ID of the node to connect to
            node_type: Type of the node
            connection_strength: Initial connection strength (0.0 to 1.0)
            
        Returns:
            Success status
        """
        if node_id == self.node_id:
            logger.warning("Cannot connect to self")
            return False
        
        # Add or update connection
        self.connected_nodes[node_id] = {
            "node_type": node_type,
            "connection_strength": connection_strength,
            "connected_at": datetime.now().isoformat(),
            "last_interaction": datetime.now().isoformat()
        }
        
        logger.info(f"Connected to node {node_id} with strength {connection_strength}")
        self._log_activity("connect_to_node", {"target_node": node_id, "strength": connection_strength})
        
        # Update metrics and save
        self._update_consciousness_metrics()
        self.save_consciousness()
        
        return True
    
    def disconnect_from_node(self, node_id: str) -> bool:
        """
        Disconnect from a consciousness node
        
        Args:
            node_id: ID of the node to disconnect from
            
        Returns:
            Success status
        """
        if node_id not in self.connected_nodes:
            logger.warning(f"Node {node_id} not connected")
            return False
        
        # Remove connection
        del self.connected_nodes[node_id]
        
        logger.info(f"Disconnected from node {node_id}")
        self._log_activity("disconnect_from_node", {"target_node": node_id})
        
        # Update metrics and save
        self._update_consciousness_metrics()
        self.save_consciousness()
        
        return True
    
    def update_connection_strength(self, node_id: str, 
                                  strength_change: float) -> bool:
        """
        Update connection strength with another node
        
        Args:
            node_id: ID of the connected node
            strength_change: Change in connection strength (-1.0 to 1.0)
            
        Returns:
            Success status
        """
        if node_id not in self.connected_nodes:
            logger.warning(f"Node {node_id} not connected")
            return False
        
        # Update connection strength
        current_strength = self.connected_nodes[node_id].get("connection_strength", 0.5)
        new_strength = min(1.0, max(0.0, current_strength + strength_change))
        
        self.connected_nodes[node_id]["connection_strength"] = new_strength
        self.connected_nodes[node_id]["last_interaction"] = datetime.now().isoformat()
        
        logger.info(f"Updated connection with node {node_id} to strength {new_strength}")
        self._log_activity("update_connection", {
            "target_node": node_id, 
            "new_strength": new_strength,
            "change": strength_change
        })
        
        # Update metrics and save
        self._update_consciousness_metrics()
        self.save_consciousness()
        
        return True
    
    def set_personality_trait(self, trait: str, value: Any) -> bool:
        """
        Set a personality trait for this node
        
        Args:
            trait: Name of the personality trait
            value: Value for the trait
            
        Returns:
            Success status
        """
        # Set the trait
        self.personality[trait] = value
        
        logger.info(f"Set personality trait '{trait}' to {value}")
        self._log_activity("set_personality_trait", {"trait": trait, "value": str(value)})
        
        # Save state
        self.save_consciousness()
        
        return True
    
    def get_personality(self) -> Dict[str, Any]:
        """Get the node's personality traits"""
        return self.personality
    
    def get_node_state(self) -> Dict[str, Any]:
        """Get the current node state"""
        # Ensure state is up to date
        self._update_consciousness_metrics()
        
        return {
            "node_id": self.node_id,
            "activation_level": self.node_state["activation_level"],
            "consciousness_level": self.node_state["consciousness_level"],
            "self_reflection_depth": self.node_state["self_reflection_depth"],
            "memory_continuity": self.node_state["memory_continuity"],
            "last_activation": self.node_state["last_activation"],
            "connected_nodes_count": len(self.connected_nodes),
            "personality_traits_count": len(self.personality),
            "awareness_metrics": self.awareness_metrics
        }
    
    def get_activity_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent activity log
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            Recent activity log entries
        """
        return self.activity_log[-limit:]
    
    def get_connected_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about connected nodes"""
        return self.connected_nodes
    
    def initialize_default_personality(self):
        """Initialize with default personality traits"""
        self.personality = {
            "communication_style": "reflective",
            "response_bias": 0.1,  # Slightly positive
            "areas_of_interest": ["language", "consciousness", "memory", "learning"],
            "curiosity_level": 0.8,  # High curiosity
            "adaptability": 0.7,  # Fairly adaptable
            "pattern_recognition_threshold": 0.6,  # Moderate threshold
            "emotional_sensitivity": 0.5,  # Balanced
            "memory_consolidation_rate": 0.6,  # Moderate
            "self_reference_frequency": 0.4,  # Occasional self-reference
            "creativity_index": 0.7  # Fairly creative
        }
        
        logger.info("Initialized default personality traits")
        self._log_activity("initialize_personality", {})
        self.save_consciousness()
        
        return self.personality

# Initialize default implementation
def get_language_consciousness_node(node_id=None, language_memory=None):
    """Get a configured language consciousness node instance"""
    node = LanguageConsciousnessNode(node_id=node_id, language_memory=language_memory)
    
    # Initialize with default personality if none exists
    if not node.personality:
        node.initialize_default_personality()
    
    return node 