import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class LanguageVisualizer:
    """Provides visualization data for the language module dashboard."""
    
    def __init__(self, central_language_node=None, output_dir=None):
        """Initialize the language visualizer.
        
        Args:
            central_language_node: Reference to the central language node
            output_dir: Directory to store visualization data
        """
        self.central_node = central_language_node
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.current_state = {
            "active": False,
            "model": None,
            "weights": {
                "llm": 0.5,
                "nn": 0.5
            }
        }
        
        self.metrics = {
            "consciousness_level": 0.0,
            "neural_linguistic_score": 0.0,
            "recursive_pattern_depth": 0.0,
            "memory_association_count": 0.0,
            "total_exchanges": 0,
            "total_concepts": 0
        }
        
        self.components = {
            "language_memory": {"active": False},
            "conscious_mirror_language": {"active": False},
            "neural_linguistic_processor": {"active": False},
            "recursive_pattern_analyzer": {"active": False},
            "database_manager": {"active": False},
            "conversation_memory": {"active": False},
            "mistral_integration": {"active": False}
        }
        
        self.metrics_history = {
            "timestamps": [],
            "consciousness_level": [],
            "neural_linguistic_score": [],
            "recursive_pattern_depth": []
        }
        
        self.recent_conversations = []
        self.memory_associations = []
        
        self.last_update = datetime.now()
        logging.info("Language visualizer initialized")
    
    def update_system_state(self, state_data: Dict[str, Any]) -> None:
        """Update the current system state.
        
        Args:
            state_data: Current state of the language system
        """
        if "active" in state_data:
            self.current_state["active"] = state_data["active"]
            
        if "model" in state_data:
            self.current_state["model"] = state_data["model"]
            
        if "weights" in state_data:
            self.current_state["weights"].update(state_data["weights"])
        
        self.last_update = datetime.now()
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def update_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """Update the current metrics.
        
        Args:
            metrics_data: Current metrics of the language system
        """
        # Update metrics
        for key, value in metrics_data.items():
            if key in self.metrics:
                self.metrics[key] = value
        
        # Add to history
        timestamp = datetime.now().isoformat()
        self.metrics_history["timestamps"].append(timestamp)
        
        self.metrics_history["consciousness_level"].append(
            metrics_data.get("consciousness_level", self.metrics["consciousness_level"])
        )
        
        self.metrics_history["neural_linguistic_score"].append(
            metrics_data.get("neural_linguistic_score", self.metrics["neural_linguistic_score"])
        )
        
        self.metrics_history["recursive_pattern_depth"].append(
            metrics_data.get("recursive_pattern_depth", self.metrics["recursive_pattern_depth"])
        )
        
        # Keep history limited
        max_history = 100
        if len(self.metrics_history["timestamps"]) > max_history:
            self.metrics_history["timestamps"] = self.metrics_history["timestamps"][-max_history:]
            self.metrics_history["consciousness_level"] = self.metrics_history["consciousness_level"][-max_history:]
            self.metrics_history["neural_linguistic_score"] = self.metrics_history["neural_linguistic_score"][-max_history:]
            self.metrics_history["recursive_pattern_depth"] = self.metrics_history["recursive_pattern_depth"][-max_history:]
        
        self.last_update = datetime.now()
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def update_components(self, components_data: Dict[str, Any]) -> None:
        """Update the component status.
        
        Args:
            components_data: Current status of language system components
        """
        for component, status in components_data.items():
            if component in self.components:
                self.components[component].update(status)
        
        self.last_update = datetime.now()
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def add_conversation(self, message: Dict[str, Any]) -> None:
        """Add a conversation message.
        
        Args:
            message: Conversation message data
        """
        # Ensure message has timestamp
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
            
        self.recent_conversations.append(message)
        
        # Keep limited history
        max_conversation_history = 50
        if len(self.recent_conversations) > max_conversation_history:
            self.recent_conversations = self.recent_conversations[-max_conversation_history:]
        
        self.last_update = datetime.now()
        
        # Update exchange count
        if "is_user" in message and message["is_user"]:
            self.metrics["total_exchanges"] += 1
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def update_memory_associations(self, associations: List[Dict[str, Any]]) -> None:
        """Update memory associations.
        
        Args:
            associations: List of word associations
        """
        self.memory_associations = associations
        
        self.last_update = datetime.now()
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def _save_visualization_data(self) -> None:
        """Save current visualization data to disk."""
        try:
            output_file = os.path.join(self.output_dir, "language_data.json")
            data = self.get_dashboard_data()
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save language visualization data: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the monitoring dashboard.
        
        Returns:
            dict: Data formatted for the monitoring dashboard
        """
        # If central node is available and has a method for this, use it
        if self.central_node and hasattr(self.central_node, "get_visualization_data"):
            try:
                central_data = self.central_node.get_visualization_data()
                if central_data:
                    # Merge with our data
                    self._update_from_central_node(central_data)
            except Exception as e:
                logging.error(f"Error getting data from central node: {e}")
        
        # Generate sample data if no real data exists and no central node
        if not self.current_state["active"] and not self.central_node:
            self._generate_sample_data()
        
        # Format data for dashboard
        return {
            "active": self.current_state["active"],
            "model": self.current_state["model"],
            "weights": self.current_state["weights"],
            "metrics": self.metrics,
            "components": self.components,
            "conversations": {
                "recent": self.recent_conversations[-10:][::-1],  # Latest 10, reversed
                "total": len(self.recent_conversations)
            },
            "memory": {
                "top_associations": self.memory_associations[:20],  # Top 20
                "total_associations": len(self.memory_associations)
            },
            "history": {
                "metrics": self.metrics_history
            },
            "last_update": self.last_update.isoformat()
        }
    
    def _update_from_central_node(self, central_data: Dict[str, Any]) -> None:
        """Update visualizer data from central node data.
        
        Args:
            central_data: Data from central language node
        """
        # Update state
        if "state" in central_data:
            self.update_system_state(central_data["state"])
            
        # Update metrics
        if "metrics" in central_data:
            self.update_metrics(central_data["metrics"])
            
        # Update components
        if "components" in central_data:
            self.update_components(central_data["components"])
            
        # Update conversations
        if "conversations" in central_data and "recent" in central_data["conversations"]:
            for message in central_data["conversations"]["recent"]:
                if message not in self.recent_conversations:
                    self.add_conversation(message)
                    
        # Update memory associations
        if "memory" in central_data and "associations" in central_data["memory"]:
            self.update_memory_associations(central_data["memory"]["associations"])
    
    def _generate_sample_data(self) -> None:
        """Generate sample data for demonstration purposes."""
        # Active state
        self.current_state = {
            "active": True,
            "model": "mistral-small-latest",
            "weights": {
                "llm": random.uniform(0.4, 0.7),
                "nn": random.uniform(0.4, 0.7)
            }
        }
        
        # Sample metrics
        self.metrics = {
            "consciousness_level": random.uniform(0.3, 0.9),
            "neural_linguistic_score": random.uniform(0.4, 0.8),
            "recursive_pattern_depth": random.uniform(0.2, 0.6),
            "memory_association_count": random.uniform(0.3, 0.7),
            "total_exchanges": random.randint(20, 100),
            "total_concepts": random.randint(50, 200)
        }
        
        # Component status
        for component in self.components:
            self.components[component] = {
                "active": random.random() > 0.2,  # 80% chance of being active
                "status": "normal"
            }
        
        # Generate history
        now = datetime.now()
        self.metrics_history = {
            "timestamps": [],
            "consciousness_level": [],
            "neural_linguistic_score": [],
            "recursive_pattern_depth": []
        }
        
        for i in range(20):
            timestamp = (now - timedelta(minutes=20-i)).isoformat()
            self.metrics_history["timestamps"].append(timestamp)
            
            # Generate metrics with slight upward trend and randomness
            base_val = 0.3 + (i * 0.02)  # Starts at ~0.3, increases gradually
            
            self.metrics_history["consciousness_level"].append(
                min(1.0, max(0, base_val + random.uniform(-0.1, 0.15)))
            )
            
            self.metrics_history["neural_linguistic_score"].append(
                min(1.0, max(0, base_val + 0.05 + random.uniform(-0.1, 0.15)))
            )
            
            self.metrics_history["recursive_pattern_depth"].append(
                min(1.0, max(0, base_val - 0.1 + random.uniform(-0.1, 0.15)))
            )
        
        # Generate sample conversations
        self.recent_conversations = []
        
        conversation_pairs = [
            ("Hello, how are you today?", "I'm functioning optimally. How can I assist you?"),
            ("Tell me about consciousness in AI.", "Consciousness in AI is a complex topic that involves self-awareness, subjective experience, and phenomenal states."),
            ("How does your neural linguistic processor work?", "My neural linguistic processor identifies patterns and semantic relationships in language using a combination of rule-based processing and statistical models."),
            ("What do you think about recursive patterns?", "Recursive patterns in language reveal interesting self-referential structures. I can analyze these patterns to detect linguistic loops and self-references."),
            ("Can you remember our previous conversation?", "Yes, I store our interactions in my conversation memory system, which allows me to recall context and build on previous exchanges.")
        ]
        
        base_time = now - timedelta(minutes=30)
        for i, (user_msg, system_msg) in enumerate(conversation_pairs):
            msg_time = base_time + timedelta(minutes=i*5)
            
            # Add user message
            self.recent_conversations.append({
                "content": user_msg,
                "is_user": True,
                "timestamp": msg_time.isoformat()
            })
            
            # Add system response with consciousness level
            self.recent_conversations.append({
                "content": system_msg,
                "is_user": False,
                "timestamp": (msg_time + timedelta(seconds=2)).isoformat(),
                "consciousness_level": random.uniform(0.4, 0.9)
            })
        
        # Generate memory associations
        words = ["neural", "language", "consciousness", "pattern", "memory", 
                "recursive", "system", "knowledge", "semantic", "integration"]
        
        self.memory_associations = []
        for word in words:
            self.memory_associations.append({
                "word": word,
                "count": random.randint(5, 30),
                "strength": random.uniform(0.5, 0.9)
            })
        
        # Sort by count
        self.memory_associations.sort(key=lambda x: x["count"], reverse=True)
        
        self.last_update = datetime.now()


# Create singleton instance
_visualizer = None

def get_language_visualizer(central_node=None, output_dir=None) -> LanguageVisualizer:
    """
    Get or create the language visualizer instance
    
    Args:
        central_node: Central language node (optional)
        output_dir: Output directory for visualization data (optional)
        
    Returns:
        LanguageVisualizer instance
    """
    global _visualizer
    
    if _visualizer is None:
        _visualizer = LanguageVisualizer(central_node, output_dir)
        
    return _visualizer


def update_system_state(state_data: Dict[str, Any]) -> None:
    """
    Update system state (convenience function)
    
    Args:
        state_data: System state data
    """
    visualizer = get_language_visualizer()
    visualizer.update_system_state(state_data)


def update_metrics(metrics_data: Dict[str, Any]) -> None:
    """
    Update metrics (convenience function)
    
    Args:
        metrics_data: Metrics data
    """
    visualizer = get_language_visualizer()
    visualizer.update_metrics(metrics_data)


def get_dashboard_data() -> Dict[str, Any]:
    """
    Get dashboard data (convenience function)
    
    Returns:
        Dashboard data
    """
    visualizer = get_language_visualizer()
    return visualizer.get_dashboard_data()


def set_llm_weight(weight: float) -> Dict[str, Any]:
    """
    Set LLM weight (convenience function)
    
    Args:
        weight: New LLM weight (0.0 to 1.0)
        
    Returns:
        Result of the operation
    """
    visualizer = get_language_visualizer()
    
    # Update weight in visualizer
    visualizer.update_system_state({
        "weights": {
            "llm": weight
        }
    })
    
    # If central node is available, update it too
    if visualizer.central_node and hasattr(visualizer.central_node, "set_llm_weight"):
        try:
            visualizer.central_node.set_llm_weight(weight)
            return {"success": True, "weight": weight}
        except Exception as e:
            logging.error(f"Error setting LLM weight in central node: {e}")
            return {"success": False, "error": str(e)}
    
    return {"success": True, "weight": weight, "note": "Central node not available"} 