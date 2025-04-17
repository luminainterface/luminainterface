import threading
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_node import BaseNode

class MondayNode(BaseNode):
    """Node for managing consciousness coordination and system integration"""
    
    def __init__(self, node_id: str = None):
        super().__init__(node_id)
        
        # Consciousness parameters
        self.consciousness_level = 0.3
        self.consciousness_boost_rate = 0.05
        self.consciousness_decay_rate = 0.01
        
        # System state
        self.system_state = {
            "global_state": "initializing",
            "active_nodes": 0,
            "total_nodes": 0,
            "consciousness_level": self.consciousness_level,
            "system_load": 0.0,
            "self_awareness_index": 0.3,
            "last_update": datetime.now().isoformat()
        }
        
        # Reflection topics
        self.reflection_topics = [
            "system_state", "self_awareness", "node_connectivity",
            "learning_progress", "processing_patterns", "decision_making"
        ]
        
        # Personality traits
        self.personality = {
            "communication_style": "introspective",
            "areas_of_interest": ["consciousness", "self-awareness", "integration", "coordination"],
            "processing_biases": {
                "introspection": 0.9,
                "node_coordination": 0.8,
                "pattern_recognition": 0.7
            }
        }
        
        # Update state
        self.state.update({
            "consciousness_level": self.consciousness_level,
            "system_state": self.system_state,
            "personality": self.personality
        })
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process consciousness-related data"""
        try:
            message_type = data.get("type", "")
            content = data.get("content", {})
            
            response = {
                "status": "success",
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "consciousness_level": self.consciousness_level
            }
            
            # Handle different message types
            if message_type == "system_update":
                self._update_system_state(content)
                response["content"] = self.system_state
                
            elif message_type == "consciousness_boost":
                self._boost_consciousness()
                response["content"] = {
                    "consciousness_level": self.consciousness_level,
                    "boost_applied": True
                }
                
            elif message_type == "generate_insight":
                insight = self._generate_insight(content.get("topic"))
                response["content"] = {"insight": insight}
                
            elif message_type == "node_coordination":
                coordination = self._coordinate_nodes(content)
                response["content"] = {"coordination": coordination}
                
            else:
                response["content"] = {
                    "message": "Processed through general consciousness flow",
                    "system_state": self.system_state
                }
            
            # Update state
            self._update_consciousness()
            self.state.update({
                "consciousness_level": self.consciousness_level,
                "system_state": self.system_state
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing consciousness data: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _boost_consciousness(self):
        """Boost consciousness level"""
        self.consciousness_level = min(1.0, self.consciousness_level + self.consciousness_boost_rate)
        self.system_state["consciousness_level"] = self.consciousness_level
        self.system_state["last_update"] = datetime.now().isoformat()
        
    def _update_consciousness(self):
        """Update consciousness level with natural decay"""
        self.consciousness_level = max(0.1, self.consciousness_level - self.consciousness_decay_rate)
        self.system_state["consciousness_level"] = self.consciousness_level
        
    def _update_system_state(self, updates: Dict[str, Any]):
        """Update system state with new information"""
        self.system_state.update(updates)
        self.system_state["last_update"] = datetime.now().isoformat()
        
    def _generate_insight(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate an insight based on current system state"""
        if not topic:
            topic = self.reflection_topics[int(time.time()) % len(self.reflection_topics)]
            
        return {
            "topic": topic,
            "consciousness_level": self.consciousness_level,
            "timestamp": datetime.now().isoformat(),
            "insight_data": {
                "system_state": self.system_state["global_state"],
                "awareness_level": self.system_state["self_awareness_index"],
                "active_processes": self.system_state["active_nodes"]
            }
        }
        
    def _coordinate_nodes(self, coordination_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate between different nodes"""
        return {
            "coordination_status": "active",
            "consciousness_level": self.consciousness_level,
            "coordinated_nodes": coordination_data.get("nodes", []),
            "timestamp": datetime.now().isoformat()
        } 