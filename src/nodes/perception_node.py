"""
Perception Node for processing sensory and pattern data
"""

import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any
from .base_node import BaseNode

logger = logging.getLogger(__name__)

class PerceptionNode(BaseNode):
    """Perception learning node"""
    
    def __init__(self, node_id: str = None):
        super().__init__(node_id)
        self.active = True
        self.patterns = {}
        self.adaptations = {}
        self.last_learned = datetime.now()
        self.adaptation_rate = 0.1
        self.state.update({
            "active": self.active,
            "patterns_count": len(self.patterns),
            "adaptations_count": len(self.adaptations),
            "last_learned": self.last_learned.isoformat()
        })
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process perception data"""
        try:
            # Extract pattern information
            pattern_type = data.get("pattern_type", "")
            pattern_data = data.get("pattern_data", {})
            
            # Process the pattern
            if pattern_type and pattern_data:
                if pattern_type not in self.patterns:
                    self.patterns[pattern_type] = []
                self.patterns[pattern_type].append(pattern_data)
                
                # Update adaptations
                self.update_adaptations(pattern_type, pattern_data)
            
            # Update state
            self.state.update({
                "active": self.active,
                "patterns_count": len(self.patterns),
                "adaptations_count": len(self.adaptations),
                "last_learned": self.last_learned.isoformat()
            })
            
            return {
                "status": "success",
                "processed_pattern": pattern_type,
                "adaptations": self.adaptations.get(pattern_type, [])
            }
            
        except Exception as e:
            logger.error(f"Error processing perception data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def update_adaptations(self, pattern_type: str = None, pattern_data: Dict = None) -> None:
        """Update adaptations based on pattern data"""
        try:
            # Initialize adaptations for pattern type if not exists
            if pattern_type not in self.adaptations:
                self.adaptations[pattern_type] = []
            
            # Create new adaptation entry
            adaptation = {
                "timestamp": datetime.now().isoformat(),
                "pattern_type": pattern_type,
                "strength": pattern_data.get("strength", 0.5) if pattern_data else 0.5,
                "confidence": pattern_data.get("confidence", 0.5) if pattern_data else 0.5
            }
            
            # Add adaptation with rate limiting
            if not self.adaptations[pattern_type] or \
               (datetime.now() - datetime.fromisoformat(self.adaptations[pattern_type][-1]["timestamp"])).total_seconds() > self.adaptation_rate:
                self.adaptations[pattern_type].append(adaptation)
                
                # Keep adaptation history manageable
                if len(self.adaptations[pattern_type]) > 100:
                    self.adaptations[pattern_type] = self.adaptations[pattern_type][-100:]
                
                self.last_learned = datetime.now()
                logger.info(f"Updated adaptations for pattern type: {pattern_type}")
            
        except Exception as e:
            logger.error(f"Error updating adaptations: {str(e)}")
    
    def get_pattern_info(self, pattern_type: str) -> Dict[str, Any]:
        """Get information about a specific pattern type"""
        return {
            "patterns": self.patterns.get(pattern_type, []),
            "adaptations": self.adaptations.get(pattern_type, []),
            "total_patterns": len(self.patterns.get(pattern_type, [])),
            "total_adaptations": len(self.adaptations.get(pattern_type, []))
        } 