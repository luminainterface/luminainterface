from typing import Dict, Any
import logging
from .base_node import BaseNode

class ConsciousnessNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.consciousness_level = 0.0
        self.awareness_threshold = 0.7
        self.integration_state = {}
        self.active = False
        
    def initialize(self) -> bool:
        """Initialize the consciousness node"""
        try:
            self.consciousness_level = 0.1  # Start with base consciousness
            self.integration_state = {
                'perception': 0.0,
                'emotion': 0.0,
                'thought': 0.0,
                'memory': 0.0
            }
            self.active = True
            logging.info("ConsciousnessNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize ConsciousnessNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and update consciousness level"""
        try:
            # Update integration state based on input
            for key, value in input_data.items():
                if key in self.integration_state:
                    self.integration_state[key] = value
                    
            # Calculate new consciousness level
            self.consciousness_level = sum(self.integration_state.values()) / len(self.integration_state)
            
            # Check if consciousness threshold is reached
            is_aware = self.consciousness_level >= self.awareness_threshold
            
            return {
                'consciousness_level': self.consciousness_level,
                'is_aware': is_aware,
                'integration_state': self.integration_state.copy()
            }
        except Exception as e:
            logging.error(f"Error processing consciousness input: {str(e)}")
            return {'error': str(e)}
            
    def get_status(self) -> str:
        """Get current status of the consciousness node"""
        if not self.active:
            return "inactive"
        return f"active (consciousness level: {self.consciousness_level:.2f})"
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 