"""
HybridNode for LUMINA v7.5
Combines LLM and neural network responses with configurable weights
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from .base_node import Node, NodeMetadata, NodeType

logger = logging.getLogger(__name__)

class HybridNode(Node):
    """Node for combining LLM and neural network responses"""
    
    def __init__(self):
        metadata = NodeMetadata(
            name="Hybrid Response Processor",
            description="Combines LLM and neural network responses with configurable weights",
            category="Processing",
            type=NodeType.PROCESSOR,
            color="#9B59B6",  # Purple for hybrid processing
            icon="🔄"  # Recycle symbol for combination
        )
        super().__init__(metadata)
        
        # Add input ports
        self.add_input_port("llm_response", str, "Response from LLM")
        self.add_input_port("nn_response", str, "Response from neural network")
        self.add_input_port("llm_weight", float, "Weight for LLM response (0-1)")
        self.add_input_port("nn_weight", float, "Weight for neural network response (0-1)")
        
        # Add output ports
        self.add_output_port("combined_response", str, "Combined weighted response")
        self.add_output_port("weights_used", dict, "Actual weights used in combination")
        self.add_output_port("status", str, "Processing status")
        
        # Initialize state
        self._processing = False
        
    async def process(self) -> None:
        """Process and combine the responses"""
        try:
            self._processing = True
            self.set_output_value("status", "Processing")
            
            # Get input values
            llm_response = self.get_input_value("llm_response")
            nn_response = self.get_input_value("nn_response")
            llm_weight = self.get_input_value("llm_weight") or 0.7  # Default LLM weight
            nn_weight = self.get_input_value("nn_weight") or 0.3    # Default NN weight
            
            # Validate inputs
            if not llm_response and not nn_response:
                self.set_output_value("status", "No input responses")
                return
                
            # Normalize weights
            total = llm_weight + nn_weight
            if total == 0:
                llm_weight = 0.7
                nn_weight = 0.3
            else:
                llm_weight = llm_weight / total
                nn_weight = nn_weight / total
            
            # Combine responses
            combined = ""
            if llm_response:
                combined += llm_response
            if nn_response:
                if combined:
                    combined += "\n\nNeural Network Insights:\n"
                combined += nn_response
            
            # Set outputs
            self.set_output_value("combined_response", combined)
            self.set_output_value("weights_used", {
                "llm_weight": llm_weight,
                "nn_weight": nn_weight
            })
            self.set_output_value("status", "Complete")
            
        except Exception as e:
            logger.error(f"Error in HybridNode: {e}")
            self.set_output_value("status", f"Error: {str(e)}")
        finally:
            self._processing = False
            
    def cleanup(self):
        """Clean up resources when node is deleted"""
        self._processing = False 