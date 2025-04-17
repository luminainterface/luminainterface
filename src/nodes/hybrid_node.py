"""
HybridNode for Neural Network Node Manager - Combines Neural Network and LLM capabilities
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from .base_node import BaseNode

class HybridNode(BaseNode):
    """Neural-LLM hybrid node for advanced processing"""
    
    def __init__(self, node_id: Optional[str] = None):
        super().__init__(node_id or "HybridNode")
        self.neural_weight = 0.5  # Balance between neural and LLM processing
        self.llm_weight = 0.5
        self.neural_state: Dict[str, Any] = {
            'activation': 0.0,
            'memory': [],
            'patterns': {}
        }
        self.llm_state: Dict[str, Any] = {
            'temperature': 0.7,
            'top_p': 0.9,
            'context_window': []
        }
        
    def initialize(self) -> bool:
        """Initialize the hybrid node"""
        try:
            # Initialize neural components
            self._initialize_neural_components()
            
            # Initialize LLM components
            self._initialize_llm_components()
            
            # Call parent initialization
            result = super().initialize()
            if result:
                self.logger.info("HybridNode neural and LLM components initialized")
            return result
        except Exception as e:
            self.logger.error(f"Failed to initialize HybridNode: {str(e)}")
            return False
            
    def _initialize_neural_components(self):
        """Initialize neural network components"""
        # Set up neural memory
        self.neural_state['memory'] = []
        
        # Initialize pattern recognition
        self.neural_state['patterns'] = {
            'basic': np.random.randn(512),  # Base pattern
            'active': np.zeros(512),        # Active pattern
            'resonance': np.zeros(512)      # Resonance pattern
        }
        
    def _initialize_llm_components(self):
        """Initialize LLM components"""
        # Clear context window
        self.llm_state['context_window'] = []
        
        # Set default parameters
        self.llm_state.update({
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 100,
            'stop_sequences': ["\n\n", "###"]
        })
        
    def process(self, data: Any) -> Optional[Dict[str, Any]]:
        """Process input through hybrid neural-LLM pipeline"""
        if not self._active:
            self.logger.error("Cannot process data - HybridNode is not active")
            return None
            
        try:
            # Neural processing
            neural_result = self._neural_process(data)
            
            # LLM processing
            llm_result = self._llm_process(data, neural_result)
            
            # Combine results
            combined_result = self._combine_results(neural_result, llm_result)
            
            # Update states
            self._update_states(neural_result, llm_result)
            
            # Prepare result
            result = {
                'status': 'success',
                'node_id': self.node_id,
                'timestamp': datetime.now().isoformat(),
                'neural_contribution': neural_result,
                'llm_contribution': llm_result,
                'combined_result': combined_result,
                'weights': {
                    'neural': self.neural_weight,
                    'llm': self.llm_weight
                }
            }
            
            self.last_process_time = result['timestamp']
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing data in HybridNode: {str(e)}")
            return None
            
    def _neural_process(self, data: Any) -> Dict[str, Any]:
        """Process data through neural components"""
        try:
            # Convert input to vector
            input_vector = self._data_to_vector(data)
            
            # Update active pattern
            self.neural_state['patterns']['active'] = input_vector
            
            # Compute resonance with stored patterns
            resonances = self._compute_resonances(input_vector)
            
            # Update neural state
            self.neural_state['activation'] = float(np.mean(resonances))
            
            return {
                'vector': input_vector.tolist(),
                'resonances': resonances,
                'activation': self.neural_state['activation']
            }
            
        except Exception as e:
            self.logger.error(f"Neural processing error: {str(e)}")
            return {'error': str(e)}
            
    def _llm_process(self, data: Any, neural_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through LLM components"""
        try:
            # Prepare context
            context = self._prepare_context(data, neural_result)
            
            # Add to context window
            self.llm_state['context_window'].append(context)
            if len(self.llm_state['context_window']) > 5:  # Keep last 5 contexts
                self.llm_state['context_window'].pop(0)
                
            # Simulate LLM processing (replace with actual LLM integration)
            processed = {
                'context': context,
                'temperature': self.llm_state['temperature'],
                'top_p': self.llm_state['top_p']
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"LLM processing error: {str(e)}")
            return {'error': str(e)}
            
    def _combine_results(self, neural_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine neural and LLM results"""
        try:
            # Check for errors
            if 'error' in neural_result or 'error' in llm_result:
                raise ValueError("Error in component results")
                
            # Weighted combination
            combined = {
                'neural_weight': self.neural_weight,
                'llm_weight': self.llm_weight,
                'neural_activation': neural_result['activation'],
                'llm_context': llm_result['context'],
                'combined_confidence': (
                    self.neural_weight * neural_result['activation'] +
                    self.llm_weight * float(llm_result['top_p'])
                )
            }
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining results: {str(e)}")
            return {'error': str(e)}
            
    def _update_states(self, neural_result: Dict[str, Any], llm_result: Dict[str, Any]):
        """Update internal states based on processing results"""
        try:
            # Update neural state
            if 'activation' in neural_result:
                self.neural_state['activation'] = neural_result['activation']
                
            # Update LLM state
            if 'temperature' in llm_result:
                self.llm_state['temperature'] = llm_result['temperature']
                
            # Adjust weights based on performance
            self._adjust_weights(neural_result, llm_result)
            
        except Exception as e:
            self.logger.error(f"Error updating states: {str(e)}")
            
    def _adjust_weights(self, neural_result: Dict[str, Any], llm_result: Dict[str, Any]):
        """Dynamically adjust neural and LLM weights"""
        try:
            # Simple adaptive weighting based on component performance
            neural_confidence = neural_result.get('activation', 0.5)
            llm_confidence = float(llm_result.get('top_p', 0.5))
            
            # Normalize weights
            total_confidence = neural_confidence + llm_confidence
            if total_confidence > 0:
                self.neural_weight = neural_confidence / total_confidence
                self.llm_weight = llm_confidence / total_confidence
            
        except Exception as e:
            self.logger.error(f"Error adjusting weights: {str(e)}")
            
    def _data_to_vector(self, data: Any) -> np.ndarray:
        """Convert input data to vector representation"""
        try:
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, (list, tuple)):
                return np.array(data)
            elif isinstance(data, dict):
                return np.array(list(data.values()))
            elif isinstance(data, (int, float)):
                return np.array([data])
            else:
                # Default random projection for unsupported types
                return np.random.randn(512)
        except Exception as e:
            self.logger.error(f"Error converting data to vector: {str(e)}")
            return np.zeros(512)
            
    def _compute_resonances(self, input_vector: np.ndarray) -> List[float]:
        """Compute resonances with stored patterns"""
        resonances = []
        for pattern in self.neural_state['patterns'].values():
            if isinstance(pattern, np.ndarray):
                resonance = np.dot(input_vector, pattern)
                resonances.append(float(resonance))
        return resonances
        
    def _prepare_context(self, data: Any, neural_result: Dict[str, Any]) -> str:
        """Prepare context for LLM processing"""
        context = f"Input: {str(data)}\n"
        context += f"Neural activation: {neural_result.get('activation', 0.0)}\n"
        context += f"Resonances: {neural_result.get('resonances', [])}\n"
        return context
        
    def get_status(self) -> Dict[str, Any]:
        """Get extended status including neural and LLM states"""
        status = super().get_status()
        status.update({
            'neural_weight': self.neural_weight,
            'llm_weight': self.llm_weight,
            'neural_state': self.neural_state,
            'llm_state': self.llm_state
        })
        return status 