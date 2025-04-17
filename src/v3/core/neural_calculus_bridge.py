"""
Neural Calculus Bridge for Version 3
This module integrates the CalculusEngine and NeuralProcessor with the
mirror/superposition system and other V3 components.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .mirror_superposition import MirrorSuperposition, VersionType, StateType, StateInfo
from .fractal_simulator import FractalSimulator, FractalType, FractalConfig
from .color_module import ColorModule, ColorSpace, ColorEffect, ColorConfig
from calculus_engine import CalculusEngine
from src.processors.neural_processor import NeuralProcessor

logger = logging.getLogger(__name__)

class OperationType(Enum):
    CALCULUS = 'calculus'
    NEURAL = 'neural'
    HYBRID = 'hybrid'

@dataclass
class OperationState:
    """Information about an operation state."""
    operation_type: OperationType
    version: VersionType
    data: Any
    timestamp: float
    metadata: Dict[str, Any] = None

class NeuralCalculusBridge:
    def __init__(self, mirror_super: MirrorSuperposition,
                 fractal_sim: Optional[FractalSimulator] = None,
                 color_module: Optional[ColorModule] = None):
        """
        Initialize the neural calculus bridge.
        
        Args:
            mirror_super: MirrorSuperposition instance
            fractal_sim: Optional FractalSimulator instance
            color_module: Optional ColorModule instance
        """
        self.mirror_super = mirror_super
        self.fractal_sim = fractal_sim
        self.color_module = color_module
        
        # Initialize processors
        self.calculus_engine = CalculusEngine()
        self.neural_processor = NeuralProcessor()
        
        self.operation_states: Dict[str, List[OperationState]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = {
            'operation_count': 0,
            'calculus_count': 0,
            'neural_count': 0,
            'hybrid_count': 0,
            'average_latency': 0.0,
            'error_count': 0
        }
        
        # Connect components
        self._connect_components()
        
        logger.info("Initialized NeuralCalculusBridge V3")
    
    def _connect_components(self) -> None:
        """Connect all components together."""
        # Connect calculus engine
        self.calculus_engine.set_central_node(self)
        if self.fractal_sim:
            self.calculus_engine.add_dependency('fractal_simulator', self.fractal_sim)
        if self.color_module:
            self.calculus_engine.add_dependency('color_module', self.color_module)
        
        # Initialize neural processor
        self.neural_processor.initialize()
        self.neural_processor.activate()
        
        # Connect processors to each other
        self.calculus_engine.add_dependency('neural_processor', self.neural_processor)
    
    async def process_hybrid_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through both calculus and neural operations.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data with both calculus and neural results
        """
        start_time = time.time()
        
        try:
            # Process through calculus engine
            calculus_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.calculus_engine.process_data,
                data.copy()
            )
            
            # Process through neural processor
            neural_data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.neural_processor.process,
                data.copy()
            )
            
            # Combine results
            result = {
                'calculus_results': calculus_data.get('calculus_results', {}),
                'neural_results': neural_data,
                'timestamp': time.time()
            }
            
            # Create operation state
            state_id = f"hybrid_{int(time.time())}"
            self.operation_states[state_id] = [OperationState(
                operation_type=OperationType.HYBRID,
                version=VersionType.V3,
                data=result,
                timestamp=time.time(),
                metadata={'input_data': data}
            )]
            
            # Update metrics
            latency = time.time() - start_time
            self.metrics['hybrid_count'] += 1
            self.metrics['operation_count'] += 1
            self.metrics['average_latency'] = (
                (self.metrics['average_latency'] * (self.metrics['hybrid_count'] - 1) +
                 latency) / self.metrics['hybrid_count']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hybrid operation: {str(e)}")
            self.metrics['error_count'] += 1
            return {'error': str(e)}
    
    async def analyze_fractal_neural(self, fractal_type: FractalType,
                                   config: FractalConfig) -> Dict[str, Any]:
        """
        Analyze a fractal using both calculus and neural operations.
        
        Args:
            fractal_type: Type of fractal to analyze
            config: Fractal configuration
            
        Returns:
            Analysis results
        """
        if not self.fractal_sim:
            logger.error("FractalSimulator not initialized")
            return {}
        
        start_time = time.time()
        
        try:
            # Generate fractal
            iterations, _ = await self.fractal_sim.generate_fractal(fractal_type)
            
            # Prepare data for processing
            fractal_data = {
                'fractal_type': fractal_type.value,
                'iterations': iterations.tolist(),
                'config': config.__dict__
            }
            
            # Process through hybrid operation
            result = await self.process_hybrid_operation(fractal_data)
            
            # Add fractal-specific metrics
            result['fractal_metrics'] = {
                'complexity': np.std(iterations),
                'resolution': iterations.shape,
                'generation_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fractal neural analysis: {str(e)}")
            self.metrics['error_count'] += 1
            return {'error': str(e)}
    
    async def propagate_operation_state(self, state_id: str,
                                      target_versions: List[VersionType]) -> bool:
        """
        Propagate an operation state to other versions.
        
        Args:
            state_id: ID of the operation state
            target_versions: Versions to propagate to
            
        Returns:
            True if propagation successful, False otherwise
        """
        if state_id not in self.operation_states:
            logger.error(f"Operation state {state_id} does not exist")
            return False
        
        state = self.operation_states[state_id][-1]
        
        # Create mirrored state
        success = await self.mirror_super.create_mirror_state(
            state_id,
            state.data,
            target_versions
        )
        
        return success
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics of the module.
        
        Returns:
            Dictionary containing metrics
        """
        return {
            **self.metrics,
            'neural_status': self.neural_processor.get_status(),
            'calculus_dependencies': list(self.calculus_engine.dependencies.keys())
        }
    
    def get_operation_state(self, state_id: str) -> Optional[OperationState]:
        """
        Get an operation state.
        
        Args:
            state_id: ID of the operation state
            
        Returns:
            Operation state if exists, None otherwise
        """
        if state_id not in self.operation_states:
            return None
        
        return self.operation_states[state_id][-1]

# Export functionality for node integration
functionality = {
    'classes': {
        'NeuralCalculusBridge': NeuralCalculusBridge,
        'OperationType': OperationType,
        'OperationState': OperationState
    },
    'description': 'Neural and calculus operations integration system'
} 