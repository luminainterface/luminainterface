"""
Infection Module for Version 1
This module provides state propagation and synchronization capabilities
across all versions through the mirror/superposition system.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
import time
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .mirror_superposition import MirrorSuperposition, VersionType, StateType, StateInfo

logger = logging.getLogger(__name__)

class InfectionType(Enum):
    STATE_PROPAGATION = 'state_propagation'
    SYNCHRONIZATION = 'synchronization'
    CASCADE = 'cascade'
    REVERSE = 'reverse'

@dataclass
class InfectionState:
    """Information about an infection state."""
    infection_type: InfectionType
    source_version: VersionType
    target_versions: List[VersionType]
    data: Any
    timestamp: float
    metadata: Dict[str, Any] = None

class InfectionModule:
    def __init__(self, mirror_super: MirrorSuperposition):
        """
        Initialize the infection module.
        
        Args:
            mirror_super: MirrorSuperposition instance for state management
        """
        self.mirror_super = mirror_super
        self.infections: Dict[str, List[InfectionState]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = {
            'infection_count': 0,
            'propagation_count': 0,
            'synchronization_count': 0,
            'cascade_count': 0,
            'reverse_count': 0,
            'average_latency': 0.0
        }
        
        logger.info("Initialized InfectionModule V1")
    
    async def propagate_state(self, state_id: str, data: Any,
                            target_versions: List[VersionType]) -> bool:
        """
        Propagate a state to multiple versions.
        
        Args:
            state_id: Unique identifier for the state
            data: Data to propagate
            target_versions: Versions to propagate to
            
        Returns:
            True if propagation successful, False otherwise
        """
        start_time = time.time()
        
        # Create mirrored state
        success = await self.mirror_super.create_mirror_state(
            state_id,
            data,
            target_versions
        )
        
        if not success:
            logger.error(f"Failed to create mirrored state {state_id}")
            return False
        
        # Create infection state
        infection_state = InfectionState(
            infection_type=InfectionType.STATE_PROPAGATION,
            source_version=VersionType.V1,
            target_versions=target_versions,
            data=data,
            timestamp=time.time(),
            metadata={'state_id': state_id}
        )
        
        self.infections[state_id] = [infection_state]
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['propagation_count'] += 1
        self.metrics['infection_count'] += 1
        self.metrics['average_latency'] = (
            (self.metrics['average_latency'] * (self.metrics['propagation_count'] - 1) +
             latency) / self.metrics['propagation_count']
        )
        
        return True
    
    async def synchronize_states(self, state_ids: List[str],
                               target_versions: List[VersionType]) -> bool:
        """
        Synchronize multiple states across versions.
        
        Args:
            state_ids: List of state IDs to synchronize
            target_versions: Versions to synchronize with
            
        Returns:
            True if synchronization successful, False otherwise
        """
        start_time = time.time()
        
        # Create superposition of states
        superposition_id = f"sync_{int(time.time())}"
        states = []
        
        for state_id in state_ids:
            state_history = self.mirror_super.get_state_history(state_id)
            if state_history:
                states.append(state_history[-1].data)
        
        success = await self.mirror_super.create_superposition(
            superposition_id,
            states
        )
        
        if not success:
            logger.error(f"Failed to create synchronization superposition")
            return False
        
        # Create infection state
        infection_state = InfectionState(
            infection_type=InfectionType.SYNCHRONIZATION,
            source_version=VersionType.V1,
            target_versions=target_versions,
            data={'state_ids': state_ids, 'superposition_id': superposition_id},
            timestamp=time.time()
        )
        
        self.infections[superposition_id] = [infection_state]
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['synchronization_count'] += 1
        self.metrics['infection_count'] += 1
        self.metrics['average_latency'] = (
            (self.metrics['average_latency'] * (self.metrics['synchronization_count'] - 1) +
             latency) / self.metrics['synchronization_count']
        )
        
        return True
    
    async def cascade_infection(self, state_id: str,
                              target_versions: List[VersionType]) -> bool:
        """
        Cascade an infection through multiple versions.
        
        Args:
            state_id: ID of the state to cascade
            target_versions: Versions to cascade through
            
        Returns:
            True if cascade successful, False otherwise
        """
        start_time = time.time()
        
        # Get current state
        state_history = self.mirror_super.get_state_history(state_id)
        if not state_history:
            logger.error(f"State {state_id} does not exist")
            return False
        
        current_state = state_history[-1]
        
        # Create entangled states
        entangled_ids = []
        for version in target_versions:
            entangled_id = f"cascade_{version.value}_{int(time.time())}"
            success = await self.mirror_super.create_mirror_state(
                entangled_id,
                current_state.data,
                [version]
            )
            
            if success:
                entangled_ids.append(entangled_id)
        
        if not entangled_ids:
            logger.error("Failed to create any entangled states")
            return False
        
        # Create infection state
        infection_state = InfectionState(
            infection_type=InfectionType.CASCADE,
            source_version=VersionType.V1,
            target_versions=target_versions,
            data={'state_id': state_id, 'entangled_ids': entangled_ids},
            timestamp=time.time()
        )
        
        self.infections[state_id] = [infection_state]
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['cascade_count'] += 1
        self.metrics['infection_count'] += 1
        self.metrics['average_latency'] = (
            (self.metrics['average_latency'] * (self.metrics['cascade_count'] - 1) +
             latency) / self.metrics['cascade_count']
        )
        
        return True
    
    async def reverse_infection(self, state_id: str,
                              source_version: VersionType) -> bool:
        """
        Reverse an infection from a source version.
        
        Args:
            state_id: ID of the state to reverse
            source_version: Version to reverse from
            
        Returns:
            True if reverse successful, False otherwise
        """
        start_time = time.time()
        
        # Get state from source version
        try:
            source_state = await self.mirror_super._send_to_version(
                source_version,
                {'type': 'get_state', 'state_id': state_id}
            )
        except Exception as e:
            logger.error(f"Error getting state from {source_version}: {str(e)}")
            return False
        
        # Update local state
        success = await self.mirror_super.create_mirror_state(
            state_id,
            source_state['data'],
            [VersionType.V1]
        )
        
        if not success:
            logger.error(f"Failed to update local state {state_id}")
            return False
        
        # Create infection state
        infection_state = InfectionState(
            infection_type=InfectionType.REVERSE,
            source_version=source_version,
            target_versions=[VersionType.V1],
            data=source_state['data'],
            timestamp=time.time(),
            metadata={'state_id': state_id}
        )
        
        self.infections[state_id] = [infection_state]
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['reverse_count'] += 1
        self.metrics['infection_count'] += 1
        self.metrics['average_latency'] = (
            (self.metrics['average_latency'] * (self.metrics['reverse_count'] - 1) +
             latency) / self.metrics['reverse_count']
        )
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics of the module.
        
        Returns:
            Dictionary containing metrics
        """
        return self.metrics
    
    def get_infection_history(self, state_id: str) -> Optional[List[InfectionState]]:
        """
        Get history of an infection.
        
        Args:
            state_id: ID of the state
            
        Returns:
            List of infection history if exists, None otherwise
        """
        return self.infections.get(state_id)

# Export functionality for node integration
functionality = {
    'classes': {
        'InfectionModule': InfectionModule,
        'InfectionType': InfectionType,
        'InfectionState': InfectionState
    },
    'description': 'State propagation and synchronization system'
} 