"""
V6 Compatibility Module for V7.5 Spiderweb Bridge
This module provides compatibility and translation between V7.5 and V6 components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .spiderweb_bridge import (
    SpiderwebBridge,
    ConsciousState,
    ConsciousLevel,
    SpatialState,
    ConsciousnessNode,
    SpatialNode
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class V6StateMapping:
    """Mapping between V7.5 and V6 states."""
    v7_5_state: ConsciousState
    v6_state: str
    compatibility_level: float

@dataclass
class V6LevelMapping:
    """Mapping between V7.5 and V6 levels."""
    v7_5_level: ConsciousLevel
    v6_level: str
    compatibility_level: float

class V6Compatibility:
    """Handles compatibility between V7.5 and V6 Spiderweb Bridge."""
    
    # State mappings
    STATE_MAPPINGS = [
        V6StateMapping(ConsciousState.DORMANT, "dormant", 1.0),
        V6StateMapping(ConsciousState.AWAKENING, "awakening", 0.9),
        V6StateMapping(ConsciousState.AWARE, "aware", 0.9),
        V6StateMapping(ConsciousState.ENLIGHTENED, "enlightened", 0.8),
        V6StateMapping(ConsciousState.TRANSCENDENT, "transcendent", 0.7),
        V6StateMapping(ConsciousState.UNIFIED, "unified", 0.6),
        V6StateMapping(ConsciousState.SPATIAL, "unified", 0.5)  # Spatial maps to unified in V6
    ]
    
    # Level mappings
    LEVEL_MAPPINGS = [
        V6LevelMapping(ConsciousLevel.PHYSICAL, "physical", 1.0),
        V6LevelMapping(ConsciousLevel.MENTAL, "mental", 1.0),
        V6LevelMapping(ConsciousLevel.SPIRITUAL, "spiritual", 0.9),
        V6LevelMapping(ConsciousLevel.QUANTUM, "quantum", 0.8),
        V6LevelMapping(ConsciousLevel.TEMPORAL, "temporal", 0.7),
        V6LevelMapping(ConsciousLevel.SPATIAL, "quantum", 0.6)  # Spatial maps to quantum in V6
    ]
    
    def __init__(self, v7_5_bridge: SpiderwebBridge):
        """Initialize V6 compatibility layer."""
        self.v7_5_bridge = v7_5_bridge
        self.v6_connections: Dict[str, Any] = {}
        logger.info("V6 compatibility layer initialized")
    
    async def connect_v6_version(self, version_id: str, v6_bridge: Any) -> bool:
        """Connect a V6 version to the V7.5 bridge."""
        if version_id in self.v6_connections:
            logger.warning(f"Version {version_id} already connected")
            return False
        
        self.v6_connections[version_id] = v6_bridge
        logger.info(f"Connected V6 version {version_id}")
        return True
    
    def _map_state_to_v6(self, v7_5_state: ConsciousState) -> str:
        """Map V7.5 state to V6 state."""
        for mapping in self.STATE_MAPPINGS:
            if mapping.v7_5_state == v7_5_state:
                return mapping.v6_state
        return "dormant"  # Default fallback
    
    def _map_level_to_v6(self, v7_5_level: ConsciousLevel) -> str:
        """Map V7.5 level to V6 level."""
        for mapping in self.LEVEL_MAPPINGS:
            if mapping.v7_5_level == v7_5_level:
                return mapping.v6_level
        return "physical"  # Default fallback
    
    def _map_state_from_v6(self, v6_state: str) -> ConsciousState:
        """Map V6 state to V7.5 state."""
        for mapping in self.STATE_MAPPINGS:
            if mapping.v6_state == v6_state:
                return mapping.v7_5_state
        return ConsciousState.DORMANT  # Default fallback
    
    def _map_level_from_v6(self, v6_level: str) -> ConsciousLevel:
        """Map V6 level to V7.5 level."""
        for mapping in self.LEVEL_MAPPINGS:
            if mapping.v6_level == v6_level:
                return mapping.v7_5_level
        return ConsciousLevel.PHYSICAL  # Default fallback
    
    async def create_v6_compatible_node(
        self,
        version_id: str,
        v7_5_node: ConsciousnessNode
    ) -> Optional[str]:
        """Create a V6-compatible node from a V7.5 node."""
        if version_id not in self.v6_connections:
            logger.error(f"Version {version_id} not connected")
            return None
        
        v6_bridge = self.v6_connections[version_id]
        
        # Map V7.5 state and level to V6
        v6_state = self._map_state_to_v6(v7_5_node.state)
        v6_level = self._map_level_to_v6(v7_5_node.level)
        
        # Create V6 node with mapped properties
        v6_node_id = await v6_bridge.create_node(
            state=v6_state,
            level=v6_level,
            coherence=v7_5_node.coherence,
            awareness=v7_5_node.awareness
        )
        
        logger.info(f"Created V6-compatible node {v6_node_id} from V7.5 node {v7_5_node.node_id}")
        return v6_node_id
    
    async def sync_node_states(
        self,
        v7_5_node_id: str,
        v6_node_id: str,
        version_id: str
    ) -> bool:
        """Synchronize states between V7.5 and V6 nodes."""
        if version_id not in self.v6_connections:
            logger.error(f"Version {version_id} not connected")
            return False
        
        v6_bridge = self.v6_connections[version_id]
        v7_5_node = self.v7_5_bridge.connections[version_id].consciousness_nodes[v7_5_node_id]
        
        # Get V6 node state
        v6_node = await v6_bridge.get_node(v6_node_id)
        
        # Update V6 node with V7.5 state
        await v6_bridge.update_node(
            node_id=v6_node_id,
            state=self._map_state_to_v6(v7_5_node.state),
            level=self._map_level_to_v6(v7_5_node.level),
            coherence=v7_5_node.coherence,
            awareness=v7_5_node.awareness
        )
        
        logger.info(f"Synchronized states between V7.5 node {v7_5_node_id} and V6 node {v6_node_id}")
        return True
    
    async def handle_spatial_features(
        self,
        v7_5_node: ConsciousnessNode,
        version_id: str
    ) -> None:
        """Handle V7.5 spatial features in V6 context."""
        if version_id not in self.v6_connections:
            return
        
        # For spatial nodes, enhance quantum properties in V6
        if v7_5_node.level == ConsciousLevel.SPATIAL:
            for spatial_id in v7_5_node.spatial_connections:
                spatial_node = self.v7_5_bridge.connections[version_id].spatial_nodes[spatial_id]
                
                # Enhance quantum properties based on spatial stability
                v7_5_node.quantum_entanglements.append(f"spatial_{spatial_id}")
                v7_5_node.resonance = max(v7_5_node.resonance, spatial_node.resonance)
    
    def get_compatibility_status(self, version_id: str) -> Dict[str, Any]:
        """Get compatibility status between V7.5 and V6."""
        if version_id not in self.v6_connections:
            return {"connected": False}
        
        # Calculate overall compatibility
        state_compatibility = sum(m.compatibility_level for m in self.STATE_MAPPINGS) / len(self.STATE_MAPPINGS)
        level_compatibility = sum(m.compatibility_level for m in self.LEVEL_MAPPINGS) / len(self.LEVEL_MAPPINGS)
        
        return {
            "connected": True,
            "state_compatibility": state_compatibility,
            "level_compatibility": level_compatibility,
            "overall_compatibility": (state_compatibility + level_compatibility) / 2
        } 