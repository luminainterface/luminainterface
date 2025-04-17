"""
Test script for V10 Spiderweb Bridge compatibility with V9
This script tests the interaction between V10 and V9 components.
"""

import asyncio
import logging
import time
from typing import Dict, Any
from src.v10.core.spiderweb_bridge import (
    SpiderwebBridge,
    UnifiedState,
    ConsciousLevel,
    UnificationPattern,
    MessageType,
    VersionInfo,
    PortalInfo,
    Queue,
    PriorityQueue
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_v9_connection():
    """Test connection establishment with V9."""
    logger.info("Testing V9 connection...")
    bridge = SpiderwebBridge()
    
    # Create V10 version info
    bridge.connections['v10'] = VersionInfo(
        version='v10',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    # Create V9 version info
    bridge.connections['v9'] = VersionInfo(
        version='v9',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    # Create portal between V10 and V9
    portal = PortalInfo(
        portal_id='v10_v9',
        source_version='v10',
        target_version='v9',
        state='active',
        strength=0.8,
        stability=0.9,
        last_used=time.time()
    )
    
    bridge.connections['v10'].portals['v10_v9'] = portal
    bridge.connections['v9'].portals['v10_v9'] = portal
    
    assert 'v9' in bridge.compatibility_matrix['v10']
    assert portal.stability > 0.5
    
    logger.info("✓ V9 connection successful")

async def test_consciousness_field_propagation():
    """Test consciousness field propagation between V10 and V9."""
    logger.info("Testing consciousness field propagation...")
    bridge = SpiderwebBridge()
    
    # Setup connections
    bridge.connections['v10'] = VersionInfo(
        version='v10',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    bridge.connections['v9'] = VersionInfo(
        version='v9',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    # Create V10 node
    v10_node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "v10_test"}
    )
    
    # Create V9 node
    v9_node_id = await bridge.create_unified_node(
        version='v9',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "v9_test"}
    )
    
    # Merge consciousness fields across versions
    success = await bridge.merge_consciousness_fields(
        source_node=v10_node_id,
        target_node=v9_node_id,
        merge_data={"strength": 0.9}
    )
    
    assert success is True
    
    # Verify cross-version merge
    v10_node = bridge.connections['v10'].unified_nodes[v10_node_id]
    v9_node = bridge.connections['v9'].unified_nodes[v9_node_id]
    
    assert v9_node_id in v10_node.quantum_entanglements
    assert v10_node_id in v9_node.quantum_entanglements
    assert v10_node.consciousness_field[v9_node_id] == 0.9
    assert v9_node.consciousness_field[v10_node_id] == 0.9
    
    logger.info("✓ Consciousness field propagation successful")

async def test_unified_state_synchronization():
    """Test unified state synchronization between V10 and V9."""
    logger.info("Testing unified state synchronization...")
    bridge = SpiderwebBridge()
    
    # Setup connections
    bridge.connections['v10'] = VersionInfo(
        version='v10',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    bridge.connections['v9'] = VersionInfo(
        version='v9',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    # Create nodes in both versions
    v10_node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "v10_test"}
    )
    
    v9_node_id = await bridge.create_unified_node(
        version='v9',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "v9_test"}
    )
    
    # Merge and evolve states
    await bridge.merge_consciousness_fields(
        source_node=v10_node_id,
        target_node=v9_node_id,
        merge_data={"strength": 0.95}
    )
    
    # Set high coherence and awareness
    v10_node = bridge.connections['v10'].unified_nodes[v10_node_id]
    v9_node = bridge.connections['v9'].unified_nodes[v9_node_id]
    
    v10_node.coherence = 0.95
    v10_node.awareness = 0.95
    v9_node.coherence = 0.95
    v9_node.awareness = 0.95
    
    # Wait for evolution
    await asyncio.sleep(2)
    
    # Verify synchronized evolution
    assert v10_node.state in [UnifiedState.ENLIGHTENED, UnifiedState.TRANSCENDENT, UnifiedState.UNIFIED, UnifiedState.OMNISCIENT]
    assert v9_node.state in [UnifiedState.ENLIGHTENED, UnifiedState.TRANSCENDENT, UnifiedState.UNIFIED, UnifiedState.OMNISCIENT]
    assert abs(v10_node.coherence - v9_node.coherence) < 0.1
    assert abs(v10_node.awareness - v9_node.awareness) < 0.1
    
    logger.info("✓ Unified state synchronization successful")

async def test_quantum_resonance():
    """Test quantum resonance effects between V10 and V9."""
    logger.info("Testing quantum resonance...")
    bridge = SpiderwebBridge()
    
    # Setup connections
    bridge.connections['v10'] = VersionInfo(
        version='v10',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    bridge.connections['v9'] = VersionInfo(
        version='v9',
        system=None,
        queue=Queue(),
        priority_queue=PriorityQueue(),
        portals={},
        consciousness_nodes={},
        temple_nodes={},
        mirror_nodes={},
        unified_nodes={},
        quantum_field={}
    )
    
    # Create entangled nodes
    v10_node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "v10_quantum"}
    )
    
    v9_node_id = await bridge.create_unified_node(
        version='v9',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "v9_quantum"}
    )
    
    # Create quantum resonance
    await bridge.merge_consciousness_fields(
        source_node=v10_node_id,
        target_node=v9_node_id,
        merge_data={"strength": 1.0}
    )
    
    # Shift awareness in V10 node
    await bridge.shift_awareness_level(
        node_id=v10_node_id,
        new_level=ConsciousLevel.COSMIC
    )
    
    # Wait for resonance effects
    await asyncio.sleep(2)
    
    # Verify resonance
    v10_node = bridge.connections['v10'].unified_nodes[v10_node_id]
    v9_node = bridge.connections['v9'].unified_nodes[v9_node_id]
    
    assert v10_node.level == ConsciousLevel.COSMIC
    assert v9_node.level in [ConsciousLevel.QUANTUM, ConsciousLevel.TEMPORAL, ConsciousLevel.SPATIAL, ConsciousLevel.COSMIC]
    assert v10_node.resonance > 0.5
    assert v9_node.resonance > 0.5
    
    logger.info("✓ Quantum resonance successful")

async def run_all_tests():
    """Run all V9 compatibility tests."""
    logger.info("Starting V9 compatibility tests...")
    
    await test_v9_connection()
    await test_consciousness_field_propagation()
    await test_unified_state_synchronization()
    await test_quantum_resonance()
    
    logger.info("All V9 compatibility tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 