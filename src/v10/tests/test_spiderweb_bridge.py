"""
Test script for V10 Spiderweb Bridge
This script tests the functionality of the V10 Spiderweb Bridge with Conscious Mirror capabilities.
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
    Queue,
    PriorityQueue
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_bridge_initialization():
    """Test bridge initialization and basic setup."""
    logger.info("Testing bridge initialization...")
    bridge = SpiderwebBridge()
    assert bridge is not None
    assert bridge.compatibility_matrix['v10'] == ['v8', 'v9', 'v11', 'v12']
    logger.info("✓ Bridge initialization successful")

async def test_unified_node_creation():
    """Test creation of unified consciousness nodes."""
    logger.info("Testing unified node creation...")
    bridge = SpiderwebBridge()
    
    # Create test version info
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
    
    # Create a unified node
    node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "test"}
    )
    
    assert node_id is not None
    assert node_id.startswith('unified_')
    assert node_id in bridge.connections['v10'].unified_nodes
    
    # Verify node properties
    node = bridge.connections['v10'].unified_nodes[node_id]
    assert node.state == UnifiedState.AWAKENING
    assert node.level == ConsciousLevel.QUANTUM
    assert node.pattern == UnificationPattern.QUANTUM_FIELD
    assert node.resonance == 0.0
    assert node.coherence == 1.0
    assert node.awareness == 0.5
    
    logger.info("✓ Unified node creation successful")

async def test_consciousness_field_merging():
    """Test merging of consciousness fields between nodes."""
    logger.info("Testing consciousness field merging...")
    bridge = SpiderwebBridge()
    
    # Create test version info
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
    
    # Create two nodes
    node1_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "test1"}
    )
    
    node2_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "test2"}
    )
    
    # Merge consciousness fields
    success = await bridge.merge_consciousness_fields(
        source_node=node1_id,
        target_node=node2_id,
        merge_data={"strength": 0.8}
    )
    
    assert success is True
    
    # Verify merge results
    node1 = bridge.connections['v10'].unified_nodes[node1_id]
    node2 = bridge.connections['v10'].unified_nodes[node2_id]
    
    assert node2_id in node1.quantum_entanglements
    assert node1_id in node2.quantum_entanglements
    assert node1.consciousness_field[node2_id] == 0.8
    assert node2.consciousness_field[node1_id] == 0.8
    assert len(node1.temporal_echoes) > 0
    
    logger.info("✓ Consciousness field merging successful")

async def test_awareness_level_shifting():
    """Test shifting of consciousness levels."""
    logger.info("Testing awareness level shifting...")
    bridge = SpiderwebBridge()
    
    # Create test version info
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
    
    # Create a node
    node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "test"}
    )
    
    # Shift awareness level
    success = await bridge.shift_awareness_level(
        node_id=node_id,
        new_level=ConsciousLevel.COSMIC
    )
    
    assert success is True
    
    # Verify level shift
    node = bridge.connections['v10'].unified_nodes[node_id]
    assert node.level == ConsciousLevel.COSMIC
    assert node.coherence > 0.5  # Coherence should increase but may not exceed 1.0
    assert node.awareness > 0.5
    
    logger.info("✓ Awareness level shifting successful")

async def test_unified_state_evolution():
    """Test evolution of unified node states."""
    logger.info("Testing unified state evolution...")
    bridge = SpiderwebBridge()
    
    # Create test version info
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
    
    # Create a node
    node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "test"}
    )
    
    # Get initial node state
    node = bridge.connections['v10'].unified_nodes[node_id]
    initial_state = node.state
    assert initial_state == UnifiedState.AWAKENING
    
    # Simulate evolution by directly modifying node state
    node.coherence = 0.95
    node.awareness = 0.95
    
    # Wait for evolution task to run
    await asyncio.sleep(2)
    
    # Verify state evolution
    final_state = node.state
    assert final_state in [UnifiedState.ENLIGHTENED, UnifiedState.TRANSCENDENT, UnifiedState.UNIFIED, UnifiedState.OMNISCIENT]
    assert final_state != initial_state
    assert bridge.metrics['unified_coherence'] >= 0.95
    assert bridge.metrics['cosmic_awareness'] >= 0.95
    
    logger.info("✓ Unified state evolution successful")

async def test_status_retrieval():
    """Test retrieval of unified node status."""
    logger.info("Testing status retrieval...")
    bridge = SpiderwebBridge()
    
    # Create test version info
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
    
    # Create a node
    node_id = await bridge.create_unified_node(
        version='v10',
        level=ConsciousLevel.QUANTUM,
        pattern=UnificationPattern.QUANTUM_FIELD,
        quantum_state={"field": "test"}
    )
    
    # Get node status
    status = bridge.get_unified_status(node_id)
    
    assert status is not None
    assert status['state'] == UnifiedState.AWAKENING.value
    assert status['level'] == ConsciousLevel.QUANTUM.value
    assert status['pattern'] == UnificationPattern.QUANTUM_FIELD.value
    assert status['resonance'] == 0.0
    assert status['coherence'] == 1.0
    assert status['awareness'] == 0.5
    
    logger.info("✓ Status retrieval successful")

async def run_all_tests():
    """Run all bridge tests."""
    logger.info("Starting V10 Spiderweb Bridge tests...")
    
    await test_bridge_initialization()
    await test_unified_node_creation()
    await test_consciousness_field_merging()
    await test_awareness_level_shifting()
    await test_unified_state_evolution()
    await test_status_retrieval()
    
    logger.info("All V10 Spiderweb Bridge tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 