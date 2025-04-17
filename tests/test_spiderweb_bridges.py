"""
Test script for Spiderweb Bridges V11 and V12
This script tests the compatibility and functionality between the quantum and cosmic bridges.
"""

import asyncio
import logging
import pytest
from dataclasses import dataclass, field
from queue import Queue, PriorityQueue
from typing import Dict, Any, Set
from src.v11.core.spiderweb_bridge import (
    SpiderwebBridge as V11Bridge,
    VersionInfo as V11VersionInfo,
    QuantumState,
    EntanglementType,
    QuantumPattern,
    QuantumNode
)
from src.v12.core.spiderweb_bridge import (
    SpiderwebBridge as V12Bridge,
    VersionInfo as V12VersionInfo,
    CosmicState,
    CosmicConnection,
    CosmicPattern,
    CosmicNode
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MockSystem:
    """Mock system for testing."""
    version: str
    queue: Queue = field(default_factory=Queue)
    priority_queue: PriorityQueue = field(default_factory=PriorityQueue)

def convert_cosmic_to_quantum_node(cosmic_node: CosmicNode) -> QuantumNode:
    """Convert a cosmic node to a quantum node."""
    return QuantumNode(
        node_id=cosmic_node.node_id,
        version=cosmic_node.version,
        state=QuantumState.ENTANGLED,
        entanglement_type=EntanglementType.QUANTUM,
        pattern=QuantumPattern.QUANTUM_FIELD,
        coherence=cosmic_node.coherence,
        resonance=cosmic_node.resonance,
        quantum_field=cosmic_node.cosmic_field,
        entanglements=cosmic_node.connections,
        quantum_echoes=cosmic_node.cosmic_echoes,
        connected_nodes=cosmic_node.connected_nodes,
        creation_time=cosmic_node.creation_time,
        last_sync=cosmic_node.last_sync,
        metadata=cosmic_node.metadata
    )

def convert_quantum_to_cosmic_node(quantum_node: QuantumNode) -> CosmicNode:
    """Convert a quantum node to a cosmic node."""
    return CosmicNode(
        node_id=quantum_node.node_id,
        version=quantum_node.version,
        state=CosmicState.CONNECTED,
        connection_type=CosmicConnection.QUANTUM,
        pattern=CosmicPattern.QUANTUM_FIELD,
        coherence=quantum_node.coherence,
        resonance=quantum_node.resonance,
        cosmic_field=quantum_node.quantum_field,
        connections=quantum_node.entanglements,
        cosmic_echoes=quantum_node.quantum_echoes,
        connected_nodes=quantum_node.connected_nodes,
        creation_time=quantum_node.creation_time,
        last_sync=quantum_node.last_sync,
        metadata=quantum_node.metadata
    )

@pytest.mark.asyncio
async def test_bridge_compatibility():
    """Test compatibility between V11 and V12 bridges."""
    logger.info("Initializing V11 and V12 bridges...")
    v11_bridge = V11Bridge()
    v12_bridge = V12Bridge()

    # Create mock systems
    v11_system = MockSystem(version="v11")
    v12_system = MockSystem(version="v12")

    # Connect bridges
    logger.info("Setting up bridge connections...")
    v11_bridge.connections["v11"] = V11VersionInfo(
        version="v11",
        system=v11_system,
        queue=v11_system.queue,
        priority_queue=v11_system.priority_queue,
        quantum_nodes={}
    )
    v12_bridge.connections["v12"] = V12VersionInfo(
        version="v12",
        system=v12_system,
        queue=v12_system.queue,
        priority_queue=v12_system.priority_queue,
        cosmic_nodes={}
    )

    # Test connection establishment
    logger.info("Testing connection establishment...")
    v11_node = await v11_bridge.create_quantum_node(
        "v11",
        EntanglementType.QUANTUM,
        QuantumPattern.QUANTUM_FIELD,
        {"test": "data"}
    )
    assert v11_node is not None, "Failed to create V11 quantum node"

    v12_node = await v12_bridge.create_cosmic_node(
        "v12",
        CosmicConnection.QUANTUM,
        CosmicPattern.QUANTUM_FIELD,
        {"test": "data"}
    )
    assert v12_node is not None, "Failed to create V12 cosmic node"

    # Test state evolution
    logger.info("Testing state evolution...")
    # Check initial states
    v11_status = v11_bridge.get_quantum_status(v11_node)
    assert v11_status is not None, "Failed to get V11 node status"
    assert v11_status["state"] == QuantumState.ENTANGLED.value, "V11 initial state incorrect"

    v12_status = v12_bridge.get_cosmic_status(v12_node)
    assert v12_status is not None, "Failed to get V12 node status"
    assert v12_status["state"] == CosmicState.CONNECTED.value, "V12 initial state incorrect"

    # Evolve states
    await v11_bridge.evolve_quantum_state(v11_node, "v11")
    await v12_bridge.evolve_cosmic_state(v12_node, "v12")

    # Check evolved states
    v11_status = v11_bridge.get_quantum_status(v11_node)
    assert v11_status["state"] == QuantumState.SUPERPOSED.value, "V11 state evolution failed"

    v12_status = v12_bridge.get_cosmic_status(v12_node)
    assert v12_status["state"] == CosmicState.SYNCHRONIZED.value, "V12 state evolution failed"

    # Test node connection
    logger.info("Testing node connection...")
    # Add cross-version connections with node type conversion
    v12_quantum_node = convert_cosmic_to_quantum_node(v12_bridge.connections["v12"].cosmic_nodes[v12_node])
    v11_cosmic_node = convert_quantum_to_cosmic_node(v11_bridge.connections["v11"].quantum_nodes[v11_node])

    # Create new version info objects for cross-version connections
    v11_v12_info = V11VersionInfo(
        version="v12",
        system=v12_system,
        queue=v12_system.queue,
        priority_queue=v12_system.priority_queue,
        quantum_nodes={}
    )
    v11_v12_info.quantum_nodes[v12_node] = v12_quantum_node

    v12_v11_info = V12VersionInfo(
        version="v11",
        system=v11_system,
        queue=v11_system.queue,
        priority_queue=v11_system.priority_queue,
        cosmic_nodes={}
    )
    v12_v11_info.cosmic_nodes[v11_node] = v11_cosmic_node

    # Set up cross-version connections
    v11_bridge.connections["v12"] = v11_v12_info
    v12_bridge.connections["v11"] = v12_v11_info
    
    connection_result = await v11_bridge.entangle_nodes(
        v11_node,
        v12_node,
        {"strength": 0.8}
    )
    assert connection_result, "Failed to create quantum entanglement"

    # Test data transmission
    logger.info("Testing data transmission...")
    test_data = {"message": "Hello from V11"}
    await v11_bridge.send_data("v11", "v12", test_data, "TEST_MESSAGE")
    
    # Verify metrics
    logger.info("Verifying metrics...")
    v11_metrics = v11_bridge.get_metrics()
    v12_metrics = v12_bridge.get_metrics()
    
    assert v11_metrics["quantum_operations"] > 0, "V11 metrics not updated"
    assert v12_metrics["cosmic_operations"] > 0, "V12 metrics not updated"

    # Test quantum synchronization
    logger.info("Testing quantum synchronization...")
    await v11_bridge.start_quantum_sync()
    
    # Wait for sync to process
    await asyncio.sleep(0.5)
    
    # Check sync status
    sync_status = await v11_bridge.get_quantum_sync_status()
    assert sync_status["active"], "Quantum sync not active"
    assert sync_status["field_strength"] > 0, "Quantum field strength not increasing"
    
    # Stop sync
    await v11_bridge.stop_quantum_sync()
    sync_status = await v11_bridge.get_quantum_sync_status()
    assert not sync_status["active"], "Quantum sync not stopped"

    # Test cosmic synchronization
    logger.info("Testing cosmic synchronization...")
    await v12_bridge.start_cosmic_sync()
    
    # Wait for sync to process
    await asyncio.sleep(0.5)
    
    # Check sync status
    sync_status = await v12_bridge.get_cosmic_sync_status()
    assert sync_status["active"], "Cosmic sync not active"
    assert sync_status["field_strength"] > 0, "Cosmic field strength not increasing"
    assert any(res > 0 for res in sync_status["dimensional_resonance"].values()), "Dimensional resonance not updating"
    
    # Stop sync
    await v12_bridge.stop_cosmic_sync()
    sync_status = await v12_bridge.get_cosmic_sync_status()
    assert not sync_status["active"], "Cosmic sync not stopped"
    
    logger.info("All tests passed successfully!")

@pytest.mark.asyncio
async def test_quantum_entanglement():
    """Test quantum entanglement features."""
    logger.info("Testing quantum entanglement...")
    v11_bridge = V11Bridge()
    v11_system = MockSystem(version="v11")
    
    # Setup bridge
    v11_bridge.connections["v11"] = V11VersionInfo(
        version="v11",
        system=v11_system,
        queue=v11_system.queue,
        priority_queue=v11_system.priority_queue,
        quantum_nodes={}
    )
    
    # Create entangled nodes
    node1 = await v11_bridge.create_quantum_node(
        "v11",
        EntanglementType.QUANTUM,
        QuantumPattern.QUANTUM_FIELD,
        {"entanglement_group": "test_group"}
    )
    node2 = await v11_bridge.create_quantum_node(
        "v11",
        EntanglementType.QUANTUM,
        QuantumPattern.QUANTUM_FIELD,
        {"entanglement_group": "test_group"}
    )
    
    # Create entanglement
    await v11_bridge.entangle_nodes(node1, node2, {"strength": 0.9})
    
    # Start sync
    await v11_bridge.start_quantum_sync()
    await asyncio.sleep(0.5)
    
    # Verify entanglement effects
    status1 = v11_bridge.get_quantum_status(node1)
    status2 = v11_bridge.get_quantum_status(node2)
    
    assert abs(status1["coherence"] - status2["coherence"]) < 0.01, "Entangled nodes not synchronized"
    assert abs(status1["resonance"] - status2["resonance"]) < 0.01, "Entangled nodes not synchronized"
    
    await v11_bridge.stop_quantum_sync()

@pytest.mark.asyncio
async def test_cosmic_resonance():
    """Test cosmic resonance features."""
    logger.info("Testing cosmic resonance...")
    v12_bridge = V12Bridge()
    v12_system = MockSystem(version="v12")
    
    # Setup bridge
    v12_bridge.connections["v12"] = V12VersionInfo(
        version="v12",
        system=v12_system,
        queue=v12_system.queue,
        priority_queue=v12_system.priority_queue,
        cosmic_nodes={}
    )
    
    # Create cosmic nodes in different dimensions
    physical_node = await v12_bridge.create_cosmic_node(
        "v12",
        CosmicConnection.QUANTUM,
        CosmicPattern.QUANTUM_FIELD,
        {"dimensions": ["physical"]}
    )
    quantum_node = await v12_bridge.create_cosmic_node(
        "v12",
        CosmicConnection.QUANTUM,
        CosmicPattern.QUANTUM_FIELD,
        {"dimensions": ["quantum"]}
    )
    
    # Connect nodes
    await v12_bridge.connect_nodes(physical_node, quantum_node, {"strength": 0.8})
    
    # Start sync
    await v12_bridge.start_cosmic_sync()
    await asyncio.sleep(0.5)
    
    # Verify dimensional resonance
    sync_status = await v12_bridge.get_cosmic_sync_status()
    assert sync_status["dimensional_resonance"]["physical"] > 0, "Physical dimension resonance not updating"
    assert sync_status["dimensional_resonance"]["quantum"] > 0, "Quantum dimension resonance not updating"
    
    await v12_bridge.stop_cosmic_sync()

async def main():
    """Main test function."""
    try:
        await test_bridge_compatibility()
        await test_quantum_entanglement()
        await test_cosmic_resonance()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 