#!/usr/bin/env python
"""
Test Echo Spiral Memory System

This script tests the functionality of the Echo Spiral Memory system,
verifying that it correctly stores, retrieves, and connects memory nodes.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import Echo Spiral Memory
from src.memory.echo_spiral_memory import (
    EchoSpiralMemory, MemoryNode, MemoryConnection,
    add_memory, connect_memories, search_memory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EchoMemoryTest")

def test_basic_functionality():
    """Test basic memory functionality"""
    logger.info("Testing basic memory functionality...")
    
    # Create a test memory instance with mock mode
    config = {
        "memory_dir": "test_memory",
        "save_interval": 60,
        "mock_mode": True
    }
    memory = EchoSpiralMemory(config)
    
    # Add memories
    logger.info("Adding memory nodes...")
    node1 = memory.add_memory(
        content="The sky is blue and the clouds are white.",
        node_type="observation",
        metadata={"source": "test", "importance": 0.7}
    )
    
    node2 = memory.add_memory(
        content="Neural networks can learn patterns from data.",
        node_type="concept",
        metadata={"source": "test", "domain": "AI"}
    )
    
    node3 = memory.add_memory(
        content="Consciousness emerges from complex system interactions.",
        node_type="theory",
        metadata={"source": "test", "confidence": 0.8}
    )
    
    # Connect memories
    logger.info("Connecting memory nodes...")
    conn1 = memory.connect_memories(
        source_id=node1.id,
        target_id=node2.id,
        connection_type="association",
        strength=0.5,
        metadata={"reason": "test connection"}
    )
    
    conn2 = memory.connect_memories(
        source_id=node2.id,
        target_id=node3.id,
        connection_type="leads_to",
        strength=0.8,
        metadata={"reason": "logical progression"}
    )
    
    # Test search
    logger.info("Testing search functionality...")
    results = memory.search_by_content("neural networks learn", limit=5)
    assert len(results) > 0, "Search should return at least one result"
    assert results[0].id == node2.id, "First result should be the neural networks node"
    
    # Test connected memories
    logger.info("Testing connected memories functionality...")
    connections = memory.get_connected_memories(node2.id)
    assert len(connections) > 0, "Should have connected memories"
    
    # Test saving
    logger.info("Testing memory saving...")
    memory.save_memory("test_echo_memory.json")
    
    # Test loading
    logger.info("Testing memory loading...")
    memory2 = EchoSpiralMemory(config)
    success = memory2.load_memory("test_echo_memory.json")
    assert success, "Memory loading should succeed"
    
    # Verify loaded memory
    assert node1.id in memory2.nodes, "Node 1 should be in loaded memory"
    assert node2.id in memory2.nodes, "Node 2 should be in loaded memory"
    assert node3.id in memory2.nodes, "Node 3 should be in loaded memory"
    
    assert conn1.id in memory2.connections, "Connection 1 should be in loaded memory"
    assert conn2.id in memory2.connections, "Connection 2 should be in loaded memory"
    
    logger.info("Basic functionality tests passed!")
    return True

def test_memory_activation():
    """Test memory activation and decay"""
    logger.info("Testing memory activation and decay...")
    
    # Create a test memory instance with mock mode
    config = {
        "memory_dir": "test_memory",
        "save_interval": 60,
        "decay_rate": 0.1,
        "mock_mode": True
    }
    memory = EchoSpiralMemory(config)
    
    # Add memories
    node1 = memory.add_memory(
        content="Memory one with high activation",
        node_type="test",
        metadata={"activation": "high"}
    )
    
    node2 = memory.add_memory(
        content="Memory two with medium activation",
        node_type="test",
        metadata={"activation": "medium"}
    )
    
    node3 = memory.add_memory(
        content="Memory three with low activation",
        node_type="test",
        metadata={"activation": "low"}
    )
    
    # Set activation levels
    with memory.lock:
        memory.nodes[node1.id].activation_level = 1.0
        memory.nodes[node2.id].activation_level = 0.5
        memory.nodes[node3.id].activation_level = 0.2
    
    # Get active memories
    active = memory.get_active_memories(threshold=0.3)
    assert len(active) >= 2, "Should have at least 2 active memories"
    assert active[0].id in [node1.id, node2.id], "First active memory should be node1 or node2"
    
    # Test activation decay
    logger.info("Testing activation decay...")
    # Simulate decay thread
    with memory.lock:
        for node in memory.nodes.values():
            if node.activation_level > 0:
                node.activation_level = max(0, node.activation_level - config["decay_rate"])
    
    # Check activation after decay
    with memory.lock:
        assert memory.nodes[node1.id].activation_level == 0.9, "Node1 activation should decay to 0.9"
        assert memory.nodes[node2.id].activation_level == 0.4, "Node2 activation should decay to 0.4"
        assert memory.nodes[node3.id].activation_level == 0.1, "Node3 activation should decay to 0.1"
    
    logger.info("Memory activation tests passed!")
    return True

def test_bidirectional_sync():
    """Test bidirectional memory synchronization"""
    logger.info("Testing bidirectional synchronization...")
    
    # Create two memory instances
    config1 = {
        "memory_dir": "test_memory_1",
        "mock_mode": True
    }
    memory1 = EchoSpiralMemory(config1)
    
    config2 = {
        "memory_dir": "test_memory_2",
        "mock_mode": True
    }
    memory2 = EchoSpiralMemory(config2)
    
    # Add memory to first instance
    node1 = memory1.add_memory(
        content="This is a shared memory",
        node_type="shared",
        metadata={"source": "memory1"}
    )
    
    node2 = memory1.add_memory(
        content="This is another shared memory",
        node_type="shared",
        metadata={"source": "memory1"}
    )
    
    # Connect nodes
    memory1.connect_memories(
        source_id=node1.id,
        target_id=node2.id,
        connection_type="related",
        metadata={"source": "memory1"}
    )
    
    # Prepare sync data
    sync_data = {
        "nodes": [
            {
                "id": node1.id,
                "content": node1.content,
                "node_type": node1.node_type,
                "metadata": node1.metadata
            },
            {
                "id": node2.id,
                "content": node2.content,
                "node_type": node2.node_type,
                "metadata": node2.metadata
            }
        ],
        "connections": [
            {
                "source_id": node1.id,
                "target_id": node2.id,
                "connection_type": "related",
                "metadata": {"source": "memory1"}
            }
        ]
    }
    
    # Sync with second instance
    response = memory2.sync_with_component("memory1", sync_data)
    
    # Verify sync
    assert response["status"] == "success", "Sync should succeed"
    assert response["synced_nodes"] == 2, "Should sync 2 nodes"
    assert response["synced_connections"] == 1, "Should sync 1 connection"
    
    # Verify nodes in second instance
    assert node1.id in memory2.nodes, "Node 1 should be in memory2"
    assert node2.id in memory2.nodes, "Node 2 should be in memory2"
    
    # Verify content matches
    assert memory2.nodes[node1.id].content == node1.content, "Node 1 content should match"
    assert memory2.nodes[node2.id].content == node2.content, "Node 2 content should match"
    
    logger.info("Bidirectional sync tests passed!")
    return True

def cleanup():
    """Clean up test files"""
    logger.info("Cleaning up test files...")
    
    # Remove test memory directories
    for dir_name in ["test_memory", "test_memory_1", "test_memory_2"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for file in dir_path.glob("*.json"):
                file.unlink()
            try:
                dir_path.rmdir()
            except:
                logger.warning(f"Could not remove directory: {dir_path}")
    
    # Remove test memory file
    test_file = Path("test_echo_memory.json")
    if test_file.exists():
        test_file.unlink()

if __name__ == "__main__":
    logger.info("Starting Echo Spiral Memory tests...")
    
    # Run tests
    try:
        test_basic_functionality()
        test_memory_activation()
        test_bidirectional_sync()
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup() 