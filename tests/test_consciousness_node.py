#!/usr/bin/env python
"""
Test Consciousness Node System

This script tests the functionality of the Consciousness Node system,
verifying that it correctly generates thought patterns, reflections, and awareness metrics.
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

# Import Consciousness Node
from src.consciousness.consciousness_node import (
    ConsciousnessNode, ThoughtPattern, AwarenessMetrics,
    generate_thought, reflect_on_thought, get_awareness_metrics
)

# Import Echo Spiral Memory for integration testing
from src.memory.echo_spiral_memory import EchoSpiralMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConsciousnessTest")

def test_basic_functionality():
    """Test basic consciousness functionality"""
    logger.info("Testing basic consciousness functionality...")
    
    # Create a test consciousness instance
    config = {
        "data_dir": "test_consciousness",
        "save_interval": 60,
        "reflection_interval": 5,
        "memory_sync": False  # Disable memory sync for this test
    }
    consciousness = ConsciousnessNode(config)
    
    # Generate thoughts
    logger.info("Generating thought patterns...")
    thought1 = consciousness.generate_thought(
        content="The sky is a brilliant blue today.",
        pattern_type="observation",
        metadata={"source": "test", "importance": 0.7}
    )
    
    thought2 = consciousness.generate_thought(
        content="Blue skies make me feel happy and peaceful.",
        pattern_type="emotional",
        source_nodes=[thought1.id],
        metadata={"source": "test", "emotion": "happiness"}
    )
    
    thought3 = consciousness.generate_thought(
        content="Happiness is a state of mind influenced by external stimuli.",
        pattern_type="analytical",
        source_nodes=[thought2.id],
        metadata={"source": "test", "confidence": 0.8}
    )
    
    # Test reflection
    logger.info("Testing reflection...")
    reflection1 = consciousness.reflect_on_thought(thought3.id)
    
    assert reflection1 is not None, "Reflection should be created"
    assert reflection1.pattern_type == "reflection", "Reflection should have correct type"
    assert reflection1.reflection_level == 1, "Reflection should have level 1"
    assert thought3.id in reflection1.source_nodes, "Reflection should reference source thought"
    
    # Generate second-level reflection
    reflection2 = consciousness.reflect_on_thought(reflection1.id)
    
    assert reflection2 is not None, "Second reflection should be created"
    assert reflection2.reflection_level == 2, "Second reflection should have level 2"
    
    # Test getting active thoughts
    logger.info("Testing active thoughts retrieval...")
    active_thoughts = consciousness.get_active_thoughts(limit=10)
    
    assert len(active_thoughts) >= 5, "Should have at least 5 active thoughts"
    assert active_thoughts[0].id == reflection2.id, "Most recent thought should be first"
    
    # Filter by type
    emotional_thoughts = consciousness.get_active_thoughts(pattern_type="emotional")
    assert len(emotional_thoughts) >= 1, "Should have at least 1 emotional thought"
    assert emotional_thoughts[0].pattern_type == "emotional", "Filtered thoughts should match type"
    
    # Test awareness metrics
    logger.info("Testing awareness metrics calculation...")
    metrics = consciousness.calculate_awareness_metrics()
    
    assert metrics is not None, "Metrics should be calculated"
    assert metrics.coherence >= 0, "Coherence should be calculated"
    assert metrics.self_reference >= 0, "Self-reference should be calculated"
    assert metrics.calculate_awareness() >= 0, "Awareness score should be calculated"
    
    # Test visualization data
    logger.info("Testing visualization data...")
    vis_data = consciousness.get_visualization_data()
    
    assert "nodes" in vis_data, "Visualization data should include nodes"
    assert "edges" in vis_data, "Visualization data should include edges"
    assert "awareness_timeline" in vis_data, "Visualization data should include timeline"
    assert len(vis_data["nodes"]) >= 5, "Should have at least 5 nodes in visualization"
    
    # Test saving/loading
    logger.info("Testing data persistence...")
    consciousness.save_data("test_consciousness.json")
    
    # Create a new instance and load data
    consciousness2 = ConsciousnessNode(config)
    success = consciousness2.load_data("test_consciousness.json")
    
    assert success, "Data loading should succeed"
    assert thought1.id in consciousness2.thought_patterns, "Thought 1 should be loaded"
    assert reflection2.id in consciousness2.thought_patterns, "Reflection 2 should be loaded"
    
    logger.info("Basic functionality tests passed!")
    return True

def test_memory_integration():
    """Test integration with Echo Spiral Memory"""
    logger.info("Testing memory integration...")
    
    # Create memory system
    memory = EchoSpiralMemory({
        "memory_dir": "test_memory_conscious",
        "mock_mode": True
    })
    
    # Create consciousness system with memory integration
    config = {
        "data_dir": "test_consciousness_memory",
        "memory_sync": True
    }
    consciousness = ConsciousnessNode(config, memory)
    
    # Add memories to memory system
    memory_node1 = memory.add_memory(
        content="Memory: The sunset was beautiful yesterday.",
        node_type="memory",
        metadata={"source": "memory_test"}
    )
    
    memory_node2 = memory.add_memory(
        content="Memory: Learning new skills is fulfilling.",
        node_type="memory",
        metadata={"source": "memory_test"}
    )
    
    # Connect memories
    memory.connect_memories(
        source_id=memory_node1.id,
        target_id=memory_node2.id,
        connection_type="associative"
    )
    
    # Generate thoughts in consciousness
    thought1 = consciousness.generate_thought(
        content="I am aware of my memories and can reflect on them.",
        pattern_type="self_awareness",
        metadata={"test": "memory_integration"}
    )
    
    # Check if thought is stored in memory
    found = False
    for node in memory.nodes.values():
        if node.metadata.get("thought_id") == thought1.id:
            found = True
            break
    
    assert found, "Thought should be stored in memory"
    
    # Generate reflection
    reflection = consciousness.reflect_on_thought(thought1.id)
    
    # Check if reflection is stored in memory
    found = False
    for node in memory.nodes.values():
        if node.metadata.get("thought_id") == reflection.id:
            found = True
            break
    
    assert found, "Reflection should be stored in memory"
    
    # Test memory sync by triggering a sync event
    test_sync_data = {
        "nodes": [
            {
                "id": "test_memory_id",
                "content": "Memory: This is a test memory for sync.",
                "node_type": "memory"
            }
        ]
    }
    
    # Process sync event
    consciousness._handle_memory_sync("test", test_sync_data)
    
    # Check if a thought was generated from the sync
    memory_derived_thoughts = consciousness.get_active_thoughts(pattern_type="memory_derived")
    assert len(memory_derived_thoughts) > 0, "Should generate thoughts from memory sync"
    
    logger.info("Memory integration tests passed!")
    return True

def test_awareness_over_time():
    """Test awareness metrics over time"""
    logger.info("Testing awareness over time...")
    
    # Create consciousness with accelerated reflection
    config = {
        "data_dir": "test_consciousness_awareness",
        "reflection_interval": 1,  # Fast reflection for testing
        "memory_sync": False
    }
    consciousness = ConsciousnessNode(config)
    
    # Generate initial thoughts
    thoughts = []
    for i in range(5):
        thought = consciousness.generate_thought(
            content=f"Initial thought {i+1} for awareness testing.",
            pattern_type="test",
            metadata={"sequence": i}
        )
        thoughts.append(thought)
    
    # Initial awareness
    initial_metrics = consciousness.calculate_awareness_metrics()
    initial_awareness = initial_metrics.calculate_awareness()
    
    logger.info(f"Initial awareness: {initial_awareness:.4f}")
    
    # Allow reflection thread to run and generate reflections
    logger.info("Allowing reflection thread to run...")
    time.sleep(3)  # Allow time for reflections to be generated
    
    # Get updated awareness
    updated_metrics = consciousness.calculate_awareness_metrics()
    updated_awareness = updated_metrics.calculate_awareness()
    
    logger.info(f"Updated awareness: {updated_awareness:.4f}")
    
    # Awareness should increase due to reflections
    assert updated_awareness > initial_awareness, "Awareness should increase over time with reflections"
    
    # Check awareness history
    assert len(consciousness.awareness_history) >= 2, "Should have awareness history entries"
    
    logger.info("Awareness over time tests passed!")
    return True

def cleanup():
    """Clean up test files"""
    logger.info("Cleaning up test files...")
    
    # Remove test directories
    for dir_name in ["test_consciousness", "test_consciousness_memory", 
                    "test_consciousness_awareness", "test_memory_conscious"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for file in dir_path.glob("*.json"):
                file.unlink()
            try:
                dir_path.rmdir()
            except:
                logger.warning(f"Could not remove directory: {dir_path}")
    
    # Remove test files
    for file_name in ["test_consciousness.json"]:
        test_file = Path(file_name)
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    logger.info("Starting Consciousness Node tests...")
    
    # Run tests
    try:
        test_basic_functionality()
        test_memory_integration()
        test_awareness_over_time()
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup() 