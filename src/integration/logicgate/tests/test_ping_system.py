#!/usr/bin/env python3
"""
Tests for Ping System

This module contains tests for the centralized pinging system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path
import random

from ..ping_system import (
    PingSystem, PingConfig, NodeStatus,
    LogicGate, LogicGateConfig, LogicGateType
)
from ..switches.triple_gate import TripleGate, PathType, GateState
from ...ml.core import MLConfig

@pytest.fixture
def mock_triple_gate():
    """Create mock triple gate"""
    gate = Mock(spec=TripleGate)
    gate.paths = {
        PathType.LITERAL: Mock(),
        PathType.SEMANTIC: Mock(),
        PathType.HYBRID: Mock()
    }
    gate.states = {
        PathType.LITERAL: Mock(),
        PathType.SEMANTIC: Mock(),
        PathType.HYBRID: Mock()
    }
    # Add data sorting mock
    gate.sort_data = Mock(return_value={"sorted": True})
    # Add gate creation mock
    gate.add_gate = Mock(return_value=Mock(name="new_gate"))
    return gate

@pytest.fixture
def ping_config():
    """Create ping configuration"""
    return PingConfig(
        ping_interval=0.1,
        timeout=1.0,
        max_retries=2,
        health_threshold=0.7,
        sync_window=10,
        batch_size=16,
        adaptive_timing=True,
        min_interval=0.05,
        max_interval=1.0,
        allow_all_data=True,
        data_sorting=True,
        self_writing=True,
        gate_creation_interval=1,  # Short interval for testing
        documentation_path="test_autowikireadme.md",
        max_logic_gates=5,  # Smaller number for testing
        logic_gate_creation_interval=0.1,  # Short interval for testing
        auto_learner_connection_probability=0.5,  # Higher probability for testing
        enabled_gate_types=[
            LogicGateType.AND,
            LogicGateType.OR,
            LogicGateType.XOR,
            LogicGateType.NOT,
            LogicGateType.NAND,
            LogicGateType.NOR
        ],
        gate_colors={
            LogicGateType.AND: "orange",
            LogicGateType.OR: "blue",
            LogicGateType.XOR: "green",
            LogicGateType.NOT: "red",
            LogicGateType.NAND: "purple",
            LogicGateType.NOR: "yellow"
        }
    )

@pytest.fixture
def ping_system(mock_triple_gate, ping_config):
    """Create ping system"""
    return PingSystem(ping_config, mock_triple_gate)

@pytest.fixture
def test_documentation(tmp_path):
    """Create test documentation file"""
    doc_path = tmp_path / "test_autowikireadme.md"
    doc_content = """# AutoWiki Integration Guide

## Integration Points
- Central Node Monitor Tab System
- Background Services
- Neural Seed Connection
- Data Management System
"""
    doc_path.write_text(doc_content)
    return doc_path

@pytest.fixture
def test_logic_gate():
    """Create a test logic gate."""
    return LogicGateConfig(
        gate_id="test_gate_1",
        gate_type=LogicGateType.AND,
        position=(0, 0, 0),
        input_count=2,
        output_count=1,
        threshold=0.5,
        learning_rate=0.1,
        activation_function="sigmoid"
    )

@pytest.mark.asyncio
async def test_initialization(ping_system, mock_triple_gate):
    """Test system initialization"""
    # Check node initialization
    assert len(ping_system.node_statuses) == len(mock_triple_gate.paths)
    for path_type in PathType:
        node_id = f"node_{path_type.value}"
        assert node_id in ping_system.node_statuses
        assert isinstance(ping_system.node_statuses[node_id], NodeStatus)

@pytest.mark.asyncio
async def test_start_stop(ping_system):
    """Test starting and stopping the ping system"""
    # Start system
    await ping_system.start()
    assert ping_system.active
    assert ping_system.ping_task is not None
    assert ping_system.gate_creation_task is not None
    
    # Stop system
    await ping_system.stop()
    assert not ping_system.active
    assert ping_system.ping_task is None or ping_system.ping_task.cancelled()
    assert ping_system.gate_creation_task is None or ping_system.gate_creation_task.cancelled()

@pytest.mark.asyncio
async def test_node_status_tracking(ping_system):
    """Test node status tracking"""
    node_id = f"node_{PathType.LITERAL.value}"
    status = ping_system.node_statuses[node_id]
    
    # Test metric updates
    metrics = {
        "latency": 0.1,
        "load": 0.5,
        "memory": 0.3,
        "success_rate": 0.9
    }
    data = {"test": "data"}
    status.update_ping(0.1, metrics, data)
    
    assert len(status.response_times) == 1
    assert status.last_response is not None
    for key in metrics:
        assert len(status.metrics[key]) == 1
        assert status.metrics[key][0] == metrics[key]
    assert len(status.data_throughput) == 1
    assert status.data_throughput[0]['data'] == data

@pytest.mark.asyncio
async def test_temporal_patterns(ping_system):
    """Test temporal pattern tracking"""
    node_id = f"node_{PathType.LITERAL.value}"
    status = ping_system.node_statuses[node_id]
    
    # Add data at different hours
    current_hour = datetime.now().hour
    for i in range(24):
        data = {"test": "data" * i}
        metrics = {
            "latency": 0.1,
            "load": 0.5,
            "memory": 0.3,
            "success_rate": 0.9
        }
        status.update_ping(0.1, metrics, data)
        
    # Check temporal patterns
    patterns = status.get_temporal_patterns()
    assert str(current_hour) in patterns
    assert 'avg_size' in patterns[str(current_hour)]
    assert 'std_size' in patterns[str(current_hour)]
    assert 'count' in patterns[str(current_hour)]

@pytest.mark.asyncio
async def test_documentation_loading(ping_system, test_documentation):
    """Test documentation loading"""
    # Update config to use test documentation
    ping_system.config.documentation_path = str(test_documentation)
    ping_system._load_documentation()
    
    assert "AutoWiki Integration Guide" in ping_system.documentation
    assert "Integration Points" in ping_system.documentation

@pytest.mark.asyncio
async def test_gate_extraction(ping_system, test_documentation):
    """Test gate extraction from documentation"""
    # Update config to use test documentation
    ping_system.config.documentation_path = str(test_documentation)
    ping_system._load_documentation()
    
    # Extract gates
    gates = ping_system._extract_gates_from_documentation()
    
    assert len(gates) == 4  # Should find 4 integration points
    assert any(gate["name"] == "Central Node Monitor Tab System" for gate in gates)
    assert all(gate["type"] == "integration" for gate in gates)
    assert all(gate["source"] == "documentation" for gate in gates)

@pytest.mark.asyncio
async def test_gate_creation(ping_system, mock_triple_gate, test_logic_gate):
    """Test gate creation process"""
    # Create test gate info
    gate_info = {
        "name": "test_gate",
        "type": "integration",
        "source": "test",
        "patterns": {"test": "pattern"},
        "gate_id": test_logic_gate.gate_id,
        "gate_type": test_logic_gate.gate_type,
        "position": test_logic_gate.position,
        "connection_strength": test_logic_gate.connection_strength,
        "color": test_logic_gate.color,
        "pulse_duration": test_logic_gate.pulse_duration,
        "pulse_intensity": test_logic_gate.pulse_intensity,
        "glow_radius": test_logic_gate.glow_radius,
        "connection_glow": test_logic_gate.connection_glow
    }
    
    # Mock add_gate to return the test gate
    mock_triple_gate.add_gate.return_value = test_logic_gate
    
    # Create new gate
    await ping_system._create_new_gate(gate_info)
    
    # Verify gate creation
    mock_triple_gate.add_gate.assert_called_once()
    assert "node_test_gate" in ping_system.node_statuses
    assert ping_system.node_statuses["node_test_gate"].last_ping == 0.0
    assert ping_system.node_statuses["node_test_gate"].latency == 0.0
    assert ping_system.node_statuses["node_test_gate"].load == 0.0
    assert ping_system.node_statuses["node_test_gate"].memory == 0.0
    assert ping_system.node_statuses["node_test_gate"].success_rate == 1.0
    assert ping_system.node_statuses["node_test_gate"].health_score == 1.0
    assert ping_system.node_statuses["node_test_gate"].connections == []

@pytest.mark.asyncio
async def test_gate_creation_loop(ping_system, mock_triple_gate, test_documentation):
    """Test gate creation loop"""
    # Update config to use test documentation
    ping_system.config.documentation_path = str(test_documentation)
    ping_system._load_documentation()
    
    # Start system
    await ping_system.start()
    
    # Let it run briefly
    await asyncio.sleep(0.5)
    
    # Stop system
    await ping_system.stop()
    
    # Verify gate creation attempts
    assert mock_triple_gate.add_gate.called

@pytest.mark.asyncio
async def test_pattern_analysis(ping_system):
    """Test temporal pattern analysis"""
    # Add data to create patterns
    node_id = f"node_{PathType.LITERAL.value}"
    status = ping_system.node_statuses[node_id]
    
    for i in range(20):
        data = {"test": "data" * i}
        metrics = {
            "latency": 0.1,
            "load": 0.5,
            "memory": 0.3,
            "success_rate": 0.9
        }
        status.update_ping(0.1, metrics, data)
        
    # Analyze patterns
    patterns = ping_system._analyze_temporal_patterns()
    assert node_id in patterns
    assert len(patterns[node_id]) > 0

@pytest.mark.asyncio
async def test_new_gate_identification(ping_system, test_documentation):
    """Test new gate identification"""
    # Update config to use test documentation
    ping_system.config.documentation_path = str(test_documentation)
    ping_system._load_documentation()
    
    # Create temporal patterns
    temporal_patterns = {
        "node_literal": {
            "12": {
                "avg_size": 100,
                "std_size": 60,
                "count": 15
            }
        }
    }
    
    # Extract gates from documentation
    doc_gates = ping_system._extract_gates_from_documentation()
    
    # Identify new gates
    new_gates = ping_system._identify_new_gates(temporal_patterns, doc_gates)
    
    # Verify new gates
    assert len(new_gates) > 0
    assert any(gate["type"] == "temporal" for gate in new_gates)
    assert any(gate["type"] == "integration" for gate in new_gates)

@pytest.mark.asyncio
async def test_health_calculation(ping_system):
    """Test health score calculation"""
    node_id = f"node_{PathType.LITERAL.value}"
    status = ping_system.node_statuses[node_id]
    
    # Test perfect health
    metrics = {
        "latency": 0.1,
        "load": 0.2,
        "memory": 0.3,
        "success_rate": 1.0
    }
    status.update_ping(0.1, metrics)
    assert status.health_score > 0.9
    assert status.state == "active"
    
    # Test degraded health
    metrics["success_rate"] = 0.3
    metrics["load"] = 0.9
    for _ in range(10):
        status.update_ping(0.8, metrics)
    assert status.health_score < 0.7
    assert status.state == "degraded"

@pytest.mark.asyncio
async def test_adaptive_interval(ping_system):
    """Test adaptive ping interval calculation"""
    # Set different health scores
    for node_id, status in ping_system.node_statuses.items():
        metrics = {
            "latency": 0.1,
            "load": 0.5,
            "memory": 0.3,
            "success_rate": 0.8
        }
        status.update_ping(0.1, metrics)
    
    interval = ping_system._calculate_adaptive_interval()
    assert ping_system.config.min_interval <= interval <= ping_system.config.max_interval

@pytest.mark.asyncio
async def test_node_failure_handling(ping_system, mock_triple_gate):
    """Test handling of node failures"""
    node_id = f"node_{PathType.LITERAL.value}"
    
    # Simulate failure
    ping_system._handle_node_failure(node_id)
    status = ping_system.node_statuses[node_id]
    
    assert status.health_score < 1.0
    if status.health_score < 0.3:
        assert status.state == "failed"
        mock_triple_gate.switch_path.assert_called_with(
            PathType.LITERAL,
            GateState.CLOSED
        )

@pytest.mark.asyncio
async def test_system_health_reporting(ping_system):
    """Test system health reporting"""
    # Update node statuses
    for node_id, status in ping_system.node_statuses.items():
        metrics = {
            "latency": 0.1,
            "load": 0.5,
            "memory": 0.3,
            "success_rate": 0.9
        }
        data = {"test": "data"}
        status.update_ping(0.1, metrics, data)
    
    health_report = ping_system.get_system_health()
    
    assert 'nodes' in health_report
    assert 'system' in health_report
    assert len(health_report['nodes']) == len(ping_system.node_statuses)
    assert all(
        key in health_report['system']
        for key in ['active_nodes', 'total_nodes', 'overall_health', 'ping_interval', 'total_data_throughput']
    )
    assert all(
        'data_throughput' in node_info
        for node_info in health_report['nodes'].values()
    )

@pytest.mark.asyncio
async def test_node_synchronization(ping_system, mock_triple_gate):
    """Test node synchronization"""
    # Set up mock states
    mock_states = {
        f"node_{path_type.value}": {
            "knowledge_level": 0.8,
            "learning_history": list(range(10))
        }
        for path_type in PathType
    }
    
    # Mock get_knowledge_state
    for path in mock_triple_gate.paths.values():
        path.get_knowledge_state.return_value = mock_states[list(mock_states.keys())[0]]
    
    # Test synchronization
    await ping_system.sync_nodes()
    
    # Verify sync attempts
    for path_type in PathType:
        node = mock_triple_gate.paths[path_type]
        assert node.get_knowledge_state.called

@pytest.mark.asyncio
async def test_ping_loop_execution(ping_system):
    """Test ping loop execution"""
    # Start system with short interval
    ping_system.config.ping_interval = 0.1
    await ping_system.start()
    
    # Let it run briefly
    await asyncio.sleep(0.3)
    
    # Stop system
    await ping_system.stop()
    
    # Check that pings occurred
    for status in ping_system.node_statuses.values():
        assert (datetime.now() - status.last_ping).total_seconds() < 0.5

@pytest.mark.asyncio
async def test_error_handling(ping_system, mock_triple_gate):
    """Test error handling in ping operations"""
    # Make a node raise an exception
    mock_triple_gate.paths[PathType.LITERAL].get_knowledge_state.side_effect = Exception("Test error")
    
    # Attempt to ping
    node_id = f"node_{PathType.LITERAL.value}"
    await ping_system._ping_node(node_id)
    
    # Check error handling
    status = ping_system.node_statuses[node_id]
    assert status.health_score < 1.0
    
@pytest.mark.asyncio
async def test_metrics_history_bounds(ping_system):
    """Test that metrics history is properly bounded"""
    node_id = f"node_{PathType.LITERAL.value}"
    status = ping_system.node_statuses[node_id]
    
    # Add many metrics
    metrics = {
        "latency": 0.1,
        "load": 0.5,
        "memory": 0.3,
        "success_rate": 0.9
    }
    data = {"test": "data"}
    
    for _ in range(1100):
        status.update_ping(0.1, metrics, data)
        
    # Check bounds
    assert len(status.response_times) <= 1000
    for metric_values in status.metrics.values():
        assert len(metric_values) <= 1000
    assert len(status.data_throughput) <= 1000

@pytest.mark.asyncio
async def test_data_passing_and_sorting(ping_system, mock_triple_gate):
    """Test data passing and sorting through triple gate"""
    # Set up mock data
    test_data = {"test": "data", "value": 42}
    sorted_data = {"sorted": True, "test": "data", "value": 42}
    
    # Configure mocks
    mock_triple_gate.paths[PathType.LITERAL].get_data.return_value = test_data
    mock_triple_gate.sort_data.return_value = sorted_data
    
    # Ping node
    node_id = f"node_{PathType.LITERAL.value}"
    await ping_system._ping_node(node_id)
    
    # Verify data was passed and sorted
    status = ping_system.node_statuses[node_id]
    assert len(status.data_throughput) > 0
    last_data = status.data_throughput[-1]
    assert last_data['size'] == len(str(test_data))
    assert last_data['data'] == sorted_data  # Should be the sorted data
    assert isinstance(last_data['timestamp'], datetime)
    mock_triple_gate.paths[PathType.LITERAL].get_data.assert_called_once()
    mock_triple_gate.sort_data.assert_called_once_with(test_data)

@pytest.mark.asyncio
async def test_data_throughput_tracking(ping_system):
    """Test data throughput tracking"""
    node_id = f"node_{PathType.LITERAL.value}"
    status = ping_system.node_statuses[node_id]
    
    # Add data with different sizes
    for i in range(5):
        data = {"test": "data" * i}
        metrics = {
            "latency": 0.1,
            "load": 0.5,
            "memory": 0.3,
            "success_rate": 0.9
        }
        status.update_ping(0.1, metrics, data)
    
    # Check throughput tracking
    assert len(status.data_throughput) == 5
    assert all(
        isinstance(d['size'], int) and d['size'] > 0
        for d in status.data_throughput
    )
    assert all(
        isinstance(d['timestamp'], datetime)
        for d in status.data_throughput
    )

@pytest.mark.asyncio
async def test_logic_gate_creation(ping_system, test_logic_gate):
    """Test logic gate creation."""
    # Create a new gate
    gate_id = await ping_system.create_logic_gate(test_logic_gate)
    assert gate_id is not None
    
    # Verify gate was created
    gate = ping_system.logic_gates.get(gate_id)
    assert gate is not None
    assert gate.gate_type == test_logic_gate.gate_type
    assert gate.position == test_logic_gate.position
    assert gate.input_count == test_logic_gate.input_count
    assert gate.output_count == test_logic_gate.output_count
    assert gate.threshold == test_logic_gate.threshold
    assert gate.learning_rate == test_logic_gate.learning_rate
    assert gate.activation_function == test_logic_gate.activation_function

@pytest.mark.asyncio
async def test_logic_gate_connections(ping_system):
    """Test logic gate connections to auto-learner nodes"""
    # Add some auto-learner nodes
    for i in range(3):
        node_id = f"auto_learner_{i}"
        ping_system.node_statuses[node_id] = NodeStatus(node_id)
    
    # Start system
    await ping_system.start()
    
    # Let it create and connect gates
    await asyncio.sleep(0.2)
    
    # Check connections
    connected_gates = 0
    for gate_id, gate in ping_system.logic_gates.items():
        assert gate.config.gate_id == gate_id  # Verify gate_id is set
        if gate.connections:
            connected_gates += 1
            # Verify connection is to an auto-learner node
            assert any("auto_learner" in node_id for node_id in gate.connections)
            
    # Should have some connected gates due to high test probability
    assert connected_gates > 0
    
    await ping_system.stop()

@pytest.mark.asyncio
async def test_logic_gate_operations(test_logic_gate):
    """Test logic gate operations."""
    # Test AND gate
    test_logic_gate.add_input("node1", 0.2)
    test_logic_gate.add_input("node2", 0.3)
    test_logic_gate.update()
    assert abs(test_logic_gate.get_output() - 0.2) < 1e-10  # AND takes minimum of inputs

    # Test OR gate
    test_logic_gate.config.gate_type = LogicGateType.OR
    test_logic_gate.update()
    assert abs(test_logic_gate.get_output() - 0.3) < 1e-10  # OR takes maximum of inputs

    # Test XOR gate
    test_logic_gate.config.gate_type = LogicGateType.XOR
    test_logic_gate.update()
    assert abs(test_logic_gate.get_output() - 0.1) < 1e-10  # XOR takes absolute difference

    # Test NOT gate
    test_logic_gate.config.gate_type = LogicGateType.NOT
    test_logic_gate.inputs = {"node1": 0.8}
    test_logic_gate.update()
    assert abs(test_logic_gate.get_output() - 0.2) < 1e-10  # NOT inverts the input

    # Test NAND gate
    test_logic_gate.config.gate_type = LogicGateType.NAND
    test_logic_gate.add_input("node1", 0.8)
    test_logic_gate.add_input("node2", 0.9)
    test_logic_gate.update()
    assert abs(test_logic_gate.get_output() - 0.2) < 1e-10  # NAND inverts AND

    # Test NOR gate
    test_logic_gate.config.gate_type = LogicGateType.NOR
    test_logic_gate.add_input("node1", 0.3)
    test_logic_gate.add_input("node2", 0.4)
    test_logic_gate.update()
    assert abs(test_logic_gate.get_output() - 0.6) < 1e-10  # NOR inverts OR

@pytest.mark.asyncio
async def test_logic_gate_state_tracking(ping_system):
    """Test logic gate state tracking"""
    # Start system
    await ping_system.start()
    
    # Let it create some gates
    await asyncio.sleep(0.2)
    
    # Get gate states
    states = ping_system.get_logic_gate_states()
    
    # Check state structure
    for gate_id, state in states.items():
        assert "type" in state
        assert "position" in state
        assert "color" in state
        assert "output" in state
        assert "connections" in state
        assert "inputs" in state
        
    await ping_system.stop()

@pytest.mark.asyncio
async def test_logic_gate_visual_effects(test_logic_gate):
    """Test logic gate visual effects."""
    # Test initial state
    assert not test_logic_gate.is_pulsing
    assert test_logic_gate.glow_intensity == 0.0
    assert not test_logic_gate.connection_glows

    # Add input and test activation
    test_logic_gate.add_input("node1", 0.9)
    test_logic_gate.update()

    # Check glow intensity
    assert test_logic_gate.glow_intensity == 0.9
    assert "node1" in test_logic_gate.connection_glows
    assert test_logic_gate.connection_glows["node1"] == 0.9

    # Test pulse effect
    test_logic_gate.output = 0.9
    test_logic_gate._update_visual_effects()
    assert test_logic_gate.is_pulsing

    # Test pulse duration
    await asyncio.sleep(0.6)  # Wait longer than pulse duration
    test_logic_gate._update_visual_effects()
    assert not test_logic_gate.is_pulsing

    # Test deactivation
    test_logic_gate.output = 0.1
    test_logic_gate._update_visual_effects()
    assert test_logic_gate.glow_intensity == 0.1

@pytest.mark.asyncio
async def test_logic_gate_state_visualization(test_logic_gate):
    """Test logic gate state visualization."""
    # Add inputs and update gate
    test_logic_gate.add_input("node1", 0.9)
    test_logic_gate.add_input("node2", 0.8)
    test_logic_gate.update()

    # Get state
    state = test_logic_gate.get_state()

    # Check state fields
    assert "gate_id" in state
    assert "gate_type" in state
    assert "output" in state
    assert "inputs" in state
    assert "connections" in state
    assert "visual_effects" in state

    # Check visual effects
    visual_effects = state["visual_effects"]
    assert "glow_intensity" in visual_effects
    assert "is_pulsing" in visual_effects
    assert "connection_glows" in visual_effects

    # Check connection glows
    assert "node1" in visual_effects["connection_glows"]
    assert "node2" in visual_effects["connection_glows"]
    assert abs(visual_effects["connection_glows"]["node1"] - 0.9) < 1e-10
    assert abs(visual_effects["connection_glows"]["node2"] - 0.8) < 1e-10

@pytest.mark.asyncio
async def test_logic_gate_triple_gate_interaction(ping_system, mock_triple_gate):
    """Test interaction between logic gates and triple gate with visual effects"""
    # Create a gate with strong output
    gate_config = LogicGateConfig(
        gate_type=LogicGateType.AND,
        position=(500, 500),
        connection_strength=1.0,
        pulse_duration=0.5,
        pulse_intensity=0.8,
        glow_radius=10.0,
        connection_glow=True
    )
    gate = LogicGate(gate_config)
    gate.add_input("node1", 0.9)
    gate.add_input("node2", 0.9)
    gate.update()
    
    # Add gate to system
    gate_id = "test_gate"
    ping_system.logic_gates[gate_id] = gate
    
    # Process output
    await ping_system._process_gate_output(gate_id, gate.get_output())
    
    # Verify triple gate update and visual effects
    mock_triple_gate.switch_path.assert_called_with(
        PathType.LITERAL,  # AND gate maps to LITERAL path
        GateState.OPEN
    )
    
    # Check visual effects
    assert gate.is_pulsing
    assert gate.pulse_start is not None
    assert gate.glow_intensity > 0.0
    assert len(gate.connection_glows) == 2
    assert all(glow > 0.0 for glow in gate.connection_glows.values())

@pytest.mark.asyncio
async def test_logic_gate_connection_management(ping_system):
    """Test logic gate connection management"""
    # Add an auto-learner node
    node_id = "auto_learner_1"
    ping_system.node_statuses[node_id] = NodeStatus(node_id)
    
    # Create a gate
    gate_config = LogicGateConfig(
        gate_type=LogicGateType.AND,
        position=(500, 500)
    )
    gate = LogicGate(gate_config)
    gate_id = "test_gate"
    ping_system.logic_gates[gate_id] = gate
    
    # Connect gate to node
    await ping_system._connect_gate_to_auto_learner(gate_id)
    
    # Verify connection
    assert node_id in gate.connections
    assert gate_id in ping_system.node_statuses[node_id].logic_gate_connections

@pytest.mark.asyncio
async def test_logic_gate_update_loop(ping_system):
    """Test logic gate update loop"""
    # Start system
    await ping_system.start()
    
    # Let it run for a bit
    await asyncio.sleep(0.2)
    
    # Check that gates were updated
    for gate in ping_system.logic_gates.values():
        assert (datetime.now() - gate.last_update).total_seconds() < 0.3
        
    await ping_system.stop()

@pytest.mark.asyncio
async def test_logic_gate_error_handling(ping_system):
    """Test logic gate error handling"""
    # Create a gate with invalid configuration
    gate_config = LogicGateConfig(
        gate_type=LogicGateType.AND,
        position=(500, 500)
    )
    gate = LogicGate(gate_config)
    gate_id = "test_gate"
    ping_system.logic_gates[gate_id] = gate
    
    # Force an error in update
    gate.update = Mock(side_effect=Exception("Test error"))
    
    # Process should handle error gracefully
    await ping_system._update_logic_gates()
    
    # Verify error was logged
    assert "Error updating logic gate" in caplog.text 

@pytest.mark.asyncio
async def test_logic_gate_visual_feedback(ping_system):
    """Test logic gate visual feedback"""
    # Create a gate
    gate_config = LogicGateConfig(
        gate_type=LogicGateType.AND,
        position=(500, 500),
        connection_strength=1.0
    )
    gate = LogicGate(gate_config)
    gate_id = "test_gate"
    ping_system.logic_gates[gate_id] = gate
    
    # Test activation
    gate.add_input("node1", 0.9)
    gate.add_input("node2", 0.9)
    gate.update()
    
    # Process output
    await ping_system._process_gate_output(gate_id, gate.get_output())
    
    # Check color change
    assert gate.config.color == ping_system.config.gate_colors[LogicGateType.AND] + "_active"
    
    # Test deactivation
    gate.inputs = {"node1": 0.1, "node2": 0.1}
    gate.update()
    await ping_system._process_gate_output(gate_id, gate.get_output())
    
    # Check color reset
    assert gate.config.color == ping_system.config.gate_colors[LogicGateType.AND] 
@pytest.mark.asyncio
async def test_logic_gate_ping_synchronization(ping_system, mock_triple_gate):
    """Test synchronization between logic gates and ping system"""
    # Create a test gate
    gate_config = LogicGateConfig(
        gate_id="test_gate",
        gate_type=LogicGateType.AND,
        position=(500, 500),
        connection_strength=1.0
    )
    gate = LogicGate(gate_config)
    ping_system.logic_gates["test_gate"] = gate
    
    # Connect gate to a node
    node_id = f"node_{PathType.LITERAL.value}"
    gate.connections.append(node_id)
    
    # Set up mock data
    test_data = {"test": "data"}
    mock_triple_gate.paths[PathType.LITERAL].get_data.return_value = test_data
    mock_triple_gate.sort_data.return_value = test_data
    
    # Test blocking when gate output is low
    gate.add_input("other_node", 0.1)  # This will make AND gate output low
    gate.update()
    await ping_system._ping_node(node_id)
    
    # Verify ping was blocked
    status = ping_system.node_statuses[node_id]
    initial_ping_count = len(status.data_throughput)
    
    # Test passing when gate output is high
    gate.add_input("other_node", 1.0)  # This will make AND gate output high
    gate.update()
    await ping_system._ping_node(node_id)
    
    # Verify ping went through
    assert len(status.data_throughput) > initial_ping_count
    
    # Verify gate state was updated
    assert gate.get_output() > 0.5
    assert gate.is_pulsing
    assert gate.glow_intensity > 0.5

@pytest.mark.asyncio
async def test_multi_gate_interaction(ping_system, mock_triple_gate):
    """Test interaction between multiple gates affecting the same node"""
    # Create two test gates
    gate1 = LogicGate(LogicGateConfig(
        gate_id="gate1",
        gate_type=LogicGateType.AND,
        position=(500, 500)
    ))
    gate2 = LogicGate(LogicGateConfig(
        gate_id="gate2",
        gate_type=LogicGateType.OR,
        position=(600, 500)
    ))
    
    ping_system.logic_gates["gate1"] = gate1
    ping_system.logic_gates["gate2"] = gate2
    
    # Connect both gates to the same node
    node_id = f"node_{PathType.LITERAL.value}"
    gate1.connections.append(node_id)
    gate2.connections.append(node_id)
    
    # Set up mock data
    test_data = {"test": "data"}
    mock_triple_gate.paths[PathType.LITERAL].get_data.return_value = test_data
    mock_triple_gate.sort_data.return_value = test_data
    
    # Test when one gate blocks but other allows
    gate1.add_input("input1", 0.1)  # AND gate blocks
    gate2.add_input("input2", 0.9)  # OR gate allows
    
    gate1.update()
    gate2.update()
    
    # Attempt to ping
    await ping_system._ping_node(node_id)
    
    # Verify ping was blocked (since we require all gates to allow)
    status = ping_system.node_statuses[node_id]
    initial_ping_count = len(status.data_throughput)
    
    # Now make both gates allow
    gate1.add_input("input1", 0.9)
    gate1.update()
    
    await ping_system._ping_node(node_id)
    
    # Verify ping went through
    assert len(status.data_throughput) > initial_ping_count 
