#!/usr/bin/env python3
"""
Tests for Triple Flip Switch Gate

This module contains tests for the triple gate switching mechanism.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from ..switches.triple_gate import (
    TripleGate,
    GateConfig,
    GateState,
    FlowDirection,
    PathState
)
from ..core import PathType
from ..paths.literal_path import LiteralPath, LiteralProcessorConfig
from ..paths.semantic_path import SemanticPath, SemanticProcessorConfig
from ..paths.hybrid_path import HybridPath, HybridProcessorConfig
from ...ml.core import MLConfig

@pytest.fixture
def ml_config():
    """Create ML config"""
    return MLConfig()

@pytest.fixture
def gate_config():
    """Create gate config"""
    return GateConfig(
        switching_threshold=0.75,
        min_confidence=0.6,
        max_active_paths=2,
        auto_switch=True,
        feedback_window=10
    )

@pytest.fixture
def paths(ml_config):
    """Create processing paths"""
    literal_config = LiteralProcessorConfig()
    semantic_config = SemanticProcessorConfig()
    hybrid_config = HybridProcessorConfig()
    
    return {
        'literal': LiteralPath(ml_config, literal_config),
        'semantic': SemanticPath(ml_config, semantic_config),
        'hybrid': HybridPath(
            ml_config,
            literal_config,
            semantic_config,
            hybrid_config
        )
    }

@pytest.fixture
def triple_gate(gate_config, paths):
    """Create triple gate"""
    return TripleGate(
        gate_config,
        paths['literal'],
        paths['semantic'],
        paths['hybrid']
    )

@pytest.mark.asyncio
async def test_initialization(triple_gate):
    """Test gate initialization"""
    # Check initial states
    assert triple_gate.states[PathType.HYBRID].state == GateState.OPEN
    assert triple_gate.states[PathType.LITERAL].state == GateState.CLOSED
    assert triple_gate.states[PathType.SEMANTIC].state == GateState.CLOSED
    
    # Check active paths
    assert PathType.HYBRID in triple_gate.active_paths
    assert len(triple_gate.active_paths) == 1

@pytest.mark.asyncio
async def test_manual_switching(triple_gate):
    """Test manual path switching"""
    # Switch literal path to open
    triple_gate.switch_path(
        PathType.LITERAL,
        GateState.OPEN,
        FlowDirection.FORWARD
    )
    
    assert triple_gate.states[PathType.LITERAL].state == GateState.OPEN
    assert triple_gate.states[PathType.LITERAL].flow == FlowDirection.FORWARD
    assert PathType.LITERAL in triple_gate.active_paths
    
    # Switch semantic path to partial
    triple_gate.switch_path(
        PathType.SEMANTIC,
        GateState.PARTIAL,
        FlowDirection.BIDIRECTIONAL
    )
    
    assert triple_gate.states[PathType.SEMANTIC].state == GateState.PARTIAL
    assert triple_gate.states[PathType.SEMANTIC].flow == FlowDirection.BIDIRECTIONAL
    
    # Verify switch history
    assert len(triple_gate.switch_history) == 2
    assert triple_gate.switch_history[-1]['path_type'] == PathType.SEMANTIC

@pytest.mark.asyncio
async def test_auto_switching(triple_gate):
    """Test automatic path switching"""
    # Process data to trigger auto-switching
    test_data = "test input"
    result = await triple_gate.process(test_data)
    
    assert 'matches' in result
    assert 'active_paths' in result
    assert 'path_states' in result
    
    # Process more data to accumulate metrics
    for _ in range(5):
        await triple_gate.process(test_data)
        
    # Verify auto-switching behavior
    state = triple_gate.get_state()
    assert len(state['switch_history']) > 0

@pytest.mark.asyncio
async def test_path_metrics(triple_gate):
    """Test path metrics tracking"""
    test_data = "test input"
    
    # Process data multiple times
    for _ in range(3):
        await triple_gate.process(test_data)
        
    # Check metrics for each path
    for path_type, state in triple_gate.states.items():
        assert len(state.processing_history) > 0
        assert 'confidence' in state.processing_history[0]
        assert 'timestamp' in state.processing_history[0]

@pytest.mark.asyncio
async def test_optimization(triple_gate):
    """Test gate optimization"""
    test_data = "test input"
    
    # Process data to build history
    for _ in range(triple_gate.config.feedback_window + 1):
        await triple_gate.process(test_data)
        
    # Run optimization
    await triple_gate.optimize()
    
    # Verify optimization effects
    assert triple_gate.config.switching_threshold >= 0.5
    assert triple_gate.config.switching_threshold <= 0.95

@pytest.mark.asyncio
async def test_result_combination(triple_gate):
    """Test result combination from multiple paths"""
    # Enable multiple paths
    triple_gate.switch_path(PathType.LITERAL, GateState.OPEN)
    triple_gate.switch_path(PathType.SEMANTIC, GateState.OPEN)
    
    # Process data
    result = await triple_gate.process("test input")
    
    # Verify combined results
    assert 'matches' in result
    assert 'confidence' in result
    assert 'active_paths' in result
    assert len(result['active_paths']) > 1

@pytest.mark.asyncio
async def test_error_handling(triple_gate):
    """Test error handling"""
    # Close all paths
    for path_type in PathType:
        triple_gate.switch_path(path_type, GateState.CLOSED)
        
    # Process data with no active paths
    result = await triple_gate.process("test input")
    assert 'error' in result

@pytest.mark.asyncio
async def test_state_persistence(triple_gate):
    """Test state persistence"""
    # Set initial states
    triple_gate.switch_path(PathType.LITERAL, GateState.OPEN)
    initial_state = triple_gate.get_state()
    
    # Reset gate
    triple_gate.reset()
    
    # Verify reset state
    reset_state = triple_gate.get_state()
    assert reset_state != initial_state
    assert len(reset_state['active_paths']) == 1
    assert PathType.HYBRID in reset_state['active_paths']

@pytest.mark.asyncio
async def test_performance_tracking(triple_gate):
    """Test performance tracking"""
    test_data = "test input"
    
    # Process data multiple times
    for _ in range(5):
        await triple_gate.process(test_data)
        
    # Check performance metrics
    for path_type, state in triple_gate.states.items():
        assert hasattr(state, 'success_rate')
        assert hasattr(state, 'confidence')
        assert len(state.processing_history) > 0

@pytest.mark.asyncio
async def test_flow_control(triple_gate):
    """Test flow direction control"""
    # Set different flow directions
    triple_gate.switch_path(
        PathType.LITERAL,
        GateState.OPEN,
        FlowDirection.FORWARD
    )
    triple_gate.switch_path(
        PathType.SEMANTIC,
        GateState.OPEN,
        FlowDirection.BACKWARD
    )
    triple_gate.switch_path(
        PathType.HYBRID,
        GateState.OPEN,
        FlowDirection.BIDIRECTIONAL
    )
    
    # Verify flow directions
    state = triple_gate.get_state()
    assert state['states']['literal']['flow'] == FlowDirection.FORWARD.value
    assert state['states']['semantic']['flow'] == FlowDirection.BACKWARD.value
    assert state['states']['hybrid']['flow'] == FlowDirection.BIDIRECTIONAL.value 
 