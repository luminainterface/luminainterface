#!/usr/bin/env python3
"""
Test suite for LUMINA v7.5 Version Bridge
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from ..version_bridge import VersionBridge
from ..lumina_core import LUMINACore
from ..signal_system import SignalBus

class MockSystem:
    """Mock system instance for testing"""
    def __init__(self, version: str):
        self.version = version
        self.received_messages = []

    def handle_message(self, message: Dict[str, Any]):
        self.received_messages.append(message)

@pytest.fixture
async def setup_bridge():
    """Setup test environment with VersionBridge"""
    signal_bus = SignalBus()
    core = LUMINACore(signal_bus)
    bridge = VersionBridge(core)
    yield bridge
    await bridge.cleanup()

@pytest.mark.asyncio
async def test_version_connection(setup_bridge):
    """Test version connection and disconnection"""
    bridge = setup_bridge
    mock_system = MockSystem("v7.0")
    
    # Test connection
    await bridge._handle_version_connect({
        "version": "v7.0",
        "system": mock_system
    })
    
    assert "v7.0" in bridge.nodes
    assert bridge.nodes["v7.0"].system_instance == mock_system
    assert bridge.nodes["v7.0"].is_running
    
    # Test disconnection
    await bridge._handle_version_disconnect({
        "version": "v7.0"
    })
    
    assert "v7.0" not in bridge.nodes

@pytest.mark.asyncio
async def test_version_compatibility(setup_bridge):
    """Test version compatibility checks"""
    bridge = setup_bridge
    
    # Test valid compatibility
    assert bridge._check_compatibility("v7.5", "v7.0")
    assert bridge._check_compatibility("v7.0", "v6.0")
    
    # Test invalid compatibility
    assert not bridge._check_compatibility("v7.5", "v1.0")
    assert not bridge._check_compatibility("invalid", "v7.0")

@pytest.mark.asyncio
async def test_message_transformation(setup_bridge):
    """Test message transformation between versions"""
    bridge = setup_bridge
    
    # Setup source and target systems
    source_system = MockSystem("v7.5")
    target_system = MockSystem("v7.0")
    
    await bridge._handle_version_connect({
        "version": "v7.5",
        "system": source_system
    })
    
    await bridge._handle_version_connect({
        "version": "v7.0",
        "system": target_system
    })
    
    # Test message routing
    test_message = {
        "type": "test",
        "content": "Hello",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"source": "test"}
    }
    
    await bridge._handle_version_message({
        "source": "v7.5",
        "target": "v7.0",
        "message": test_message
    })
    
    # Allow time for async processing
    await asyncio.sleep(0.1)
    
    # Verify transformed message
    assert len(target_system.received_messages) > 0
    transformed = target_system.received_messages[0]
    assert transformed["type"] == test_message["type"]
    assert transformed["content"] == test_message["content"]
    assert "metadata" in transformed

@pytest.mark.asyncio
async def test_error_handling(setup_bridge):
    """Test error handling scenarios"""
    bridge = setup_bridge
    
    # Test invalid version format
    await bridge._handle_version_connect({
        "version": "invalid",
        "system": MockSystem("invalid")
    })
    
    assert "invalid" not in bridge.nodes
    
    # Test missing required fields
    await bridge._handle_version_message({
        "source": "v7.5",
        "target": "v7.0"  # Missing message field
    })
    
    # Test incompatible versions
    await bridge._handle_version_message({
        "source": "v7.5",
        "target": "v1.0",
        "message": {"type": "test"}
    })

@pytest.mark.asyncio
async def test_monitoring_integration(setup_bridge):
    """Test monitoring system integration"""
    bridge = setup_bridge
    
    # Connect test version
    mock_system = MockSystem("v7.0")
    await bridge._handle_version_connect({
        "version": "v7.0",
        "system": mock_system
    })
    
    # Verify monitoring started
    assert "v7.0" in bridge.monitor.version_metrics
    
    # Test message monitoring
    test_message = {
        "type": "test",
        "content": "Hello",
        "timestamp": datetime.now().isoformat()
    }
    
    await bridge._handle_version_message({
        "source": "v7.0",
        "target": "v7.5",
        "message": test_message
    })
    
    # Verify metrics
    metrics = bridge.monitor.get_version_health("v7.0")
    assert metrics["status"] == "connected"
    assert metrics["message_stats"]["total"] > 0

@pytest.mark.asyncio
async def test_cleanup(setup_bridge):
    """Test cleanup procedures"""
    bridge = setup_bridge
    
    # Connect multiple versions
    versions = ["v7.0", "v6.0", "v5.0"]
    for ver in versions:
        await bridge._handle_version_connect({
            "version": ver,
            "system": MockSystem(ver)
        })
    
    # Verify connections
    for ver in versions:
        assert ver in bridge.nodes
    
    # Perform cleanup
    await bridge.cleanup()
    
    # Verify all versions disconnected
    assert len(bridge.nodes) == 0
    for ver in versions:
        assert ver not in bridge.nodes

@pytest.mark.asyncio
async def test_status_requests(setup_bridge):
    """Test status request handling"""
    bridge = setup_bridge
    
    # Connect test version
    mock_system = MockSystem("v7.0")
    await bridge._handle_version_connect({
        "version": "v7.0",
        "system": mock_system
    })
    
    # Request status
    await bridge._handle_status_request({
        "version": "v7.0"
    })
    
    # Verify health metrics
    health = bridge.monitor.get_version_health("v7.0")
    assert "status" in health
    assert "message_stats" in health
    assert "errors" in health 