"""
Backend System Tests

This module contains tests for the backend system components.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from src.integration.backend import BackendSystem
from src.integration.config import (
    LOGGING_CONFIG, PING_CONFIG, AUTOWIKI_CONFIG,
    ML_CONFIG, MONITORING_CONFIG
)
from src.integration.logicgate.ping_system import PingSystem
from src.integration.logicgate.switches.triple_gate import TripleGate
from src.integration.ml.model import NeuralProcessor
from src.integration.monitoring.metrics import MetricsCollector, HealthMonitor, PerformanceMonitor

@pytest.fixture
def backend():
    """Create a backend system instance."""
    return BackendSystem()

@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    with patch('src.integration.backend.TripleGate') as mock_triple_gate, \
         patch('src.integration.backend.ArticleManager') as mock_article_manager, \
         patch('src.integration.backend.SuggestionEngine') as mock_suggestion_engine, \
         patch('src.integration.backend.ContentGenerator') as mock_content_generator, \
         patch('src.integration.backend.AutoLearningEngine') as mock_learning_engine, \
         patch('src.integration.backend.AutoWiki') as mock_autowiki, \
         patch('src.integration.backend.NeuralProcessor') as mock_neural_processor, \
         patch('src.integration.backend.MetricsCollector') as mock_metrics_collector, \
         patch('src.integration.backend.HealthMonitor') as mock_health_monitor, \
         patch('src.integration.backend.PerformanceMonitor') as mock_performance_monitor, \
         patch('src.integration.backend.PingSystem') as mock_ping_system:
            
        # Setup mock return values
        mock_triple_gate.return_value = MagicMock()
        mock_article_manager.return_value = MagicMock()
        mock_suggestion_engine.return_value = MagicMock()
        mock_content_generator.return_value = MagicMock()
        mock_learning_engine.return_value = MagicMock()
        mock_autowiki.return_value = MagicMock()
        mock_neural_processor.return_value = MagicMock()
        mock_metrics_collector.return_value = MagicMock()
        mock_health_monitor.return_value = MagicMock()
        mock_performance_monitor.return_value = MagicMock()
        mock_ping_system.return_value = MagicMock()
        
        yield {
            'triple_gate': mock_triple_gate,
            'article_manager': mock_article_manager,
            'suggestion_engine': mock_suggestion_engine,
            'content_generator': mock_content_generator,
            'learning_engine': mock_learning_engine,
            'autowiki': mock_autowiki,
            'neural_processor': mock_neural_processor,
            'metrics_collector': mock_metrics_collector,
            'health_monitor': mock_health_monitor,
            'performance_monitor': mock_performance_monitor,
            'ping_system': mock_ping_system
        }

@pytest.mark.asyncio
async def test_backend_initialization(backend, mock_components):
    """Test backend system initialization."""
    # Verify components are initialized
    assert backend.triple_gate is not None
    assert backend.article_manager is not None
    assert backend.suggestion_engine is not None
    assert backend.content_generator is not None
    assert backend.learning_engine is not None
    assert backend.autowiki is not None
    assert backend.neural_processor is not None
    assert backend.metrics_collector is not None
    assert backend.health_monitor is not None
    assert backend.performance_monitor is not None
    assert backend.ping_system is not None
    
    # Verify connection status
    assert all(not status for status in backend.connection_status.values())

@pytest.mark.asyncio
async def test_backend_start(backend, mock_components):
    """Test backend system startup."""
    # Mock component methods
    backend.triple_gate.start = MagicMock(return_value=None)
    backend.article_manager.initialize = MagicMock(return_value=None)
    backend.suggestion_engine.initialize = MagicMock(return_value=None)
    backend.content_generator.initialize = MagicMock(return_value=None)
    backend.learning_engine.initialize = MagicMock(return_value=None)
    backend.autowiki.shutdown = MagicMock(return_value=None)
    backend.neural_processor.load_model = MagicMock(return_value=None)
    backend.ping_system.start = MagicMock(return_value=None)
    
    # Start the backend
    await backend.start()
    
    # Verify components are started
    assert backend.connection_status['database'] is True
    assert backend.connection_status['autowiki'] is True
    assert backend.connection_status['ml_model'] is True
    assert backend.connection_status['monitoring'] is True
    
    # Verify component methods are called
    backend.neural_processor.load_model.assert_called_once()
    backend.ping_system.start.assert_called_once()
    backend.article_manager.initialize.assert_called_once()
    backend.suggestion_engine.initialize.assert_called_once()
    backend.content_generator.initialize.assert_called_once()
    backend.learning_engine.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_backend_stop(backend, mock_components):
    """Test backend system shutdown."""
    # Mock component methods
    backend.ping_system.stop = MagicMock(return_value=None)
    backend.neural_processor.save_model = MagicMock(return_value=None)
    backend.autowiki.shutdown = MagicMock(return_value=None)
    
    # Start and then stop the backend
    await backend.start()
    await backend.stop()
    
    # Verify components are stopped
    assert all(not status for status in backend.connection_status.values())
    
    # Verify component methods are called
    backend.ping_system.stop.assert_called_once()
    backend.neural_processor.save_model.assert_called_once()
    backend.autowiki.shutdown.assert_called_once()

@pytest.mark.asyncio
async def test_connection_health_check(backend, mock_components):
    """Test connection health check functionality."""
    # Start the backend
    await backend.start()
    
    # Simulate connection loss
    backend.connection_status['ml_model'] = False
    backend.connection_status['monitoring'] = False
    backend.connection_status['autowiki'] = False
    backend.connection_status['database'] = False
    
    # Mock reconnection methods
    backend.neural_processor.load_model = MagicMock(return_value=None)
    backend.metrics_collector = MagicMock()
    backend.health_monitor = MagicMock()
    backend.performance_monitor = MagicMock()
    backend._initialize_autowiki = MagicMock(return_value=None)
    
    # Run health check
    await backend._connection_health_check()
    
    # Verify reconnection attempts
    backend.neural_processor.load_model.assert_called_once()
    backend._initialize_autowiki.assert_called_once()

@pytest.mark.asyncio
async def test_monitoring_loop(backend, mock_components):
    """Test monitoring loop functionality."""
    # Start the backend
    await backend.start()
    
    # Mock metrics collection
    backend.metrics_collector.collect_system_metrics = MagicMock(return_value={
        'cpu_usage': 50.0,
        'memory_usage': 60.0,
        'disk_usage': 70.0
    })
    
    # Mock gate states
    backend.ping_system.logic_gates = {
        'gate1': MagicMock(output=0.9),
        'gate2': MagicMock(output=0.5)
    }
    
    # Mock AutoWiki metrics
    backend.article_manager.get_all_articles = MagicMock(return_value=['article1', 'article2'])
    backend.suggestion_engine.get_pending_suggestions = MagicMock(return_value=['suggestion1'])
    backend.learning_engine.get_progress = MagicMock(return_value=0.8)
    
    # Run monitoring loop
    await backend._monitoring_loop()
    
    # Verify metrics collection
    backend.metrics_collector.collect_system_metrics.assert_called_once()
    backend.metrics_collector.update_gate_metrics.assert_called_once()
    backend.metrics_collector.update_autowiki_metrics.assert_called_once()

@pytest.mark.asyncio
async def test_ml_processing_loop(backend, mock_components):
    """Test ML processing loop functionality."""
    # Start the backend
    await backend.start()
    
    # Mock gate states
    backend.ping_system.logic_gates = {
        'gate1': MagicMock(
            get_state=MagicMock(return_value={
                'output': 0.9,
                'gate_state': 'OPEN',
                'connections': ['gate2'],
                'inputs': [0.8, 0.9]
            })
        )
    }
    
    # Mock ML processing
    backend.neural_processor.recognize_patterns = MagicMock(return_value=[
        {'temporal': 0.8, 'spatial': 0.7, 'causal': 0.6, 'structural': 0.5}
    ])
    backend.neural_processor.predict_states = MagicMock(return_value=[
        {'output_prob': 0.9, 'state_change_prob': 0.8, 'connection_change_prob': 0.7, 'stability_score': 0.6}
    ])
    backend.neural_processor.generate_code = MagicMock(return_value="print('Hello, World!')")
    
    # Run ML processing loop
    await backend._ml_processing_loop()
    
    # Verify ML processing
    backend.neural_processor.recognize_patterns.assert_called_once()
    backend.neural_processor.predict_states.assert_called_once()
    backend.neural_processor.generate_code.assert_called_once()

@pytest.mark.asyncio
async def test_signal_storage_loop(backend, mock_components):
    """Test signal storage loop functionality."""
    # Start the backend
    await backend.start()
    
    # Mock gate states
    backend.ping_system.logic_gates = {
        'gate1': MagicMock(
            config=MagicMock(gate_type=MagicMock(name='AND')),
            output=0.9,
            inputs=[0.8, 0.9],
            connections=['gate2'],
            get_state=MagicMock(return_value={
                'visual_effects': {'pulse': True, 'glow': 0.8}
            })
        )
    }
    
    # Mock node statuses
    backend.ping_system.node_statuses = {
        'gate1': MagicMock(
            health_score=0.9,
            metrics={
                'latency': [0.1],
                'load': [0.5],
                'memory': [0.6],
                'success_rate': [0.95]
            },
            get_temporal_patterns=MagicMock(return_value={
                'pattern1': {'type': 'temporal', 'score': 0.8}
            })
        )
    }
    
    # Run signal storage loop
    await backend._signal_storage_loop()
    
    # Verify metrics recording
    backend.metrics_collector.record_signal_processed.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(backend, mock_components):
    """Test error handling in backend system."""
    # Start the backend
    await backend.start()
    
    # Simulate errors in different components
    backend.neural_processor.load_model = MagicMock(side_effect=Exception("ML model error"))
    backend.ping_system.start = MagicMock(side_effect=Exception("Ping system error"))
    backend.autowiki.shutdown = MagicMock(side_effect=Exception("AutoWiki error"))
    
    # Verify error handling
    with pytest.raises(Exception):
        await backend.start()
    
    with pytest.raises(Exception):
        await backend.stop()
    
    # Verify connection status updates
    assert not backend.connection_status['ml_model']
    assert not backend.connection_status['autowiki'] 