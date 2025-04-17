#!/usr/bin/env python3
"""
Backend Entry Point

This module serves as the entry point for the backend system,
integrating the ping system with the database for signal storage
and AutoWiki components.
"""

import asyncio
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from .logicgate.ping_system import PingSystem, PingConfig
from .logicgate.switches.triple_gate import TripleGate, PathType, GateState
from .database import (
    init_db, store_signal, store_metrics,
    store_pattern, get_recent_signals
)
from .wiki.article_manager import ArticleManager
from .wiki.suggestion_engine import SuggestionEngine
from .wiki.content_generator import ContentGenerator
from .wiki.auto_learning import AutoLearningEngine
from .wiki.auto_wiki import AutoWiki
from .ml.model import NeuralProcessor
from .monitoring.metrics import (
    MetricsCollector, HealthMonitor, PerformanceMonitor,
    SystemMetrics, PerformanceMetrics
)
from .config import (
    get_full_config, LOGGING_CONFIG, PING_CONFIG,
    AUTOWIKI_CONFIG, ML_CONFIG, MONITORING_CONFIG
)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class BackendSystem:
    """Main backend system that integrates all components."""
    
    def __init__(self):
        """Initialize the backend system."""
        # Initialize core components
        self.triple_gate = TripleGate()
        self.article_manager = ArticleManager()
        self.suggestion_engine = SuggestionEngine()
        self.content_generator = ContentGenerator()
        self.learning_engine = AutoLearningEngine()
        
        # Initialize AutoWiki
        self.autowiki = AutoWiki(
            article_manager=self.article_manager,
            suggestion_engine=self.suggestion_engine,
            content_generator=self.content_generator,
            learning_engine=self.learning_engine
        )
        
        # Initialize ML model with proper configuration
        self.neural_processor = NeuralProcessor(ML_CONFIG)
        
        # Initialize monitoring with proper configuration
        self.metrics_collector = MetricsCollector(MONITORING_CONFIG)
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        
        # Configure ping system with proper configuration
        self.ping_config = PingConfig(**PING_CONFIG)
        self.ping_system = PingSystem(self.ping_config, self.triple_gate)
        
        # Initialize connection status
        self.connection_status = {
            'ml_model': False,
            'monitoring': False,
            'autowiki': False,
            'database': False
        }
        
    async def start(self) -> None:
        """Start the backend system."""
        try:
            # Initialize database
            init_db()
            self.connection_status['database'] = True
            
            # Initialize AutoWiki components
            await self._initialize_autowiki()
            self.connection_status['autowiki'] = True
            
            # Load ML model
            self.neural_processor.load_model()
            self.connection_status['ml_model'] = True
            
            # Start ping system
            await self.ping_system.start()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._signal_storage_loop())
            asyncio.create_task(self._ml_processing_loop())
            self.connection_status['monitoring'] = True
            
            # Start connection health check
            asyncio.create_task(self._connection_health_check())
            
            logger.info("Backend system started")
            
        except Exception as e:
            logger.error(f"Error starting backend system: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the backend system."""
        try:
            # Stop ping system
            await self.ping_system.stop()
            
            # Save ML model
            self.neural_processor.save_model()
            
            # Shutdown AutoWiki
            await self.autowiki.shutdown()
            
            # Reset connection status
            self.connection_status = {k: False for k in self.connection_status}
            
            logger.info("Backend system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping backend system: {e}")
            raise
            
    async def _connection_health_check(self) -> None:
        """Periodically check connection health."""
        while self.ping_system.active:
            try:
                # Check ML model connection
                if not self.connection_status['ml_model']:
                    logger.warning("ML model connection lost, attempting to reconnect...")
                    self.neural_processor.load_model()
                    self.connection_status['ml_model'] = True
                    
                # Check monitoring connection
                if not self.connection_status['monitoring']:
                    logger.warning("Monitoring connection lost, attempting to reconnect...")
                    self.metrics_collector = MetricsCollector(MONITORING_CONFIG)
                    self.health_monitor = HealthMonitor(self.metrics_collector)
                    self.performance_monitor = PerformanceMonitor(self.metrics_collector)
                    self.connection_status['monitoring'] = True
                    
                # Check AutoWiki connection
                if not self.connection_status['autowiki']:
                    logger.warning("AutoWiki connection lost, attempting to reconnect...")
                    await self._initialize_autowiki()
                    self.connection_status['autowiki'] = True
                    
                # Check database connection
                if not self.connection_status['database']:
                    logger.warning("Database connection lost, attempting to reconnect...")
                    init_db()
                    self.connection_status['database'] = True
                    
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in connection health check: {e}")
                await asyncio.sleep(1.0)
                
    async def _initialize_autowiki(self) -> None:
        """Initialize AutoWiki system components."""
        try:
            # Initialize core components
            await self.suggestion_engine.initialize()
            await self.content_generator.initialize()
            await self.learning_engine.initialize()
            
            # Connect to neural seed
            await self._connect_autowiki_neural_seed()
            
            # Start background services
            await self._start_autowiki_services()
            
            logger.info("AutoWiki system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutoWiki: {e}")
            self.connection_status['autowiki'] = False
            raise
            
    async def _connect_autowiki_neural_seed(self) -> None:
        """Connect AutoWiki to Neural Seed."""
        try:
            # Register message handlers
            self.ping_system.register_handler(
                'content_request',
                self.autowiki.handle_content_request
            )
            self.ping_system.register_handler(
                'learning_update',
                self.autowiki.handle_learning_update
            )
            
            # Setup data bridges
            self.autowiki.set_neural_seed(self.ping_system)
            self.ping_system.add_extension('autowiki', self.autowiki)
            
            logger.info("AutoWiki connected to Neural Seed")
            
        except Exception as e:
            logger.error(f"Failed to connect AutoWiki to Neural Seed: {e}")
            self.connection_status['autowiki'] = False
            raise
            
    async def _start_autowiki_services(self) -> None:
        """Start AutoWiki background services."""
        try:
            # Start content generation service
            await self.content_generator.start_service()
            
            # Start suggestion service
            await self.suggestion_engine.start_service()
            
            # Start learning service
            await self.learning_engine.start_service()
            
            logger.info("AutoWiki services started")
            
        except Exception as e:
            logger.error(f"Failed to start AutoWiki services: {e}")
            self.connection_status['autowiki'] = False
            raise
            
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.ping_system.active:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                # Update gate metrics
                gate_states = {
                    gate_id: gate.output > 0.8
                    for gate_id, gate in self.ping_system.logic_gates.items()
                }
                self.metrics_collector.update_gate_metrics(
                    gate_count=len(self.ping_system.logic_gates),
                    gate_states=gate_states,
                    connection_count=sum(len(gate.connections) for gate in self.ping_system.logic_gates.values())
                )
                
                # Update AutoWiki metrics
                self.metrics_collector.update_autowiki_metrics(
                    article_count=len(await self.article_manager.get_all_articles()),
                    suggestion_count=len(await self.suggestion_engine.get_pending_suggestions()),
                    learning_progress=await self.learning_engine.get_progress()
                )
                
                # Check system health
                health_status = self.health_monitor.check_health()
                if not health_status['overall']['healthy']:
                    logger.warning(f"System health check failed: {health_status}")
                    
                await asyncio.sleep(self.ping_config.ping_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.connection_status['monitoring'] = False
                await asyncio.sleep(1.0)
                
    async def _signal_storage_loop(self) -> None:
        """Loop for storing signals in the database."""
        while self.ping_system.active:
            try:
                start_time = datetime.now()
                
                # Get recent signals
                signals = get_recent_signals(limit=100)
                
                # Process each gate
                for gate_id, gate in self.ping_system.logic_gates.items():
                    # Store signal
                    store_signal(
                        gate_id=gate_id,
                        gate_type=gate.config.gate_type.name,
                        output=gate.output,
                        inputs=gate.inputs,
                        connections=gate.connections,
                        visual_effects=gate.get_state()["visual_effects"],
                        path_type=self._get_path_type(gate).name,
                        gate_state=self._get_gate_state(gate).name
                    )
                    
                    # Store metrics
                    if gate_id in self.ping_system.node_statuses:
                        status = self.ping_system.node_statuses[gate_id]
                        store_metrics(
                            gate_id=gate_id,
                            health_score=status.health_score,
                            latency=status.metrics["latency"][-1] if status.metrics["latency"] else 0.0,
                            load=status.metrics["load"][-1] if status.metrics["load"] else 0.0,
                            memory=status.metrics["memory"][-1] if status.metrics["memory"] else 0.0,
                            success_rate=status.metrics["success_rate"][-1] if status.metrics["success_rate"] else 0.0
                        )
                        
                    # Store patterns if any
                    patterns = status.get_temporal_patterns() if gate_id in self.ping_system.node_statuses else {}
                    for pattern_type, pattern_data in patterns.items():
                        store_pattern(
                            gate_id=gate_id,
                            pattern_type=pattern_type,
                            pattern_data=pattern_data
                        )
                        
                # Record performance
                processing_time = (datetime.now() - start_time).total_seconds()
                self.metrics_collector.record_signal_processed(processing_time)
                
                await asyncio.sleep(0.1)  # Store signals every 100ms
                
            except Exception as e:
                logger.error(f"Error in signal storage loop: {e}")
                self.connection_status['database'] = False
                self.metrics_collector.record_error()
                await asyncio.sleep(1.0)
                
    async def _ml_processing_loop(self) -> None:
        """Machine learning processing loop."""
        while self.ping_system.active:
            try:
                start_time = datetime.now()
                
                # Get current gate states
                gate_states = [
                    gate.get_state()
                    for gate in self.ping_system.logic_gates.values()
                ]
                
                if gate_states:
                    # Recognize patterns
                    patterns = self.neural_processor.recognize_patterns(gate_states)
                    
                    # Predict future states
                    predicted_states = self.neural_processor.predict_states(gate_states)
                    
                    # Generate code if needed
                    if any(state['state_change_prob'] > 0.8 for state in predicted_states):
                        prompt = self._generate_code_prompt(gate_states, patterns, predicted_states)
                        generated_code = self.neural_processor.generate_code(prompt)
                        if generated_code:
                            logger.info(f"Generated code: {generated_code}")
                            
                    # Record performance
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.performance_monitor.record_performance(
                        PerformanceMetrics(
                            signal_processing_time=processing_time,
                            gate_update_time=0.0,  # Updated in signal loop
                            pattern_recognition_time=processing_time / 2,
                            state_prediction_time=processing_time / 2,
                            timestamp=datetime.now()
                        )
                    )
                    
                await asyncio.sleep(0.2)  # Process ML every 200ms
                
            except Exception as e:
                logger.error(f"Error in ML processing loop: {e}")
                self.connection_status['ml_model'] = False
                self.metrics_collector.record_error()
                await asyncio.sleep(1.0)
                
    def _generate_code_prompt(
        self,
        gate_states: List[Dict[str, Any]],
        patterns: List[Dict[str, float]],
        predicted_states: List[Dict[str, float]]
    ) -> str:
        """Generate a prompt for code generation."""
        prompt_parts = [
            "Generate Python code based on the following system state:",
            "\nGate States:",
            *[f"- Gate {i}: {state}" for i, state in enumerate(gate_states)],
            "\nRecognized Patterns:",
            *[f"- Pattern {i}: {pattern}" for i, pattern in enumerate(patterns)],
            "\nPredicted States:",
            *[f"- State {i}: {state}" for i, state in enumerate(predicted_states)],
            "\nRequirements:",
            "- Use asyncio for asynchronous operations",
            "- Include error handling",
            "- Follow PEP 8 style guidelines",
            "- Optimize for performance",
            "\nGenerate code:"
        ]
        return "\n".join(prompt_parts)
                
    def _get_path_type(self, gate: Any) -> PathType:
        """Get path type for a gate."""
        if gate.config.gate_type in [LogicGateType.AND, LogicGateType.NAND]:
            return PathType.LITERAL
        elif gate.config.gate_type in [LogicGateType.OR, LogicGateType.NOR]:
            return PathType.SEMANTIC
        else:
            return PathType.HYBRID
            
    def _get_gate_state(self, gate: Any) -> GateState:
        """Get gate state based on output."""
        return GateState.OPEN if gate.output > 0.8 else GateState.CLOSED

async def main():
    """Main entry point for the backend system."""
    try:
        # Initialize configuration
        config = PingConfig()
        
        # Initialize ping system
        ping_system = PingSystem(config)
        
        # Initialize triple gate with configuration
        triple_gate = TripleGate(
            gate_id="triple_gate_1",
            config=config,
            ping_system=ping_system
        )
        
        # Start the ping system
        await ping_system.start()
        
        # Start the triple gate
        await triple_gate.start()
        
        # Start the signal storage loop
        await store_signals_loop(ping_system)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 