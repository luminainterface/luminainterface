#!/usr/bin/env python3
"""
Main Entry Point for Backend Integration System

This module provides the main entry point for initializing and running
the backend integration system, including bridges and services.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add src directory to Python path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.integration.backend_system import BackendSystem
from src.integration.bridges import BridgeFactory, create_bridge_config
from src.integration.services import create_service_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend_integration")

class BackendIntegrationSystem:
    """
    Main backend integration system.
    
    This class:
    1. Initializes the backend system
    2. Sets up bridges and services
    3. Manages system lifecycle
    4. Handles monitoring and metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the backend integration system"""
        self.config = config or {}
        self.backend = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the system"""
        logger.info("Initializing backend integration system")
        
        try:
            # Create backend system
            self.backend = BackendSystem(self.config)
            
            # Initialize bridges
            await self._initialize_bridges()
            
            # Initialize services
            await self._initialize_services()
            
            self.running = True
            logger.info("Backend integration system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize backend integration system: {e}")
            return False
            
    async def _initialize_bridges(self):
        """Initialize bridge components"""
        # Create V1-V2 bridge
        v1v2_config = create_bridge_config(
            'v1_to_v2',
            ['base_node', 'neural_processor']
        )
        await self._create_and_start_bridge('v1_to_v2', v1v2_config)
        
        # Create V2-V3 bridge
        v2v3_config = create_bridge_config(
            'v2_to_v3',
            ['base_node', 'neural_processor', 'language_processor'],
            data_transformation=True
        )
        await self._create_and_start_bridge('v2_to_v3', v2v3_config)
        
        # Create V3-V4 bridge
        v3v4_config = create_bridge_config(
            'v3_to_v4',
            [
                'base_node',
                'neural_processor',
                'language_processor',
                'hyperdimensional_thought'
            ],
            data_transformation=True
        )
        await self._create_and_start_bridge('v3_to_v4', v3v4_config)
        
    async def _create_and_start_bridge(self, bridge_type: str, config: Dict[str, Any]):
        """Create and start a bridge"""
        try:
            bridge = BridgeFactory.create_bridge(bridge_type, config)
            bridge.start()
            logger.info(f"Started bridge: {bridge_type}")
        except Exception as e:
            logger.error(f"Failed to start bridge {bridge_type}: {e}")
            
    async def _initialize_services(self):
        """Initialize background services"""
        # Create bridge manager service
        bridge_manager_config = create_service_config(
            'critical',
            100,  # 100ms interval
            dependencies=[]
        )
        await self._create_and_start_service('bridge_manager', bridge_manager_config)
        
        # Create version controller service
        version_controller_config = create_service_config(
            'high',
            200,  # 200ms interval
            dependencies=['bridge_manager']
        )
        await self._create_and_start_service('version_controller', version_controller_config)
        
        # Create stability monitor service
        stability_monitor_config = create_service_config(
            'high',
            100,  # 100ms interval
            dependencies=['bridge_manager']
        )
        await self._create_and_start_service('stability_monitor', stability_monitor_config)
        
    async def _create_and_start_service(self, service_type: str, config: Dict[str, Any]):
        """Create and start a service"""
        try:
            service = self.backend.services.get(service_type)
            if service:
                service.start()
                logger.info(f"Started service: {service_type}")
        except Exception as e:
            logger.error(f"Failed to start service {service_type}: {e}")
            
    async def run(self):
        """Run the system"""
        if not self.running:
            logger.error("System not initialized")
            return
            
        logger.info("Starting backend integration system")
        
        try:
            # Start event processing
            event_task = asyncio.create_task(self.backend.process_events())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel event task
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
                
            # Stop services
            await self._stop_services()
            
            logger.info("Backend integration system stopped")
            
        except Exception as e:
            logger.error(f"Error running backend integration system: {e}")
            
    async def _stop_services(self):
        """Stop all services"""
        for service_type, service in self.backend.services.items():
            try:
                service.stop()
                logger.info(f"Stopped service: {service_type}")
            except Exception as e:
                logger.error(f"Failed to stop service {service_type}: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.backend:
            return {"status": "not_initialized"}
            
        return {
            "status": "running" if self.running else "stopped",
            "bridges": self.backend.get_all_bridge_status(),
            "metrics": self.backend.get_metrics()
        }
        
async def main():
    """Main entry point"""
    # Create and run the system
    system = BackendIntegrationSystem()
    
    if await system.initialize():
        await system.run()
        
def run():
    """Run the system"""
    asyncio.run(main())
    
if __name__ == "__main__":
    run() 