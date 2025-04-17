"""
Fractal Bridge for Version 3
This module integrates the fractal simulator with the spiderweb bridge system,
enabling fractal generation and processing across different versions.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from .spiderweb_bridge import SpiderwebBridge, VersionType, VersionInfo
from .fractal_simulator import FractalSimulator, FractalType, FractalConfig
import numpy as np
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class FractalBridge:
    def __init__(self, bridge: SpiderwebBridge):
        """
        Initialize the fractal bridge.
        
        Args:
            bridge: Spiderweb bridge instance
        """
        self.bridge = bridge
        self.simulators: Dict[VersionType, FractalSimulator] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.metrics = {
            'generation_count': 0,
            'error_count': 0,
            'average_generation_time': 0.0,
            'total_pixels_processed': 0
        }
        self._register_message_handlers()
        
        logger.info("Initialized FractalBridge V3")
    
    def _register_message_handlers(self) -> None:
        """Register message handlers for fractal operations."""
        for version in VersionType:
            if version in self.bridge.versions:
                self.bridge.register_message_handler(
                    version,
                    'generate_fractal',
                    self._handle_generate_fractal
                )
                self.bridge.register_message_handler(
                    version,
                    'update_fractal_config',
                    self._handle_update_config
                )
                self.bridge.register_message_handler(
                    version,
                    'get_fractal_metrics',
                    self._handle_get_metrics
                )
                self.bridge.register_message_handler(
                    version,
                    'batch_generate_fractals',
                    self._handle_batch_generate
                )
    
    async def _handle_generate_fractal(self, message: Dict[str, Any]) -> None:
        """
        Handle fractal generation requests asynchronously.
        
        Args:
            message: Message containing generation parameters
        """
        version = VersionType(message['version'])
        fractal_type = FractalType(message['fractal_type'])
        
        if version not in self.simulators:
            self.simulators[version] = FractalSimulator()
        
        simulator = self.simulators[version]
        start_time = time.time()
        
        try:
            iterations, rendered = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                simulator.generate_fractal,
                fractal_type
            )
            
            # Update metrics
            generation_time = time.time() - start_time
            self.metrics['generation_count'] += 1
            self.metrics['average_generation_time'] = (
                (self.metrics['average_generation_time'] * (self.metrics['generation_count'] - 1) +
                 generation_time) / self.metrics['generation_count']
            )
            self.metrics['total_pixels_processed'] += iterations.size
            
            # Send results back to requesting version
            await self.bridge.send_data(
                VersionType.V3,  # Current version
                version,
                {
                    'type': 'fractal_result',
                    'iterations': iterations.tolist(),
                    'rendered': rendered.tolist(),
                    'generation_time': generation_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating fractal: {str(e)}")
            self.metrics['error_count'] += 1
            await self.bridge.send_data(
                VersionType.V3,
                version,
                {
                    'type': 'fractal_error',
                    'error': str(e)
                }
            )
    
    async def _handle_update_config(self, message: Dict[str, Any]) -> None:
        """
        Handle fractal configuration updates asynchronously.
        
        Args:
            message: Message containing new configuration
        """
        version = VersionType(message['version'])
        config = message['config']
        
        if version not in self.simulators:
            self.simulators[version] = FractalSimulator()
        
        simulator = self.simulators[version]
        simulator.update_config(**config)
        
        # Broadcast configuration update to compatible versions
        await self.bridge.broadcast(
            VersionType.V3,
            {
                'type': 'fractal_config_updated',
                'version': version.value,
                'config': config
            }
        )
    
    async def _handle_get_metrics(self, message: Dict[str, Any]) -> None:
        """
        Handle fractal metrics requests asynchronously.
        
        Args:
            message: Message containing request parameters
        """
        version = VersionType(message['version'])
        
        if version not in self.simulators:
            self.simulators[version] = FractalSimulator()
        
        simulator = self.simulators[version]
        metrics = simulator.get_metrics()
        
        # Combine simulator metrics with bridge metrics
        combined_metrics = {
            'simulator': metrics,
            'bridge': self.metrics
        }
        
        # Send metrics back to requesting version
        await self.bridge.send_data(
            VersionType.V3,
            version,
            {
                'type': 'fractal_metrics',
                'metrics': combined_metrics
            }
        )
    
    async def _handle_batch_generate(self, message: Dict[str, Any]) -> None:
        """
        Handle batch fractal generation requests asynchronously.
        
        Args:
            message: Message containing batch generation parameters
        """
        version = VersionType(message['version'])
        fractal_types = [FractalType(ft) for ft in message['fractal_types']]
        
        if version not in self.simulators:
            self.simulators[version] = FractalSimulator()
        
        simulator = self.simulators[version]
        results = []
        
        for fractal_type in fractal_types:
            try:
                iterations, rendered = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    simulator.generate_fractal,
                    fractal_type
                )
                results.append({
                    'type': fractal_type.value,
                    'iterations': iterations.tolist(),
                    'rendered': rendered.tolist()
                })
            except Exception as e:
                logger.error(f"Error in batch generation for {fractal_type}: {str(e)}")
                self.metrics['error_count'] += 1
        
        # Send batch results back to requesting version
        await self.bridge.send_data(
            VersionType.V3,
            version,
            {
                'type': 'batch_fractal_results',
                'results': results
            }
        )
    
    async def generate_fractal(self, version: VersionType, fractal_type: FractalType) -> None:
        """
        Generate a fractal for a specific version asynchronously.
        
        Args:
            version: Target version
            fractal_type: Type of fractal to generate
        """
        await self.bridge.send_data(
            VersionType.V3,
            version,
            {
                'type': 'generate_fractal',
                'fractal_type': fractal_type.value
            }
        )
    
    async def update_config(self, version: VersionType, config: Dict[str, Any]) -> None:
        """
        Update fractal configuration for a specific version asynchronously.
        
        Args:
            version: Target version
            config: New configuration
        """
        await self.bridge.send_data(
            VersionType.V3,
            version,
            {
                'type': 'update_fractal_config',
                'config': config
            }
        )
    
    async def get_metrics(self, version: VersionType) -> None:
        """
        Get fractal metrics for a specific version asynchronously.
        
        Args:
            version: Target version
        """
        await self.bridge.send_data(
            VersionType.V3,
            version,
            {
                'type': 'get_fractal_metrics'
            }
        )
    
    async def batch_generate_fractals(self, version: VersionType,
                                    fractal_types: List[FractalType]) -> None:
        """
        Generate multiple fractals for a specific version asynchronously.
        
        Args:
            version: Target version
            fractal_types: List of fractal types to generate
        """
        await self.bridge.send_data(
            VersionType.V3,
            version,
            {
                'type': 'batch_generate_fractals',
                'fractal_types': [ft.value for ft in fractal_types]
            }
        )
    
    async def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get status of the fractal bridge asynchronously.
        
        Returns:
            Bridge status dictionary
        """
        return {
            'connected_versions': [v.value for v in self.simulators.keys()],
            'bridge_status': await self.bridge.get_bridge_status(),
            'metrics': self.metrics
        }

# Export functionality for node integration
functionality = {
    'classes': {
        'FractalBridge': FractalBridge
    },
    'description': 'Fractal bridge system for version interoperability'
} 