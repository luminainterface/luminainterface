"""
Fractal Bridge for Version 1
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
        self._register_message_handlers()
        
        logger.info("Initialized FractalBridge")
    
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
    
    def _handle_generate_fractal(self, message: Dict[str, Any]) -> None:
        """
        Handle fractal generation requests.
        
        Args:
            message: Message containing generation parameters
        """
        version = VersionType(message['version'])
        fractal_type = FractalType(message['fractal_type'])
        
        if version not in self.simulators:
            self.simulators[version] = FractalSimulator()
        
        simulator = self.simulators[version]
        iterations, rendered = simulator.generate_fractal(fractal_type)
        
        # Send results back to requesting version
        self.bridge.send_data(
            VersionType.V1,  # Current version
            version,
            {
                'type': 'fractal_result',
                'iterations': iterations.tolist(),
                'rendered': rendered.tolist()
            }
        )
    
    def _handle_update_config(self, message: Dict[str, Any]) -> None:
        """
        Handle fractal configuration updates.
        
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
        self.bridge.broadcast(
            VersionType.V1,
            {
                'type': 'fractal_config_updated',
                'version': version.value,
                'config': config
            }
        )
    
    def _handle_get_metrics(self, message: Dict[str, Any]) -> None:
        """
        Handle fractal metrics requests.
        
        Args:
            message: Message containing request parameters
        """
        version = VersionType(message['version'])
        
        if version not in self.simulators:
            self.simulators[version] = FractalSimulator()
        
        simulator = self.simulators[version]
        metrics = simulator.get_metrics()
        
        # Send metrics back to requesting version
        self.bridge.send_data(
            VersionType.V1,
            version,
            {
                'type': 'fractal_metrics',
                'metrics': metrics
            }
        )
    
    def generate_fractal(self, version: VersionType, fractal_type: FractalType) -> None:
        """
        Generate a fractal for a specific version.
        
        Args:
            version: Target version
            fractal_type: Type of fractal to generate
        """
        self.bridge.send_data(
            VersionType.V1,
            version,
            {
                'type': 'generate_fractal',
                'fractal_type': fractal_type.value
            }
        )
    
    def update_config(self, version: VersionType, config: Dict[str, Any]) -> None:
        """
        Update fractal configuration for a specific version.
        
        Args:
            version: Target version
            config: New configuration
        """
        self.bridge.send_data(
            VersionType.V1,
            version,
            {
                'type': 'update_fractal_config',
                'config': config
            }
        )
    
    def get_metrics(self, version: VersionType) -> None:
        """
        Get fractal metrics for a specific version.
        
        Args:
            version: Target version
        """
        self.bridge.send_data(
            VersionType.V1,
            version,
            {
                'type': 'get_fractal_metrics'
            }
        )
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get status of the fractal bridge.
        
        Returns:
            Bridge status dictionary
        """
        return {
            'connected_versions': [v.value for v in self.simulators.keys()],
            'bridge_status': self.bridge.get_bridge_status()
        }

# Export functionality for node integration
functionality = {
    'classes': {
        'FractalBridge': FractalBridge
    },
    'description': 'Fractal bridge system for version interoperability'
} 