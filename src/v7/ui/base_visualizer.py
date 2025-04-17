#!/usr/bin/env python
"""
Base Visualizer Component for V7 Node Consciousness Visualization

This module defines the base class and interface that all V7 visualizer
components should inherit from to ensure consistent behavior and integration
with the V7 visualization connector.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class BaseVisualizer(ABC):
    """
    Abstract base class for all V7 visualization components.
    
    All visualizers in the V7 system should inherit from this class
    and implement its abstract methods to ensure proper integration
    with the visualization connector.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base visualizer.
        
        Args:
            config: Optional configuration dictionary with settings for this visualizer
        """
        self.config = config or {}
        self.active = False
        self.data = {}
        self.last_update_time = 0
        self.update_interval = self.config.get('update_interval', 100)  # ms
        
        # Default color palette that can be overridden by subclasses
        self.colors = {
            'background': '#1E1E1E',
            'foreground': '#FFFFFF',
            'accent': '#3498DB',
            'highlight': '#E74C3C',
            'neutral': '#95A5A6',
            'active': '#2ECC71',
            'inactive': '#7F8C8D',
            'warning': '#F39C12',
            'error': '#C0392B'
        }
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the visualizer with new data.
        
        Args:
            data: Dictionary containing the data to visualize
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """
        Render the current visualization state.
        
        This method should be called whenever the visualization needs to be
        redrawn, typically in response to a paint event or after an update.
        """
        pass
    
    @abstractmethod
    def create_widget(self) -> Any:
        """
        Create and return a widget that can be added to a Qt layout.
        
        Returns:
            A Qt widget that displays this visualization
        """
        pass
    
    def start(self) -> None:
        """
        Start the visualizer.
        
        This method is called when the visualizer should begin active visualization.
        """
        self.active = True
        logger.debug(f"Started {self.__class__.__name__}")
    
    def stop(self) -> None:
        """
        Stop the visualizer.
        
        This method is called when the visualizer should pause visualization.
        """
        self.active = False
        logger.debug(f"Stopped {self.__class__.__name__}")
    
    def resize(self, width: int, height: int) -> None:
        """
        Handle resize events.
        
        Args:
            width: New width of the visualization area
            height: New height of the visualization area
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            The current configuration dictionary
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration.
        
        Args:
            config: New configuration dictionary to merge with the existing one
        """
        self.config.update(config)
        
    def is_active(self) -> bool:
        """
        Check if the visualizer is currently active.
        
        Returns:
            True if the visualizer is active, False otherwise
        """
        return self.active
    
    def get_name(self) -> str:
        """
        Get the name of this visualizer.
        
        Returns:
            The name of the visualizer (defaults to class name)
        """
        return self.config.get('name', self.__class__.__name__)
    
    def get_description(self) -> str:
        """
        Get a description of this visualizer.
        
        Returns:
            A description of what this visualizer displays
        """
        return self.config.get('description', f"{self.__class__.__name__} Visualization")
    
    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Handle events from the visualization connector.
        
        Args:
            event_type: The type of event
            event_data: Data associated with the event
        """
        logger.debug(f"{self.__class__.__name__} received event: {event_type}")
        
        # Default implementation just updates the visualizer with event data
        self.update(event_data)
    
    def set_color_palette(self, colors: Dict[str, str]) -> None:
        """
        Set a custom color palette for this visualizer.
        
        Args:
            colors: Dictionary mapping color names to hex color values
        """
        self.colors.update(colors)
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the visualizer.
        
        Returns:
            A dictionary with the current status information
        """
        return {
            'active': self.active,
            'name': self.get_name(),
            'last_update': self.last_update_time
        } 