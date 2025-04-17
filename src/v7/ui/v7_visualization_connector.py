#!/usr/bin/env python
"""
V7 Visualization Connector

This module provides a bridge between the backend V6-V7 Connector
and the frontend visualization components in the V7 UI system.
It handles the communication, data transformation, and event
routing necessary for visualization of V7 components.
"""

import os
import logging
import threading
import time
import json
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

# Set up logging
logger = logging.getLogger("v7.visualization_connector")

class V7VisualizationConnector:
    """
    Connects the backend V6-V7 connector to visualization components.
    
    This connector:
    1. Transforms backend data into visualization-friendly formats
    2. Routes events to appropriate visualization components
    3. Manages state synchronization between backend and frontend
    4. Provides a unified interface for UI components to access backend data
    """
    
    def __init__(self, v6v7_connector=None, config=None):
        """
        Initialize the visualization connector.
        
        Args:
            v6v7_connector: The backend connector that bridges V6 and V7
            config: Configuration dictionary with visualization settings
        """
        self._v6v7_connector = v6v7_connector
        self._visualizers = {}
        self._event_handlers = {}
        self._last_update = 0
        self._update_interval = 1.0  # seconds between updates
        
        # Default configuration
        self._config = {
            "enable_breath_visualization": True,
            "enable_contradiction_visualization": True,
            "enable_monday_visualization": True,
            "enable_node_visualization": True,
            "enable_memory_visualization": True,
            "visualization_update_interval": 1.0,
            "memory_update_interval": 5.0,
            "max_visualized_memories": 100
        }
        
        # Override defaults with provided config
        if config:
            self._config.update(config)
            
        # Set update interval from config
        self._update_interval = self._config.get("visualization_update_interval", 1.0)
        
        # Initialize visualizers dictionary
        self._visualizers = {
            "breath": None,
            "contradiction": None,
            "monday": None,
            "node_consciousness": None,
            "memory": None
        }
        
        # Initialize event handlers
        self._event_handlers = {
            "breath": set(),
            "contradiction": set(),
            "monday": set(),
            "node_status": set(),
            "memory": set(),
            "system": set()
        }
        
        logger.info("V7 Visualization Connector initialized")
        
        # Schedule first update
        self._setup_update_timer()
        
        # Bridge to breath contradiction system
        self.breath_bridge = None
        self._try_find_breath_bridge()
        
        # Connect to the v6v7_connector if provided
        if self._v6v7_connector:
            self._connect_to_connector()
    
    def _try_find_breath_bridge(self):
        """Attempt to find and connect to the breath contradiction bridge"""
        if self._v6v7_connector:
            try:
                # First check if it's in the connector components
                self.breath_bridge = self._v6v7_connector.get_component("breath_contradiction_bridge")
                
                if not self.breath_bridge:
                    # Try to import it directly
                    from src.v7.breath_contradiction_bridge import BreathContradictionBridge
                    self.breath_bridge = BreathContradictionBridge()
                    
                if self.breath_bridge:
                    logger.info("Connected to Breath Contradiction Bridge")
                    # Register for events
                    if hasattr(self.breath_bridge, "register_event_handler"):
                        self.breath_bridge.register_event_handler(self._handle_breath_event)
            except Exception as e:
                logger.warning(f"Failed to connect to Breath Contradiction Bridge: {e}")
    
    def register_visualizer(self, name: str, visualizer: Any) -> bool:
        """
        Register a visualization component
        
        Args:
            name: Unique identifier for the visualizer
            visualizer: The visualization component to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        if name in self._visualizers:
            logger.warning(f"Visualizer '{name}' already registered")
            return False
        
        self._visualizers[name] = visualizer
        logger.debug(f"Registered visualizer: {name}")
        return True
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Register a handler for specific event types
        
        Args:
            event_type: Type of event to handle
            handler: Callback function to handle the event
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = set()
            
        if handler not in self._event_handlers[event_type]:
            self._event_handlers[event_type].add(handler)
            logger.debug(f"Registered handler for event type: {event_type}")
            return True
        
        return False
    
    def _notify_event_handlers(self, event: Dict[str, Any]):
        """Notify all registered event handlers for this event type"""
        event_type = event.get("type", "unknown")
        
        # Notify specific handlers
        for handler in self._event_handlers.get(event_type, set()):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in {event_type} event handler: {e}")
        
        # Notify general handlers
        for handler in self._event_handlers.get("all", set()):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in general event handler: {e}")
    
    def _handle_breath_event(self, event: Dict[str, Any]):
        """Handle events from the breath contradiction bridge"""
        if not self._config["enable_breath_visualization"]:
            return
        
        # Transform the event for visualization purposes
        viz_event = {
            "source": "breath_bridge",
            "original_event": event,
            "timestamp": time.time(),
            "visualization_data": {}
        }
        
        # Add visualization-specific data based on event type
        if event.get("type") == "breath_pattern_changed":
            viz_event["visualization_data"] = {
                "pattern": event.get("pattern", "unknown"),
                "confidence": event.get("confidence", 0.0),
                "emotional_state": event.get("emotional_state", "neutral"),
                "color": self._get_color_for_pattern(event.get("pattern", "unknown"))
            }
        
        elif event.get("type") == "contradiction_processed":
            viz_event["visualization_data"] = {
                "contradiction_type": event.get("contradiction_type", "unknown"),
                "suggested_pattern": event.get("suggested_pattern", "unknown"),
                "pulse_effect": True,
                "color": self._get_color_for_contradiction(
                    event.get("contradiction_type", "unknown")
                )
            }
        
        # Notify event handlers
        self._notify_event_handlers(viz_event)
    
    def _get_color_for_pattern(self, pattern: str) -> str:
        """Get a color representation for a breath pattern"""
        color_map = {
            "relaxed": "#3498db",  # Blue
            "focused": "#2ecc71",  # Green
            "creative": "#9b59b6", # Purple
            "stressed": "#e74c3c", # Red
            "meditative": "#1abc9c" # Teal
        }
        return color_map.get(pattern, "#95a5a6")  # Default to gray
    
    def _get_color_for_contradiction(self, contradiction_type: str) -> str:
        """Get a color representation for a contradiction type"""
        color_map = {
            "logical": "#3498db",    # Blue
            "temporal": "#f39c12",   # Orange
            "spatial": "#2ecc71",    # Green
            "causal": "#e74c3c",     # Red
            "linguistic": "#9b59b6"  # Purple
        }
        return color_map.get(contradiction_type, "#95a5a6")  # Default to gray
    
    def start(self) -> bool:
        """
        Start the visualization connector
        
        Returns:
            bool: True if start successful, False otherwise
        """
        if self._config["enable_memory_visualization"]:
            return False
        
        self._config["enable_memory_visualization"] = True
        
        # Start update thread
        self._setup_update_timer()
        
        logger.info("V7 Visualization Connector started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the visualization connector
        
        Returns:
            bool: True if stop successful, False otherwise
        """
        if not self._config["enable_memory_visualization"]:
            return False
        
        self._config["enable_memory_visualization"] = False
        logger.info("V7 Visualization Connector stopped")
        return True
    
    def _update_loop(self):
        """Background thread for updating visualizations"""
        while self._config["enable_memory_visualization"]:
            try:
                # Sleep for update interval
                time.sleep(self._update_interval)
                
                # Update visualizations
                self._update_visualizations()
                
                # Update timestamp
                self._last_update = time.time()
            except Exception as e:
                logger.error(f"Error in visualization update loop: {e}")
    
    def _update_visualizations(self):
        """Retrieve the latest status from the backend and update visualizations."""
        try:
            current_time = time.time()
            
            # Only update at specified interval
            if current_time - self._last_update < self._update_interval:
                return
                
            self._last_update = current_time
            
            # Skip if no connector
            if self._v6v7_connector is None:
                return
                
            # Get backend status
            status = self._get_backend_status()
            
            # Process memory data if memory visualization is enabled
            if (self._config.get("enable_memory_visualization", True) and
                hasattr(self._v6v7_connector, "memory_node")):
                
                # Memory data is handled by the memory visualizer plugin
                # But we can still process memory events here if needed
                for handler in self._event_handlers.get("memory", []):
                    try:
                        handler({
                            "memory_count": status.get("memory_count", 0),
                            "type_distribution": status.get("memory_types", {}),
                            "event_time": current_time
                        })
                    except Exception as e:
                        logger.error(f"Error in memory event handler: {e}")
            
            # Update other visualizations
            for name, visualizer in self._visualizers.items():
                try:
                    if hasattr(visualizer, "update_from_status"):
                        visualizer.update_from_status(status)
                except Exception as e:
                    logger.error(f"Error updating visualizer {name}: {e}")
            
            # Notify about update
            self._notify_event_handlers({
                "type": "visualization_updated",
                "timestamp": time.time(),
                "components_updated": list(self._visualizers.keys())
            })
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
    
    def get_color_palette(self) -> Dict[str, str]:
        """
        Get a consistent color palette for visualization
        
        Returns:
            Dict[str, str]: Mapping of semantic names to hex color codes
        """
        return {
            # Base colors
            "primary": "#3498db",       # Blue
            "secondary": "#2ecc71",     # Green
            "tertiary": "#9b59b6",      # Purple
            "quaternary": "#e74c3c",    # Red
            "quinary": "#f39c12",       # Orange
            
            # Semantic colors
            "relaxed": "#3498db",       # Blue
            "focused": "#2ecc71",       # Green
            "creative": "#9b59b6",      # Purple
            "stressed": "#e74c3c",      # Red
            "meditative": "#1abc9c",    # Teal
            
            # Contradiction types
            "logical": "#3498db",       # Blue
            "temporal": "#f39c12",      # Orange
            "spatial": "#2ecc71",       # Green
            "causal": "#e74c3c",        # Red
            "linguistic": "#9b59b6",    # Purple
            
            # UI elements
            "background": "#2c3e50",    # Dark blue
            "text": "#ecf0f1",          # White
            "accent": "#f1c40f"         # Yellow
        }
    
    def create_visualization_data(self, component: str) -> Dict[str, Any]:
        """
        Create visualization data for a specific component
        
        Args:
            component: Name of the component to create data for
            
        Returns:
            Dict[str, Any]: Visualization data structure
        """
        if component == "breath":
            return self._create_breath_visualization_data()
        elif component == "contradiction":
            return self._create_contradiction_visualization_data()
        elif component == "monday":
            return self._create_monday_visualization_data()
        elif component == "node_consciousness":
            return self._create_node_consciousness_visualization_data()
        else:
            logger.warning(f"Unknown visualization component: {component}")
            return {}
    
    def _create_breath_visualization_data(self) -> Dict[str, Any]:
        """Create data structure for breath visualization"""
        if not self.breath_bridge:
            return {"available": False}
            
        # Get current breath pattern if available
        current_pattern = "unknown"
        confidence = 0.0
        
        if hasattr(self.breath_bridge, "breath_detector"):
            detector = self.breath_bridge.breath_detector
            if hasattr(detector, "current_pattern"):
                current_pattern = detector.current_pattern
                confidence = detector.current_confidence
        
        # Build visualization data
        return {
            "available": True,
            "current_pattern": current_pattern,
            "confidence": confidence,
            "color": self._get_color_for_pattern(current_pattern),
            "patterns": [
                {"name": "relaxed", "color": "#3498db"},
                {"name": "focused", "color": "#2ecc71"},
                {"name": "creative", "color": "#9b59b6"},
                {"name": "stressed", "color": "#e74c3c"},
                {"name": "meditative", "color": "#1abc9c"}
            ]
        }
    
    def _create_contradiction_visualization_data(self) -> Dict[str, Any]:
        """Create data structure for contradiction visualization"""
        # Find contradiction processor
        contradiction_processor = None
        if self._v6v7_connector:
            contradiction_processor = self._v6v7_connector.get_component("contradiction_processor")
        
        if not contradiction_processor:
            return {"available": False}
        
        # Get recent contradictions if available
        recent_contradictions = []
        if hasattr(contradiction_processor, "get_recent_contradictions"):
            recent = contradiction_processor.get_recent_contradictions(limit=5)
            for c in recent:
                recent_contradictions.append({
                    "id": c.id,
                    "type": c.type,
                    "description": c.description,
                    "resolved": c.resolved,
                    "color": self._get_color_for_contradiction(c.type)
                })
        
        # Build visualization data
        return {
            "available": True,
            "recent_contradictions": recent_contradictions,
            "contradiction_types": [
                {"name": "logical", "color": "#3498db"},
                {"name": "temporal", "color": "#f39c12"},
                {"name": "spatial", "color": "#2ecc71"},
                {"name": "causal", "color": "#e74c3c"},
                {"name": "linguistic", "color": "#9b59b6"}
            ]
        }
    
    def _create_monday_visualization_data(self) -> Dict[str, Any]:
        """Create data structure for Monday visualization"""
        # Find Monday component
        monday = None
        if self._v6v7_connector:
            monday = self._v6v7_connector.get_component("monday")
        
        if not monday:
            return {"available": False}
        
        # Get consciousness level
        consciousness_level = 0
        if hasattr(monday, "consciousness_level"):
            consciousness_level = monday.consciousness_level
        
        # Build visualization data
        return {
            "available": True,
            "consciousness_level": consciousness_level,
            "active": hasattr(monday, "active") and monday.active,
            "color": self._get_color_based_on_level(consciousness_level)
        }
    
    def _create_node_consciousness_visualization_data(self) -> Dict[str, Any]:
        """Create data structure for node consciousness visualization"""
        # Find language consciousness
        language_consciousness = None
        if self._v6v7_connector:
            language_consciousness = self._v6v7_connector.get_component("language_consciousness")
        
        if not language_consciousness:
            return {"available": False}
        
        # Build visualization data
        return {
            "available": True,
            "active": hasattr(language_consciousness, "active") and language_consciousness.active,
            "nodes": []  # Would populate from actual node consciousness data
        }
    
    def _get_color_based_on_level(self, level: float) -> str:
        """Get color based on consciousness level"""
        if level < 0.2:
            return "#3498db"  # Blue
        elif level < 0.4:
            return "#2ecc71"  # Green
        elif level < 0.6:
            return "#f39c12"  # Orange
        elif level < 0.8:
            return "#9b59b6"  # Purple
        else:
            return "#e74c3c"  # Red
    
    def _connect_to_connector(self):
        """Connect to the V6V7Connector events."""
        if not self._v6v7_connector:
            return
            
        # Register event handlers
        if hasattr(self._v6v7_connector, 'on_breath'):
            self._v6v7_connector.on_breath(self._handle_breath_event)
            
        if hasattr(self._v6v7_connector, 'on_contradiction'):
            self._v6v7_connector.on_contradiction(self._handle_contradiction_event)
        
        # Register for memory events if available
        if hasattr(self._v6v7_connector, 'on_memory_event'):
            self._v6v7_connector.on_memory_event(self._handle_memory_event)
            
        # Set up timer for periodic updates
        self._setup_update_timer()
    
    def _handle_memory_event(self, event_data):
        """
        Handle memory events from the V6V7Connector.
        
        Args:
            event_data: Memory event data
        """
        if not self._config['enable_memory_visualization']:
            return
            
        try:
            # Transform the memory event into visualization data
            viz_data = self._create_memory_visualization_data(event_data)
            
            # Notify all registered handlers
            for handler in self._event_handlers['memory']:
                try:
                    handler(viz_data)
                except Exception as e:
                    logger.error(f"Error in memory visualization handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling memory event: {e}")
    
    def _create_memory_visualization_data(self, event_data):
        """
        Create visualization data for memory events.
        
        Args:
            event_data: Memory event data
            
        Returns:
            Dictionary with visualization data
        """
        memory = event_data.get('memory', {})
        operation = event_data.get('operation', 'unknown')
        
        # Create the base visualization data
        viz_data = {
            'type': 'memory',
            'operation': operation,
            'timestamp': time.time(),
            'memory_id': memory.get('id', 'unknown'),
            'memory_type': memory.get('memory_type', 'generic'),
            'strength': memory.get('strength', 0.5),
            'tags': memory.get('tags', []),
            'age': time.time() - memory.get('created_at', time.time()),
            'freshness': memory.get('last_accessed', 0) - memory.get('created_at', 0)
        }
        
        # Add colors based on memory type and operation
        colors = self._get_memory_colors(memory.get('memory_type', 'generic'), operation)
        viz_data.update(colors)
        
        # Add additional visualization metadata
        if operation == 'store':
            viz_data['animation'] = 'grow'
            viz_data['duration'] = 1000  # ms
        elif operation == 'retrieve':
            viz_data['animation'] = 'pulse'
            viz_data['duration'] = 500  # ms
        elif operation == 'decay':
            viz_data['animation'] = 'fade'
            viz_data['duration'] = 1500  # ms
        
        return viz_data
    
    def _get_memory_colors(self, memory_type, operation):
        """
        Get color scheme for memory visualization based on type and operation.
        
        Args:
            memory_type: Type of memory
            operation: Memory operation type
            
        Returns:
            Dictionary with color information
        """
        # Base colors for different memory types
        type_colors = {
            'experience': '#8BC34A',    # Light green
            'conversation': '#2196F3',  # Blue
            'emotional': '#FF9800',     # Orange
            'contradiction': '#F44336', # Red
            'insight': '#9C27B0',       # Purple
            'generic': '#607D8B'        # Blue gray
        }
        
        # Default color if type not recognized
        base_color = type_colors.get(memory_type, type_colors['generic'])
        
        # Modify colors based on operation
        if operation == 'store':
            return {
                'primary_color': base_color,
                'secondary_color': self._lighten_color(base_color, 0.3),
                'text_color': '#FFFFFF'
            }
        elif operation == 'retrieve':
            return {
                'primary_color': self._lighten_color(base_color, 0.1),
                'secondary_color': self._darken_color(base_color, 0.1),
                'text_color': '#FFFFFF'
            }
        elif operation == 'decay':
            return {
                'primary_color': self._darken_color(base_color, 0.3),
                'secondary_color': self._darken_color(base_color, 0.5),
                'text_color': '#DDDDDD'
            }
        else:
            return {
                'primary_color': base_color,
                'secondary_color': self._darken_color(base_color, 0.2),
                'text_color': '#FFFFFF'
            }
    
    def _lighten_color(self, hex_color, factor=0.2):
        """
        Lighten a hex color by the given factor.
        
        Args:
            hex_color: Hex color string (e.g., '#RRGGBB')
            factor: How much to lighten (0-1)
            
        Returns:
            Lightened hex color
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _darken_color(self, hex_color, factor=0.2):
        """
        Darken a hex color by the given factor.
        
        Args:
            hex_color: Hex color string (e.g., '#RRGGBB')
            factor: How much to darken (0-1)
            
        Returns:
            Darkened hex color
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def register_memory_visualizer(self, handler):
        """
        Register a handler for memory visualizations.
        
        Args:
            handler: Function to call with memory visualization data
            
        Returns:
            True if registered successfully, False otherwise
        """
        if callable(handler):
            self._event_handlers['memory'].add(handler)
            return True
        return False
    
    def _setup_update_timer(self):
        """Set up a timer for periodic updates"""
        if not self._v6v7_connector:
            return
        
        # Set up a timer for periodic updates
        self._update_loop()
    
    def _handle_contradiction_event(self, event: Dict[str, Any]):
        """Handle contradiction events from the V6V7Connector"""
        if not self._config['enable_contradiction_visualization']:
            return
        
        # Transform the event for visualization purposes
        viz_event = {
            "source": "contradiction_processor",
            "original_event": event,
            "timestamp": time.time(),
            "visualization_data": {}
        }
        
        # Add visualization-specific data based on event type
        if event.get("type") == "contradiction_processed":
            viz_event["visualization_data"] = {
                "contradiction_type": event.get("contradiction_type", "unknown"),
                "suggested_pattern": event.get("suggested_pattern", "unknown"),
                "pulse_effect": True,
                "color": self._get_color_for_contradiction(event.get("contradiction_type", "unknown"))
            }
        
        # Notify event handlers
        self._notify_event_handlers(viz_event)
    
    def _setup_ui_visualizers(self):
        """Set up and configure UI visualizers if connected to a UI."""
        # Try to get parent widget for visualizers
        parent = None
        
        # If we have a memory_node in the connector, set up memory visualization
        if (self._config.get("enable_memory_visualization", True) and 
            hasattr(self._v6v7_connector, "memory_node")):
            try:
                from src.v7.ui.memory_visualizer import get_memory_visualizer
                
                memory_config = {
                    "update_interval": self._config.get("memory_update_interval", 5.0),
                    "memory_limit": self._config.get("max_visualized_memories", 100),
                    "color_by_type": True,
                    "show_labels": True
                }
                
                memory_viz = get_memory_visualizer(
                    v7_connector=self._v6v7_connector,
                    memory_node=self._v6v7_connector.memory_node,
                    config=memory_config
                )
                
                if memory_viz:
                    self._visualizers["memory"] = memory_viz
                    logger.info("Memory visualization enabled")
            except ImportError as e:
                logger.warning(f"Memory visualization not available: {e}")
                
        # Set up other visualizers (existing code)
    
    def get_memory_visualization_widget(self):
        """
        Get the memory visualization widget.
        
        Returns:
            The memory visualization widget or None if not available
        """
        if not self._config.get("enable_memory_visualization", True):
            return None
            
        if self._visualizers["memory"] is None:
            try:
                from src.v7.ui.memory_visualizer import get_memory_visualizer
                
                memory_config = {
                    "update_interval": self._config.get("memory_update_interval", 5.0),
                    "memory_limit": self._config.get("max_visualized_memories", 100),
                    "color_by_type": True,
                    "show_labels": True
                }
                
                memory_viz = get_memory_visualizer(
                    v7_connector=self._v6v7_connector,
                    memory_node=getattr(self._v6v7_connector, "memory_node", None),
                    config=memory_config
                )
                
                if memory_viz:
                    self._visualizers["memory"] = memory_viz
                    logger.info("Memory visualization widget created")
            except ImportError as e:
                logger.warning(f"Memory visualization not available: {e}")
                
        return self._visualizers["memory"]
        
    def highlight_memories(self, memory_ids):
        """
        Highlight specific memories in the visualization.
        
        Args:
            memory_ids: List of memory IDs to highlight
        """
        if self._visualizers["memory"] is not None:
            self._visualizers["memory"].highlight_memories(memory_ids)
            
    def select_memory(self, memory_id):
        """
        Select a specific memory to show details.
        
        Args:
            memory_id: ID of memory to select or None to deselect
        """
        if self._visualizers["memory"] is not None:
            self._visualizers["memory"].select_memory(memory_id)
    
    def _get_backend_status(self):
        """Retrieve the latest status from the backend"""
        if self._v6v7_connector and hasattr(self._v6v7_connector, 'get_status'):
            return self._v6v7_connector.get_status()
        else:
            return {} 