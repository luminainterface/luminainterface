"""
Visualization Connector for V7 Lumina

This module connects various V7 components to visualization systems,
facilitating the display of metrics, states, and interactive elements.
It acts as a bridge between data sources and visualization components.
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger("lumina_v7.ui.visualization_connector")

class VisualizationConnector:
    """
    Connects V7 components to visualization systems
    
    Key features:
    - Real-time data collection from various components
    - Data transformation for visualization-friendly formats
    - Event handling and propagation
    - Dashboard integration
    - Performance metrics visualization
    """
    
    def __init__(self, node_manager=None, system_integrator=None, config=None):
        """
        Initialize the visualization connector
        
        Args:
            node_manager: NodeConsciousnessManager instance (optional)
            system_integrator: SystemIntegrator instance (optional)
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = {
            "update_interval": 1.0,      # Seconds between updates
            "metrics_history": 100,      # Number of history points to keep
            "dream_visualization": True,  # Enable dream visualization
            "performance_metrics": True,  # Enable performance metrics
            "consciousness_metrics": True,  # Enable consciousness metrics
            "dashboard_integration": True  # Enable dashboard integration
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Component references
        self.node_manager = node_manager
        self.system_integrator = system_integrator
        self.dream_controller = None
        self.consciousness_node = None
        self.monday_node = None
        
        # Visualization data
        self.metrics = {}
        self.metrics_history = {}
        self.visualization_data = {}
        
        # Event handlers
        self.event_handlers = {}
        
        # State
        self.active = False
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Connect to components
        self._connect_components()
        
        logger.info("VisualizationConnector initialized")
    
    def _connect_components(self):
        """Connect to available system components"""
        # Connect to Dream Controller
        if self.system_integrator and hasattr(self.system_integrator, "get_component"):
            self.dream_controller = self.system_integrator.get_component("dream_controller")
            if self.dream_controller:
                logger.info("Connected to Dream Controller")
        
        # Connect to Consciousness Node
        if self.node_manager and hasattr(self.node_manager, "get_node"):
            self.consciousness_node = self.node_manager.get_node("language_consciousness")
            if self.consciousness_node:
                logger.info("Connected to Language Consciousness Node")
        
        # Connect to Monday Node
        if self.node_manager and hasattr(self.node_manager, "get_node"):
            self.monday_node = self.node_manager.get_node("monday")
            if self.monday_node:
                logger.info("Connected to Monday Node")
        
        # Connect to Dashboard if enabled
        if self.config["dashboard_integration"]:
            try:
                from src.monitoring.dashboard import register_visualization_source
                register_visualization_source("v7", self.get_dashboard_data)
                logger.info("Registered with monitoring dashboard")
            except ImportError:
                logger.warning("Dashboard module not available")
    
    def start(self):
        """Start the visualization connector"""
        if self.active:
            logger.warning("VisualizationConnector already active")
            return
        
        # Start update thread
        self.stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="visualization-connector"
        )
        self.update_thread.start()
        
        self.active = True
        logger.info("VisualizationConnector started")
    
    def stop(self):
        """Stop the visualization connector"""
        if not self.active:
            return
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        self.active = False
        logger.info("VisualizationConnector stopped")
    
    def _update_loop(self):
        """Main update loop for collecting and processing data"""
        while not self.stop_event.is_set():
            try:
                # Update visualization data
                self._update_dream_visualization()
                self._update_consciousness_visualization()
                self._update_performance_metrics()
                
                # Sleep until next update
                time.sleep(self.config["update_interval"])
                
            except Exception as e:
                logger.error(f"Error in visualization update: {e}")
                time.sleep(1.0)  # Sleep on error
    
    def _update_dream_visualization(self):
        """Update dream visualization data"""
        if not self.config["dream_visualization"] or not self.dream_controller:
            return
        
        try:
            # Get dream state
            dream_state = self.dream_controller.get_dream_state()
            
            # Get current dream if active
            current_dream = None
            if dream_state["active"] and self.dream_controller.current_dream:
                current_dream = self.dream_controller.current_dream
            
            # Create visualization data
            viz_data = {
                "active": dream_state["active"],
                "phase": dream_state.get("phase"),
                "intensity": dream_state.get("intensity", 0.0),
                "start_time": dream_state.get("start_time"),
                "end_time": dream_state.get("planned_end_time"),
                "metrics": dream_state.get("metrics", {}),
                "phase_index": dream_state.get("phase_index", 0),
                "dream_id": dream_state.get("current_dream_id")
            }
            
            # Add dream details if available
            if current_dream:
                viz_data["dream_details"] = {
                    "id": current_dream.get("id"),
                    "insight_count": len(current_dream.get("insights", [])),
                    "connection_count": len(current_dream.get("connections", [])),
                    "phases": current_dream.get("phases", [])
                }
            
            # Store visualization data
            self.visualization_data["dream"] = viz_data
            
            # Add to metrics
            if current_dream:
                self._update_metric("dream.insights", len(current_dream.get("insights", [])))
                self._update_metric("dream.connections", len(current_dream.get("connections", [])))
                
                # Calculate dream progress percentage
                if dream_state["active"] and dream_state.get("start_time") and dream_state.get("planned_end_time"):
                    try:
                        start = datetime.fromisoformat(dream_state["start_time"])
                        end = datetime.fromisoformat(dream_state["planned_end_time"])
                        now = datetime.now()
                        
                        total_duration = (end - start).total_seconds()
                        elapsed = (now - start).total_seconds()
                        
                        if total_duration > 0:
                            progress = min(100, max(0, (elapsed / total_duration) * 100))
                            self._update_metric("dream.progress", progress)
                    except (ValueError, TypeError):
                        pass
        
        except Exception as e:
            logger.error(f"Error updating dream visualization: {e}")
    
    def _update_consciousness_visualization(self):
        """Update consciousness visualization data"""
        if not self.config["consciousness_metrics"] or not self.consciousness_node:
            return
        
        try:
            # Get awareness metrics if available
            awareness_metrics = None
            if hasattr(self.consciousness_node, "calculate_awareness_metrics"):
                awareness_metrics = self.consciousness_node.calculate_awareness_metrics()
            
            # Get visualization data if available
            viz_data = None
            if hasattr(self.consciousness_node, "get_visualization_data"):
                viz_data = self.consciousness_node.get_visualization_data()
            
            if awareness_metrics:
                # Extract metrics
                metrics_dict = awareness_metrics.to_dict() if hasattr(awareness_metrics, "to_dict") else {}
                
                # Update individual metrics
                for key, value in metrics_dict.items():
                    self._update_metric(f"consciousness.{key}", value)
                
                # Store in visualization data
                self.visualization_data["consciousness_metrics"] = metrics_dict
            
            if viz_data:
                # Store visualization graph data
                self.visualization_data["consciousness_graph"] = viz_data
        
        except Exception as e:
            logger.error(f"Error updating consciousness visualization: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if not self.config["performance_metrics"]:
            return
        
        try:
            # Collect system metrics
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Update metrics
            self._update_metric("system.cpu", cpu_percent)
            self._update_metric("system.memory", memory.percent)
            self._update_metric("system.memory_available", memory.available / (1024 * 1024))  # MB
            
            # Thread count
            thread_count = threading.active_count()
            self._update_metric("system.threads", thread_count)
            
            # Component-specific metrics
            if self.dream_controller:
                if hasattr(self.dream_controller, "get_stats"):
                    dream_stats = self.dream_controller.get_stats()
                    self._update_metric("dream.total_dreams", dream_stats.get("total_dreams", 0))
                    self._update_metric("dream.total_insights", dream_stats.get("total_insights", 0))
                    self._update_metric("dream.total_connections", dream_stats.get("total_connections", 0))
            
            # Store in visualization data
            self.visualization_data["performance"] = {
                "cpu": cpu_percent,
                "memory": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "thread_count": thread_count
            }
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _update_metric(self, name: str, value: Any):
        """
        Update a metric value and its history
        
        Args:
            name: Metric name
            value: Current value
        """
        # Update current value
        self.metrics[name] = value
        
        # Add to history
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        # Add data point with timestamp
        self.metrics_history[name].append({
            "timestamp": datetime.now().isoformat(),
            "value": value
        })
        
        # Trim history if needed
        if len(self.metrics_history[name]) > self.config["metrics_history"]:
            self.metrics_history[name] = self.metrics_history[name][-self.config["metrics_history"]:]
    
    def get_metric(self, name: str) -> Any:
        """
        Get current value of a metric
        
        Args:
            name: Metric name
            
        Returns:
            Current metric value or None if not found
        """
        return self.metrics.get(name)
    
    def get_metric_history(self, name: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get historical values for a metric
        
        Args:
            name: Metric name
            limit: Maximum number of history points to return
            
        Returns:
            List of data points with timestamps
        """
        history = self.metrics_history.get(name, [])
        
        if limit is not None and limit > 0:
            return history[-limit:]
        
        return history
    
    def get_visualization_data(self, component: str = None) -> Dict[str, Any]:
        """
        Get visualization data for a component
        
        Args:
            component: Component name (dream, consciousness, performance)
            
        Returns:
            Visualization data dictionary
        """
        if component:
            return self.visualization_data.get(component, {})
        
        return self.visualization_data
    
    def update_dream_visualization(self, dream_state: Dict[str, Any]):
        """
        Update dream visualization with new state
        
        Args:
            dream_state: Current dream state
        """
        # Store the raw dream state
        self.visualization_data["raw_dream_state"] = dream_state
        
        # Trigger immediate update
        self._update_dream_visualization()
        
        # Notify event handlers
        self._notify_event_handlers("dream_update", dream_state)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """
        Register an event handler
        
        Args:
            event_type: Type of event to handle
            handler: Callback function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    def _notify_event_handlers(self, event_type: str, data: Any):
        """
        Notify registered event handlers
        
        Args:
            event_type: Type of event
            data: Event data
        """
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for dashboard integration
        
        Returns:
            Dict with visualization data for dashboard
        """
        dashboard_data = {
            "dream_mode": self.visualization_data.get("dream", {}),
            "consciousness": self.visualization_data.get("consciousness_metrics", {}),
            "performance": self.visualization_data.get("performance", {}),
            "metrics": {k: v for k, v in self.metrics.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add language module data if available
        if "language_module" in self.visualization_data:
            dashboard_data["language_module"] = self.visualization_data.get("language_module", {})
        
        return dashboard_data 