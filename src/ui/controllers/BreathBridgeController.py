from PySide6.QtCore import QObject, Slot, Signal, QTimer
import time
import math

class BreathBridgeController(QObject):
    """
    Controller that manages the breath-neural network integration between
    the GlyphInterfacePanel and NetworkVisualizationPanelPySide6.
    
    This controller:
    1. Establishes connections between the two panels
    2. Facilitates breath data exchange
    3. Manages synchronization of breath patterns and network activation
    4. Provides logging and diagnostic information about the breath bridge
    """
    
    # Signals
    connection_status_changed = Signal(bool, str)  # connected, message
    bridge_stats_updated = Signal(dict)  # statistics about the bridge
    
    def __init__(self, glyph_panel=None, network_panel=None, parent=None):
        super().__init__(parent)
        
        # Store references to panels
        self.glyph_panel = glyph_panel
        self.network_panel = network_panel
        
        # Bridge status
        self.is_connected = False
        self.connection_timestamp = None
        self.breath_cycles = 0
        self.network_updates = 0
        
        # Connection metrics
        self.metrics = {
            "breath_data_sent": 0,
            "network_data_sent": 0,
            "last_breath_intensity": 0.0,
            "last_breath_pattern": "none",
            "last_network_activation": 0.0,
            "connection_duration": 0,
            "last_sync_time": 0
        }
        
        # Initialize metrics update timer
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.setInterval(1000)  # Update once per second
        
        # Connect panels if provided
        if glyph_panel and network_panel:
            self.connect_panels(glyph_panel, network_panel)
    
    def connect_panels(self, glyph_panel, network_panel):
        """Connect the glyph panel to the network panel"""
        self.glyph_panel = glyph_panel
        self.network_panel = network_panel
        
        # Connect signals between panels
        self.glyph_panel.breath_data_updated.connect(self.on_breath_data_updated)
        self.network_panel.breath_data_updated.connect(self.on_network_data_updated)
        
        # Connect the toggle network connection signals
        if hasattr(self.glyph_panel, 'connect_button'):
            self.glyph_panel.connect_button.clicked.connect(self.toggle_bridge_connection)
        
        # Set connected state
        self.is_connected = True
        self.connection_timestamp = time.time()
        self.metrics_timer.start()
        
        # Emit connection status
        self.connection_status_changed.emit(True, "Breath Bridge initialized and connected")
    
    def disconnect_panels(self):
        """Disconnect the panels and stop the breath bridge"""
        if not self.is_connected:
            return
            
        # Disconnect signals
        if self.glyph_panel and self.network_panel:
            self.glyph_panel.breath_data_updated.disconnect(self.on_breath_data_updated)
            self.network_panel.breath_data_updated.disconnect(self.on_network_data_updated)
            
            # Disconnect the toggle buttons
            if hasattr(self.glyph_panel, 'connect_button'):
                self.glyph_panel.connect_button.clicked.disconnect(self.toggle_bridge_connection)
        
        # Stop metrics timer
        self.metrics_timer.stop()
        
        # Set disconnected state
        self.is_connected = False
        self.connection_timestamp = None
        
        # Emit connection status
        self.connection_status_changed.emit(False, "Breath Bridge disconnected")
    
    @Slot(dict)
    def on_breath_data_updated(self, breath_data):
        """
        Handle breath data updates from the glyph panel and forward to network panel
        """
        if not self.is_connected or not self.network_panel:
            return
            
        # Only forward if the connection is active
        if breath_data.get("connected_to_network", False):
            # Forward breath data to network panel
            if hasattr(self.network_panel, 'receive_breath_data'):
                self.network_panel.receive_breath_data(breath_data)
                
                # Update metrics
                self.metrics["breath_data_sent"] += 1
                self.metrics["last_breath_intensity"] = breath_data.get("intensity", 0.0)
                self.metrics["last_breath_pattern"] = breath_data.get("pattern", "none")
                self.metrics["last_sync_time"] = time.time()
                
                # Check if this is a new breath cycle (when phase crosses 0)
                if breath_data.get("phase", 0) < 0.05 and breath_data.get("phase", 0) > 0:
                    self.breath_cycles += 1
    
    @Slot(dict)
    def on_network_data_updated(self, network_data):
        """
        Handle network data updates from the network panel and forward to glyph panel
        """
        if not self.is_connected or not self.glyph_panel:
            return
            
        # Forward network data to glyph panel
        if hasattr(self.glyph_panel, 'process_network_data'):
            self.glyph_panel.process_network_data(network_data)
            
            # Update metrics
            self.metrics["network_data_sent"] += 1
            
            # Calculate average activation if available
            if "available_nodes" in network_data:
                nodes = network_data["available_nodes"]
                if nodes:
                    total_activation = sum(node.get("activation", 0) for node in nodes.values())
                    avg_activation = total_activation / len(nodes)
                    self.metrics["last_network_activation"] = avg_activation
            
            self.network_updates += 1
    
    def toggle_bridge_connection(self, checked=None):
        """Toggle the breath bridge connection state"""
        if checked is None:
            checked = not self.is_connected
            
        if checked and not self.is_connected:
            # Reconnect panels
            if self.glyph_panel and self.network_panel:
                self.connect_panels(self.glyph_panel, self.network_panel)
        elif not checked and self.is_connected:
            # Disconnect panels
            self.disconnect_panels()
        
        # Ensure glyph panel button reflects the state
        if self.glyph_panel and hasattr(self.glyph_panel, 'connect_button'):
            self.glyph_panel.connect_button.setChecked(self.is_connected)
            
        # Ensure network panel is aware of the connection state
        if self.network_panel and hasattr(self.network_panel, 'update_connection_status'):
            self.network_panel.update_connection_status(self.is_connected)
            
        return self.is_connected
    
    def update_metrics(self):
        """Update and emit bridge metrics"""
        if not self.is_connected or not self.connection_timestamp:
            return
            
        # Calculate connection duration
        duration = time.time() - self.connection_timestamp
        self.metrics["connection_duration"] = int(duration)
        
        # Calculate integration metrics like data flow rates
        if duration > 0:
            self.metrics["breath_rate"] = self.breath_cycles / (duration / 60.0)  # breaths per minute
            self.metrics["network_update_rate"] = self.network_updates / duration  # updates per second
        
        # Emit updated metrics
        self.bridge_stats_updated.emit(self.metrics)
    
    def get_status_report(self):
        """Get a status report on the breath bridge"""
        if not self.is_connected:
            return {
                "status": "disconnected",
                "message": "Breath Bridge is currently disconnected"
            }
            
        # Create a status report
        report = {
            "status": "connected",
            "connection_duration": self.format_duration(self.metrics["connection_duration"]),
            "breath_cycles": self.breath_cycles,
            "breath_pattern": self.metrics["last_breath_pattern"],
            "breath_intensity": f"{self.metrics['last_breath_intensity'] * 100:.1f}%",
            "network_activation": f"{self.metrics['last_network_activation'] * 100:.1f}%",
            "data_exchange": f"{self.metrics['breath_data_sent']} breath updates, {self.metrics['network_data_sent']} network updates"
        }
        
        return report
    
    @staticmethod
    def format_duration(seconds):
        """Format duration in seconds to a readable string"""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m" 