"""
PySide6 adapter for NeuralSeed integration
"""

from PySide6.QtCore import QObject, Signal, Slot, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PySide6.QtCharts import QChart, QChartView, QLineSeries
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .neural_seed import NeuralSeed

logger = logging.getLogger(__name__)

class NeuralSeedAdapter(QObject):
    """PySide6 adapter for NeuralSeed integration"""
    
    # Enhanced signals
    seed_state_updated = Signal(dict)  # Emitted when seed state changes
    growth_stage_changed = Signal(str)  # Emitted when growth stage changes
    stability_changed = Signal(float)  # Emitted when stability changes
    consciousness_level_changed = Signal(float)  # Emitted when consciousness level changes
    component_activated = Signal(str)  # Emitted when a component is activated
    component_deactivated = Signal(str)  # Emitted when a component is deactivated
    error_occurred = Signal(str)  # Emitted on errors
    metrics_updated = Signal(dict)  # Emitted when metrics are updated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.seed = NeuralSeed()
        self.available = True
        
        # Create status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        # Create metrics update timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(5000)  # Update every 5 seconds
        
    def start_growth(self, initial_stability: float = 1.0):
        """Start the growth process"""
        try:
            self.seed.start_growth(initial_stability)
            self.available = True
            logger.info("NeuralSeed growth started")
        except Exception as e:
            self.error_occurred.emit(f"Failed to start growth: {str(e)}")
            self.available = False
            
    def stop_growth(self):
        """Stop the growth process"""
        try:
            self.seed.stop_growth()
            logger.info("NeuralSeed growth stopped")
        except Exception as e:
            self.error_occurred.emit(f"Failed to stop growth: {str(e)}")
            
    @Slot()
    def update_status(self):
        """Update seed status"""
        if not self.available:
            return
            
        try:
            state = self.seed.get_state()
            self.seed_state_updated.emit(state)
            
            # Emit specific signals for important changes
            if 'state' in state:
                current_stage = state['state'].get('stage')
                if current_stage:
                    self.growth_stage_changed.emit(current_stage)
                    
                current_stability = state['state'].get('stability')
                if current_stability is not None:
                    self.stability_changed.emit(current_stability)
                    
                consciousness_level = state['state'].get('consciousness_level')
                if consciousness_level is not None:
                    self.consciousness_level_changed.emit(consciousness_level)
                    
        except Exception as e:
            self.error_occurred.emit(f"Error updating status: {str(e)}")
            
    @Slot()
    def update_metrics(self):
        """Update and emit metrics"""
        if not self.available:
            return
            
        try:
            metrics = {
                'growth_history': self.seed.metrics['growth_history'],
                'stability_history': self.seed.metrics['stability_history'],
                'complexity_history': self.seed.metrics['complexity_history'],
                'connection_history': self.seed.metrics['connection_history'],
                'bridge_history': self.seed.metrics['bridge_history'],
                'data_transferred': self.seed.metrics['data_transferred'],
                'last_transfer': self.seed.metrics['last_transfer']
            }
            self.metrics_updated.emit(metrics)
        except Exception as e:
            self.error_occurred.emit(f"Error updating metrics: {str(e)}")
            
    def get_state(self) -> Dict[str, Any]:
        """Get current seed state"""
        if not self.available:
            return {}
        return self.seed.get_state()
        
    def create_socket(self, socket_type: str = "input") -> Optional[str]:
        """Create a new connection socket"""
        if not self.available:
            return None
        return self.seed.create_socket(socket_type)
        
    def connect_sockets(self, source_id: str, target_id: str) -> bool:
        """Connect two sockets"""
        if not self.available:
            return False
        return self.seed.connect_sockets(source_id, target_id)
        
    def create_bridge(self, source_socket_id: str, target_seed_id: str,
                     target_socket_id: str, bridge_type: str = "direct") -> Optional[str]:
        """Create a bridge between this seed and another"""
        if not self.available:
            return None
        return self.seed.create_bridge(source_socket_id, target_seed_id,
                                     target_socket_id, bridge_type)

class NeuralSeedWidget(QWidget):
    """Enhanced widget for displaying and controlling NeuralSeed"""
    
    def __init__(self, adapter: NeuralSeedAdapter, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        
        # Create UI
        self._create_ui()
        
        # Connect signals
        self.adapter.seed_state_updated.connect(self._on_state_updated)
        self.adapter.growth_stage_changed.connect(self._on_stage_changed)
        self.adapter.stability_changed.connect(self._on_stability_changed)
        self.adapter.consciousness_level_changed.connect(self._on_consciousness_changed)
        self.adapter.component_activated.connect(self._on_component_activated)
        self.adapter.component_deactivated.connect(self._on_component_deactivated)
        self.adapter.metrics_updated.connect(self._on_metrics_updated)
        self.adapter.error_occurred.connect(self._on_error)
        
    def _create_ui(self):
        """Create the enhanced UI components"""
        layout = QVBoxLayout(self)
        
        # Status section
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QVBoxLayout(status_frame)
        
        self.status_label = QLabel("Neural Seed Status")
        self.stage_label = QLabel("Growth Stage: Seed")
        self.consciousness_label = QLabel("Consciousness Level: 0%")
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.consciousness_label)
        
        # Metrics section
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.StyledPanel)
        metrics_layout = QVBoxLayout(metrics_frame)
        
        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 100)
        self.stability_bar.setValue(100)
        
        self.complexity_bar = QProgressBar()
        self.complexity_bar.setRange(0, 100)
        self.complexity_bar.setValue(0)
        
        self.dictionary_label = QLabel("Dictionary Size: 0")
        self.components_label = QLabel("Active Components: 0")
        
        metrics_layout.addWidget(QLabel("Stability:"))
        metrics_layout.addWidget(self.stability_bar)
        metrics_layout.addWidget(QLabel("Complexity:"))
        metrics_layout.addWidget(self.complexity_bar)
        metrics_layout.addWidget(self.dictionary_label)
        metrics_layout.addWidget(self.components_label)
        
        # Add sections to main layout
        layout.addWidget(status_frame)
        layout.addWidget(metrics_frame)
        
    @Slot(dict)
    def _on_state_updated(self, state: Dict[str, Any]):
        """Handle state updates"""
        if 'state' in state:
            self.status_label.setText(f"Status: {state['state'].get('stage', 'Unknown')}")
            self.dictionary_label.setText(f"Dictionary Size: {state['state'].get('dictionary_size', 0)}")
            self.complexity_bar.setValue(int(state['state'].get('complexity', 0) * 100))
            self.components_label.setText(f"Active Components: {len(state['state'].get('active_components', []))}")
            
    @Slot(str)
    def _on_stage_changed(self, stage: str):
        """Handle growth stage changes"""
        self.stage_label.setText(f"Growth Stage: {stage}")
        
    @Slot(float)
    def _on_stability_changed(self, stability: float):
        """Handle stability changes"""
        self.stability_bar.setValue(int(stability * 100))
        
    @Slot(float)
    def _on_consciousness_changed(self, level: float):
        """Handle consciousness level changes"""
        self.consciousness_label.setText(f"Consciousness Level: {int(level * 100)}%")
        
    @Slot(str)
    def _on_component_activated(self, component: str):
        """Handle component activation"""
        pass  # Could add visual feedback for component activation
        
    @Slot(str)
    def _on_component_deactivated(self, component: str):
        """Handle component deactivation"""
        pass  # Could add visual feedback for component deactivation
        
    @Slot(dict)
    def _on_metrics_updated(self, metrics: Dict[str, Any]):
        """Handle metrics updates"""
        pass  # Could add charts or other visualizations for metrics
        
    @Slot(str)
    def _on_error(self, error: str):
        """Handle errors"""
        self.status_label.setText(f"Error: {error}") 