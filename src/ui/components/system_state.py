#!/usr/bin/env python3
"""
SystemState

This module implements the system state widget that synchronizes with the backend.
"""

import logging
from typing import Dict, Any, List, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemState(QWidget):
    """System state widget that synchronizes with backend."""
    
    # Signals
    state_updated = Signal(dict)
    health_changed = Signal(dict)
    stability_changed = Signal(float)
    growth_stage_changed = Signal(str)
    
    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)
        self._config = {}
        self._state = {
            'neural_seed': {
                'connected': False,
                'growth_stage': 'SEED',
                'progress': 0.0,
                'health_score': 0.0,
                'latency': 0.0,
                'load': 0.0,
                'memory': 0.0,
                'success_rate': 0.0,
                'last_activity': None,
                'error_count': 0,
                'warning_count': 0
            },
            'signal_system': {
                'connected': False,
                'message_count': 0,
                'active_signals': [],
                'processing_time': 0.0,
                'queue_size': 0,
                'error_rate': 0.0,
                'throughput': 0.0,
                'last_processed': None,
                'pending_requests': 0,
                'retry_count': 0
            },
            'spiderweb': {
                'connected': False,
                'active_bridges': 0,
                'bridge_health': {},
                'connection_quality': 0.0,
                'throughput': 0.0,
                'latency': 0.0,
                'error_count': 0,
                'last_sync': None,
                'bridge_status': {}
            },
            'health': {
                'overall': 0.0,
                'components': {
                    'neural_seed': 0.0,
                    'signal_system': 0.0,
                    'spiderweb': 0.0
                },
                'last_update': None,
                'trend': 'stable',
                'history': [],
                'thresholds': {
                    'critical': 0.3,
                    'warning': 0.6,
                    'healthy': 0.8
                }
            },
            'stability': {
                'value': 0.0,
                'trend': 'stable',
                'fluctuations': [],
                'last_update': None,
                'history': [],
                'thresholds': {
                    'unstable': 0.4,
                    'stable': 0.7,
                    'very_stable': 0.9
                }
            },
            'metrics': {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_io': 0.0,
                'network_traffic': 0.0,
                'timestamp': None,
                'history': {
                    'cpu': [],
                    'memory': [],
                    'disk': [],
                    'network': []
                },
                'averages': {
                    'cpu': 0.0,
                    'memory': 0.0,
                    'disk': 0.0,
                    'network': 0.0
                }
            }
        }
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create status labels with detailed information
        self._neural_seed_label = QLabel("Neural Seed: Disconnected")
        self._neural_seed_details = QLabel()
        self._neural_seed_warnings = QLabel()
        self._signal_system_label = QLabel("Signal System: Disconnected")
        self._signal_system_details = QLabel()
        self._signal_system_queue = QLabel()
        self._spiderweb_label = QLabel("Spiderweb: Disconnected")
        self._spiderweb_details = QLabel()
        self._spiderweb_bridges = QLabel()
        
        # Create detailed health indicators
        self._health_bar = QProgressBar()
        self._health_bar.setRange(0, 100)
        self._health_bar.setValue(0)
        self._health_details = QLabel()
        self._health_trend = QLabel()
        
        # Create detailed stability indicator
        self._stability_bar = QProgressBar()
        self._stability_bar.setRange(0, 100)
        self._stability_bar.setValue(0)
        self._stability_details = QLabel()
        self._stability_trend = QLabel()
        
        # Create metrics display
        self._metrics_label = QLabel("System Metrics:")
        self._metrics_details = QLabel()
        self._metrics_trends = QLabel()
        
        # Add widgets to layout
        layout.addWidget(self._neural_seed_label)
        layout.addWidget(self._neural_seed_details)
        layout.addWidget(self._neural_seed_warnings)
        layout.addWidget(self._signal_system_label)
        layout.addWidget(self._signal_system_details)
        layout.addWidget(self._signal_system_queue)
        layout.addWidget(self._spiderweb_label)
        layout.addWidget(self._spiderweb_details)
        layout.addWidget(self._spiderweb_bridges)
        layout.addWidget(QLabel("System Health:"))
        layout.addWidget(self._health_bar)
        layout.addWidget(self._health_details)
        layout.addWidget(self._health_trend)
        layout.addWidget(QLabel("System Stability:"))
        layout.addWidget(self._stability_bar)
        layout.addWidget(self._stability_details)
        layout.addWidget(self._stability_trend)
        layout.addWidget(self._metrics_label)
        layout.addWidget(self._metrics_details)
        layout.addWidget(self._metrics_trends)
        
    def set_config(self, config: Dict[str, Any]):
        """Set widget configuration."""
        self._config = config
        
    def update_state(self, state: Dict[str, Any], metrics: Dict[str, Any], health: Dict[str, Any]):
        """Update system state from backend."""
        try:
            # Update neural seed state with detailed metrics
            if 'neural_seed' in state:
                self._update_neural_seed_state(state['neural_seed'])
                
            # Update signal system state with detailed metrics
            if 'signal_system' in state:
                self._update_signal_system_state(state['signal_system'])
                
            # Update spiderweb state with detailed metrics
            if 'spiderweb' in state:
                self._update_spiderweb_state(state['spiderweb'])
                
            # Update health metrics with component details
            if health:
                self._update_health_state(health)
                
            # Update stability metrics with trend analysis
            if 'stability' in metrics:
                self._update_stability_state(metrics['stability'])
                
            # Update system metrics
            if 'system_metrics' in metrics:
                self._update_system_metrics(metrics['system_metrics'])
                
            # Emit state updated signal
            self.state_updated.emit(self._state)
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
            
    def _update_neural_seed_state(self, state: Dict[str, Any]):
        """Update neural seed state with detailed metrics."""
        try:
            self._state['neural_seed'].update(state)
            if 'metrics' in state:
                self._state['neural_seed'].update(state['metrics'])
                
            # Update history
            self._update_metric_history('neural_seed', state)
            
            # Update display
            self._update_neural_seed_display()
            
        except Exception as e:
            logger.error(f"Error updating neural seed state: {e}")
            
    def _update_signal_system_state(self, state: Dict[str, Any]):
        """Update signal system state with detailed metrics."""
        try:
            self._state['signal_system'].update(state)
            if 'metrics' in state:
                self._state['signal_system'].update(state['metrics'])
                
            # Update history
            self._update_metric_history('signal_system', state)
            
            # Update display
            self._update_signal_system_display()
            
        except Exception as e:
            logger.error(f"Error updating signal system state: {e}")
            
    def _update_spiderweb_state(self, state: Dict[str, Any]):
        """Update spiderweb state with detailed metrics."""
        try:
            self._state['spiderweb'].update(state)
            if 'metrics' in state:
                self._state['spiderweb'].update(state['metrics'])
                
            # Update history
            self._update_metric_history('spiderweb', state)
            
            # Update display
            self._update_spiderweb_display()
            
        except Exception as e:
            logger.error(f"Error updating spiderweb state: {e}")
            
    def _update_health_state(self, health: Dict[str, Any]):
        """Update health metrics with component details."""
        try:
            self._state['health'].update(health)
            if 'components' in health:
                self._state['health']['components'].update(health['components'])
                
            # Update history
            self._update_health_history(health)
            
            # Update trend
            self._update_health_trend()
            
            # Update display
            self._update_health_display()
            
        except Exception as e:
            logger.error(f"Error updating health state: {e}")
            
    def _update_stability_state(self, stability: float):
        """Update stability metrics with trend analysis."""
        try:
            self._state['stability']['value'] = stability
            self._state['stability']['trend'] = self._calculate_stability_trend(stability)
            self._state['stability']['fluctuations'].append(stability)
            if len(self._state['stability']['fluctuations']) > 10:
                self._state['stability']['fluctuations'].pop(0)
            self._state['stability']['last_update'] = datetime.now()
            
            # Update history
            self._update_stability_history(stability)
            
            # Update display
            self._update_stability_display()
            
        except Exception as e:
            logger.error(f"Error updating stability state: {e}")
            
    def _update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics."""
        try:
            self._state['metrics'].update(metrics)
            self._state['metrics']['timestamp'] = datetime.now()
            
            # Update history
            for metric in ['cpu_usage', 'memory_usage', 'disk_io', 'network_traffic']:
                if metric in metrics:
                    self._update_metric_history(metric, metrics[metric])
                    
            # Update averages
            self._update_metric_averages()
            
            # Update display
            self._update_metrics_display()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            
    def _update_metric_history(self, metric: str, value: Any):
        """Update metric history."""
        try:
            if metric in self._state['metrics']['history']:
                self._state['metrics']['history'][metric].append(value)
                if len(self._state['metrics']['history'][metric]) > 100:
                    self._state['metrics']['history'][metric].pop(0)
        except Exception as e:
            logger.error(f"Error updating metric history: {e}")
            
    def _update_health_history(self, health: Dict[str, Any]):
        """Update health history."""
        try:
            self._state['health']['history'].append({
                'overall': health['overall'],
                'components': health['components'],
                'timestamp': datetime.now()
            })
            if len(self._state['health']['history']) > 100:
                self._state['health']['history'].pop(0)
        except Exception as e:
            logger.error(f"Error updating health history: {e}")
            
    def _update_stability_history(self, stability: float):
        """Update stability history."""
        try:
            self._state['stability']['history'].append({
                'value': stability,
                'timestamp': datetime.now()
            })
            if len(self._state['stability']['history']) > 100:
                self._state['stability']['history'].pop(0)
        except Exception as e:
            logger.error(f"Error updating stability history: {e}")
            
    def _update_metric_averages(self):
        """Update metric averages."""
        try:
            for metric in ['cpu_usage', 'memory_usage', 'disk_io', 'network_traffic']:
                history = self._state['metrics']['history'][metric]
                if history:
                    self._state['metrics']['averages'][metric] = sum(history) / len(history)
        except Exception as e:
            logger.error(f"Error updating metric averages: {e}")
            
    def _update_health_trend(self):
        """Update health trend."""
        try:
            history = self._state['health']['history']
            if len(history) >= 2:
                current = history[-1]['overall']
                previous = history[-2]['overall']
                if current > previous + 0.05:
                    self._state['health']['trend'] = 'improving'
                elif current < previous - 0.05:
                    self._state['health']['trend'] = 'degrading'
                else:
                    self._state['health']['trend'] = 'stable'
        except Exception as e:
            logger.error(f"Error updating health trend: {e}")
            
    def _update_neural_seed_display(self):
        """Update neural seed display with detailed metrics."""
        try:
            seed_state = self._state['neural_seed']
            status = "Connected" if seed_state['connected'] else "Disconnected"
            self._neural_seed_label.setText(
                f"Neural Seed: {status} ({seed_state['growth_stage']})"
            )
            
            # Update details
            details = [
                f"Health: {seed_state['health_score']:.1%}",
                f"Latency: {seed_state['latency']:.2f}ms",
                f"Load: {seed_state['load']:.1%}",
                f"Memory: {seed_state['memory']:.1%}",
                f"Success: {seed_state['success_rate']:.1%}"
            ]
            self._neural_seed_details.setText(" | ".join(details))
            
            self.growth_stage_changed.emit(seed_state['growth_stage'])
            
        except Exception as e:
            logger.error(f"Error updating neural seed display: {e}")
            
    def _update_signal_system_display(self):
        """Update signal system display with detailed metrics."""
        try:
            signal_state = self._state['signal_system']
            status = "Connected" if signal_state['connected'] else "Disconnected"
            self._signal_system_label.setText(
                f"Signal System: {status} ({signal_state['message_count']} messages)"
            )
            
            # Update details
            details = [
                f"Processing: {signal_state['processing_time']:.2f}ms",
                f"Queue: {signal_state['queue_size']}",
                f"Error Rate: {signal_state['error_rate']:.1%}"
            ]
            self._signal_system_details.setText(" | ".join(details))
            
        except Exception as e:
            logger.error(f"Error updating signal system display: {e}")
            
    def _update_spiderweb_display(self):
        """Update spiderweb display with detailed metrics."""
        try:
            spiderweb_state = self._state['spiderweb']
            status = "Connected" if spiderweb_state['connected'] else "Disconnected"
            self._spiderweb_label.setText(
                f"Spiderweb: {status} ({spiderweb_state['active_bridges']} bridges)"
            )
            
            # Update details
            details = [
                f"Quality: {spiderweb_state['connection_quality']:.1%}",
                f"Throughput: {spiderweb_state['throughput']:.2f} MB/s"
            ]
            self._spiderweb_details.setText(" | ".join(details))
            
        except Exception as e:
            logger.error(f"Error updating spiderweb display: {e}")
            
    def _update_health_display(self):
        """Update health display with component details."""
        try:
            health = self._state['health']
            overall_health = health['overall']
            self._health_bar.setValue(int(overall_health * 100))
            
            # Update details
            details = []
            for component, score in health['components'].items():
                details.append(f"{component}: {score:.1%}")
            self._health_details.setText(" | ".join(details))
            
            self.health_changed.emit(health)
            
        except Exception as e:
            logger.error(f"Error updating health display: {e}")
            
    def _update_stability_display(self):
        """Update stability display with trend analysis."""
        try:
            stability = self._state['stability']
            self._stability_bar.setValue(int(stability['value'] * 100))
            
            # Update details
            details = [
                f"Trend: {stability['trend'].upper()}",
                f"Fluctuation: {self._calculate_fluctuation(stability['fluctuations']):.2%}"
            ]
            self._stability_details.setText(" | ".join(details))
            
            self.stability_changed.emit(stability['value'])
            
        except Exception as e:
            logger.error(f"Error updating stability display: {e}")
            
    def _update_metrics_display(self):
        """Update system metrics display."""
        try:
            metrics = self._state['metrics']
            details = [
                f"CPU: {metrics['cpu_usage']:.1%}",
                f"Memory: {metrics['memory_usage']:.1%}",
                f"Disk I/O: {metrics['disk_io']:.2f} MB/s",
                f"Network: {metrics['network_traffic']:.2f} MB/s"
            ]
            self._metrics_details.setText(" | ".join(details))
            
        except Exception as e:
            logger.error(f"Error updating metrics display: {e}")
            
    def _calculate_stability_trend(self, current_value: float) -> str:
        """Calculate stability trend based on recent values."""
        try:
            if not self._state['stability']['fluctuations']:
                return 'stable'
                
            avg = sum(self._state['stability']['fluctuations']) / len(self._state['stability']['fluctuations'])
            
            if current_value > avg + 0.1:
                return 'increasing'
            elif current_value < avg - 0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating stability trend: {e}")
            return 'unknown'
            
    def _calculate_fluctuation(self, values: List[float]) -> float:
        """Calculate fluctuation range of values."""
        try:
            if not values:
                return 0.0
            return max(values) - min(values)
        except Exception as e:
            logger.error(f"Error calculating fluctuation: {e}")
            return 0.0
        
    def get_growth_stage(self) -> str:
        """Get current growth stage."""
        return self._state['neural_seed']['growth_stage']
        
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics."""
        return self._state['health']
        
    def get_stability(self) -> float:
        """Get current stability value."""
        return self._state['stability']['value']
        
    def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get active nodes from current state."""
        try:
            nodes = []
            
            # Add neural seed node
            if self._state['neural_seed']['connected']:
                nodes.append({
                    'id': 'neural_seed',
                    'type': 'seed',
                    'x': 100,
                    'y': 100,
                    'size': 30,
                    'label': 'Neural Seed'
                })
                
            # Add signal system nodes
            if self._state['signal_system']['connected']:
                for i, signal in enumerate(self._state['signal_system']['active_signals']):
                    nodes.append({
                        'id': f'signal_{i}',
                        'type': 'signal',
                        'x': 200 + i * 50,
                        'y': 100,
                        'size': 20,
                        'label': f'Signal {i}'
                    })
                    
            # Add spiderweb nodes
            if self._state['spiderweb']['connected']:
                for i in range(self._state['spiderweb']['active_bridges']):
                    nodes.append({
                        'id': f'bridge_{i}',
                        'type': 'bridge',
                        'x': 100 + i * 50,
                        'y': 200,
                        'size': 25,
                        'label': f'Bridge {i}'
                    })
                    
            return nodes
            
        except Exception as e:
            logger.error(f"Error getting active nodes: {e}")
            return []
            
    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get active connections from current state."""
        try:
            connections = []
            
            # Connect neural seed to signals
            if (self._state['neural_seed']['connected'] and 
                self._state['signal_system']['connected']):
                for i in range(len(self._state['signal_system']['active_signals'])):
                    connections.append({
                        'from': 'neural_seed',
                        'to': f'signal_{i}',
                        'type': 'literal'
                    })
                    
            # Connect signals to bridges
            if (self._state['signal_system']['connected'] and 
                self._state['spiderweb']['connected']):
                for i in range(min(
                    len(self._state['signal_system']['active_signals']),
                    self._state['spiderweb']['active_bridges']
                )):
                    connections.append({
                        'from': f'signal_{i}',
                        'to': f'bridge_{i}',
                        'type': 'semantic'
                    })
                    
            return connections
            
        except Exception as e:
            logger.error(f"Error getting active connections: {e}")
            return [] 