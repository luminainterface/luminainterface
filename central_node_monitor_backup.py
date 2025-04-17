import logging
import sys
import os
import json
import datetime
from typing import Dict, List, Any, Optional

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QMessageBox, QSplitter,
                              QFrame, QToolBar, QStatusBar, QComboBox, QCheckBox,
                              QLineEdit, QListWidget, QTreeWidget, QTreeWidgetItem,
                              QTableWidget, QTableWidgetItem, QHeaderView, QSpacerItem,
                              QSizePolicy, QGridLayout, QStackedWidget)
from PySide6.QtCore import Qt, QTimer, QSize, Signal, Slot, QThread, QPoint, QObject
from PySide6.QtGui import QFont, QColor, QIcon, QPixmap, QCursor
from pathlib import Path
import time
import numpy as np
import torch
import random

# Import UI components
from ui.theme import LuminaTheme
from ui.components import ModernCard, ModernProgressCircle, ModernButton, ModernMetricsCard, ModernLogViewer
from ui.tab_bar import ModernTabBar

# Import central node and managers
from central_node import CentralNode
from spiderweb.spiderweb_manager import SpiderwebManager
from autowiki.autowiki import AutoWiki

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/central_node_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SystemMetricsThread(QThread):
    """Thread for collecting system metrics"""
    metrics_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self._paused = False
        self._reset = False
        self._articles = [
            {
                'title': 'Introduction to Neural Networks',
                'content': 'Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains...',
                'progress': 0
            },
            {
                'title': 'Deep Learning Fundamentals',
                'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks...',
                'progress': 0
            },
            {
                'title': 'Reinforcement Learning',
                'content': 'Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment...',
                'progress': 0
            }
        ]
        self._current_article_index = 0
        
    def run(self):
        """Main thread loop for updating metrics"""
        while True:
            if not self._paused:
                try:
                    metrics = self._collect_metrics()
                    self.metrics_updated.emit(metrics)
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(0.1)  # Update every 100ms for smoother progress
            
    def _collect_metrics(self):
        """Collect and return current system metrics"""
        if self._reset:
            self._reset = False
            return {
                'growth_stage': 'Seed',
                'stability': 65,
                'consciousness': 45,
                'complexity': 78,
                'growth_rate': 1.2,
                'system_age': '0h 0m',
                'active_connections': 8,
                'status': 'reset'
            }
            
        # Update current article progress
        current_article = self._articles[self._current_article_index]
        current_article['progress'] += 33  # Increment by 33% (complete in 3 steps for 3 seconds)
        
        # Move to next article if current one is complete
        if current_article['progress'] >= 100:
            current_article['progress'] = 100  # Cap at 100%
            self._current_article_index = (self._current_article_index + 1) % len(self._articles)
            # Reset the next article's progress
            self._articles[self._current_article_index]['progress'] = 0
            
        return {
            'current_article': current_article,
            'growth_stage': 'Growing',
            'stability': min(100, random.randint(60, 95)),
            'consciousness': min(100, random.randint(40, 85)),
            'complexity': min(100, random.randint(70, 90)),
            'growth_rate': round(random.uniform(1.0, 2.0), 2),
            'system_age': self._get_system_age(),
            'active_connections': random.randint(150, 200)
        }
        
    def _get_system_age(self):
        """Calculate and return the system age"""
        current_time = time.time()
        if not hasattr(self, '_start_time'):
            self._start_time = current_time
            
        age_seconds = int(current_time - self._start_time)
        days = age_seconds // (24 * 3600)
        hours = (age_seconds % (24 * 3600)) // 3600
        return f"{days}d {hours}h"
        
    def pause(self):
        """Pause metrics updates"""
        self._paused = True
        
    def resume(self):
        """Resume metrics updates"""
        self._paused = False
        
    def reset(self):
        """Reset metrics to initial state"""
        self._reset = True
        self._current_article_index = 0
        for article in self._articles:
            article['progress'] = 0

class VersionDataManager(QObject):
    version_updated = Signal(dict)  # Emits version summary when changes occur
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        self.versions_dir = Path("model_versions")
        self.versions_dir.mkdir(exist_ok=True)
        self.current_version = None
        self.version_summary = {}
        
        # Auto-save settings
        self.auto_save_enabled = False
        self.auto_save_interval = 300  # 5 minutes in seconds
        self.max_auto_saves = 5
        self.last_auto_save = None
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save)
        
        # Load initial versions
        self.load_all_versions()
        
    def toggle_auto_save(self, enabled: bool):
        """Toggle auto-save functionality"""
        try:
            self.auto_save_enabled = enabled
            if enabled:
                self.auto_save_timer.start(self.auto_save_interval * 1000)  # Convert to milliseconds
                logger.info(f"Auto-save enabled with interval of {self.auto_save_interval} seconds")
            else:
                self.auto_save_timer.stop()
                logger.info("Auto-save disabled")
        except Exception as e:
            logger.error(f"Error toggling auto-save: {str(e)}")
            
    def set_auto_save_interval(self, interval: int):
        """Set auto-save interval in seconds"""
        try:
            self.auto_save_interval = max(60, interval)  # Minimum 1 minute
            if self.auto_save_enabled:
                self.auto_save_timer.setInterval(self.auto_save_interval * 1000)
            logger.info(f"Auto-save interval set to {self.auto_save_interval} seconds")
        except Exception as e:
            logger.error(f"Error setting auto-save interval: {str(e)}")
            
    def set_max_auto_saves(self, max_saves: int):
        """Set maximum number of auto-saves to keep"""
        try:
            self.max_auto_saves = max(1, max_saves)
            logger.info(f"Maximum auto-saves set to {self.max_auto_saves}")
            self._cleanup_old_auto_saves()
        except Exception as e:
            logger.error(f"Error setting max auto-saves: {str(e)}")
            
    def _auto_save(self):
        """Perform auto-save"""
        try:
            if not self.auto_save_enabled:
                return
                
            # Save current state
            timestamp = self.save_current_version()
            if timestamp:
                self.last_auto_save = timestamp
                logger.info(f"Auto-saved version: {timestamp}")
                
                # Cleanup old auto-saves
                self._cleanup_old_auto_saves()
        except Exception as e:
            logger.error(f"Error during auto-save: {str(e)}")
            
    def _cleanup_old_auto_saves(self):
        """Remove old auto-saves exceeding the maximum limit"""
        try:
            # Get list of auto-save versions sorted by timestamp
            auto_saves = []
            for version_dir in self.versions_dir.glob("version_*"):
                timestamp = version_dir.name.replace("version_", "")
                if timestamp in self.version_summary:
                    auto_saves.append(timestamp)
                    
            auto_saves.sort(reverse=True)  # Most recent first
            
            # Remove excess auto-saves
            if len(auto_saves) > self.max_auto_saves:
                for timestamp in auto_saves[self.max_auto_saves:]:
                    self.delete_version(timestamp)
                    
        except Exception as e:
            logger.error(f"Error cleaning up old auto-saves: {str(e)}")
            
    def get_auto_save_status(self) -> dict:
        """Get current auto-save settings and status"""
        return {
            "enabled": self.auto_save_enabled,
            "interval": self.auto_save_interval,
            "max_saves": self.max_auto_saves,
            "last_save": self.last_auto_save
        }
    
    def save_current_version(self):
        """Save current model state with metadata"""
        try:
            # Create timestamp-based version name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version_path = self.versions_dir / f"version_{timestamp}"
            version_path.mkdir(exist_ok=True)
            
            # Get current model state and metrics
            model_state = self.parent().get_model_state()
            metrics = self.parent().current_metrics
            
            # Save model state
            with open(version_path / "model_state.json", "w") as f:
                json.dump(model_state, f, indent=4)
            
            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "accuracy": metrics.get("accuracy", 0),
                "complexity": metrics.get("complexity", 0),
                "growth_rate": metrics.get("growth_rate", 0),
                "active_connections": metrics.get("active_connections", 0),
                "system_age": metrics.get("system_age", "0h 0m")
            }
            
            with open(version_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            
            # Update version summary
            self.version_summary[timestamp] = metadata
            self.version_updated.emit(self.version_summary)
            
            logger.info(f"Saved model version: {timestamp}")
            return timestamp
            
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            return None
    
    def load_version(self, timestamp):
        """Load a specific model version"""
        try:
            version_path = self.versions_dir / f"version_{timestamp}"
            if not version_path.exists():
                logger.error(f"Version {timestamp} not found")
                return False
            
            # Load model state
            with open(version_path / "model_state.json", "r") as f:
                model_state = json.load(f)
            
            # Load into parent
            if self.parent().load_model_state(model_state):
                self.current_version = timestamp
                logger.info(f"Loaded model version: {timestamp}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading model version: {e}")
            return False
    
    def delete_version(self, timestamp):
        """Delete a specific model version"""
        try:
            version_path = self.versions_dir / f"version_{timestamp}"
            if not version_path.exists():
                logger.error(f"Version {timestamp} not found")
                return False
            
            # Remove version files
            for file in version_path.glob("*"):
                file.unlink()
            version_path.rmdir()
            
            # Update version summary
            if timestamp in self.version_summary:
                del self.version_summary[timestamp]
                self.version_updated.emit(self.version_summary)
            
            logger.info(f"Deleted model version: {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model version: {str(e)}")
            return False
    
    def get_version_summary(self):
        """Get summary of all saved versions."""
        try:
            summary = {}
            for version_dir in self.versions_dir.glob("version_*"):
                timestamp = version_dir.name.replace("version_", "")
                metadata_path = version_dir / "metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    summary[timestamp] = metadata
            
            self.version_summary = summary
        return summary
        
        except Exception as e:
            self.logger.error(f"Error getting version summary: {str(e)}")
            return {}
            
    def load_all_versions(self):
        """Load all saved versions from disk."""
        try:
            # Create versions directory if it doesn't exist
            self.versions_dir.mkdir(exist_ok=True)
            
            # Clear current summary
            self.version_summary = {}
            
            # Load each version's metadata
            for version_dir in self.versions_dir.glob("version_*"):
                timestamp = version_dir.name.replace("version_", "")
                metadata_path = version_dir / "metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    self.version_summary[timestamp] = metadata
            
            # Sort versions by timestamp
            self.version_summary = dict(sorted(self.version_summary.items(), reverse=True))
            
            # Set current version to latest if none selected
            if not self.current_version and self.version_summary:
                self.current_version = next(iter(self.version_summary))
            
            self.version_updated.emit(self.version_summary)
            self.logger.info(f"Loaded {len(self.version_summary)} versions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading versions: {str(e)}")
            return False

class CentralNodeMonitor(QObject):
    # Signals for UI updates
    metrics_updated = Signal(dict)
    nodes_updated = Signal(list)
    processors_updated = Signal(list)
    article_updated = Signal(dict)
    article_progress_updated = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.current_metrics = {
            'growth_stage': 'Seed',
            'stability': 65,
            'consciousness': 45,
            'complexity': 78,
            'growth_rate': 1.2,
            'system_age': '0h 0m',
            'active_connections': 8,
            'status': 'ready'
        }
        
        # Initialize states
        self.is_learning = False
        self.growth_paused = False
        
        # Initialize spiderweb manager
        self.spiderweb = SpiderwebManager()
        self.spiderweb.initialize()
        
        # Initialize nodes and processors
        self.nodes = {}
        self.processors = {}
        self.version_manager = VersionDataManager(self)
        
        # Initialize AutoWiki system
        self.autowiki = AutoWiki()
        self.autowiki.initialize()
        
        # Initialize metrics thread
        self.metrics_thread = SystemMetricsThread()
        self.metrics_thread.metrics_updated.connect(self._handle_metrics_update)
        self.metrics_thread.start()
        
        # Load initial data
        self.load_data()
        
        # Article tracking
        self.current_article = {
            'title': '',
            'content': '',
            'progress': 0
        }
        
    def _handle_metrics_update(self, metrics):
        """Handle metrics updates from the SystemMetricsThread"""
        try:
            # Extract article info if present
            if 'current_article' in metrics:
                current_article = metrics.pop('current_article')
                self.article_updated.emit(current_article)
                self.article_progress_updated.emit(current_article)
            
            # Update current metrics
            self.current_metrics.update(metrics)
            # Emit the metrics_updated signal with the current metrics
            self.metrics_updated.emit(self.current_metrics)
            
        except Exception as e:
            logger.error(f"Error handling metrics update: {e}")
            
    def pause_growth(self):
        """Pause the growth simulation"""
        try:
            self.metrics_thread.pause()
            self.current_metrics['status'] = 'paused'
            self.metrics_updated.emit(self.current_metrics)
        except Exception as e:
            logger.error(f"Error pausing growth: {e}")
            
    def resume_growth(self):
        """Resume the growth simulation"""
        try:
            self.metrics_thread.resume()
            self.current_metrics['status'] = 'running'
            self.metrics_updated.emit(self.current_metrics)
        except Exception as e:
            logger.error(f"Error resuming growth: {e}")
            
    def reset_seed(self):
        """Reset the growth simulation"""
        try:
            self.metrics_thread.reset()
            self.current_metrics['status'] = 'reset'
            self.metrics_updated.emit(self.current_metrics)
        except Exception as e:
            logger.error(f"Error resetting seed: {e}")
            
    def start_learning(self):
        """Start the learning process"""
        try:
            self.metrics_thread.resume()
            self.current_metrics['status'] = 'learning'
            self.metrics_updated.emit(self.current_metrics)
        except Exception as e:
            logger.error(f"Error starting learning: {e}")
            
    def pause_learning(self):
        """Pause the learning process"""
        try:
            self.metrics_thread.pause()
            self.current_metrics['status'] = 'paused'
            self.metrics_updated.emit(self.current_metrics)
        except Exception as e:
            logger.error(f"Error pausing learning: {e}")
            
    def resume_learning(self):
        """Resume the learning process"""
        try:
            self.metrics_thread.resume()
            self.current_metrics['status'] = 'learning'
            self.metrics_updated.emit(self.current_metrics)
        except Exception as e:
            logger.error(f"Error resuming learning: {e}")
            
    def cleanup(self):
        """Clean up resources"""
        try:
            self.metrics_thread.terminate()
            self.metrics_thread.wait()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def load_data(self):
        """Load initial data and emit signals"""
        try:
            # Load versions
            self.version_manager.load_all_versions()
            
            # Initialize some mock nodes and processors for testing
            self.nodes = {
                'node1': {'name': 'Quantum Node A', 'status': 'STANDBY', 'version': 'v1', 'type': 'quantum'},
                'node2': {'name': 'Cosmic Node B', 'status': 'ACTIVE', 'version': 'v2', 'type': 'cosmic'},
                'node3': {'name': 'Quantum Node C', 'status': 'STANDBY', 'version': 'v1', 'type': 'quantum'}
            }
            
            # Initialize with 28+ processors
            self.processors = {
                'proc1': {'name': 'Knowledge Acquisition', 'status': 'ACTIVE', 'version': 'v1', 'type': 'knowledge'},
                'proc2': {'name': 'Pattern Recognition', 'status': 'ACTIVE', 'version': 'v1', 'type': 'pattern'},
                'proc3': {'name': 'Data Integration', 'status': 'ACTIVE', 'version': 'v1', 'type': 'data'},
                'proc4': {'name': 'Neural Synthesis', 'status': 'STANDBY', 'version': 'v1', 'type': 'neural'},
                'proc5': {'name': 'Semantic Analysis', 'status': 'ACTIVE', 'version': 'v1', 'type': 'semantic'},
                'proc6': {'name': 'Memory Consolidation', 'status': 'ACTIVE', 'version': 'v1', 'type': 'memory'},
                'proc7': {'name': 'Learning Optimization', 'status': 'STANDBY', 'version': 'v1', 'type': 'learning'},
                'proc8': {'name': 'Feature Extraction', 'status': 'ACTIVE', 'version': 'v1', 'type': 'feature'},
                'proc9': {'name': 'Context Analysis', 'status': 'ACTIVE', 'version': 'v1', 'type': 'context'},
                'proc10': {'name': 'Temporal Processing', 'status': 'STANDBY', 'version': 'v1', 'type': 'temporal'},
                'proc11': {'name': 'Spatial Analysis', 'status': 'ACTIVE', 'version': 'v1', 'type': 'spatial'},
                'proc12': {'name': 'Decision Making', 'status': 'ACTIVE', 'version': 'v1', 'type': 'decision'},
                'proc13': {'name': 'Error Correction', 'status': 'STANDBY', 'version': 'v1', 'type': 'error'},
                'proc14': {'name': 'Feedback Loop', 'status': 'ACTIVE', 'version': 'v1', 'type': 'feedback'},
                'proc15': {'name': 'Weight Adjustment', 'status': 'ACTIVE', 'version': 'v1', 'type': 'weight'},
                'proc16': {'name': 'Bias Optimization', 'status': 'STANDBY', 'version': 'v1', 'type': 'bias'},
                'proc17': {'name': 'Layer Management', 'status': 'ACTIVE', 'version': 'v1', 'type': 'layer'},
                'proc18': {'name': 'Network Topology', 'status': 'ACTIVE', 'version': 'v1', 'type': 'topology'},
                'proc19': {'name': 'Activation Control', 'status': 'STANDBY', 'version': 'v1', 'type': 'activation'},
                'proc20': {'name': 'Gradient Calculation', 'status': 'ACTIVE', 'version': 'v1', 'type': 'gradient'},
                'proc21': {'name': 'Loss Computation', 'status': 'ACTIVE', 'version': 'v1', 'type': 'loss'},
                'proc22': {'name': 'Backpropagation', 'status': 'STANDBY', 'version': 'v1', 'type': 'backprop'},
                'proc23': {'name': 'Model Validation', 'status': 'ACTIVE', 'version': 'v1', 'type': 'validation'},
                'proc24': {'name': 'Data Preprocessing', 'status': 'ACTIVE', 'version': 'v1', 'type': 'preprocess'},
                'proc25': {'name': 'Batch Processing', 'status': 'STANDBY', 'version': 'v1', 'type': 'batch'},
                'proc26': {'name': 'Epoch Management', 'status': 'ACTIVE', 'version': 'v1', 'type': 'epoch'},
                'proc27': {'name': 'Checkpoint Control', 'status': 'ACTIVE', 'version': 'v1', 'type': 'checkpoint'},
                'proc28': {'name': 'Model Persistence', 'status': 'STANDBY', 'version': 'v1', 'type': 'persistence'},
                'proc29': {'name': 'Inference Engine', 'status': 'ACTIVE', 'version': 'v1', 'type': 'inference'},
                'proc30': {'name': 'Output Generation', 'status': 'ACTIVE', 'version': 'v1', 'type': 'output'}
            }
            
            # Emit initial signals
            self.update_ui()
            logger.info("Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
        
    def start(self):
        """Start the central node monitor"""
        try:
            # Initialize spiderweb system
            self.spiderweb.initialize()
            logger.info("Spiderweb system initialized successfully")
            
            # Connect version bridges
            for version in ['v1', 'v2']:
                config = {'version': version}
                if self.spiderweb.connect_version(version, config):
                    logger.info(f"Version bridge {version} connected successfully")
                else:
                    logger.error(f"Failed to connect version bridge {version}")
            
            # Initialize AutoWiki system
            if self.autowiki.initialize():
                logger.info("AutoWiki system initialized successfully")
                self._connect_autowiki_neural_seed()
            else:
                logger.error("Failed to initialize AutoWiki system")
                    
            # Update UI with initial state
            self.update_ui()
            
        except Exception as e:
            logger.error(f"Error starting central node monitor: {str(e)}")
            
    def _connect_autowiki_neural_seed(self):
        """Connect AutoWiki to Neural Seed"""
        try:
            # Register message handlers
            self.spiderweb.register_handler(
                'content_request',
                self.autowiki.handle_content_request
            )
            self.spiderweb.register_handler(
                'learning_update',
                self.autowiki.handle_learning_update
            )
            
            # Setup data bridges
            self.autowiki.set_neural_seed(self.spiderweb)
            
        except Exception as e:
            logger.error(f"Failed to connect AutoWiki to Neural Seed: {str(e)}")
            raise

    def update_ui(self):
        """Update UI with current state"""
        try:
            # Update nodes list
            nodes_data = []
            for node_id, info in self.nodes.items():
                nodes_data.append({
                    'text': f"{node_id}: {info['name']} [{info['status']}]",
                    'color': '#00FF00' if info['status'] == 'ACTIVE' else '#888888',
                    'version': info['version'],
                    'type': info['type']
                })
            self.nodes_updated.emit(nodes_data)
            
            # Update processors list
            procs_data = []
            for proc_id, info in self.processors.items():
                procs_data.append({
                    'text': f"{proc_id}: {info['name']} [{info['status']}]",
                    'color': '#00FF00' if info['status'] == 'ACTIVE' else '#888888',
                    'version': info['version'],
                    'type': info['type']
                })
            self.processors_updated.emit(procs_data)
            
            # Update metrics
            metrics = self.spiderweb.get_metrics()
            self.metrics_updated.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error updating UI: {str(e)}")
            
    def activate_node(self, node_id):
        """Activate a node"""
        try:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node['type'] == 'quantum':
                    config = self.spiderweb.create_quantum_node()
                else:
                    config = self.spiderweb.create_cosmic_node()
                    
                node['status'] = 'ACTIVE'
                self.update_ui()
                logger.info(f"Node {node_id} activated successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error activating node {node_id}: {str(e)}")
            
        return False
        
    def deactivate_node(self, node_id):
        """Deactivate a node"""
        try:
            if node_id in self.nodes:
                self.nodes[node_id]['status'] = 'STANDBY'
                self.update_ui()
                logger.info(f"Node {node_id} deactivated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error deactivating node {node_id}: {str(e)}")
            
        return False
        
    def start_processor(self, proc_id):
        """Start a processor"""
        try:
            if proc_id in self.processors:
                self.processors[proc_id]['status'] = 'ACTIVE'
                self.update_ui()
                logger.info(f"Processor {proc_id} started successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error starting processor {proc_id}: {str(e)}")
            
        return False
        
    def stop_processor(self, proc_id):
        """Stop a processor"""
        try:
            if proc_id in self.processors:
                self.processors[proc_id]['status'] = 'STANDBY'
                self.update_ui()
                logger.info(f"Processor {proc_id} stopped successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping processor {proc_id}: {str(e)}")
            
        return False
        
    def get_network_state(self):
        """Get current network state"""
        return {
            'nodes': self.nodes,
            'processors': self.processors,
            'metrics': self.spiderweb.get_metrics()
        }

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create and show main window
    window = CentralNodeMonitor()
    window.show()
    
    # Start application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
