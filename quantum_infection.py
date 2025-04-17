"""
Quantum Infection System
This module implements the "quantum infection" mechanism that allows information
to spread between all nodes in the neural network system.

The hybrid_node acts as the central hub, and information spreads out to all connected
nodes, then returns to the RSEN_node for analysis and further information retrieval.
"""

import torch
import numpy as np
import logging
import time
import os
import threading
import json
from datetime import datetime
import traceback
import sqlite3
from queue import Queue, Empty, PriorityQueue
import random
import glob
import importlib
import requests
import re
from pathlib import Path
from threading import Lock
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_infection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuantumInfection:
    """
    Manages the spread of information between neural network nodes
    using the quantum infection mechanism.
    
    This class handles:
    1. Propagation of information from hybrid node to all other nodes
    2. Analysis of information resonance across the network
    3. Tracking of information spread and growth
    4. Background processing to continuously spread information
    """
    
    def __init__(self, node_paths=None):
        self.node_paths = node_paths or self._discover_nodes()
        self.infection_queue = PriorityQueue()
        self.infection_lock = Lock()
        self.infected_data = set()
        self.learning_rate = 0.01
        self.infection_threshold = 0.7
        self.max_infection_cycles = 100
        self.node_sockets = {}
        self.node_manager = None
        
        # New attributes for improved infection
        self.data_quality_metrics = {}  # Track quality metrics for each data type
        self.infection_history = {}  # Track infection paths to prevent loops
        self.dynamic_thresholds = {}  # Store dynamic thresholds per data type
        self.resonance_cache = {}  # Cache resonance calculations
        self.quality_factors = {
            'relevance': 0.4,
            'novelty': 0.3,
            'complexity': 0.2,
            'stability': 0.1
        }
        
    def _discover_nodes(self):
        """Automatically discover all *node*.py files"""
        nodes = []
        for file in Path('.').glob('**/*node*.py'):
            if 'node' in file.name.lower():
                nodes.append(str(file))
        return nodes
        
    def infect_data(self, data, source_node=None, priority=0, metadata=None):
        """Start infection process for new data with enhanced quality tracking"""
        try:
            data_hash = self._hash_data(data)
            if data_hash not in self.infected_data:
                # Calculate initial quality metrics
                quality_metrics = self._calculate_quality_metrics(data, metadata)
                
                # Store quality metrics
                self.data_quality_metrics[data_hash] = quality_metrics
                
                # Initialize infection history
                self.infection_history[data_hash] = {
                    'path': [source_node] if source_node else [],
                    'timestamp': datetime.now(),
                    'cycle_count': 0
                }
                
                # Calculate dynamic threshold based on data type and quality
                threshold = self._calculate_dynamic_threshold(data, quality_metrics)
                self.dynamic_thresholds[data_hash] = threshold
                
                # Add to queue with enhanced priority calculation
                enhanced_priority = self._calculate_enhanced_priority(priority, quality_metrics)
                
                self.infection_queue.put((enhanced_priority, {
                    'data': data,
                    'source': source_node,
                    'infected_nodes': set(),
                    'cycle_count': 0,
                    'quality_metrics': quality_metrics,
                    'threshold': threshold
                }))
                self.infected_data.add(data_hash)
                return True
        except Exception as e:
            logger.error(f"Error infecting data: {str(e)}")
        return False
        
    def _hash_data(self, data):
        """Create unique hash for data tracking"""
        if isinstance(data, dict):
            return hash(frozenset(data.items()))
        return hash(str(data))
        
    def initialize_node_manager(self):
        """Initialize the node manager and establish socket connections"""
        try:
            from node_manager import NodeManager
            self.node_manager = NodeManager()
            
            # Discover and register nodes
            registered = self.node_manager.discover_and_register_nodes()
            logger.info(f"Registered {registered} nodes")
            
            # Initialize sockets
            if self.node_manager.initialize_sockets():
                logger.info("Successfully initialized node sockets")
                
                # Validate connections
                validation = self.node_manager.validate_connections()
                logger.info(f"Connection validation results: {validation}")
                
                return True
        except Exception as e:
            logger.error(f"Error initializing node manager: {str(e)}")
            return False
            
    def _load_node(self, node_path):
        """Load node from path with socket initialization"""
        try:
            # Import node module dynamically
            spec = importlib.util.spec_from_file_location("node_module", node_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get main node class
            node_class = None
            for attr in dir(module):
                if 'node' in attr.lower() and attr.endswith('Node'):
                    node_class = getattr(module, attr)
                    break
                    
            if node_class:
                node = node_class()
                # Initialize socket if method exists
                if hasattr(node, 'initialize_socket'):
                    if node.initialize_socket():
                        self.node_sockets[node_path] = True
                        logger.info(f"Initialized socket for node {node_path}")
                    else:
                        self.node_sockets[node_path] = False
                        logger.warning(f"Failed to initialize socket for node {node_path}")
                return node
        except Exception as e:
            logger.error(f"Error loading node {node_path}: {str(e)}")
            self.node_sockets[node_path] = False
        return None
        
    def _process_through_node(self, node, data):
        """Process data through a node"""
        try:
            if hasattr(node, 'process_data'):
                return node.process_data(data)
            elif hasattr(node, 'train_step'):
                return node.train_step(data)
        except Exception as e:
            logger.error(f"Error processing through node: {str(e)}")
        return None
        
    def _save_node_state(self, node, node_path):
        """Save node state to disk"""
        try:
            if hasattr(node, 'save_state'):
                save_path = Path('training_data') / f"{Path(node_path).stem}_state.pt"
                node.save_state(save_path)
        except Exception as e:
            logger.error(f"Error saving node state: {str(e)}")
            
    def get_infection_stats(self):
        """Get current infection statistics"""
        return {
            'total_infected': len(self.infected_data),
            'queue_size': self.infection_queue.qsize(),
            'active_nodes': len(self.node_paths)
        }

    def propagate(self):
        """Propagate infection through all nodes with socket validation"""
        while not self.infection_queue.empty():
            try:
                with self.infection_lock:
                    _, infection = self.infection_queue.get()
                    
                    if infection['cycle_count'] >= self.max_infection_cycles:
                        continue
                        
                    # Propagate through each node
                    for node_path in self.node_paths:
                        if node_path not in infection['infected_nodes']:
                            try:
                                # Check socket status
                                if node_path in self.node_sockets and not self.node_sockets[node_path]:
                                    logger.warning(f"Skipping node {node_path} due to inactive socket")
                                    continue
                                    
                                # Load and process node
                                node = self._load_node(node_path)
                                if node:
                                    result = self._process_through_node(node, infection['data'])
                                    if result and result.get('confidence', 0) > self.infection_threshold:
                                        infection['infected_nodes'].add(node_path)
                                        infection['data'].update(result)
                                        self._save_node_state(node, node_path)
                                        
                                        new_priority = -len(infection['infected_nodes'])
                                        infection['cycle_count'] += 1
                                        self.infection_queue.put((new_priority, infection))
                                        
                            except Exception as e:
                                logger.error(f"Error processing node {node_path}: {str(e)}")
                            finally:
                                gc.collect()
                                
            except Exception as e:
                logger.error(f"Error in propagation cycle: {str(e)}")

    def _calculate_quality_metrics(self, data, metadata=None):
        """Calculate quality metrics for the data"""
        metrics = {
            'relevance': self._calculate_relevance(data, metadata),
            'novelty': self._calculate_novelty(data),
            'complexity': self._calculate_complexity(data),
            'stability': self._calculate_stability(data)
        }
        return metrics
        
    def _calculate_relevance(self, data, metadata):
        """Calculate relevance score based on data content and metadata"""
        # Implement relevance calculation logic
        return 0.8  # Placeholder
        
    def _calculate_novelty(self, data):
        """Calculate novelty score based on data uniqueness"""
        # Implement novelty calculation logic
        return 0.7  # Placeholder
        
    def _calculate_complexity(self, data):
        """Calculate complexity score based on data structure"""
        # Implement complexity calculation logic
        return 0.6  # Placeholder
        
    def _calculate_stability(self, data):
        """Calculate stability score based on data consistency"""
        # Implement stability calculation logic
        return 0.9  # Placeholder
        
    def _calculate_dynamic_threshold(self, data, quality_metrics):
        """Calculate dynamic threshold based on data type and quality"""
        base_threshold = self.infection_threshold
        quality_score = sum(
            metric * self.quality_factors[metric_name]
            for metric_name, metric in quality_metrics.items()
        )
        return base_threshold * (1.0 - (quality_score * 0.3))  # Higher quality = lower threshold
        
    def _calculate_enhanced_priority(self, base_priority, quality_metrics):
        """Calculate enhanced priority based on quality metrics"""
        quality_score = sum(
            metric * self.quality_factors[metric_name]
            for metric_name, metric in quality_metrics.items()
        )
        return base_priority * (1.0 + quality_score)
        
    def _process_infection_cycle(self):
        """Process a single infection cycle with enhanced logic"""
        try:
            with self.infection_lock:
                _, infection = self.infection_queue.get()
                
                if infection['cycle_count'] >= self.max_infection_cycles:
                    return
                    
                data_hash = self._hash_data(infection['data'])
                current_threshold = self.dynamic_thresholds.get(data_hash, self.infection_threshold)
                
                # Propagate through each node
                for node_path in self.node_paths:
                    if node_path not in infection['infected_nodes']:
                        try:
                            # Check for infection loops
                            if self._is_infection_loop(data_hash, node_path):
                                continue
                                
                            # Check socket status
                            if node_path in self.node_sockets and not self.node_sockets[node_path]:
                                logger.warning(f"Skipping node {node_path} due to inactive socket")
                                continue
                                
                            # Load and process node
                            node = self._load_node(node_path)
                            if node:
                                result = self._process_through_node(node, infection['data'])
                                if result and result.get('confidence', 0) > current_threshold:
                                    # Update infection history
                                    self.infection_history[data_hash]['path'].append(node_path)
                                    self.infection_history[data_hash]['cycle_count'] += 1
                                    
                                    infection['infected_nodes'].add(node_path)
                                    infection['data'].update(result)
                                    self._save_node_state(node, node_path)
                                    
                                    # Update quality metrics based on node response
                                    self._update_quality_metrics(data_hash, result)
                                    
                                    # Recalculate priority
                                    new_priority = self._calculate_enhanced_priority(
                                        -len(infection['infected_nodes']),
                                        self.data_quality_metrics[data_hash]
                                    )
                                    
                                    infection['cycle_count'] += 1
                                    self.infection_queue.put((new_priority, infection))
                                    
                        except Exception as e:
                            logger.error(f"Error processing node {node_path}: {str(e)}")
                        finally:
                            gc.collect()
                            
        except Exception as e:
            logger.error(f"Error in infection cycle: {str(e)}")
            
    def _is_infection_loop(self, data_hash, node_path):
        """Check if this would create an infection loop"""
        history = self.infection_history.get(data_hash, {})
        path = history.get('path', [])
        
        # Check if node is already in the path
        if node_path in path:
            # Allow re-infection if enough time has passed
            last_visit_index = len(path) - 1 - path[::-1].index(node_path)
            time_since_last_visit = datetime.now() - history.get('timestamp', datetime.now())
            if time_since_last_visit.total_seconds() < 3600:  # 1 hour cooldown
                return True
                
        return False
        
    def _update_quality_metrics(self, data_hash, result):
        """Update quality metrics based on node response"""
        if data_hash in self.data_quality_metrics:
            metrics = self.data_quality_metrics[data_hash]
            
            # Update relevance based on node response
            if 'relevance_score' in result:
                metrics['relevance'] = (metrics['relevance'] + result['relevance_score']) / 2
                
            # Update stability based on response consistency
            if 'stability_score' in result:
                metrics['stability'] = (metrics['stability'] + result['stability_score']) / 2
                
            # Store updated metrics
            self.data_quality_metrics[data_hash] = metrics

# Singleton instance
_infection_system = None

def initialize(node_paths=None):
    """Initialize the quantum infection system"""
    global _infection_system
    if _infection_system is None:
        _infection_system = QuantumInfection(node_paths)
    return _infection_system

def get_system():
    """Get the quantum infection system instance"""
    global _infection_system
    if _infection_system is None:
        _infection_system = QuantumInfection()
    return _infection_system

def infect_data(data, source_node=None, priority=0):
    """Trigger a new infection"""
    system = get_system()
    return system.infect_data(data, source_node, priority)

def propagate():
    """Start the quantum infection system"""
    system = get_system()
    return system.propagate()

def get_infection_stats():
    """Get the current status of the quantum infection system"""
    system = get_system()
    return system.get_infection_stats()

# --- Global Variables ---
active_infection_threads = {}
stop_infection_event = threading.Event()
node_registry = {} # Store imported node modules
NODE_FILE_PATTERN = "*_node.py" # Pattern to find node files

# --- Infection Logic ---

def _infect_node(node_module, data_packet, infection_id):
    """Internal function to attempt calling receive_data on a node."""
    node_name = node_module.__name__
    if hasattr(node_module, 'receive_data'):
        try:
            logger.info(f"[Infection:{infection_id}] Transmitting data packet to node: {node_name}")
            # Potentially modify data packet or get response if needed
            node_module.receive_data(data_packet) 
            logger.debug(f"[Infection:{infection_id}] Data packet successfully received by {node_name}")
            return True
        except Exception as e:
            logger.error(f"[Infection:{infection_id}] Error transmitting data to node {node_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    else:
        logger.warning(f"[Infection:{infection_id}] Node {node_name} does not have a receive_data function. Skipping.")
        return False

def broadcast_data(data: dict):
    """Broadcasts data to all discovered node modules."""
    infection_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"[Infection:{infection_id}] Initiating broadcast with data from source: {data.get('source', 'unknown')}")
    
    # Dynamically find and import node modules
    _discover_and_load_nodes()
    
    successful_transmissions = 0
    failed_transmissions = 0
    
    if not node_registry:
        logger.warning(f"[Infection:{infection_id}] No node modules found or loaded. Broadcast aborted.")
        return

    # Attempt to infect each registered node
    for node_name, node_module in node_registry.items():
        if _infect_node(node_module, data, infection_id):
            successful_transmissions += 1
        else:
            failed_transmissions += 1
            
    logger.info(f"[Infection:{infection_id}] Broadcast complete. Successful transmissions: {successful_transmissions}, Failures/Skipped: {failed_transmissions}")


def _discover_and_load_nodes():
    """Finds node files, imports them, and stores in node_registry."""
    global node_registry
    logger.debug("Discovering node files...")
    discovered_files = glob.glob(NODE_FILE_PATTERN)
    
    for filepath in discovered_files:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        if module_name == __name__: # Don't import self
            continue
            
        if module_name not in node_registry:
            try:
                logger.debug(f"Attempting to import node module: {module_name}")
                # Ensure the directory containing the module is in sys.path if needed
                # spec = importlib.util.spec_from_file_location(module_name, filepath)
                # node_module = importlib.util.module_from_spec(spec)
                # spec.loader.exec_module(node_module)
                node_module = importlib.import_module(module_name)
                node_registry[module_name] = node_module
                logger.info(f"Successfully discovered and loaded node module: {module_name}")
            except ImportError as e:
                logger.error(f"Failed to import node module {module_name} from {filepath}: {e}")
                logger.error(traceback.format_exc())
            except Exception as e:
                 logger.error(f"Unexpected error loading node module {module_name} from {filepath}: {e}")
                 logger.error(traceback.format_exc())
    # Optional: Reload existing modules if needed (careful with state)
    # for node_name, node_module in node_registry.items():
    #     try:
    #         importlib.reload(node_module)
    #         logger.debug(f"Reloaded node module: {node_name}")
    #     except Exception as e:
    #         logger.error(f"Failed to reload node module {node_name}: {e}")

# --- Legacy Infection Simulation (Keep or Remove?) ---
# The functions below seem related to a different simulation 
# (spreading infection probabilistically over time). 
# Keep them if they are still needed, otherwise remove for clarity.

def start_infection_simulation(infection_rate=0.1, spread_delay=1.0):
    """Starts the background infection simulation (legacy?)."""
    # ... (Implementation of the probabilistic infection simulation) ...
    pass # Remove or keep original code

def stop_infection_simulation():
    """Stops the background infection simulation (legacy?)."""
    # ... (Implementation to stop the simulation thread) ...
    pass # Remove or keep original code

def get_infection_status():
    """Returns the status of the infection simulation (legacy?)."""
    # ... (Implementation to return status) ...
    return {} # Remove or keep original code

# --- Main Execution (Example/Test) ---
if __name__ == "__main__":
    logger.info("Quantum Infection Module - Standalone Test")
    
    # Test discovery
    _discover_and_load_nodes()
    print(f"Discovered nodes: {list(node_registry.keys())}")
    
    # Test broadcast
    test_data = {
        'source': 'qi_test',
        'content': 'This is a test broadcast from quantum_infection.py',
        'timestamp': datetime.now().isoformat(),
        'test_value': random.random()
    }
    broadcast_data(test_data)
    
    print("\nBroadcast test complete. Check logs and node behavior.")
    
    # Optional: Start the legacy simulation if needed
    # logger.info("Starting legacy infection simulation...")
    # start_infection_simulation()
    # try:
    #     while True:
    #         time.sleep(5)
    #         print(f"Infection Status: {get_infection_status()}")
    # except KeyboardInterrupt:
    #     logger.info("Stopping legacy infection simulation...")
    #     stop_infection_simulation()
    #     logger.info("Simulation stopped.") 