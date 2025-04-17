"""
Neural Seed Module

This module implements an enhanced neural seed system with non-linear growth rates,
adaptive dictionary size, and stability-based component activation.
"""

import math
import random
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
import socket
import threading
import json
from queue import Queue
import time
import queue

logger = logging.getLogger(__name__)

class ConnectionSocket:
    """Represents a socket connection for neural pattern bridging"""
    
    def __init__(self, socket_id: str, socket_type: str = "input"):
        self.id = socket_id
        self.type = socket_type  # "input" or "output"
        self.connections: Set[str] = set()  # Connected socket IDs
        self.buffer = Queue()
        self.active = False
        self.last_activity = datetime.now()
        self.stability = 1.0
        self.metrics = {
            'data_transferred': 0,
            'last_transfer': 0.0,
            'error_count': 0
        }
        
    def connect(self, target_socket_id: str) -> bool:
        """Connect to another socket"""
        if target_socket_id not in self.connections:
            self.connections.add(target_socket_id)
            self.last_activity = datetime.now()
            self.stability = min(1.0, self.stability + 0.1)
            return True
        return False
        
    def disconnect(self, target_socket_id: str) -> bool:
        """Disconnect from a socket"""
        if target_socket_id in self.connections:
            self.connections.remove(target_socket_id)
            self.stability = max(0.0, self.stability - 0.2)
            return True
        return False
        
    def send(self, data: Any) -> bool:
        """Send data through the socket"""
        if not self.active:
            return False
        try:
            self.buffer.put(data)
            self.last_activity = datetime.now()
            self.metrics['data_transferred'] += 1
            self.metrics['last_transfer'] = time.time()
            self.stability = min(1.0, self.stability + 0.05)
            return True
        except Exception as e:
            logger.error(f"Error sending data through socket {self.id}: {str(e)}")
            self.metrics['error_count'] += 1
            self.stability = max(0.0, self.stability - 0.1)
            return False
            
    def receive(self) -> Optional[Any]:
        """Receive data from the socket"""
        if not self.active:
            return None
        try:
            if not self.buffer.empty():
                data = self.buffer.get_nowait()
                self.last_activity = datetime.now()
                self.stability = min(1.0, self.stability + 0.05)
                return data
            return None
        except Exception as e:
            logger.error(f"Error receiving data from socket {self.id}: {str(e)}")
            self.metrics['error_count'] += 1
            self.stability = max(0.0, self.stability - 0.1)
            return None

class NeuralSeed:
    """
    Enhanced Neural Seed with advanced growth capabilities.
    Implements non-linear growth patterns and stability monitoring.
    """
    
    # Growth stage thresholds with detailed characteristics
    GROWTH_STAGES = {
        'seed': {
            'threshold': 0.0,
            'max_components': 2,
            'growth_rate': 0.1,
            'complexity_limit': 0.3,
            'description': 'Initial growth phase with basic pattern formation'
        },
        'sprout': {
            'threshold': 0.3,
            'max_components': 5,
            'growth_rate': 0.2,
            'complexity_limit': 0.6,
            'description': 'Accelerated growth with pattern expansion'
        },
        'sapling': {
            'threshold': 0.6,
            'max_components': 8,
            'growth_rate': 0.3,
            'complexity_limit': 0.8,
            'description': 'Complex pattern development with full component activation'
        },
        'mature': {
            'threshold': 0.9,
            'max_components': 10,
            'growth_rate': 0.4,
            'complexity_limit': 1.0,
            'description': 'Optimized growth with maximum system capacity'
        }
    }
    
    # Stability thresholds with detailed behaviors
    STABILITY_THRESHOLDS = {
        'unstable': {
            'threshold': 0.5,
            'growth_multiplier': 0.5,
            'component_activation': False,
            'description': 'Growth paused, components may deactivate'
        },
        'moderate': {
            'threshold': 0.7,
            'growth_multiplier': 0.8,
            'component_activation': True,
            'description': 'Limited component operation with reduced growth rate'
        },
        'stable': {
            'threshold': 0.8,
            'growth_multiplier': 1.2,
            'component_activation': True,
            'description': 'Full component operation with maximum growth potential'
        }
    }
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.state = {
            'growth_rate': 0.0,
            'stability': 1.0,
            'complexity': 0.0,
            'consciousness_level': 0.0,
            'stage': 'seed',
            'age': 0,
            'dictionary_size': 100,
            'active_components': set(),
            'dormant_components': set(),
            'connection_stability': 1.0,
            'bridge_count': 0,
            'last_growth_update': datetime.now(),
            'stability_history': [],
            'growth_history': [],
            'complexity_history': [],
            'stage_history': [],
            'component_history': [],
            'growth_pattern': 'linear',  # linear, exponential, or logistic
            'last_stage_transition': datetime.now(),
            'stage_duration': 0,
            'growth_acceleration': 0.0,
            'pattern_coherence': 1.0
        }
        self.metrics = {
            'growth_history': [],
            'stability_history': [],
            'complexity_history': [],
            'connection_history': [],
            'bridge_history': [],
            'data_transferred': 0,
            'last_transfer': 0.0,
            'component_stability': {},
            'bridge_stability': {},
            'stage_metrics': {
                stage: {
                    'duration': 0,
                    'growth_rate': 0.0,
                    'stability_avg': 0.0,
                    'component_count': 0
                }
                for stage in self.GROWTH_STAGES.keys()
            },
            'pattern_metrics': {
                'coherence': 1.0,
                'adaptation_rate': 0.0,
                'complexity_trend': 'stable'
            }
        }
        self.dictionary = {}
        self.sockets: Dict[str, ConnectionSocket] = {}
        self.bridges: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.last_update = datetime.now()
        self._socket_thread = None
        self._bridge_thread = None
        self._growth_thread = None
        self._monitor_thread = None
        
        # Initialize component stability tracking
        self._initialize_component_stability()
        
    def _initialize_component_stability(self):
        """Initialize component stability tracking"""
        self.metrics['component_stability'] = {
            'consciousness': 1.0,
            'linguistic': 1.0,
            'neural_plasticity': 1.0
        }
        
    def create_socket(self, socket_type: str = "input") -> Optional[str]:
        """Create a new connection socket"""
        if socket_type not in ["input", "output"]:
            return None
        socket_id = f"socket_{len(self.sockets)}"
        self.sockets[socket_id] = ConnectionSocket(socket_id, socket_type)
        return socket_id
        
    def remove_socket(self, socket_id: str) -> bool:
        """Remove a connection socket"""
        if socket_id in self.sockets:
            # Disconnect all connections
            for target_id in self.sockets[socket_id].connections.copy():
                self.disconnect_sockets(socket_id, target_id)
            del self.sockets[socket_id]
            return True
        return False
        
    def connect_sockets(self, source_id: str, target_id: str) -> bool:
        """Connect two sockets"""
        if (source_id in self.sockets and 
            target_id in self.sockets and
            self.sockets[source_id].type == "output" and
            self.sockets[target_id].type == "input"):
            return self.sockets[source_id].connect(target_id)
        return False
        
    def disconnect_sockets(self, source_id: str, target_id: str) -> bool:
        """Disconnect two sockets"""
        if source_id in self.sockets and target_id in self.sockets:
            return self.sockets[source_id].disconnect(target_id)
        return False
        
    def create_bridge(self, 
                     source_socket_id: str, 
                     target_seed_id: str,
                     target_socket_id: str,
                     bridge_type: str = "direct") -> Optional[str]:
        """Create a bridge between this seed and another"""
        if (source_socket_id not in self.sockets or
            self.sockets[source_socket_id].type != "output"):
            return None
            
        bridge_id = f"bridge_{len(self.bridges)}"
        self.bridges[bridge_id] = {
            'source_socket': source_socket_id,
            'target_seed': target_seed_id,
            'target_socket': target_socket_id,
            'type': bridge_type,
            'stability': 1.0,
            'created_at': datetime.now().isoformat(),
            'metrics': {
                'data_transferred': 0,
                'last_transfer': 0.0
            }
        }
        self.state['bridge_count'] += 1
        return bridge_id
        
    def remove_bridge(self, bridge_id: str) -> bool:
        """Remove a bridge"""
        if bridge_id in self.bridges:
            del self.bridges[bridge_id]
            self.state['bridge_count'] -= 1
            return True
        return False
        
    def _process_socket_connections(self):
        """Process socket connections in a separate thread"""
        while self.running:
            for socket_id, socket in self.sockets.items():
                if socket.active and socket.type == "output":
                    # Process outgoing connections
                    for target_id in socket.connections:
                        try:
                            if target_id in self.sockets and self.sockets[target_id].active:
                                # Get data from output socket's buffer
                                data = socket.buffer.get_nowait() if not socket.buffer.empty() else None
                                if data:
                                    # Send data to input socket
                                    self.sockets[target_id].buffer.put(data)
                                    # Update metrics
                                    self.metrics['data_transferred'] += 1
                                    self.metrics['last_transfer'] = time.time()
                        except queue.Empty:
                            pass
                        except Exception as e:
                            logger.error(f"Error processing socket {socket_id}: {str(e)}")
            # Sleep to prevent CPU overuse
            time.sleep(0.01)
            
    def _process_bridges(self):
        """Process data transfer through bridges"""
        while self.running:
            for bridge_id, bridge in self.bridges.items():
                try:
                    # Calculate bridge stability
                    bridge['stability'] = self._calculate_bridge_stability(bridge)
                    
                    # Process data transfer through bridge
                    source_socket = self.sockets[bridge['source_socket']]
                    target_socket = self.sockets[bridge['target_socket']]
                    
                    # Receive data from source socket
                    data = source_socket.receive()
                    if data:
                        # Transfer data through bridge
                        target_socket.send(data)
                        bridge['metrics']['data_transferred'] += 1
                        bridge['metrics']['last_transfer'] = time.time()
                        
                        # Update bridge stability based on successful transfer
                        bridge['stability'] = min(1.0, bridge['stability'] + 0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing bridge {bridge_id}: {str(e)}")
                    bridge['stability'] = max(0.0, bridge['stability'] - 0.2)
                    
                # Sleep to prevent CPU overuse
                time.sleep(0.01)
                    
    def _calculate_bridge_stability(self, bridge: Dict[str, Any]) -> float:
        """Calculate stability of a bridge connection"""
        # Consider multiple factors for bridge stability
        age = (datetime.now() - datetime.fromisoformat(bridge['created_at'])).total_seconds()
        age_factor = math.exp(-age / 3600)  # Decay over time
        
        # Activity factor based on data transfer
        activity_factor = 1.0
        if bridge['metrics']['data_transferred'] > 0:
            time_since_last = time.time() - bridge['metrics']['last_transfer']
            activity_factor = math.exp(-time_since_last / 3600)
            
        # Error factor
        error_factor = 1.0
        if 'error_count' in bridge['metrics']:
            error_factor = max(0.1, 1.0 - (bridge['metrics']['error_count'] * 0.1))
            
        return min(1.0, age_factor * activity_factor * error_factor)
        
    def start_growth(self, initial_stability: float = 1.0):
        """Start the growth process"""
        if not self.running:
            self.running = True
            self.state['stability'] = initial_stability
            
            # Start socket processing thread
            self._socket_thread = threading.Thread(target=self._process_socket_connections)
            self._socket_thread.daemon = True
            self._socket_thread.start()
            
            # Start bridge processing thread
            self._bridge_thread = threading.Thread(target=self._process_bridges)
            self._bridge_thread.daemon = True
            self._bridge_thread.start()
            
            # Start growth loop in a separate thread
            self._growth_thread = threading.Thread(target=self._growth_loop)
            self._growth_thread.daemon = True
            self._growth_thread.start()
            
            # Start monitoring thread
            self._monitor_thread = threading.Thread(target=self._monitor_system)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
    def stop_growth(self):
        """Stop the growth process"""
        self.running = False
        if self._socket_thread:
            self._socket_thread.join(timeout=1.0)
        if self._bridge_thread:
            self._bridge_thread.join(timeout=1.0)
        if self._growth_thread:
            self._growth_thread.join(timeout=1.0)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
    def _growth_loop(self):
        """Main growth loop that runs in a separate thread"""
        try:
            while self.running:
                # Monitor stability and adjust growth
                is_stable = self._monitor_growth_stability()
                
                if is_stable:
                    # Calculate growth parameters
                    growth_factor = self._calculate_growth_factor()
                    complexity_increase = self._calculate_complexity_increase()
                    
                    # Apply growth and update complexity
                    self._apply_growth(growth_factor)
                    self._update_complexity(complexity_increase)
                    
                    # Update consciousness level
                    self._update_consciousness_level()
                    
                    # Check for stage transitions
                    self._check_stage_transition()
                    
                    # Adapt dictionary size based on complexity
                    self._adapt_dictionary_size()
                    
                    # Update metrics
                    self._update_metrics()
                    
                    # Update age
                    self.state['age'] += 1
                    
                else:
                    # If unstable, focus on stabilization
                    logger.warning("System unstable - focusing on stabilization")
                    time.sleep(2)  # Wait longer between iterations when unstable
                    
                    # Deactivate unstable components
                    for component in list(self.state['active_components']):
                        if not self._is_component_stable(component):
                            self.deactivate_component(component)
                    
                # Brief sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in growth loop: {str(e)}")
            self.running = False
            
    def _monitor_system(self):
        """Monitor system health and stability"""
        while self.running:
            try:
                # Update component stability
                for component in self.state['active_components']:
                    self.metrics['component_stability'][component] = self._calculate_component_stability(component)
                
                # Update bridge stability
                for bridge_id, bridge in self.bridges.items():
                    self.metrics['bridge_stability'][bridge_id] = self._calculate_bridge_stability(bridge)
                
                # Check for unstable components
                for component, stability in self.metrics['component_stability'].items():
                    if stability < self.STABILITY_THRESHOLDS['unstable']['threshold']:
                        self.deactivate_component(component)
                
                # Update overall stability
                self._check_stability()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(5.0)  # Wait longer on error
                
    def _calculate_component_stability(self, component: str) -> float:
        """Calculate stability for a specific component"""
        if component not in self.state['active_components']:
            return 0.0
            
        # Base stability on component age and activity
        age = (datetime.now() - self.last_update).total_seconds()
        age_factor = math.exp(-age / 3600)  # Decay over time
        
        # Activity factor based on recent operations
        activity_factor = 1.0
        if component in self.metrics['component_stability']:
            activity_factor = self.metrics['component_stability'][component]
            
        return min(1.0, age_factor * activity_factor)
        
    def _calculate_growth_factor(self) -> float:
        """Calculate non-linear growth factor with pattern analysis"""
        current_stage = self.GROWTH_STAGES[self.state['stage']]
        
        # Base growth factor from current stage
        base_factor = current_stage['growth_rate']
        
        # Age-based growth (logarithmic scaling with acceleration)
        age_factor = math.log(self.state['age'] + 1) / 10
        self.state['growth_acceleration'] = age_factor * 0.1
        
        # Complexity penalty (adaptive scaling)
        complexity_limit = current_stage['complexity_limit']
        complexity_ratio = self.state['complexity'] / complexity_limit
        complexity_penalty = math.exp(complexity_ratio - 1)  # Exponential penalty near limit
        
        # Stability boost with pattern analysis
        stability_boost = 1.0
        if self.state['stability'] > self.STABILITY_THRESHOLDS['stable']['threshold']:
            stability_boost = self.STABILITY_THRESHOLDS['stable']['growth_multiplier']
        elif self.state['stability'] < self.STABILITY_THRESHOLDS['unstable']['threshold']:
            stability_boost = self.STABILITY_THRESHOLDS['unstable']['growth_multiplier']
            
        # Pattern coherence factor
        pattern_factor = self.state['pattern_coherence']
        
        # Calculate final growth factor with all components
        growth_factor = (
            base_factor * 
            (1 + age_factor) * 
            (1 - complexity_penalty) * 
            stability_boost * 
            pattern_factor
        )
        
        # Update growth pattern based on recent history
        self._update_growth_pattern(growth_factor)
        
        return max(0.0, min(1.0, growth_factor))
        
    def _update_growth_pattern(self, current_growth: float):
        """Update growth pattern analysis"""
        history = self.metrics['growth_history'][-10:]  # Last 10 growth rates
        if len(history) >= 3:
            # Calculate trend
            changes = [history[i] - history[i-1] for i in range(1, len(history))]
            avg_change = sum(changes) / len(changes)
            
            # Determine pattern
            if abs(avg_change) < 0.01:
                self.state['growth_pattern'] = 'linear'
            elif avg_change > 0:
                self.state['growth_pattern'] = 'exponential'
            else:
                self.state['growth_pattern'] = 'logistic'
                
            # Update pattern coherence
            self.state['pattern_coherence'] = self._calculate_pattern_coherence(history)
            
    def _calculate_pattern_coherence(self, history: List[float]) -> float:
        """Calculate how well the growth follows its current pattern"""
        if len(history) < 3:
            return 1.0
            
        pattern = self.state['growth_pattern']
        changes = [history[i] - history[i-1] for i in range(1, len(history))]
        
        if pattern == 'linear':
            # Check variance in changes
            variance = sum((x - sum(changes)/len(changes)) ** 2 for x in changes) / len(changes)
            return math.exp(-variance)
        elif pattern == 'exponential':
            # Check ratio consistency
            ratios = [history[i]/history[i-1] for i in range(1, len(history))]
            variance = sum((x - sum(ratios)/len(ratios)) ** 2 for x in ratios) / len(ratios)
            return math.exp(-variance)
        else:  # logistic
            # Check convergence
            max_val = max(history)
            distances = [abs(x - max_val) for x in history[-3:]]
            return 1.0 - (sum(distances) / (3 * max_val))
            
    def _calculate_complexity_increase(self) -> float:
        """Calculate complexity increase based on growth"""
        return random.uniform(0.01, 0.05) * self.state['growth_rate']
        
    def _apply_growth(self, growth_factor: float):
        """Apply growth to the system"""
        consciousness_increase = growth_factor * (1 - self.state['consciousness_level'])
        self.state['consciousness_level'] = min(
            1.0,
            self.state['consciousness_level'] + consciousness_increase
        )
        self.state['growth_rate'] = growth_factor
        
    def _update_complexity(self, increase: float):
        """Update system complexity"""
        self.state['complexity'] = min(1.0, self.state['complexity'] + increase)
        
    def _update_metrics(self):
        """Update system metrics"""
        metrics_map = {
            'growth_rate': 'growth_history',
            'stability': 'stability_history',
            'complexity': 'complexity_history'
        }
        for state_key, metric_key in metrics_map.items():
            self.metrics[metric_key].append(self.state[state_key])
            # Keep only last 10 entries
            self.metrics[metric_key] = self.metrics[metric_key][-10:]
            
    def _check_stage_transition(self):
        """Check and update growth stage"""
        consciousness = self.state['consciousness_level']
        current_stage = self.state['stage']
        
        for stage, details in sorted(
            self.GROWTH_STAGES.items(),
            key=lambda x: x[1]['threshold'],
            reverse=True
        ):
            if consciousness >= details['threshold']:
                if stage != current_stage:
                    logger.info(f"Transitioning from {current_stage} to {stage}")
                    self.state['stage'] = stage
                    self.state['stage_history'].append(stage)
                    self.state['stage_duration'] = (datetime.now() - self.state['last_stage_transition']).total_seconds()
                    self.state['last_stage_transition'] = datetime.now()
                break
                
    def _adapt_dictionary_size(self):
        """Adapt dictionary size with enhanced scaling"""
        current_stage = self.GROWTH_STAGES[self.state['stage']]
        
        # Base size with stage-based scaling
        base_size = 100 * (1 + current_stage['growth_rate'])
        
        # Consciousness-based scaling (adaptive square root)
        consciousness_factor = math.sqrt(self.state['consciousness_level'])
        if self.state['growth_pattern'] == 'exponential':
            consciousness_factor *= 1.2  # Boost for exponential growth
            
        # Complexity-based scaling (adaptive logarithmic)
        complexity_factor = math.log(self.state['complexity'] + 1) + 1
        if self.state['stability'] < self.STABILITY_THRESHOLDS['moderate']['threshold']:
            complexity_factor *= 0.8  # Reduce during instability
            
        # Pattern coherence factor
        pattern_factor = self.state['pattern_coherence']
        
        # Calculate new size with all factors
        new_size = int(base_size * consciousness_factor * complexity_factor * pattern_factor)
        
        # Update dictionary size
        self.state['dictionary_size'] = new_size
        
        # Manage dictionary content
        self._manage_dictionary_content(new_size)
        
    def _manage_dictionary_content(self, target_size: int):
        """Manage dictionary content based on usage and importance"""
        current_size = len(self.dictionary)
        
        if current_size > target_size:
            # Calculate word importance scores
            word_scores = {}
            for word, embedding in self.dictionary.items():
                # Score based on usage frequency and embedding magnitude
                usage_score = self._calculate_word_usage_score(word)
                magnitude_score = math.sqrt(sum(x**2 for x in embedding))
                word_scores[word] = (usage_score + magnitude_score) / 2
                
            # Remove least important words
            excess = current_size - target_size
            words_to_remove = sorted(
                word_scores.items(),
                key=lambda x: x[1]
            )[:excess]
            
            for word, _ in words_to_remove:
                del self.dictionary[word]
                logger.info(f"Removed word from dictionary: {word}")
                
    def _calculate_word_usage_score(self, word: str) -> float:
        """Calculate usage score for a word"""
        # Placeholder for actual usage tracking
        # In a real implementation, this would track word usage frequency
        return random.uniform(0.0, 1.0)  # Temporary random score
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the neural seed"""
        return {
            'id': self.id,
            'state': self.state.copy(),
            'metrics': {
                k: v[-100:] if isinstance(v, list) else v
                for k, v in self.metrics.items()
            },
            'sockets': {
                socket_id: {
                    'type': socket.type,
                    'connections': list(socket.connections),
                    'active': socket.active,
                    'stability': socket.stability,
                    'metrics': socket.metrics
                }
                for socket_id, socket in self.sockets.items()
            },
            'bridges': {
                bridge_id: {
                    **bridge,
                    'stability': self.metrics['bridge_stability'].get(bridge_id, 0.0)
                }
                for bridge_id, bridge in self.bridges.items()
            },
            'components': {
                'active': list(self.state['active_components']),
                'dormant': list(self.state['dormant_components']),
                'stability': self.metrics['component_stability']
            }
        }
        
    def add_word(self, word: str, embedding: List[float]) -> bool:
        """Add a word to the dictionary"""
        if len(self.dictionary) >= self.state['dictionary_size']:
            logger.warning("Dictionary size limit reached")
            return False
        self.dictionary[word] = embedding
        logger.info(f"Added word to dictionary: {word}")
        return True
        
    def activate_component(self, component: str) -> bool:
        """Activate a component if system is stable enough"""
        if self.state['stability'] >= self.STABILITY_THRESHOLDS['moderate']['threshold']:
            self.state['active_components'].add(component)
            self.state['dormant_components'].discard(component)
            self.metrics['component_stability'][component] = 1.0
            logger.info(f"Activated component: {component}")
            return True
        logger.warning(f"Cannot activate component {component}: system not stable enough")
        return False
        
    def deactivate_component(self, component: str):
        """Deactivate a component"""
        if component in self.state['active_components']:
            self.state['active_components'].discard(component)
            self.state['dormant_components'].add(component)
            self.metrics['component_stability'][component] = 0.0
            logger.info(f"Deactivated component: {component}")
            
    def connect_to_consciousness(self, consciousness_node):
        """Connect to the consciousness node"""
        try:
            if self.state['stability'] < self.STABILITY_THRESHOLDS['moderate']['threshold']:
                logger.warning("System not stable enough for consciousness connection")
                return False
                
            # Create output socket for consciousness connection
            socket_id = self.create_socket("output")
            if socket_id:
                # Store consciousness node reference
                self.consciousness_node = consciousness_node
                if self.activate_component('consciousness'):
                    logger.info(f"Connected to consciousness node via socket {socket_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to consciousness node: {str(e)}")
            return False

    def connect_to_linguistic_processor(self, linguistic_processor):
        """Connect to the linguistic processor"""
        try:
            if self.state['stability'] < self.STABILITY_THRESHOLDS['moderate']['threshold']:
                logger.warning("System not stable enough for linguistic processor connection")
                return False
                
            # Create input/output socket pair for linguistic processing
            input_socket = self.create_socket("input")
            output_socket = self.create_socket("output")
            if input_socket and output_socket:
                # Store linguistic processor reference
                self.linguistic_processor = linguistic_processor
                if self.activate_component('linguistic'):
                    logger.info(f"Connected to linguistic processor via sockets {input_socket}, {output_socket}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to linguistic processor: {str(e)}")
            return False

    def connect_to_neural_plasticity(self, neural_processor):
        """Connect to the neural plasticity processor"""
        try:
            if self.state['stability'] < self.STABILITY_THRESHOLDS['moderate']['threshold']:
                logger.warning("System not stable enough for neural plasticity connection")
                return False
                
            # Create bidirectional connection for neural plasticity
            plasticity_socket = self.create_socket("output")
            if plasticity_socket:
                # Store neural processor reference
                self.neural_processor = neural_processor
                if self.activate_component('neural_plasticity'):
                    logger.info(f"Connected to neural plasticity via socket {plasticity_socket}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to neural plasticity: {str(e)}")
            return False

    def _monitor_growth_stability(self):
        """Monitor growth stability with enhanced analysis"""
        try:
            # Calculate stability components
            component_stability = self._calculate_component_stability()
            growth_stability = self._calculate_growth_stability()
            complexity_stability = self._calculate_complexity_stability()
            pattern_stability = self._calculate_pattern_stability()
            
            # Weighted stability calculation
            weights = {
                'component': 0.4,
                'growth': 0.3,
                'complexity': 0.2,
                'pattern': 0.1
            }
            
            stability = (
                weights['component'] * component_stability +
                weights['growth'] * growth_stability +
                weights['complexity'] * complexity_stability +
                weights['pattern'] * pattern_stability
            )
            
            # Update state
            self.state['stability'] = stability
            
            # Adjust growth based on stability level
            current_stage = self.GROWTH_STAGES[self.state['stage']]
            max_components = current_stage['max_components']
            
            if stability < self.STABILITY_THRESHOLDS['unstable']['threshold']:
                # Reduce active components if exceeding stage limit
                while len(self.state['active_components']) > max_components * 0.5:
                    self.deactivate_component(next(iter(self.state['active_components'])))
                    
            # Update metrics
            self.metrics['stability_history'].append(stability)
            self._update_stage_metrics()
            
            return stability >= self.STABILITY_THRESHOLDS['unstable']['threshold']
            
        except Exception as e:
            logger.error(f"Error monitoring growth stability: {str(e)}")
            return False
            
    def _update_stage_metrics(self):
        """Update metrics for current growth stage"""
        current_stage = self.state['stage']
        stage_metrics = self.metrics['stage_metrics'][current_stage]
        
        # Update duration
        stage_metrics['duration'] = (datetime.now() - self.state['last_stage_transition']).total_seconds()
        
        # Update average growth rate
        recent_growth = self.metrics['growth_history'][-10:]
        if recent_growth:
            stage_metrics['growth_rate'] = sum(recent_growth) / len(recent_growth)
            
        # Update average stability
        recent_stability = self.metrics['stability_history'][-10:]
        if recent_stability:
            stage_metrics['stability_avg'] = sum(recent_stability) / len(recent_stability)
            
        # Update component count
        stage_metrics['component_count'] = len(self.state['active_components'])
        
    def _calculate_pattern_stability(self) -> float:
        """Calculate stability based on growth pattern coherence"""
        pattern = self.state['growth_pattern']
        coherence = self.state['pattern_coherence']
        
        # Pattern-specific stability factors
        if pattern == 'linear':
            return coherence * 0.8  # Linear growth is generally more stable
        elif pattern == 'exponential':
            return coherence * 0.6  # Exponential growth can be less stable
        else:  # logistic
            return coherence * 0.9  # Logistic growth is generally stable
            
    def _update_consciousness_level(self):
        """Update consciousness level based on active components and stability"""
        try:
            # Base consciousness on number of active components and their stability
            component_count = len(self.state['active_components'])
            stability_factor = self.state['stability']
            complexity_factor = self.state['complexity']
            
            # Calculate consciousness level (0.0 to 1.0)
            self.state['consciousness_level'] = min(1.0, (
                0.4 * (component_count / 10) +  # Component factor
                0.3 * stability_factor +        # Stability factor
                0.3 * complexity_factor         # Complexity factor
            ))
            
            # Notify consciousness node if connected
            if hasattr(self, 'consciousness_node'):
                self.consciousness_node.update_consciousness_level(
                    self.state['consciousness_level']
                )
                
        except Exception as e:
            logger.error(f"Error updating consciousness level: {str(e)}") 