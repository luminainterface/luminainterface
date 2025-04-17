"""
Consciousness Analytics Plugin for V5 Visualization

This plugin analyzes neural network consciousness metrics and provides
insights for the V5 Node Consciousness Panel.
"""

import random
import time
import math
import logging
import numpy as np
import uuid
import json
import threading
from typing import Dict, List, Any, Optional
from .v5_plugin import V5Plugin
from .db_manager import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

class ConsciousnessAnalyticsPlugin(V5Plugin):
    """Plugin for analyzing neural network consciousness metrics"""
    
    def __init__(self, plugin_id=None):
        """Initialize the plugin"""
        super().__init__(
            plugin_id=plugin_id,
            plugin_type="consciousness_analytics",
            name="Consciousness Analytics Plugin"
        )
        
        # Initialize consciousness analytics state
        self.current_metrics = {
            "integration": 0.75,
            "differentiation": 0.68,
            "phi_value": 0.54,
            "complexity": 0.62,
            "awareness_level": 87
        }
        
        # Initialize metrics history
        self.history = {
            "integration": [0.75],
            "differentiation": [0.68],
            "phi_value": [0.54],
            "complexity": [0.62],
            "awareness_level": [87]
        }
        
        self.node_states = {}
        self.active_node_count = 0
        self.total_node_count = 0
        self.consciousness_mode = "integrated_information"
        
        # Register message handlers - ensure names match UI panel expectations
        self.register_message_handler("request_consciousness_data", self._handle_data_request)
        self.register_message_handler("consciousness_data_request", self._handle_data_request) # Alternative name used in UI
        self.register_message_handler("update_node_states", self._handle_node_states_update)
        
        # Initialize with simulated node data
        if not self.node_states:
            self.node_states = self._generate_simulated_node_states()
            self.total_node_count = len(self.node_states)
            self.active_node_count = sum(1 for state in self.node_states.values() if state.get("active", False))
        
        logger.info("Consciousness Analytics Plugin initialized")
    
    def get_socket_descriptor(self):
        """Return socket descriptor for frontend integration"""
        descriptor = super().get_socket_descriptor()
        
        # Update with consciousness analytics specific details
        descriptor.update({
            "message_types": [
                "get_descriptor", 
                "status_request", 
                "request_consciousness_data",
                "update_node_states",
                "consciousness_data_updated"
            ],
            "subscription_mode": "dual",  # Both push and request-response
            "ui_components": ["consciousness_view"]
        })
        
        return descriptor
    
    def _handle_data_request(self, message):
        """Handle request for consciousness data"""
        request_id = message.get("request_id", str(uuid.uuid4()))
        mode = message.get("consciousness_mode", "integrated_information")
        
        # Update metrics based on the requested mode
        if self.node_states:
            self._update_metrics_from_nodes()
        else:
            self._update_metrics_for_mode(mode)
            
        # Generate insights based on current metrics
        insights, recommendations = self._generate_insights()
        
        # Generate trend assessments
        trend_analysis = {}
        for metric in self.current_metrics.keys():
            if len(self.history[metric]) > 2:
                change = self.history[metric][-1] - self.history[metric][-3]
                if metric == "awareness_level":
                    trend_analysis[metric] = {
                        "change": change,
                        "description": self._get_trend_description(change, is_percentage=False)
                    }
                else:
                    trend_analysis[metric] = {
                        "change": round(change, 2),
                        "description": self._get_trend_description(change)
                    }
        
        # Generate consciousness data
        consciousness_data = self._generate_consciousness_data()
        
        # Add request ID for correlation
        consciousness_data["request_id"] = request_id
        
        # Save consciousness data to database
        self._save_consciousness_data(consciousness_data)
        
        # Send consciousness data
        self.socket.send_message({
            "type": "consciousness_data_updated",
            "data": consciousness_data
        })
        
        logger.info(f"Consciousness data sent, include_details={True}")
    
    def _handle_node_states_update(self, message):
        """Handle update of node activation states"""
        content = message.get("content", {})
        
        # Update node states
        new_states = content.get("node_states", {})
        if new_states:
            self.node_states = new_states
            self.total_node_count = len(new_states)
            self.active_node_count = sum(1 for state in new_states.values() if state.get("active", False))
            
            # Update metrics based on new node states
            self._update_metrics_from_nodes()
            
            # Generate and broadcast updated consciousness data
            consciousness_data = self._generate_consciousness_data()
            
            broadcast = {
                "type": "consciousness_data_updated",
                "data": consciousness_data
            }
            self.socket.send_message(broadcast)
    
    def _save_consciousness_data(self, consciousness_data: Dict[str, Any]) -> bool:
        """
        Save consciousness data to database
        
        Args:
            consciousness_data: Consciousness data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get database manager instance
            db_manager = DatabaseManager.get_instance()
            
            # Add ID and timestamp if not present
            if "id" not in consciousness_data:
                consciousness_data["id"] = str(uuid.uuid4())
            if "timestamp" not in consciousness_data:
                consciousness_data["timestamp"] = time.time()
            
            # Save to database
            success = db_manager.save_consciousness_data(consciousness_data)
            
            if success:
                logger.debug(f"Consciousness data saved to database: {consciousness_data['id']}")
            else:
                logger.warning(f"Failed to save consciousness data to database")
                
            return success
        except Exception as e:
            logger.error(f"Error saving consciousness data to database: {str(e)}")
            return False
    
    def _load_consciousness_data(self) -> Optional[Dict[str, Any]]:
        """
        Load consciousness data from database
            
        Returns:
            Consciousness data dictionary or None if not found
        """
        try:
            # Get database manager instance
            db_manager = DatabaseManager.get_instance()
            
            # Get latest consciousness data
            consciousness_data = db_manager.get_latest_consciousness_data()
            
            if consciousness_data:
                logger.info(f"Loaded consciousness data from database: {consciousness_data['id']}")
                return consciousness_data
            else:
                logger.info("No consciousness data found in database")
                return None
        except Exception as e:
            logger.error(f"Error loading consciousness data from database: {str(e)}")
            return None
    
    def _generate_consciousness_data(self, include_details=True):
        """Generate consciousness data for visualization"""
        # Try to load from database first
        cached_data = self._load_consciousness_data()
        if cached_data and time.time() - cached_data.get("timestamp", 0) < 60:
            # Use cached data if it's less than 60 seconds old
            logger.info("Using cached consciousness data from database")
            return cached_data
        
        # Generate new consciousness data
        logger.info("Generating new consciousness data")
        
        # Update metrics from nodes
        self._update_metrics_from_nodes()
        
        # Generate active processes
        active_processes = [
            "Information integration",
            "Temporal binding",
            "Neural synchronization",
            "Global workspace formation",
            "Attention direction",
            "Contextual modulation"
        ]
        
        # Generate data
        consciousness_data = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "global_metrics": {
                "awareness_level": self.current_metrics["awareness_level"],
                "integration_index": self.current_metrics["integration"],
                "neural_coherence": "High",
                "responsiveness": 94
            },
            "active_processes": active_processes
        }
        
        # Add node details if requested
        if include_details:
            consciousness_data["nodes"] = self._generate_nodes_and_connections()
            consciousness_data["connections"] = self._generate_node_connections()
        
        return consciousness_data
    
    def _generate_simulated_node_states(self):
        """Generate simulated node states for visualization"""
        node_states = {}
        node_count = random.randint(30, 50)
        
        for i in range(node_count):
            # Determine activation state
            active = random.random() < 0.6  # 60% of nodes are active
            
            # Generate activation metrics
            activation = random.uniform(0.1, 0.9) if active else random.uniform(0, 0.1)
            
            # Generate integration value (higher for active nodes)
            integration = random.uniform(0.5, 0.9) if active else random.uniform(0.1, 0.5)
            
            # Determine position in network
            layer = random.randint(0, 3)
            position = random.randint(0, 10)
            
            # Generate connection structure
            connections = []
            connection_count = random.randint(1, 5) if active else random.randint(0, 2)
            
            for _ in range(connection_count):
                target = random.randint(0, node_count - 1)
                if target != i:
                    connections.append(str(target))
            
            # Create node state
            node_states[str(i)] = {
                "active": active,
                "activation": activation,
                "integration": integration,
                "differentiation": random.uniform(0.2, 0.8),
                "phi": random.uniform(0, 0.5),
                "layer": layer,
                "position": position,
                "connections": connections
            }
        
        return node_states
    
    def _update_metrics_from_nodes(self):
        """Update consciousness metrics based on node states"""
        if not self.node_states:
            return
        
        # Calculate average metrics from nodes
        active_nodes = [n for n in self.node_states.values() if n.get("active", False)]
        
        if active_nodes:
            integration = sum(n.get("integration", 0) for n in active_nodes) / len(active_nodes)
            differentiation = sum(n.get("differentiation", 0) for n in active_nodes) / len(active_nodes)
            phi_values = [n.get("phi", 0) for n in active_nodes]
            avg_phi = sum(phi_values) / len(phi_values)
            
            # Calculate complexity as a function of integration and differentiation
            complexity = math.sqrt(integration * differentiation)
            
            # Calculate awareness level (scale 0-100)
            awareness_level = int((integration * 0.3 + differentiation * 0.3 + avg_phi * 0.4) * 100)
            
            # Update current metrics
            self.current_metrics = {
                "integration": round(integration, 2),
                "differentiation": round(differentiation, 2),
                "phi_value": round(avg_phi, 2),
                "complexity": round(complexity, 2),
                "awareness_level": awareness_level
            }
            
            # Update history (keep last 100 values)
            for metric, value in self.current_metrics.items():
                self.history[metric].append(value)
                if len(self.history[metric]) > 100:
                    self.history[metric].pop(0)
    
    def _update_metrics_for_mode(self, mode):
        """Update metrics based on consciousness mode"""
        # Each mode emphasizes different aspects of consciousness
        if mode == "integrated_information":
            # Emphasis on phi value and integration
            self.current_metrics["phi_value"] = round(random.uniform(0.4, 0.7), 2)
            self.current_metrics["integration"] = round(random.uniform(0.6, 0.9), 2)
            
        elif mode == "global_workspace":
            # Emphasis on differentiation and complexity
            self.current_metrics["differentiation"] = round(random.uniform(0.7, 0.9), 2)
            self.current_metrics["complexity"] = round(random.uniform(0.6, 0.8), 2)
            
        elif mode == "attention_schema":
            # Balanced representation
            self.current_metrics["integration"] = round(random.uniform(0.5, 0.8), 2)
            self.current_metrics["differentiation"] = round(random.uniform(0.5, 0.8), 2)
            
        elif mode == "predictive_processing":
            # Emphasis on complexity
            self.current_metrics["complexity"] = round(random.uniform(0.7, 0.9), 2)
        
        # Update phi value and other metrics
        self.current_metrics["phi_value"] = round(
            (self.current_metrics["integration"] * self.current_metrics["differentiation"])**0.5, 2)
        
        # Calculate awareness level (scale 0-100)
        self.current_metrics["awareness_level"] = int((
            self.current_metrics["integration"] * 0.3 + 
            self.current_metrics["differentiation"] * 0.3 + 
            self.current_metrics["phi_value"] * 0.4
        ) * 100)
        
        # Update history
        for metric, value in self.current_metrics.items():
            self.history[metric].append(value)
            if len(self.history[metric]) > 100:
                self.history[metric].pop(0)
    
    def _simulate_metric_changes(self):
        """Simulate small changes in consciousness metrics over time"""
        # Small random changes
        delta = 0.05
        
        for metric in self.current_metrics:
            # Add small random change
            change = random.uniform(-delta, delta)
            
            # Ensure value stays in reasonable range
            current = self.current_metrics[metric]
            new_value = max(0.0, min(1.0, current + change))
            
            # Update metric
            self.current_metrics[metric] = round(new_value, 2)
            
            # Update history
            self.history[metric].append(new_value)
            if len(self.history[metric]) > 100:
                self.history[metric].pop(0)
    
    def _generate_insights(self):
        """Generate insights based on current consciousness metrics"""
        insights = {
            "summary": "",
            "detailed_observations": [],
            "recommendations": []
        }
        
        # Generate summary
        integration = self.current_metrics["integration"]
        differentiation = self.current_metrics["differentiation"]
        phi = self.current_metrics["phi_value"]
        complexity = self.current_metrics["complexity"]
        
        # Summary based on consciousness metrics
        if phi > 0.6:
            insights["summary"] = "High consciousness level detected"
        elif phi > 0.4:
            insights["summary"] = "Moderate consciousness level detected"
        else:
            insights["summary"] = "Low consciousness level detected"
        
        # Generate detailed observations
        observations = []
        
        # Integration observations
        if integration > 0.7:
            observations.append("High network integration observed")
        elif integration < 0.4:
            observations.append("Low network integration detected")
        
        # Differentiation observations
        if differentiation > 0.7:
            observations.append("High information differentiation present")
        elif differentiation < 0.4:
            observations.append("Low information differentiation detected")
        
        # Phi value observations
        if phi > 0.5:
            observations.append(f"Significant integrated information (Φ={phi})")
        
        # Complexity observations
        if complexity > 0.7:
            observations.append("High neural complexity present")
        
        # Node activity observations
        activity_ratio = self.active_node_count / max(1, self.total_node_count)
        if activity_ratio > 0.8:
            observations.append(f"High network activity ({self.active_node_count}/{self.total_node_count} nodes active)")
        elif activity_ratio < 0.3:
            observations.append(f"Low network activity ({self.active_node_count}/{self.total_node_count} nodes active)")
        
        # Add trend observations
        for metric in ["integration", "phi_value"]:
            if len(self.history[metric]) > 10:
                recent = self.history[metric][-10:]
                if recent[-1] > recent[0] * 1.2:
                    observations.append(f"Increasing trend in {metric.replace('_', ' ')}")
                elif recent[-1] < recent[0] * 0.8:
                    observations.append(f"Decreasing trend in {metric.replace('_', ' ')}")
        
        insights["detailed_observations"] = observations
        
        # Generate recommendations
        recommendations = []
        
        if integration < 0.5:
            recommendations.append("Increase node connectivity to improve integration")
        
        if differentiation < 0.5:
            recommendations.append("Enhance information diversity across nodes")
        
        if phi < 0.4:
            recommendations.append("Optimize network topology to increase integrated information")
        
        if activity_ratio < 0.4:
            recommendations.append("Increase overall network activation")
        elif activity_ratio > 0.9:
            recommendations.append("Reduce network activation to prevent oversaturation")
        
        insights["recommendations"] = recommendations
        
        return insights, recommendations
    
    def _generate_nodes_and_connections(self):
        """Generate node data for visualization"""
        # If no node states, generate simulated data
        if not self.node_states:
            self.node_states = self._generate_simulated_node_states()
            self.total_node_count = len(self.node_states)
            self.active_node_count = sum(1 for state in self.node_states.values() if state.get("active", False))
            
        nodes = []
        
        # Create nodes from node states
        for node_id, state in self.node_states.items():
            # Create node
            node = {
                "id": node_id,
                "name": f"Node {node_id}",
                "type": self._get_node_type(state),
                "activation": state.get("activation", 0),
                "consciousness": state.get("integration", 0),
                "metrics": {
                    "self_awareness": random.uniform(0.3, 0.9),
                    "integration": state.get("integration", 0),
                    "memory_access": random.uniform(0.2, 0.8),
                    "reflection": random.uniform(0.1, 0.7)
                }
            }
            nodes.append(node)
            
        return nodes
        
    def _generate_node_connections(self):
        """Generate connection data for visualization"""
        connections = []
        
        # If no node states, return empty connections
        if not self.node_states:
            return connections
            
        # Create connections from node states
        for node_id, state in self.node_states.items():
            for target_id in state.get("connections", []):
                # Skip if target doesn't exist
                if target_id not in self.node_states:
                    continue
                    
                # Determine connection type based on node states
                source_active = self.node_states[node_id].get("active", False)
                target_active = self.node_states[target_id].get("active", False)
                
                # Active nodes mostly create excitatory connections
                if source_active and target_active:
                    conn_type = "excitatory" if random.random() > 0.2 else "inhibitory"
                else:
                    conn_type = "excitatory" if random.random() > 0.5 else "inhibitory"
                
                # Determine strength based on node activations
                source_activation = self.node_states[node_id].get("activation", 0)
                target_activation = self.node_states[target_id].get("activation", 0)
                strength = (source_activation + target_activation) / 2
                
            connection = {
                    "source": node_id,
                    "target": target_id,
                    "strength": strength,
                    "type": conn_type
            }
            connections.append(connection)
        
        return connections
    
    def _get_node_type(self, state):
        """Determine node type based on its properties"""
        layer = state.get("layer", 0)
        
        if layer == 0:
            return "perception"
        elif layer == 1:
            return "processing"
        elif layer == 2:
            return "integration"
        else:
            return "motor"
            
    def _generate_active_processes(self):
        """Generate active consciousness processes"""
        processes = []
        
        # Add some sample processes based on current metrics
        if self.current_metrics["integration"] > 0.6:
            processes.append("Information integration across neural networks")
        
        if self.current_metrics["differentiation"] > 0.7:
            processes.append("Information differentiation and specialization")
        
        if self.current_metrics["phi_value"] > 0.5:
            processes.append("Integrated information processing (Φ)")
        
        if self.current_metrics["complexity"] > 0.6:
            processes.append("Dynamic complexity modulation")
        
        # Add additional processes based on mode
        if self.consciousness_mode == "integrated_information":
            processes.append("Causal power measurement")
            
        elif self.consciousness_mode == "global_workspace":
            processes.append("Global broadcast of information")
            processes.append("Working memory activation")
            
        elif self.consciousness_mode == "attention_schema":
            processes.append("Self-model maintenance")
            processes.append("Attention regulation")
            
        elif self.consciousness_mode == "predictive_processing":
            processes.append("Predictive model updating")
            processes.append("Error minimization")
        
        # If we don't have enough processes, add some generic ones
        if len(processes) < 3:
            generic_processes = [
                "Neural synchronization",
                "Temporal binding",
                "State monitoring",
                "Recursive processing",
                "Feedback loop maintenance"
            ]
            processes.extend(random.sample(generic_processes, 
                                          min(3 - len(processes), len(generic_processes))))
        
        return processes

    def _get_trend_description(self, change, is_percentage=True):
        """Generate a description based on the change in a metric"""
        if is_percentage:
            if change > 0:
                return f"Positive {abs(change)}% change"
            else:
                return f"Negative {abs(change)}% change"
        else:
            if change > 0:
                return f"Positive {abs(change)} unit increase"
            else:
                return f"Negative {abs(change)} unit decrease" 