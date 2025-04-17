"""
Pattern Processor Plugin for V5 Visualization

This plugin processes neural patterns and provides fractal visualizations
for the V5 Fractal Pattern Panel.
"""

import random
import time
import math
import logging
import os
import json
import threading
import uuid
from typing import Dict, List, Any, Optional
from .v5_plugin import V5Plugin
from .node_socket import NodeSocket
from .db_manager import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

class PatternProcessorPlugin(V5Plugin):
    """Plugin for processing neural patterns and generating fractal visualizations"""
    
    def __init__(self, plugin_id=None):
        """Initialize the plugin"""
        super().__init__(
            plugin_id=plugin_id,
            plugin_type="pattern_processor",
            name="Pattern Processor Plugin"
        )
        
        # Initialize pattern processing state
        self.current_pattern = None
        self.pattern_style = "neural"
        self.fractal_depth = 5
        self.metrics = {
            "fractal_dimension": 1.68,
            "complexity_index": 78,
            "pattern_coherence": 92,
            "entropy_level": "Medium"
        }
        self.insights = {
            "detected_patterns": [
                "Recursive symmetry", 
                "Bifurcation sequences",
                "Self-similar structures"
            ]
        }
        
        # Register message handlers
        self.register_message_handler("request_pattern_data", self._handle_pattern_request)
        
        logger.info("Pattern Processor Plugin initialized")
    
    def get_socket_descriptor(self):
        """Return socket descriptor for frontend integration"""
        descriptor = super().get_socket_descriptor()
        
        # Update with pattern processor specific details
        descriptor.update({
            "message_types": [
                "get_descriptor", 
                "status_request", 
                "request_pattern_data",
                "pattern_data_updated"
            ],
            "subscription_mode": "dual",  # Both push and request-response
            "ui_components": ["fractal_view"]
        })
        
        return descriptor
    
    def _handle_pattern_request(self, message):
        """Handle request for pattern data"""
        # Extract request parameters
        request_id = message.get("request_id", "unknown")
        content = message.get("content", {})
        pattern_style = content.get("pattern_style", self.pattern_style)
        fractal_depth = content.get("fractal_depth", self.fractal_depth)
        
        try:
            # Generate pattern data
            pattern_data = self._generate_pattern_data(pattern_style, fractal_depth)
            
            # Add request ID for correlation
            pattern_data["request_id"] = request_id
            
            # Update current parameters
            self.pattern_style = pattern_style
            self.fractal_depth = fractal_depth
            
            # Save pattern data to database
            self._save_pattern_to_database(pattern_data)
            
            # Send pattern data
                    self.socket.send_message({
                "type": "pattern_data_updated",
                "data": pattern_data
            })
            
            # Log successful request
            logger.info(f"Pattern data sent for style {pattern_style}, depth {fractal_depth}")
        except Exception as e:
            # Log error and send error response
            logger.error(f"Error generating pattern data: {str(e)}")
                self.socket.send_message({
                "type": "pattern_data_updated",
                "data": {
                    "error": f"Error generating pattern: {str(e)}",
                    "request_id": request_id
                }
            })
    
    def _save_pattern_to_database(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Save pattern data to database
        
        Args:
            pattern_data: Pattern data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get database manager instance
            db_manager = DatabaseManager.get_instance()
            
            # Add ID and timestamp if not present
            if "id" not in pattern_data:
                pattern_data["id"] = str(uuid.uuid4())
            if "timestamp" not in pattern_data:
                pattern_data["timestamp"] = time.time()
            
            # Save to database
            success = db_manager.save_pattern_data(pattern_data)
            
            if success:
                logger.debug(f"Pattern data saved to database: {pattern_data['id']}")
            else:
                logger.warning(f"Failed to save pattern data to database")
                
            return success
        except Exception as e:
            logger.error(f"Error saving pattern data to database: {str(e)}")
            return False
    
    def _load_pattern_from_database(self, style: str) -> Optional[Dict[str, Any]]:
        """
        Load pattern data from database
        
        Args:
            style: Pattern style to load
            
        Returns:
            Pattern data dictionary or None if not found
        """
        try:
            # Get database manager instance
            db_manager = DatabaseManager.get_instance()
            
            # Get latest pattern by style
            pattern_data = db_manager.get_latest_pattern_by_style(style)
            
            if pattern_data:
                logger.info(f"Loaded pattern data from database: {style}")
                return pattern_data
            else:
                logger.info(f"No pattern data found in database for style: {style}")
                return None
        except Exception as e:
            logger.error(f"Error loading pattern data from database: {str(e)}")
            return None
    
    def _generate_pattern_data(self, pattern_style, fractal_depth):
        """Generate pattern data based on style and depth"""
        # Check if pattern is in database first
        cached_pattern = self._load_pattern_from_database(pattern_style)
        if cached_pattern and cached_pattern.get("fractal_depth") == fractal_depth:
            # Return cached pattern if depth matches
            logger.info(f"Using cached pattern data for {pattern_style} (depth {fractal_depth})")
            return cached_pattern
        
        # Generate new pattern data
        logger.info(f"Generating new pattern data for {pattern_style} (depth {fractal_depth})")
        
        # Generate nodes based on pattern style
        nodes = self._generate_nodes(pattern_style, fractal_depth)
        
        # Update metrics
        self._update_metrics(pattern_style, fractal_depth)
        
        # Generate insights for pattern
        insights = self._generate_insights(pattern_style, nodes)
        
        # Create pattern data
        pattern_data = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "pattern_style": pattern_style,
            "fractal_depth": fractal_depth,
            "nodes": nodes,
            "metrics": {
                "fractal_dimension": self.metrics["fractal_dimension"],
                "complexity_index": self.metrics["complexity_index"],
                "pattern_coherence": self.metrics["pattern_coherence"],
                "entropy_level": self.metrics["entropy_level"]
            },
            "insights": insights
        }
        
        return pattern_data
    
    def _generate_nodes(self, pattern_style, fractal_depth):
        """
        Generate node positions for visualization
        
        Args:
            pattern_style: The style of pattern to generate
            fractal_depth: The depth of fractal recursion
            
        Returns:
            List of node dictionaries with positions
        """
        nodes = []
        
        if pattern_style == "neural":
            # Generate neural pattern nodes
            # These form interconnected networks with branch-like structures
            nodes = self._generate_neural_nodes(fractal_depth)
        
        elif pattern_style == "mandelbrot":
            # Generate Mandelbrot-like pattern nodes
            # These form circular bulb patterns
            nodes = self._generate_mandelbrot_nodes(fractal_depth)
        
        elif pattern_style == "julia":
            # Generate Julia set-like pattern nodes
            # These form intricate spiral patterns
            nodes = self._generate_julia_nodes(fractal_depth)
        
        elif pattern_style == "tree":
            # Generate tree-like pattern nodes
            # These form branching tree structures
            nodes = self._generate_tree_nodes(fractal_depth)
        
        return nodes
    
    def _generate_neural_nodes(self, depth):
        """Generate nodes for neural network pattern"""
        nodes = []
        
        # Number of nodes depends on depth
        node_count = 20 + depth * 15
        
        for i in range(node_count):
            # Determine depth of this node in the fractal
            node_depth = random.randint(0, depth - 1)
            
            # Calculate position in a spiral pattern
            angle = (i / node_count) * math.pi * 6
            distance = 0.1 + (i / node_count) * 0.5
            
            # Add variation based on depth
            distance += node_depth * 0.02
            
            # Calculate x, y coordinates (normalized to 0.0-1.0 range)
            x = 0.5 + math.cos(angle) * distance
            y = 0.5 + math.sin(angle) * distance
            
            # Connection structure
            connections = []
            connection_count = random.randint(1, max(2, min(5, depth)))
            
            # Create connections primarily to nodes of similar depth or lower
            valid_targets = list(range(i))  # Only connect to existing nodes
            if valid_targets:
                for _ in range(connection_count):
                    target = random.choice(valid_targets)
                    connections.append(str(target))
            
            # Create node data
            node = {
                "id": str(i),
                "depth": node_depth,
                "x": x,
                "y": y,
                "connections": list(set(connections)),  # Remove duplicates
                "iterations": node_depth,
                "activation": random.uniform(0.3, 1.0)
            }
            
            nodes.append(node)
        
        return nodes
    
    def _generate_mandelbrot_nodes(self, depth):
        """Generate nodes for Mandelbrot-like pattern"""
        nodes = []
        
        # Generate nodes in a Mandelbrot-like pattern
        # Use max(1, value) to prevent division by zero
        x_step = max(1, 3 // max(1, depth))
        y_step = max(1, 2 // max(1, depth))
        
        for x in range(-2, 1, x_step):
            for y in range(-1, 1, y_step):
                # Scale to [0,1] range
                sx = (x / 3) + 0.5
                sy = (y / 2) + 0.5
                
                # Mandelbrot iteration
                c_real = x / max(1, depth)
                c_imag = y / max(1, depth)
                z_real = 0
                z_imag = 0
                
                # Iterate for depth iterations or until escape
                iterations = 0
                for i in range(16):
                    iterations = i
                    
                    # z = z² + c
                    temp_real = z_real*z_real - z_imag*z_imag + c_real
                    z_imag = 2*z_real*z_imag + c_imag
                    z_real = temp_real
                    
                    # Check for escape
                    if z_real*z_real + z_imag*z_imag > 4:
                        break
                
                # Convert iteration count to depth
                node_depth = min(depth, iterations // 2)
                
                # Add node
                node_id = f"mandelbrot_{x}_{y}"
                node = {
                    "id": node_id,
                    "depth": node_depth,
                    "x": sx,
                    "y": sy,
                    "iterations": iterations,
                    "escaped": iterations < 16,
                    "connections": []
                }
                
                # Find nearby nodes to connect to
                for prev_node in nodes:
                    dx = prev_node["x"] - sx
                    dy = prev_node["y"] - sy
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Connect if nearby
                    if distance < 0.2:
                        node["connections"].append(prev_node["id"])
                        prev_node["connections"].append(node_id)
                
                nodes.append(node)
        
        return nodes
    
    def _generate_julia_nodes(self, depth):
        """Generate nodes for Julia set-like pattern"""
        nodes = []
        c_real = -0.7
        c_imag = 0.27
        
        # Generate nodes in a Julia set-like pattern
        for i in range(50 + depth * 10):
            # Start with points on a circle
            angle = (i / (50 + depth * 10)) * 2 * math.pi
            z_real = math.cos(angle) * 0.4
            z_imag = math.sin(angle) * 0.4
            
            # Track the path of this point
            path = []
            
            # Iterate
            for j in range(depth + 3):
                # Save current position
                path.append((z_real, z_imag))
                
                # Julia iteration: z = z² + c
                temp_real = z_real*z_real - z_imag*z_imag + c_real
                z_imag = 2*z_real*z_imag + c_imag
                z_real = temp_real
                
                # Check for escape
                if z_real*z_real + z_imag*z_imag > 4:
                    break
            
            # Add nodes for each point in the path
            for j, (px, py) in enumerate(path):
                # Scale to [0,1] range
                sx = (px + 1) * 0.5
                sy = (py + 1) * 0.5
                
                # Add node
                nodes.append({
                    "depth": j,
                    "x": sx, 
                    "y": sy,
                    "iterations": j,
                    "angle": angle,
                    "connections": []
                })
        
        return nodes
    
    def _generate_tree_nodes(self, depth):
        """Generate nodes for tree-like pattern"""
        nodes = []
        
        def add_branch(x, y, angle, length, current_depth):
            if current_depth > depth:
                return
                
            # Calculate end point
            end_x = x + length * math.cos(angle)
            end_y = y + length * math.sin(angle)
            
            # Create node at end point
            node_index = len(nodes)
            nodes.append({
                "depth": current_depth,
                "x": end_x,
                "y": end_y,
                "angle": angle,
                "length": length,
                "connections": []
            })
            
            # If not the first node, connect to previous
            if nodes:
                # Find closest node at lower depth
                candidates = [(i, n) for i, n in enumerate(nodes) 
                             if n["depth"] < current_depth]
                
                if candidates:
                    # Find closest
                    closest = min(candidates, 
                                 key=lambda c: ((c[1]["x"]-end_x)**2 + 
                                               (c[1]["y"]-end_y)**2))
                    
                    # Add connection
                    nodes[node_index]["connections"].append(closest[0])
                    
            # Recursive branches
            if current_depth < depth:
                # Left branch
                left_angle = angle + random.uniform(0.3, 0.5)
                left_length = length * random.uniform(0.6, 0.8)
                add_branch(end_x, end_y, left_angle, left_length, current_depth + 1)
                
                # Right branch
                right_angle = angle - random.uniform(0.3, 0.5)
                right_length = length * random.uniform(0.6, 0.8)
                add_branch(end_x, end_y, right_angle, right_length, current_depth + 1)
                
                # Sometimes add middle branch
                if random.random() < 0.3:
                    mid_angle = angle + random.uniform(-0.1, 0.1)
                    mid_length = length * random.uniform(0.7, 0.9)
                    add_branch(end_x, end_y, mid_angle, mid_length, current_depth + 1)
        
        # Start with trunk
        add_branch(0.5, 0.9, -math.pi/2, 0.15, 0)
        
        return nodes
    
    def _update_metrics(self, pattern_style, fractal_depth):
        """Update metrics based on pattern parameters"""
        
        # Update fractal dimension
        base_dimension = {
            "neural": 1.65,
            "mandelbrot": 2.0,
            "julia": 1.79,
            "tree": 1.58
        }.get(pattern_style, 1.5)
        
        # Adjust dimension based on depth
        depth_factor = 0.02 * (fractal_depth - 5)  # Deviation from default depth 5
        dimension = base_dimension + depth_factor
        
        # Update complexity index
        base_complexity = {
            "neural": 75,
            "mandelbrot": 85,
            "julia": 80,
            "tree": 70
        }.get(pattern_style, 70)
        
        # Adjust complexity based on depth
        complexity = base_complexity + (fractal_depth - 5) * 3
        
        # Update coherence
        base_coherence = {
            "neural": 90,
            "mandelbrot": 95,
            "julia": 88,
            "tree": 92
        }.get(pattern_style, 90)
        
        # Adjust coherence based on depth (higher depth can reduce coherence)
        coherence = base_coherence - abs(fractal_depth - 5) * 2
        
        # Update entropy level
        entropy_levels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
        entropy_index = min(4, max(0, (complexity // 20) - 1))
        entropy_level = entropy_levels[entropy_index]
        
        # Update metrics
        self.metrics = {
            "fractal_dimension": round(dimension, 2),
            "complexity_index": int(complexity),
            "pattern_coherence": int(coherence),
            "entropy_level": entropy_level
        }
        
        # Update insights
        patterns = []
        
        if pattern_style == "neural":
            patterns.append("Neural layer activation patterns")
            if fractal_depth > 6:
                patterns.append("Deep network recursion detected")
            if coherence > 90:
                patterns.append("High neural coherence")
                
        elif pattern_style == "mandelbrot":
            patterns.append("Classic Mandelbrot stability regions")
            if fractal_depth > 7:
                patterns.append("Deep cardioid structures")
            if complexity > 85:
                patterns.append("Complex periodicity detected")
                
        elif pattern_style == "julia":
            patterns.append("Julia set connectivity patterns")
            if fractal_depth > 5:
                patterns.append("Spiral arm formations")
            if coherence < 85:
                patterns.append("Dynamic instability detected")
                
        elif pattern_style == "tree":
            patterns.append("Recursive branching structures")
            if fractal_depth > 6:
                patterns.append("High branching factor detected")
            if coherence > 90:
                patterns.append("Symmetric growth patterns")
        
        self.insights = {
            "detected_patterns": patterns
        }
    
    def _generate_insights(self, pattern_style, nodes):
        """
        Generate insights based on pattern style and nodes
        
        Args:
            pattern_style: The pattern style
            nodes: The generated nodes
        
        Returns:
            Dictionary containing insights
        """
        # Use the existing insights if available
        if hasattr(self, 'insights') and self.insights:
            return self.insights
            
        # Generate new insights based on pattern style
        patterns = []
        
        if pattern_style == "neural":
            patterns.append("Neural layer activation patterns")
            patterns.append("Connection density analysis")
            if len(nodes) > 30:
                patterns.append("Deep network recursion detected")
            
        elif pattern_style == "mandelbrot":
            patterns.append("Classic Mandelbrot stability regions")
            patterns.append("Boundary exploration pattern")
            if len(nodes) > 40:
                patterns.append("Deep cardioid structures")
                
        elif pattern_style == "julia":
            patterns.append("Julia set connectivity patterns")
            patterns.append("Complex plane exploration")
            if len(nodes) > 35:
                patterns.append("Spiral arm formations")
                
        elif pattern_style == "tree":
            patterns.append("Recursive branching structures")
            patterns.append("Hierarchical node organization")
            if len(nodes) > 25:
                patterns.append("High branching factor detected")
        
        # Return insights
        return {
            "detected_patterns": patterns
        } 