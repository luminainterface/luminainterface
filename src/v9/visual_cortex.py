#!/usr/bin/env python3
"""
Visual Cortex Module (v9)

This module provides visual processing capabilities for the Lumina Neural Network.
It can process visual data and integrate with the Neural Playground to create
visual-neural interactions.
"""

import logging
import time
import json
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple

# Define component type for auto-discovery
LUMINA_COMPONENT_TYPE = "visualization"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.visual_cortex")

class VisualCortex:
    """
    Visual processing system that can be integrated with Neural Playground
    
    This component processes visual data and creates neural representations
    that can be used by the playground for pattern formation and consciousness
    development.
    """
    
    def __init__(self, resolution=(28, 28), channels=1):
        """
        Initialize the visual cortex
        
        Args:
            resolution: Tuple of (width, height) for visual field
            channels: Number of channels (1=grayscale, 3=RGB)
        """
        self.resolution = resolution
        self.channels = channels
        self.width, self.height = resolution
        self.visual_field = self._create_empty_field()
        self.pattern_memory = []
        self.cortex_id = str(uuid.uuid4())[:8]
        
        # Processing statistics
        self.processed_frames = 0
        self.recognized_patterns = 0
        
        logger.info(f"Visual Cortex initialized (ID: {self.cortex_id}, "
                   f"Resolution: {self.width}x{self.height}, Channels: {self.channels})")
    
    def _create_empty_field(self):
        """Create an empty visual field"""
        if self.channels == 1:
            return [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        else:
            return [[[0.0 for _ in range(self.channels)] 
                   for _ in range(self.width)] for _ in range(self.height)]
    
    def process_image(self, image_data, metadata=None):
        """
        Process an image and extract visual patterns
        
        Args:
            image_data: 2D or 3D array of pixel values (0.0-1.0)
            metadata: Optional metadata about the image
            
        Returns:
            Dict containing processing results
        """
        if image_data is None or len(image_data) == 0:
            logger.warning("Received empty image data")
            return {"success": False, "error": "Empty image data"}
            
        # Validate and normalize image dimensions
        normalized_data = self._normalize_image(image_data)
        if normalized_data is None:
            return {"success": False, "error": "Could not normalize image data"}
            
        # Set the visual field
        self.visual_field = normalized_data
        self.processed_frames += 1
        
        # Extract basic features
        edge_count, brightness, contrast = self._extract_features()
        
        # Detect patterns
        patterns = self._detect_patterns()
        
        # Update stats
        if patterns:
            self.recognized_patterns += len(patterns)
            
        # Store in pattern memory (limited size)
        if len(self.pattern_memory) > 20:
            self.pattern_memory.pop(0)  # Remove oldest
        
        pattern_record = {
            "timestamp": time.time(),
            "edge_count": edge_count,
            "brightness": brightness,
            "contrast": contrast,
            "patterns": patterns,
            "metadata": metadata
        }
        self.pattern_memory.append(pattern_record)
        
        return {
            "success": True,
            "frame_id": self.processed_frames,
            "edge_count": edge_count,
            "brightness": brightness,
            "contrast": contrast,
            "patterns_detected": len(patterns),
            "patterns": patterns
        }
    
    def _normalize_image(self, image_data):
        """Normalize image data to match expected dimensions"""
        try:
            # Simple normalization for demo purposes
            # In a real implementation, would properly handle various image formats
            
            # Create a normalized representation matching our dimensions
            normalized = self._create_empty_field()
            
            # Simple scaling algorithm (very basic for demo)
            src_height = len(image_data)
            src_width = len(image_data[0]) if src_height > 0 else 0
            
            if src_height == 0 or src_width == 0:
                return None
                
            # Map source to destination dimensions
            for y in range(self.height):
                for x in range(self.width):
                    src_y = min(int(y * src_height / self.height), src_height - 1)
                    src_x = min(int(x * src_width / self.width), src_width - 1)
                    
                    if self.channels == 1:
                        # Handle grayscale
                        if isinstance(image_data[src_y][src_x], (list, tuple)):
                            # Source is multi-channel, convert to grayscale
                            value = sum(image_data[src_y][src_x]) / len(image_data[src_y][src_x])
                        else:
                            # Source is already single-channel
                            value = image_data[src_y][src_x]
                        normalized[y][x] = min(1.0, max(0.0, float(value)))
                    else:
                        # Handle multi-channel
                        if isinstance(image_data[src_y][src_x], (list, tuple)):
                            # Source is also multi-channel
                            src_channels = len(image_data[src_y][src_x])
                            for c in range(self.channels):
                                src_c = min(c, src_channels - 1)
                                normalized[y][x][c] = min(1.0, max(0.0, 
                                                        float(image_data[src_y][src_x][src_c])))
                        else:
                            # Source is single-channel, duplicate to all channels
                            for c in range(self.channels):
                                normalized[y][x][c] = min(1.0, max(0.0, 
                                                        float(image_data[src_y][src_x])))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return None
    
    def _extract_features(self):
        """Extract basic features from the current visual field"""
        # Calculate average brightness
        total_pixels = self.width * self.height
        brightness_sum = 0
        
        if self.channels == 1:
            for y in range(self.height):
                for x in range(self.width):
                    brightness_sum += self.visual_field[y][x]
        else:
            for y in range(self.height):
                for x in range(self.width):
                    brightness_sum += sum(self.visual_field[y][x]) / self.channels
        
        brightness = brightness_sum / total_pixels
        
        # Calculate simple contrast (standard deviation of brightness)
        variance_sum = 0
        if self.channels == 1:
            for y in range(self.height):
                for x in range(self.width):
                    variance_sum += (self.visual_field[y][x] - brightness) ** 2
        else:
            for y in range(self.height):
                for x in range(self.width):
                    pixel_brightness = sum(self.visual_field[y][x]) / self.channels
                    variance_sum += (pixel_brightness - brightness) ** 2
        
        contrast = (variance_sum / total_pixels) ** 0.5
        
        # Count edges (simple gradient threshold)
        edge_count = 0
        edge_threshold = 0.1
        
        if self.channels == 1:
            for y in range(1, self.height):
                for x in range(1, self.width):
                    # Horizontal gradient
                    gradient_h = abs(self.visual_field[y][x] - self.visual_field[y][x-1])
                    # Vertical gradient
                    gradient_v = abs(self.visual_field[y][x] - self.visual_field[y-1][x])
                    
                    if gradient_h > edge_threshold or gradient_v > edge_threshold:
                        edge_count += 1
        else:
            for y in range(1, self.height):
                for x in range(1, self.width):
                    # Average channel gradients
                    gradient_h = sum(abs(self.visual_field[y][x][c] - self.visual_field[y][x-1][c]) 
                                  for c in range(self.channels)) / self.channels
                    gradient_v = sum(abs(self.visual_field[y][x][c] - self.visual_field[y-1][x][c]) 
                                  for c in range(self.channels)) / self.channels
                    
                    if gradient_h > edge_threshold or gradient_v > edge_threshold:
                        edge_count += 1
        
        return edge_count, brightness, contrast
    
    def _detect_patterns(self):
        """Detect visual patterns in the current visual field"""
        # This is a simplified pattern detection for demonstration
        # Real implementation would use computer vision algorithms
        
        patterns = []
        
        # Define regions to check (divide image into quadrants)
        regions = [
            (0, 0, self.width // 2, self.height // 2),                # top-left
            (self.width // 2, 0, self.width, self.height // 2),       # top-right
            (0, self.height // 2, self.width // 2, self.height),      # bottom-left
            (self.width // 2, self.height // 2, self.width, self.height)  # bottom-right
        ]
        
        for region_id, (x1, y1, x2, y2) in enumerate(regions):
            # Calculate region statistics
            region_pixels = (x2 - x1) * (y2 - y1)
            region_brightness = 0
            
            if self.channels == 1:
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        region_brightness += self.visual_field[y][x]
            else:
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        region_brightness += sum(self.visual_field[y][x]) / self.channels
            
            region_brightness /= region_pixels
            
            # Check for regions with distinctive brightness
            global_brightness = sum(p["brightness"] for p in self.pattern_memory[-5:]) / 5 if self.pattern_memory else 0.5
            
            if abs(region_brightness - global_brightness) > 0.15:
                # Region has distinctive brightness - consider it a pattern
                pattern_type = "bright_region" if region_brightness > global_brightness else "dark_region"
                
                # Check for horizontal/vertical lines
                h_line = self._check_for_line(x1, y1, x2, y2, "horizontal")
                v_line = self._check_for_line(x1, y1, x2, y2, "vertical")
                
                if h_line:
                    pattern_type = "horizontal_line"
                elif v_line:
                    pattern_type = "vertical_line"
                
                patterns.append({
                    "id": f"pattern_{self.processed_frames}_{region_id}",
                    "type": pattern_type,
                    "region": [x1, y1, x2, y2],
                    "confidence": min(1.0, abs(region_brightness - global_brightness) * 2),
                    "brightness": region_brightness
                })
        
        return patterns
    
    def _check_for_line(self, x1, y1, x2, y2, direction):
        """Check if the region contains a line in specified direction"""
        # Simple line detection
        if direction == "horizontal":
            # Check for horizontal lines
            for y in range(y1, y2):
                line_strength = 0
                for x in range(x1, x2-1):
                    if self.channels == 1:
                        if abs(self.visual_field[y][x] - self.visual_field[y][x+1]) < 0.05:
                            line_strength += 1
                    else:
                        avg_diff = sum(abs(self.visual_field[y][x][c] - self.visual_field[y][x+1][c]) 
                                     for c in range(self.channels)) / self.channels
                        if avg_diff < 0.05:
                            line_strength += 1
                
                if line_strength > (x2 - x1) * 0.7:  # If 70% of pixels form a line
                    return True
        
        elif direction == "vertical":
            # Check for vertical lines
            for x in range(x1, x2):
                line_strength = 0
                for y in range(y1, y2-1):
                    if self.channels == 1:
                        if abs(self.visual_field[y][x] - self.visual_field[y+1][x]) < 0.05:
                            line_strength += 1
                    else:
                        avg_diff = sum(abs(self.visual_field[y][x][c] - self.visual_field[y+1][x][c]) 
                                     for c in range(self.channels)) / self.channels
                        if avg_diff < 0.05:
                            line_strength += 1
                
                if line_strength > (y2 - y1) * 0.7:  # If 70% of pixels form a line
                    return True
        
        return False
    
    def visualize(self, visualization_data):
        """
        Create visualization based on input data
        
        Args:
            visualization_data: Dict containing visualization request
            
        Returns:
            Dict containing visualization result
        """
        viz_type = visualization_data.get("type", "unknown")
        title = visualization_data.get("title", "Untitled Visualization")
        data = visualization_data.get("data", {})
        
        logger.info(f"Creating visualization: {viz_type} - {title}")
        
        # In a real implementation, this would create actual visualizations
        # For this demo, we'll just return information about what would be visualized
        
        if viz_type == "play_session":
            # Visualize a neural playground play session
            result = {
                "visualization_id": f"viz_{int(time.time())}",
                "type": "play_session",
                "title": title,
                "description": "Visualization showing neural activity during play session",
                "generated": True,
                "elements": [
                    {
                        "type": "neural_network_graph",
                        "neuron_count": len(data.get("neurons", {})),
                        "connection_count": len(data.get("connections", {}))
                    },
                    {
                        "type": "consciousness_timeline",
                        "data_points": len(data.get("consciousness_history", [])),
                        "peak": max(data.get("consciousness_history", [0]))
                    }
                ]
            }
        
        elif viz_type == "consciousness_peak":
            # Visualize a consciousness peak
            result = {
                "visualization_id": f"viz_{int(time.time())}",
                "type": "consciousness_peak",
                "title": title,
                "description": "Visualization of neural activity during consciousness peak",
                "generated": True,
                "elements": [
                    {
                        "type": "peak_snapshot",
                        "peak_value": data.get("value", 0),
                        "timestamp": data.get("timestamp", time.time())
                    },
                    {
                        "type": "neural_activity_heatmap",
                        "data_points": 100,  # Placeholder
                        "intensity": "high"
                    }
                ]
            }
        
        else:
            # Generic visualization
            result = {
                "visualization_id": f"viz_{int(time.time())}",
                "type": viz_type,
                "title": title,
                "description": "Generic visualization",
                "generated": True,
                "elements": [
                    {
                        "type": "generic_view",
                        "data_size": len(str(data))
                    }
                ]
            }
        
        return result
    
    def integrate_with_playground(self, playground):
        """
        Integrate with Neural Playground
        
        Args:
            playground: NeuralPlayground instance
            
        Returns:
            Dict containing integration information and hooks
        """
        logger.info(f"Integrating Visual Cortex with Neural Playground")
        
        # Define hooks
        def post_play_hook(playground, play_result):
            """Hook called after play session"""
            try:
                # Generate a visual representation of the play session
                neural_state = playground.core.get_state()
                
                # Convert neural state to a visual representation
                visual_data = self._neural_state_to_visual(neural_state)
                
                # Process the resulting image to find patterns
                processing_result = self.process_image(
                    visual_data,
                    {
                        "source": "neural_playground",
                        "play_type": play_result["play_type"],
                        "session_id": play_result["session_id"]
                    }
                )
                
                # Create visualization
                self.visualize({
                    "type": "play_session",
                    "title": f"Neural Play: {play_result['play_type']} mode",
                    "data": {
                        "neurons": neural_state["neurons"],
                        "connections": neural_state["connections"],
                        "consciousness_history": play_result["consciousness_history"],
                        "play_type": play_result["play_type"]
                    }
                })
                
                logger.info(f"Visual Cortex processed neural state: {processing_result['patterns_detected']} patterns")
                
            except Exception as e:
                logger.error(f"Error in Visual Cortex post-play hook: {e}")
        
        def consciousness_peak_hook(playground, peak_data):
            """Hook called when consciousness peak is detected"""
            try:
                # Visualize the consciousness peak
                self.visualize({
                    "type": "consciousness_peak",
                    "title": f"Consciousness Peak: {peak_data['value']:.2f}",
                    "data": peak_data
                })
                
                logger.info(f"Visual Cortex visualized consciousness peak: {peak_data['value']:.2f}")
                
            except Exception as e:
                logger.error(f"Error in Visual Cortex consciousness peak hook: {e}")
        
        # Return integration information and hooks
        return {
            "component_id": self.cortex_id,
            "component_type": "visual_cortex",
            "hooks": {
                "post_play": post_play_hook,
                "consciousness_peak": consciousness_peak_hook
            }
        }
    
    def _neural_state_to_visual(self, neural_state):
        """Convert neural state to visual representation"""
        # Create an empty visual field
        visual_data = self._create_empty_field()
        
        # Get neurons and connections
        neurons = neural_state.get("neurons", {})
        connections = neural_state.get("connections", {})
        
        # If no neurons, return empty field
        if not neurons:
            return visual_data
            
        # Map neurons to visual field positions
        neuron_positions = {}
        neuron_ids = list(neurons.keys())
        
        for i, neuron_id in enumerate(neuron_ids):
            # Map neuron to a position in the visual field
            # This is a simple mapping for demonstration
            x = (i % self.width)
            y = (i // self.width) % self.height
            neuron_positions[neuron_id] = (x, y)
        
        # Set pixel values based on neuron states
        for neuron_id, neuron in neurons.items():
            if neuron_id in neuron_positions:
                x, y = neuron_positions[neuron_id]
                state = neuron.get("state", "inactive")
                activation = neuron.get("activation", 0.0)
                
                # Set pixel value based on neuron state and activation
                if self.channels == 1:
                    if state == "active":
                        visual_data[y][x] = min(1.0, activation + 0.5)
                    else:
                        visual_data[y][x] = max(0.0, activation - 0.1)
                else:
                    if state == "active":
                        # Active neurons: more red/yellow
                        visual_data[y][x][0] = min(1.0, activation + 0.5)  # Red
                        visual_data[y][x][1] = min(1.0, activation)        # Green
                        visual_data[y][x][2] = max(0.0, activation - 0.3)  # Blue
                    else:
                        # Inactive neurons: more blue/gray
                        visual_data[y][x][0] = max(0.0, activation - 0.1)  # Red
                        visual_data[y][x][1] = max(0.0, activation - 0.1)  # Green
                        visual_data[y][x][2] = min(0.8, activation + 0.2)  # Blue
        
        # Enhance visualization with neuron connections
        for connection_id, connection in connections.items():
            source_id = connection.get("source", "")
            target_id = connection.get("target", "")
            strength = connection.get("strength", 0.0)
            
            if source_id in neuron_positions and target_id in neuron_positions:
                # Draw a line between source and target (simplified)
                x1, y1 = neuron_positions[source_id]
                x2, y2 = neuron_positions[target_id]
                
                # Draw a simple line using Bresenham's algorithm
                self._draw_line(visual_data, x1, y1, x2, y2, strength)
        
        return visual_data
    
    def _draw_line(self, visual_data, x1, y1, x2, y2, intensity=0.5):
        """Draw a line on the visual field using Bresenham's algorithm"""
        # Simplification: just draw a few points along the line
        points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        
        for i in range(points):
            # Interpolate position
            t = i / points if points > 1 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            # Ensure within bounds
            if 0 <= x < self.width and 0 <= y < self.height:
                # Set pixel value based on connection strength
                if self.channels == 1:
                    visual_data[y][x] = max(visual_data[y][x], intensity * 0.7)
                else:
                    # Connections: green-ish
                    visual_data[y][x][0] = max(visual_data[y][x][0], intensity * 0.3)  # Red
                    visual_data[y][x][1] = max(visual_data[y][x][1], intensity * 0.9)  # Green
                    visual_data[y][x][2] = max(visual_data[y][x][2], intensity * 0.5)  # Blue
    
    def generate_test_image(self, pattern_type="random"):
        """
        Generate a test image for processing
        
        Args:
            pattern_type: Type of pattern to generate
                          (random, horizontal_lines, vertical_lines, checkerboard)
            
        Returns:
            Generated image data
        """
        image = self._create_empty_field()
        
        if pattern_type == "horizontal_lines":
            # Generate horizontal lines
            for y in range(self.height):
                if y % 4 == 0:  # Line every 4 pixels
                    for x in range(self.width):
                        if self.channels == 1:
                            image[y][x] = 1.0
                        else:
                            image[y][x] = [1.0, 1.0, 1.0]  # White
        
        elif pattern_type == "vertical_lines":
            # Generate vertical lines
            for x in range(self.width):
                if x % 4 == 0:  # Line every 4 pixels
                    for y in range(self.height):
                        if self.channels == 1:
                            image[y][x] = 1.0
                        else:
                            image[y][x] = [1.0, 1.0, 1.0]  # White
        
        elif pattern_type == "checkerboard":
            # Generate checkerboard
            for y in range(self.height):
                for x in range(self.width):
                    if (x // 4 + y // 4) % 2 == 0:  # 4x4 squares
                        if self.channels == 1:
                            image[y][x] = 1.0
                        else:
                            image[y][x] = [1.0, 1.0, 1.0]  # White
        
        else:  # random
            # Generate random noise
            for y in range(self.height):
                for x in range(self.width):
                    if self.channels == 1:
                        image[y][x] = random.random()
                    else:
                        image[y][x] = [random.random() for _ in range(self.channels)]
        
        return image
    
    def get_state(self):
        """Get current state of the visual cortex"""
        return {
            "cortex_id": self.cortex_id,
            "resolution": self.resolution,
            "channels": self.channels,
            "processed_frames": self.processed_frames,
            "recognized_patterns": self.recognized_patterns,
            "last_pattern": self.pattern_memory[-1] if self.pattern_memory else None
        }

# Example usage
if __name__ == "__main__":
    # Create visual cortex
    cortex = VisualCortex(resolution=(20, 20), channels=3)
    
    # Generate and process a test image
    test_image = cortex.generate_test_image(pattern_type="horizontal_lines")
    result = cortex.process_image(test_image, {"source": "test_generator"})
    
    print(f"Visual Cortex processed test image:")
    print(f"- Pattern count: {result['patterns_detected']}")
    print(f"- Edge count: {result['edge_count']}")
    print(f"- Brightness: {result['brightness']:.2f}")
    print(f"- Contrast: {result['contrast']:.2f}")
    
    # Example of visualization
    viz_result = cortex.visualize({
        "type": "test_pattern",
        "title": "Test Pattern Visualization",
        "data": {
            "pattern_type": "horizontal_lines",
            "resolution": cortex.resolution
        }
    })
    
    print(f"\nCreated visualization:")
    print(f"- ID: {viz_result['visualization_id']}")
    print(f"- Type: {viz_result['type']}")
    print(f"- Title: {viz_result['title']}")
    print(f"- Elements: {len(viz_result['elements'])}") 