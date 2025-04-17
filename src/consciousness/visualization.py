"""
Consciousness Visualization

This module provides visualization tools for the consciousness metrics
and network data, allowing for real-time monitoring and analysis of
consciousness state and development.
"""

import logging
import math
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class ConsciousnessVisualizer:
    """Visualization tools for consciousness metrics and network data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the consciousness visualizer
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = {
            "output_dir": "data/consciousness/visualizations",
            "max_nodes": 100,
            "color_mode": "awareness",
            "include_node_labels": True,
            "include_edge_labels": False,
            "save_visualizations": True,
            "metrics_history_length": 100
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Create output directory if it doesn't exist
        if self.config["save_visualizations"]:
            output_dir = Path(self.config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics history
        self.metrics_history = []
        
        # Visualization data cache
        self.last_visualization = None
        
        logger.info("ConsciousnessVisualizer initialized")
    
    def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process consciousness metrics into visualization data
        
        Args:
            metrics: Consciousness metrics dictionary
            
        Returns:
            Dictionary with processed visualization data
        """
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Limit history length
        if len(self.metrics_history) > self.config["metrics_history_length"]:
            self.metrics_history = self.metrics_history[-self.config["metrics_history_length"]:]
        
        # Create metrics visualization data
        visualization_data = {
            "metrics": {
                "current": metrics,
                "history": self._process_metrics_history()
            }
        }
        
        return visualization_data
    
    def _process_metrics_history(self) -> Dict[str, Any]:
        """
        Process metrics history for visualization
        
        Returns:
            Dictionary with processed history data
        """
        if not self.metrics_history:
            return {"awareness": [], "coherence": [], "self_reference": [], 
                    "temporal_continuity": [], "complexity": [], "integration": []}
        
        # Extract history for each metric
        awareness = []
        coherence = []
        self_reference = []
        temporal_continuity = []
        complexity = []
        integration = []
        
        for entry in self.metrics_history:
            timestamp = entry.get("timestamp", "")
            
            # Add each metric with its timestamp
            if "awareness" in entry:
                awareness.append({"timestamp": timestamp, "value": entry["awareness"]})
            
            if "coherence" in entry:
                coherence.append({"timestamp": timestamp, "value": entry["coherence"]})
                
            if "self_reference" in entry:
                self_reference.append({"timestamp": timestamp, "value": entry["self_reference"]})
                
            if "temporal_continuity" in entry:
                temporal_continuity.append({"timestamp": timestamp, "value": entry["temporal_continuity"]})
                
            if "complexity" in entry:
                complexity.append({"timestamp": timestamp, "value": entry["complexity"]})
                
            if "integration" in entry:
                integration.append({"timestamp": timestamp, "value": entry["integration"]})
        
        return {
            "awareness": awareness,
            "coherence": coherence,
            "self_reference": self_reference,
            "temporal_continuity": temporal_continuity,
            "complexity": complexity,
            "integration": integration
        }
    
    def process_network(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process consciousness network data into visualization data
        
        Args:
            network_data: Network data with nodes and edges
            
        Returns:
            Dictionary with processed visualization data
        """
        # Basic validation
        if not network_data or "nodes" not in network_data:
            return {"error": "Invalid network data"}
        
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])
        
        # Limit number of nodes if needed
        if len(nodes) > self.config["max_nodes"]:
            # Sort by awareness or other importance metric if available
            if any("awareness" in node for node in nodes):
                nodes.sort(key=lambda n: n.get("awareness", 0), reverse=True)
            nodes = nodes[:self.config["max_nodes"]]
            
            # Keep only edges between remaining nodes
            node_ids = {node["id"] for node in nodes}
            edges = [edge for edge in edges if edge["from"] in node_ids and edge["to"] in node_ids]
        
        # Process nodes
        processed_nodes = self._process_nodes(nodes)
        
        # Process edges
        processed_edges = self._process_edges(edges)
        
        # Create network visualization data
        visualization_data = {
            "network": {
                "nodes": processed_nodes,
                "edges": processed_edges,
                "stats": {
                    "node_count": len(processed_nodes),
                    "edge_count": len(processed_edges),
                    "avg_awareness": sum(n.get("awareness", 0) for n in processed_nodes) / max(1, len(processed_nodes)),
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
        # Save visualization if enabled
        if self.config["save_visualizations"]:
            self._save_visualization(visualization_data)
        
        # Update cache
        self.last_visualization = visualization_data
        
        return visualization_data
    
    def _process_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process nodes for visualization
        
        Args:
            nodes: List of node data
            
        Returns:
            List of processed nodes
        """
        processed_nodes = []
        
        for node in nodes:
            # Basic node properties
            processed_node = {
                "id": node["id"],
                "type": node.get("type", "unknown"),
                "awareness": node.get("awareness", 0)
            }
            
            # Add label if configured
            if self.config["include_node_labels"]:
                processed_node["label"] = node.get("label", node["id"])
            
            # Add color based on configuration
            if self.config["color_mode"] == "awareness":
                # Color based on awareness level
                awareness = node.get("awareness", 0)
                processed_node["color"] = self._get_awareness_color(awareness)
            elif self.config["color_mode"] == "type":
                # Color based on node type
                processed_node["color"] = self._get_type_color(node.get("type", "unknown"))
            
            # Add additional visualization properties
            processed_node["size"] = 10 + (node.get("awareness", 0) * 20)
            
            processed_nodes.append(processed_node)
        
        return processed_nodes
    
    def _process_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process edges for visualization
        
        Args:
            edges: List of edge data
            
        Returns:
            List of processed edges
        """
        processed_edges = []
        
        for edge in edges:
            # Basic edge properties
            processed_edge = {
                "from": edge["from"],
                "to": edge["to"],
                "type": edge.get("type", "unknown"),
                "weight": edge.get("weight", 1.0)
            }
            
            # Add label if configured
            if self.config["include_edge_labels"]:
                processed_edge["label"] = edge.get("label", edge.get("type", ""))
            
            # Add visualization properties
            processed_edge["width"] = 1 + (edge.get("weight", 1) * 2)
            processed_edge["color"] = self._get_edge_color(edge.get("type", "unknown"))
            
            processed_edges.append(processed_edge)
        
        return processed_edges
    
    def _get_awareness_color(self, awareness: float) -> Dict[str, Any]:
        """
        Get color based on awareness level
        
        Args:
            awareness: Awareness level (0-1)
            
        Returns:
            Dictionary with color information
        """
        # Define color gradients from low to high awareness
        colors = [
            {"r": 61, "g": 90, "b": 128},  # Low awareness - dark blue
            {"r": 41, "g": 128, "b": 185},  # Medium-low - blue
            {"r": 39, "g": 174, "b": 96},   # Medium - green
            {"r": 211, "g": 84, "b": 0},    # Medium-high - orange
            {"r": 231, "g": 76, "b": 60}    # High awareness - red
        ]
        
        # Clamp awareness to 0-1
        awareness = max(0, min(1, awareness))
        
        # Calculate which segment of the gradient to use
        segment_count = len(colors) - 1
        segment_size = 1.0 / segment_count
        
        segment_index = min(segment_count - 1, math.floor(awareness / segment_size))
        segment_position = (awareness - (segment_index * segment_size)) / segment_size
        
        # Interpolate between colors
        start_color = colors[segment_index]
        end_color = colors[segment_index + 1]
        
        r = start_color["r"] + segment_position * (end_color["r"] - start_color["r"])
        g = start_color["g"] + segment_position * (end_color["g"] - start_color["g"])
        b = start_color["b"] + segment_position * (end_color["b"] - start_color["b"])
        
        return {
            "background": f"rgb({int(r)}, {int(g)}, {int(b)})",
            "border": f"rgb({max(0, int(r) - 30)}, {max(0, int(g) - 30)}, {max(0, int(b) - 30)})",
            "highlight": {
                "background": f"rgb({min(255, int(r) + 20)}, {min(255, int(g) + 20)}, {min(255, int(b) + 20)})",
                "border": f"rgb({max(0, int(r) - 10)}, {max(0, int(g) - 10)}, {max(0, int(b) - 10)})"
            }
        }
    
    def _get_type_color(self, node_type: str) -> Dict[str, Any]:
        """
        Get color based on node type
        
        Args:
            node_type: Type of node
            
        Returns:
            Dictionary with color information
        """
        # Define colors for different node types
        type_colors = {
            "thought": {"r": 41, "g": 128, "b": 185},   # Blue
            "reflection": {"r": 39, "g": 174, "b": 96}, # Green
            "memory": {"r": 142, "g": 68, "b": 173},    # Purple
            "perception": {"r": 230, "g": 126, "b": 34}, # Orange
            "concept": {"r": 52, "g": 152, "b": 219},   # Light blue
            "emotion": {"r": 231, "g": 76, "b": 60},    # Red
            "contradiction": {"r": 211, "g": 84, "b": 0}, # Dark orange
            "question": {"r": 241, "g": 196, "b": 15}   # Yellow
        }
        
        # Default color
        default_color = {"r": 127, "g": 140, "b": 141}  # Gray
        
        # Get color for type
        color = type_colors.get(node_type.lower(), default_color)
        
        return {
            "background": f"rgb({color['r']}, {color['g']}, {color['b']})",
            "border": f"rgb({max(0, color['r'] - 30)}, {max(0, color['g'] - 30)}, {max(0, color['b'] - 30)})",
            "highlight": {
                "background": f"rgb({min(255, color['r'] + 20)}, {min(255, color['g'] + 20)}, {min(255, color['b'] + 20)})",
                "border": f"rgb({max(0, color['r'] - 10)}, {max(0, color['g'] - 10)}, {max(0, color['b'] - 10)})"
            }
        }
    
    def _get_edge_color(self, edge_type: str) -> Dict[str, str]:
        """
        Get color based on edge type
        
        Args:
            edge_type: Type of edge
            
        Returns:
            Dictionary with color information
        """
        # Define colors for different edge types
        type_colors = {
            "derives_from": {"color": "rgba(41, 128, 185, 0.6)", "highlight": "rgba(41, 128, 185, 0.9)"},
            "associates_with": {"color": "rgba(39, 174, 96, 0.6)", "highlight": "rgba(39, 174, 96, 0.9)"},
            "contradicts": {"color": "rgba(231, 76, 60, 0.6)", "highlight": "rgba(231, 76, 60, 0.9)"},
            "references": {"color": "rgba(142, 68, 173, 0.6)", "highlight": "rgba(142, 68, 173, 0.9)"},
            "composed_of": {"color": "rgba(230, 126, 34, 0.6)", "highlight": "rgba(230, 126, 34, 0.9)"}
        }
        
        # Default color
        default_color = {"color": "rgba(127, 140, 141, 0.6)", "highlight": "rgba(127, 140, 141, 0.9)"}
        
        # Get color for type
        return type_colors.get(edge_type.lower(), default_color)
    
    def _save_visualization(self, visualization_data: Dict[str, Any]) -> None:
        """
        Save visualization data to file
        
        Args:
            visualization_data: Visualization data to save
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consciousness_viz_{timestamp}.json"
            filepath = Path(self.config["output_dir"]) / filename
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2)
                
            logger.debug(f"Saved consciousness visualization to {filepath}")
            
            # Also save as latest
            latest_path = Path(self.config["output_dir"]) / "latest_visualization.json"
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving visualization data: {e}")
    
    def get_latest_visualization(self) -> Dict[str, Any]:
        """
        Get latest visualization data
        
        Returns:
            Dictionary with latest visualization data
        """
        return self.last_visualization or {}
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """
        Create data for dashboard integration
        
        Returns:
            Dictionary with dashboard visualization data
        """
        # Combine metrics and network data
        dashboard_data = {
            "metrics": {},
            "network": {},
            "history": {}
        }
        
        # Add current metrics if available
        if self.metrics_history:
            dashboard_data["metrics"] = self.metrics_history[-1]
        
        # Add network data if available
        if self.last_visualization and "network" in self.last_visualization:
            dashboard_data["network"] = self.last_visualization["network"]
        
        # Add historical awareness data
        if self.metrics_history:
            awareness_history = []
            for entry in self.metrics_history:
                if "timestamp" in entry and "awareness" in entry:
                    awareness_history.append({
                        "timestamp": entry["timestamp"],
                        "value": entry["awareness"]
                    })
            dashboard_data["history"]["awareness"] = awareness_history
        
        return dashboard_data


# Create singleton instance
_visualizer = None

def get_consciousness_visualizer(config: Dict[str, Any] = None) -> ConsciousnessVisualizer:
    """
    Get or create the consciousness visualizer instance
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        ConsciousnessVisualizer instance
    """
    global _visualizer
    
    if _visualizer is None:
        _visualizer = ConsciousnessVisualizer(config)
        
    return _visualizer


def process_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process consciousness metrics (convenience function)
    
    Args:
        metrics: Consciousness metrics dictionary
        
    Returns:
        Processed visualization data
    """
    visualizer = get_consciousness_visualizer()
    return visualizer.process_metrics(metrics)


def process_network(network_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process consciousness network (convenience function)
    
    Args:
        network_data: Network data with nodes and edges
        
    Returns:
        Processed visualization data
    """
    visualizer = get_consciousness_visualizer()
    return visualizer.process_network(network_data) 