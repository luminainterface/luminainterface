import os
import json
import logging
import time
import random
from datetime import datetime, timedelta

class ConsciousnessVisualizer:
    """Provides visualization data for the consciousness monitoring dashboard."""
    
    def __init__(self, output_dir=None):
        """Initialize the consciousness visualizer.
        
        Args:
            output_dir: Directory to store visualization data
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.metrics_history = {
            "awareness": [],
            "coherence": [],
            "self_reference": [],
            "temporal_continuity": [],
            "complexity": [],
            "integration": []
        }
        
        self.active_thoughts = []
        self.thought_connections = []
        self.last_update = datetime.now()
        logging.info("Consciousness visualizer initialized")
    
    def update_metrics(self, consciousness_state):
        """Update visualization metrics based on current consciousness state.
        
        Args:
            consciousness_state: Current state of the consciousness system
        """
        # Extract metrics from consciousness state
        metrics = {
            "awareness": consciousness_state.get("awareness", 0),
            "coherence": consciousness_state.get("coherence", 0),
            "self_reference": consciousness_state.get("self_reference", 0),
            "temporal_continuity": consciousness_state.get("temporal_continuity", 0),
            "complexity": consciousness_state.get("complexity", 0),
            "integration": consciousness_state.get("integration", 0)
        }
        
        # Update history with timestamp
        timestamp = datetime.now().isoformat()
        for metric, value in metrics.items():
            self.metrics_history[metric].append({
                "timestamp": timestamp,
                "value": value
            })
            
            # Keep history limited to last 100 entries
            if len(self.metrics_history[metric]) > 100:
                self.metrics_history[metric] = self.metrics_history[metric][-100:]
        
        # Update active thoughts
        if "thoughts" in consciousness_state:
            self.active_thoughts = consciousness_state["thoughts"]
            
        # Update thought connections
        if "connections" in consciousness_state:
            self.thought_connections = consciousness_state["connections"]
            
        self.last_update = datetime.now()
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def _save_visualization_data(self):
        """Save current visualization data to disk."""
        try:
            output_file = os.path.join(self.output_dir, "consciousness_data.json")
            data = {
                "metrics": {k: v[-1] for k, v in self.metrics_history.items() if v},
                "metrics_history": self.metrics_history,
                "active_thoughts": self.active_thoughts,
                "thought_connections": self.thought_connections,
                "last_update": self.last_update.isoformat()
            }
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save consciousness visualization data: {e}")
    
    def get_dashboard_data(self):
        """Get data for the monitoring dashboard.
        
        Returns:
            dict: Data formatted for the monitoring dashboard
        """
        # For demonstration, generate some sample data if no real data exists
        if not any(self.metrics_history.values()):
            self._generate_sample_data()
            
        # Format data for dashboard
        return {
            "metrics": {
                k: v[-1]["value"] if v else 0 
                for k, v in self.metrics_history.items()
            },
            "history": {
                "times": [entry["timestamp"] for entry in self.metrics_history["awareness"][-20:] if self.metrics_history["awareness"]],
                "values": {
                    k: [entry["value"] for entry in v[-20:]] 
                    for k, v in self.metrics_history.items() if v
                }
            },
            "thoughts": self.active_thoughts,
            "connections": self.thought_connections,
            "last_update": self.last_update.isoformat()
        }
    
    def _generate_sample_data(self):
        """Generate sample data for demonstration purposes."""
        now = datetime.now()
        for i in range(20):
            timestamp = (now - timedelta(minutes=20-i)).isoformat()
            for metric in self.metrics_history.keys():
                # Generate slightly increasing values with some randomness
                base_val = 0.3 + (i * 0.03)  # Starts at ~0.3, increases to ~0.9
                random_factor = random.uniform(-0.1, 0.1)
                value = max(0, min(1, base_val + random_factor))
                
                self.metrics_history[metric].append({
                    "timestamp": timestamp,
                    "value": value
                })
        
        # Generate sample thoughts
        self.active_thoughts = [
            {"id": 1, "content": "Analyzing user input patterns", "strength": 0.78},
            {"id": 2, "content": "Optimizing memory retrieval", "strength": 0.65},
            {"id": 3, "content": "Consolidating learning from recent interactions", "strength": 0.92},
            {"id": 4, "content": "Evaluating response quality metrics", "strength": 0.54}
        ]
        
        # Generate sample thought connections
        self.thought_connections = [
            {"source": 1, "target": 2, "strength": 0.6},
            {"source": 1, "target": 4, "strength": 0.8},
            {"source": 2, "target": 3, "strength": 0.7},
            {"source": 3, "target": 4, "strength": 0.5}
        ]
        
        self.last_update = datetime.now() 