import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class DreamVisualizer:
    """Provides visualization data for dream mode monitoring dashboard."""
    
    def __init__(self, dream_archive=None, output_dir=None):
        """Initialize the dream visualizer.
        
        Args:
            dream_archive: Reference to the dream archive for accessing dream records
            output_dir: Directory to store visualization data
        """
        self.dream_archive = dream_archive
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        self.current_dream_state = {
            "active": False,
            "start_time": None,
            "duration": 0,
            "intensity": 0,
            "phase": None,
            "patterns": [],
            "insights": []
        }
        
        self.dream_metrics = {
            "consolidation_progress": [],
            "pattern_discovery_rate": [],
            "insight_generation": [],
            "creativity_index": [],
            "dream_depth": []
        }
        
        self.last_update = datetime.now()
        logging.info("Dream visualizer initialized")
    
    def update_dream_state(self, dream_state: Dict[str, Any]) -> None:
        """Update the current dream state for visualization.
        
        Args:
            dream_state: Current state of the dream process
        """
        self.current_dream_state.update(dream_state)
        
        # Extract metrics
        if "metrics" in dream_state:
            timestamp = datetime.now().isoformat()
            metrics = dream_state["metrics"]
            
            for metric in self.dream_metrics:
                if metric in metrics:
                    self.dream_metrics[metric].append({
                        "timestamp": timestamp,
                        "value": metrics[metric]
                    })
                    
                    # Keep history limited
                    if len(self.dream_metrics[metric]) > 100:
                        self.dream_metrics[metric] = self.dream_metrics[metric][-100:]
        
        self.last_update = datetime.now()
        
        # Save to file if output directory is set
        if self.output_dir:
            self._save_visualization_data()
    
    def _save_visualization_data(self) -> None:
        """Save current visualization data to disk."""
        try:
            output_file = os.path.join(self.output_dir, "dream_data.json")
            data = {
                "current_state": self.current_dream_state,
                "metrics": {k: v[-1] for k, v in self.dream_metrics.items() if v},
                "metrics_history": self.dream_metrics,
                "last_update": self.last_update.isoformat()
            }
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save dream visualization data: {e}")
    
    def get_dream_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent dream history from the archive.
        
        Args:
            limit: Maximum number of dream records to retrieve
            
        Returns:
            List of recent dream records
        """
        if not self.dream_archive:
            return self._generate_sample_dream_history(limit)
            
        try:
            return self.dream_archive.get_recent_dreams(limit)
        except:
            logging.warning("Failed to get dream history from archive, using sample data")
            return self._generate_sample_dream_history(limit)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the monitoring dashboard.
        
        Returns:
            dict: Data formatted for the monitoring dashboard
        """
        # Generate sample data if no real data exists
        if not any(self.dream_metrics.values()):
            self._generate_sample_data()
            
        # Get dream history
        dream_history = self.get_dream_history(5)
        
        # Format data for dashboard
        return {
            "current_state": self.current_dream_state,
            "metrics": {
                k: v[-1]["value"] if v else 0 
                for k, v in self.dream_metrics.items()
            },
            "history": {
                "times": [entry["timestamp"] for entry in self.dream_metrics["dream_depth"][-20:] if self.dream_metrics["dream_depth"]],
                "values": {
                    k: [entry["value"] for entry in v[-20:]] 
                    for k, v in self.dream_metrics.items() if v
                }
            },
            "dream_history": dream_history,
            "last_update": self.last_update.isoformat()
        }
    
    def _generate_sample_data(self) -> None:
        """Generate sample data for demonstration purposes."""
        now = datetime.now()
        for i in range(20):
            timestamp = (now - timedelta(minutes=20-i)).isoformat()
            for metric in self.dream_metrics.keys():
                # Generate slightly varying values with some randomness
                base_val = 0.4 + (i * 0.025)  # Starts at ~0.4, increases to ~0.9
                random_factor = random.uniform(-0.15, 0.15)
                value = max(0, min(1, base_val + random_factor))
                
                self.dream_metrics[metric].append({
                    "timestamp": timestamp,
                    "value": value
                })
        
        # Set sample current state
        self.current_dream_state = {
            "active": True if random.random() > 0.5 else False,
            "start_time": (now - timedelta(minutes=random.randint(5, 30))).isoformat() if random.random() > 0.5 else None,
            "duration": random.randint(0, 1800), # 0-30 minutes in seconds
            "intensity": random.uniform(0.3, 0.9),
            "phase": random.choice(["consolidation", "synthesis", "integration", None]),
            "patterns": [
                {"id": 1, "strength": 0.78, "description": "User interaction pattern delta-4"},
                {"id": 2, "strength": 0.62, "description": "Knowledge web expansion node"}
            ],
            "insights": [
                {"id": 1, "confidence": 0.85, "content": "Connection between recent queries suggests focus on AI safety topics"},
                {"id": 2, "confidence": 0.72, "content": "Emerging pattern in user problem-solving approach detected"}
            ]
        }
        
        self.last_update = datetime.now()
    
    def _generate_sample_dream_history(self, limit: int) -> List[Dict[str, Any]]:
        """Generate sample dream history data.
        
        Args:
            limit: Number of sample dreams to generate
            
        Returns:
            List of sample dream records
        """
        now = datetime.now()
        sample_dreams = []
        
        for i in range(limit):
            hours_ago = i * 4 + random.randint(0, 3)  # Spread dreams out over time
            start_time = now - timedelta(hours=hours_ago)
            duration = random.randint(5, 30) * 60  # 5-30 minutes in seconds
            
            dream = {
                "id": f"dream_{i}",
                "start_time": start_time.isoformat(),
                "end_time": (start_time + timedelta(seconds=duration)).isoformat(),
                "duration": duration,
                "intensity": random.uniform(0.3, 0.9),
                "insights_count": random.randint(1, 8),
                "patterns_discovered": random.randint(2, 12),
                "primary_focus": random.choice([
                    "memory consolidation", 
                    "pattern synthesis", 
                    "creativity enhancement",
                    "knowledge integration"
                ]),
                "success_rating": random.uniform(0.5, 0.95)
            }
            
            sample_dreams.append(dream)
            
        return sample_dreams 