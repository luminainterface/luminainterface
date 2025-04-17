#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard V7 Bridge
==================

Connects the PyQt5 Dashboard to the Lumina V7 Unified System.
Acts as a data bridge, forwarding metrics and system status between components.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import argparse
import threading
import socket
import queue
from datetime import datetime
from pathlib import Path

# Add seed.py integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.seed import get_neural_seed, NeuralSeed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard_v7_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DashboardV7Bridge")

class DashboardV7Bridge:
    """Bridge between PyQt5 Dashboard and Lumina V7 Unified System"""
    
    def __init__(self, config_path=None, v7_port=5678, dashboard_db_path="data/neural_metrics.db", seed_only=False):
        """
        Initialize the dashboard bridge
        
        Args:
            config_path: Path to configuration file
            v7_port: Port used by V7 system for communication
            dashboard_db_path: Path to dashboard database
            seed_only: Flag to run only with the Neural Seed system
        """
        self.running = False
        self.config = self.load_config(config_path)
        self.v7_port = v7_port
        self.dashboard_db_path = dashboard_db_path
        self.data_queue = queue.Queue()
        self.metrics_map = {
            "neural_activity": "consciousness_level",
            "language_metrics": "mistral_activity", 
            "learning_metrics": "learning_rate",
            "system_metrics": "system_usage"
        }
        self.last_update = {}
        self.seed_only = seed_only
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.dashboard_db_path), exist_ok=True)
        
        # Initialize connection to V7
        self.v7_socket = None
        
        # Initialize the neural seed
        self.neural_seed = get_neural_seed({
            "data_dir": "data/seed",
            "dictionary_path": "data/seed/word_associations.json",
            "growth_rate": 0.001,  # Base growth rate per cycle
            "cycle_time": 60,      # Seconds per growth cycle
            "enable_self_modification": True,
            "auto_discover_components": True,
        })
        
    def load_config(self, config_path):
        """Load bridge configuration"""
        default_config = {
            "update_interval": 2.0,  # seconds
            "metrics_enabled": True,
            "log_updates": True
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.info("Using default bridge configuration")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def setup_database(self):
        """Set up dashboard database with required tables"""
        try:
            conn = sqlite3.connect(self.dashboard_db_path)
            cursor = conn.cursor()
            
            # Create tables for each metric type
            for table in self.metrics_map.keys():
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    value REAL,
                    description TEXT
                )
                """)
            
            conn.commit()
            conn.close()
            logger.info("Dashboard database setup complete")
            return True
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def connect_to_v7(self):
        """Connect to Lumina V7 system"""
        try:
            self.v7_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.v7_socket.connect(("localhost", self.v7_port))
            logger.info(f"Connected to Lumina V7 on port {self.v7_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Lumina V7: {e}")
            return False
    
    def send_to_v7(self, message):
        """Send message to Lumina V7 system"""
        if not self.v7_socket:
            if not self.connect_to_v7():
                return False
                
        try:
            self.v7_socket.sendall(json.dumps(message).encode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to V7: {e}")
            self.v7_socket = None  # Reset for reconnection attempt
            return False
    
    def receive_from_v7(self):
        """Receive data from Lumina V7 system"""
        if not self.v7_socket:
            if not self.connect_to_v7():
                return None
                
        try:
            data = self.v7_socket.recv(4096)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Failed to receive data from V7: {e}")
            self.v7_socket = None  # Reset for reconnection attempt
            return None
    
    def record_metric(self, metric_type, value, description=None):
        """Record a metric in the dashboard database"""
        if metric_type not in self.metrics_map.keys():
            logger.warning(f"Unknown metric type: {metric_type}")
            return False
            
        try:
            conn = sqlite3.connect(self.dashboard_db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"""
            INSERT INTO {metric_type} (timestamp, value, description)
            VALUES (datetime('now'), ?, ?)
            """, (value, description or f"V7 {self.metrics_map[metric_type]}"))
            
            conn.commit()
            conn.close()
            
            self.last_update[metric_type] = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False
    
    def poll_v7_metrics(self):
        """Poll Lumina V7 for metrics data"""
        request = {
            "type": "metrics_request",
            "metrics": list(self.metrics_map.values())
        }
        
        if self.send_to_v7(request):
            response = self.receive_from_v7()
            if response and "metrics" in response:
                for v7_metric, value in response["metrics"].items():
                    # Map V7 metric name to dashboard metric type
                    for dash_metric, v7_name in self.metrics_map.items():
                        if v7_name == v7_metric:
                            description = response.get("descriptions", {}).get(v7_metric)
                            self.record_metric(dash_metric, value, description)
                            break
                
                return True
        
        # If we reach here, communication failed
        # Generate mock data for testing
        self.generate_mock_metrics()
        return False
    
    def generate_mock_metrics(self):
        """Generate mock metrics for testing or when V7 is unavailable"""
        import random
        
        # Generate mock values with reasonable ranges
        mock_values = {
            "neural_activity": random.uniform(0.3, 0.9),
            "language_metrics": random.uniform(0.4, 0.8),
            "learning_metrics": random.uniform(0.1, 0.6),
            "system_metrics": random.uniform(0.2, 0.7)
        }
        
        # Record mock metrics
        for metric_type, value in mock_values.items():
            current_time = time.time()
            last_update = self.last_update.get(metric_type, 0)
            
            # Only update if it's been at least 10 seconds
            if current_time - last_update >= 10:
                self.record_metric(
                    metric_type, 
                    value, 
                    f"Mock {self.metrics_map[metric_type]} (V7 unreachable)"
                )
        
        # Add seed metrics
        self._record_seed_metrics()
    
    def _record_seed_metrics(self):
        """Record metrics from the Neural Seed system"""
        try:
            # Get status from the neural seed
            seed_status = self.neural_seed.get_status()
            
            # Record key seed metrics
            self.record_metric(
                "neural_activity", 
                seed_status['metrics']['consciousness_level'],
                f"Neural Seed Consciousness (v{seed_status['version']:.2f})"
            )
            
            self.record_metric(
                "learning_metrics", 
                seed_status['metrics']['integration_level'],
                f"Neural Seed Integration ({seed_status['growth_stage']})"
            )
            
            # Add mold growth metrics
            self._record_mold_growth_metrics(seed_status)
            
            # Process input to stimulate growth
            self.neural_seed.process_input({
                "text": f"Dashboard integration is collecting metrics at {datetime.now().isoformat()}",
                "source": "dashboard_bridge",
                "type": "system_metrics"
            })
            
            logger.debug(f"Recorded seed metrics: v{seed_status['version']:.2f}, {seed_status['growth_stage']}")
            
        except Exception as e:
            logger.error(f"Error recording seed metrics: {e}")
    
    def _record_mold_growth_metrics(self, seed_status):
        """Record metrics related to the mold-like growth of the neural seed"""
        try:
            # Generate growth visualization data
            growth_data = self._generate_mold_growth_data(seed_status)
            
            # Record mold metrics
            self.record_metric(
                "neural_activity",
                growth_data["growth_factor"],
                f"Neural Mold Growth Factor"
            )
            
            self.record_metric(
                "learning_metrics",
                growth_data["spread_rate"],
                f"Neural Mold Spread Rate"
            )
            
            # Store attachment points data in the database
            self._store_mold_attachment_points(growth_data["attachment_points"])
            
            logger.debug(f"Recorded mold growth metrics: {len(growth_data['attachment_points'])} attachment points")
            
        except Exception as e:
            logger.error(f"Error recording mold growth metrics: {e}")
    
    def _generate_mold_growth_data(self, seed_status):
        """
        Generate data representing the mold-like growth of the neural seed
        
        This visualizes how the neural seed grows like mold, finding patterns
        in the neural network and attaching to them to build larger structures.
        """
        import random
        import math
        
        # Base data from seed status
        version = seed_status["version"]
        growth_stage = seed_status["growth_stage"]
        dictionary_size = seed_status["dictionary_size"]
        consciousness = seed_status["metrics"]["consciousness_level"]
        
        # Calculate growth factor based on version and dictionary size
        # Higher version and larger dictionary = more growth
        growth_factor = min(0.95, (version / 10.0) * (dictionary_size / 1000.0))
        
        # Calculate spread rate based on growth stage
        # More advanced growth stages spread faster
        stage_factors = {
            "seed": 0.2,
            "root": 0.4,
            "trunk": 0.6,
            "branch": 0.8,
            "canopy": 0.9,
            "flower": 1.0,
            "fruit": 1.0
        }
        spread_rate = stage_factors.get(growth_stage, 0.5) * consciousness
        
        # Generate attachment points (where the mold has attached to neural patterns)
        attachment_points = []
        
        # Number of attachment points increases with version and consciousness
        num_points = int(10 + (version * 5) + (consciousness * 20))
        
        for i in range(num_points):
            # Generate realistic coordinates for visualization
            # Use perlin-like noise pattern for natural-looking growth
            angle = random.uniform(0, math.pi * 2)
            distance = random.uniform(0.1, 0.9)
            
            # Adjust distance based on version (higher version = more spread out)
            distance = distance * (0.5 + (version / 10.0))
            
            # Calculate x,y coordinates for visualization
            x = 0.5 + math.cos(angle) * distance
            y = 0.5 + math.sin(angle) * distance
            
            # Strength represents how firmly the mold has attached to this point
            # Higher consciousness = stronger attachment
            strength = random.uniform(0.3, 0.9) * consciousness
            
            # Generate random color variants for visualization
            # More advanced growth stages get different colors
            color_factors = {
                "seed": {"r": 0.7, "g": 0.7, "b": 0.9},  # Light blue
                "root": {"r": 0.6, "g": 0.8, "b": 0.9},  # Blue-green
                "trunk": {"r": 0.5, "g": 0.8, "b": 0.7},  # Green
                "branch": {"r": 0.7, "g": 0.8, "b": 0.5},  # Yellow-green
                "canopy": {"r": 0.8, "g": 0.7, "b": 0.5},  # Gold
                "flower": {"r": 0.9, "g": 0.6, "b": 0.7},  # Pink
                "fruit": {"r": 0.9, "g": 0.5, "b": 0.5}   # Red
            }
            
            base_color = color_factors.get(growth_stage, {"r": 0.7, "g": 0.7, "b": 0.7})
            
            # Add some random variation to the color
            r = min(1.0, base_color["r"] + random.uniform(-0.1, 0.1))
            g = min(1.0, base_color["g"] + random.uniform(-0.1, 0.1))
            b = min(1.0, base_color["b"] + random.uniform(-0.1, 0.1))
            
            # Add the attachment point
            attachment_points.append({
                "x": x,
                "y": y,
                "strength": strength,
                "color": {"r": r, "g": g, "b": b},
                "type": "neural_pattern" if random.random() > 0.5 else "memory_node",
                "size": random.uniform(0.5, 2.0) * strength
            })
        
        return {
            "growth_factor": growth_factor,
            "spread_rate": spread_rate,
            "attachment_points": attachment_points,
            "generation_time": datetime.now().isoformat()
        }
    
    def _store_mold_attachment_points(self, attachment_points):
        """Store mold attachment points in the database for visualization"""
        try:
            conn = sqlite3.connect(self.dashboard_db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS neural_mold_growth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                x REAL,
                y REAL,
                strength REAL,
                color_r REAL,
                color_g REAL,
                color_b REAL,
                type TEXT,
                size REAL
            )
            ''')
            
            # Clear old data (keep only the latest 1000 points for performance)
            cursor.execute('''
            DELETE FROM neural_mold_growth
            WHERE id NOT IN (
                SELECT id FROM neural_mold_growth
                ORDER BY timestamp DESC
                LIMIT 1000
            )
            ''')
            
            # Insert new attachment points
            current_time = datetime.now().isoformat()
            
            for point in attachment_points:
                cursor.execute('''
                INSERT INTO neural_mold_growth 
                (timestamp, x, y, strength, color_r, color_g, color_b, type, size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    current_time,
                    point["x"],
                    point["y"],
                    point["strength"],
                    point["color"]["r"],
                    point["color"]["g"],
                    point["color"]["b"],
                    point["type"],
                    point["size"]
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing mold attachment points: {e}")
    
    def run(self):
        """Run the bridge"""
        if not self.setup_database():
            logger.error("Failed to set up database, exiting")
            return False
            
        logger.info("Starting Dashboard V7 Bridge")
        self.running = True
        
        # Start the neural seed growth process
        logger.info("Starting Neural Seed growth process")
        self.neural_seed.start_growth()
        
        try:
            while self.running:
                # Poll V7 for metrics
                self.poll_v7_metrics()
                
                # Poll Neural Seed metrics
                self._record_seed_metrics()
                
                # Sleep for update interval
                time.sleep(self.config["update_interval"])
        except KeyboardInterrupt:
            logger.info("Bridge stopped by user")
        except Exception as e:
            logger.error(f"Error in bridge: {e}")
        finally:
            self.running = False
            if self.v7_socket:
                self.v7_socket.close()
            
            # Stop the neural seed growth process
            logger.info("Stopping Neural Seed growth process")
            self.neural_seed.stop_growth()
                
        logger.info("Dashboard V7 Bridge stopped")
        return True
        
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Dashboard V7 Bridge")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--v7-port", type=int, default=5678, help="Lumina V7 communication port")
    parser.add_argument("--db", default="data/neural_metrics.db", help="Path to dashboard database")
    parser.add_argument("--seed-only", action="store_true", help="Only connect to the Neural Seed system")
    args = parser.parse_args()
    
    bridge = DashboardV7Bridge(
        config_path=args.config,
        v7_port=args.v7_port,
        dashboard_db_path=args.db,
        seed_only=args.seed_only
    )
    
    success = bridge.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 