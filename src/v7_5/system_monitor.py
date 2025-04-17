#!/usr/bin/env python
"""
LUMINA V7.5 System Monitor
Displays system status and metrics for LUMINA components
"""

import os
import sys
import time
import random
import datetime
import argparse
import logging
import json
from threading import Thread
from collections import deque

# Configure logging
os.makedirs(os.path.join('logs', 'monitor'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'monitor', 'system_monitor.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SystemMonitor")

class SystemMonitor:
    """LUMINA V7.5 System Monitor class"""
    
    def __init__(self, mock_mode=False, update_interval=1.0):
        """Initialize the system monitor
        
        Args:
            mock_mode (bool): Whether to generate mock data if real components can't be connected
            update_interval (float): How often to update system metrics (in seconds)
        """
        self.mock_mode = mock_mode
        self.update_interval = update_interval
        self.running = False
        self.component_status_file = "component_status.json"
        
        # Component statuses
        self.component_status = {
            "neural_seed": {"active": False, "status": "Offline", "metrics": {}},
            "consciousness": {"active": False, "status": "Offline", "metrics": {}},
            "holographic_ui": {"active": False, "status": "Offline", "metrics": {}},
            "chat_interface": {"active": False, "status": "Offline", "metrics": {}},
            "database_connector": {"active": False, "status": "Offline", "metrics": {}},
            "autowiki": {"active": False, "status": "Offline", "metrics": {}}
        }
        
        # System metrics
        self.system_metrics = {
            "consciousness_level": 0.0,
            "neural_activity": 0.0,
            "topics_count": 0,
            "memory_utilization": 0.0,
            "uptime": 0,
            "exchanges_count": 0
        }
        
        # Track neural activity over time
        self.neural_activity_history = deque(maxlen=20)  # Last 20 seconds
        
        # Initialize mock data if in mock mode
        if self.mock_mode:
            self._init_mock_data()
            
        logger.info(f"System monitor initialized (mock_mode: {mock_mode})")
    
    def _init_mock_data(self):
        """Initialize mock data for simulation"""
        # Set some components to active
        for component in ["neural_seed", "consciousness", "chat_interface"]:
            self.component_status[component]["active"] = True
            self.component_status[component]["status"] = "Active"
        
        # Set initial system metrics
        self.system_metrics["consciousness_level"] = 0.5
        self.system_metrics["neural_activity"] = 0.3
        self.system_metrics["topics_count"] = 5
        self.system_metrics["memory_utilization"] = 0.25
        self.system_metrics["uptime"] = 300  # 5 minutes
        self.system_metrics["exchanges_count"] = 3
        
        # Initialize neural activity history
        self.neural_activity_history.extend([0.3] * 5)
        
        # Add some component-specific metrics
        self.component_status["neural_seed"]["metrics"] = {
            "growth_rate": "medium",
            "neural_connections": 1250
        }
        
        self.component_status["consciousness"]["metrics"] = {
            "level": 0.5,
            "active_nodes": 8
        }
        
        self.component_status["chat_interface"]["metrics"] = {
            "exchanges": 3,
            "response_time_ms": 350
        }
    
    def _update_mock_data(self):
        """Update mock data for simulation"""
        # Read latest component status from file
        if os.path.exists(self.component_status_file):
            try:
                with open(self.component_status_file, 'r') as f:
                    self.component_status = json.load(f)
                logger.debug("Read component status from file")
            except Exception as e:
                logger.error(f"Error reading component status file: {e}")
        
        # Update only the system metric values
        # Gradually increase consciousness level and neural activity
        self.system_metrics["consciousness_level"] = min(0.95, 
                                                       self.system_metrics["consciousness_level"] + random.uniform(-0.03, 0.05))
        
        # Update neural activity with some randomness
        new_activity = min(0.95, max(0.1, self.system_metrics["neural_activity"] + random.uniform(-0.1, 0.1)))
        self.system_metrics["neural_activity"] = new_activity
        self.neural_activity_history.append(new_activity)
        
        # Update other metrics
        self.system_metrics["uptime"] += self.update_interval
        
        # Update exchange count from chat interface if available
        chat_interface = self.component_status.get("chat_interface", {})
        if chat_interface.get("active", False) and "metrics" in chat_interface:
            self.system_metrics["exchanges_count"] = chat_interface["metrics"].get("exchanges", 0)
        
        # Update topic count sometimes
        if random.random() < 0.05:  # 5% chance of new topic
            self.system_metrics["topics_count"] = min(20, self.system_metrics["topics_count"] + 1)
        
        # Update memory utilization to fluctuate
        self.system_metrics["memory_utilization"] = min(0.95, max(0.1, 
                                                                self.system_metrics["memory_utilization"] + 
                                                                random.uniform(-0.05, 0.05)))
        
        # Update neural seed connections if active
        neural_seed = self.component_status.get("neural_seed", {})
        if neural_seed.get("active", False) and "metrics" in neural_seed:
            if "neural_connections" not in neural_seed["metrics"]:
                neural_seed["metrics"]["neural_connections"] = 1250
            neural_seed["metrics"]["neural_connections"] += random.randint(0, 5)
    
    def _load_component_status(self):
        """Load component status from file if available"""
        if os.path.exists(self.component_status_file):
            try:
                with open(self.component_status_file, 'r') as f:
                    status_data = json.load(f)
                    # Update our component status with file data
                    for component, data in status_data.items():
                        if component in self.component_status:
                            self.component_status[component] = data
                logger.debug("Loaded component status from file")
                return True
            except Exception as e:
                logger.error(f"Error loading component status file: {e}")
        return False
    
    def _get_system_status(self):
        """Get current system status (either from real components or mock data)"""
        # First try to load component status from file
        file_status_loaded = self._load_component_status()
        
        if self.mock_mode:
            # In mock mode, update system metrics but preserve component status
            # from the status file if available
            if file_status_loaded:
                self._update_mock_data()  # Updates metrics without changing component status
            else:
                # If no file exists, initialize and update mock data
                if not any(c.get("active", False) for c in self.component_status.values()):
                    self._init_mock_data()
                self._update_mock_data()
        else:
            # In real mode, try to get actual system status
            # For now, just use the file status if available
            if not file_status_loaded:
                logger.warning("Real component integration not implemented - using mock data")
                self._init_mock_data()
                self._update_mock_data()
            else:
                self._update_mock_data()  # Just update the metrics
            
        return {
            "components": self.component_status,
            "metrics": self.system_metrics,
            "neural_activity_history": list(self.neural_activity_history),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def display_console(self, status=None):
        """Display system status in console display"""
        if not status:
            status = self._get_system_status()
        
        components = status["components"]
        metrics = status["metrics"]
        history = status["neural_activity_history"] 
        timestamp = status["timestamp"]
        
        # Clear screen (platform independent)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display header
        print("="*80)
        print(f"    LUMINA V7.5 SYSTEM MONITOR                      {timestamp}")
        print("="*80)
        
        # Display component status
        print("\nCOMPONENT STATUS:")
        print("-----------------")
        
        # Component headers
        headers = ["Component", "Status", "Details"]
        print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]}")
        print("-" * 70)
        
        # Display each component
        for name, component in components.items():
            # Format component name for display
            display_name = name.replace("_", " ").title()
            
            # Get status with color (for terminals that support it)
            status_str = component.get("status", "Unknown")
            if status_str == "Active":
                status_display = f"\033[92m{status_str}\033[0m"  # Green
            elif status_str == "Initializing":
                status_display = f"\033[93m{status_str}\033[0m"  # Yellow
            elif status_str == "Error":
                status_display = f"\033[91m{status_str}\033[0m"  # Red
            else:
                status_display = f"\033[90m{status_str}\033[0m"  # Gray for Offline
            
            # Format metrics
            metrics_str = ""
            if "metrics" in component and component.get("active", False):
                metrics_list = []
                for k, v in component["metrics"].items():
                    metrics_list.append(f"{k.replace('_', ' ')}: {v}")
                metrics_str = ", ".join(metrics_list)
            
            # Print component line
            print(f"{display_name:<20} {status_display:<10} {metrics_str}")
        
        # Display system metrics
        print("\nSYSTEM METRICS:")
        print("--------------")
        
        # Format metrics
        print(f"Consciousness Level:   {metrics['consciousness_level']:.2f}")
        print(f"Neural Activity:       {metrics['neural_activity']:.2f}")
        print(f"Memory Utilization:    {metrics['memory_utilization']:.2f}")
        print(f"Known Topics:          {metrics['topics_count']}")
        print(f"Exchanges:             {metrics['exchanges_count']}")
        
        # Format uptime
        uptime = metrics['uptime']
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Uptime:                {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Display neural activity graph
        print("\nNEURAL ACTIVITY (last 20s):")
        print("--------------------------")
        
        if history:
            # Create a simple ASCII graph of neural activity
            max_width = 50
            for value in history:
                bar_width = int(value * max_width)
                bar = "█" * bar_width
                print(f"[{bar:<{max_width}}] {value:.2f}")
        else:
            print("No neural activity data available.")
        
        print("\nPress Ctrl+C to exit monitor")
    
    def run(self):
        """Run the system monitor continuously"""
        self.running = True
        try:
            while self.running:
                status = self._get_system_status()
                self.display_console(status)
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            logger.info("System monitor stopped by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error in system monitor: {e}")
            self.running = False

def main():
    """Main entry point for system monitor"""
    parser = argparse.ArgumentParser(description="LUMINA V7.5 System Monitor")
    parser.add_argument("--mock", action="store_true", 
                        help="Use mock data if real components can't be connected")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Update interval in seconds (default: 1.0)")
    args = parser.parse_args()
    
    try:
        monitor = SystemMonitor(mock_mode=args.mock, update_interval=args.interval)
        monitor.run()
    except Exception as e:
        logger.error(f"System monitor error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 