#!/usr/bin/env python
"""
LUMINA v7.5 System Monitor
--------------------------
A real-time monitoring tool for the LUMINA v7.5 system components.
Provides console-based visualization of system metrics and component status.
"""

import os
import sys
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from collections import deque

# Ensure the correct import paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path("logs/v7.5_monitor.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LUMINA_Monitor")

class SystemMonitor:
    """
    System Monitor for LUMINA v7.5
    Provides real-time monitoring of system components and metrics
    """
    
    def __init__(self, mock_mode=False, update_interval=1.0):
        """Initialize the system monitor"""
        self.mock_mode = mock_mode
        self.update_interval = update_interval
        self.stop_event = Event()
        
        # Component status tracking
        self.component_status = {
            "neural_seed": False,
            "chat_interface": False,
            "holographic_interface": False,
            "autowiki": False,
            "consciousness": False,
            "memory": False,
            "database": False
        }
        
        # System metrics tracking
        self.system_metrics = {
            "consciousness_level": 0.0,
            "neural_activity": 0.0,
            "memory_usage": 0.0,
            "exchange_count": 0,
            "topic_count": 0
        }
        
        # Store historical neural activity (last 20 seconds)
        self.neural_history = deque(maxlen=20)
        
        # Initialize with mock data if needed
        if self.mock_mode:
            self._init_mock_data()
            
        # Try to import the system integration if not in mock mode
        if not self.mock_mode:
            try:
                from system_integration import get_system_integration
                self.system = get_system_integration()
                logger.info("Connected to system integration")
            except ImportError:
                logger.warning("Could not import system integration, falling back to mock mode")
                self.mock_mode = True
                self._init_mock_data()
    
    def _init_mock_data(self):
        """Initialize with mock data"""
        # Set random initial component status
        for component in self.component_status:
            self.component_status[component] = random.choice([True, False])
            
        # Set random initial metrics
        self.system_metrics["consciousness_level"] = random.uniform(0.2, 0.8)
        self.system_metrics["neural_activity"] = random.uniform(0.1, 0.9)
        self.system_metrics["memory_usage"] = random.uniform(0.3, 0.7)
        self.system_metrics["exchange_count"] = random.randint(10, 100)
        self.system_metrics["topic_count"] = random.randint(5, 20)
        
        # Initialize neural history
        for _ in range(20):
            self.neural_history.append(random.uniform(0.1, 0.9))
    
    def _update_mock_data(self):
        """Update mock data with some realistic variations"""
        # Randomly toggle component status (10% chance)
        for component in self.component_status:
            if random.random() < 0.1:
                self.component_status[component] = not self.component_status[component]
        
        # Update metrics with small variations
        self.system_metrics["consciousness_level"] += random.uniform(-0.05, 0.05)
        self.system_metrics["consciousness_level"] = max(0.0, min(1.0, self.system_metrics["consciousness_level"]))
        
        self.system_metrics["neural_activity"] += random.uniform(-0.1, 0.1)
        self.system_metrics["neural_activity"] = max(0.0, min(1.0, self.system_metrics["neural_activity"]))
        
        self.system_metrics["memory_usage"] += random.uniform(-0.05, 0.05)
        self.system_metrics["memory_usage"] = max(0.0, min(1.0, self.system_metrics["memory_usage"]))
        
        if random.random() < 0.3:  # 30% chance to increment exchange count
            self.system_metrics["exchange_count"] += 1
            
        if random.random() < 0.1:  # 10% chance to change topic count
            self.system_metrics["topic_count"] += random.choice([-1, 0, 1])
            self.system_metrics["topic_count"] = max(0, self.system_metrics["topic_count"])
        
        # Update neural history
        self.neural_history.append(self.system_metrics["neural_activity"])
    
    def _update_from_system(self):
        """Update data from the actual system integration"""
        try:
            system_state = self.system.get_system_state()
            
            # Update component status
            for component in self.component_status:
                if component in system_state["components"]:
                    self.component_status[component] = system_state["components"][component]
            
            # Update metrics
            if "consciousness_level" in system_state:
                self.system_metrics["consciousness_level"] = system_state["consciousness_level"]
                
            if "neural_activity" in system_state:
                self.system_metrics["neural_activity"] = system_state["neural_activity"]
                
            if "memory_usage" in system_state:
                self.system_metrics["memory_usage"] = system_state["memory_usage"]
                
            if "conversation_context" in system_state:
                if "exchange_count" in system_state["conversation_context"]:
                    self.system_metrics["exchange_count"] = system_state["conversation_context"]["exchange_count"]
                    
                if "active_topics" in system_state["conversation_context"]:
                    self.system_metrics["topic_count"] = len(system_state["conversation_context"]["active_topics"])
            
            # Update neural history
            self.neural_history.append(self.system_metrics["neural_activity"])
            
        except Exception as e:
            logger.error(f"Error updating from system: {e}")
            # Fall back to mock data if system update fails
            self._update_mock_data()
    
    def update(self):
        """Update the system monitor data"""
        if self.mock_mode:
            self._update_mock_data()
        else:
            self._update_from_system()
    
    def get_status(self):
        """Get the current system status"""
        return {
            "component_status": self.component_status.copy(),
            "system_metrics": self.system_metrics.copy(),
            "neural_history": list(self.neural_history),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mock_mode": self.mock_mode
        }
    
    def display_console(self):
        """Display the system status in the console"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display header
        print("=" * 80)
        print(f"LUMINA v7.5 SYSTEM MONITOR                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Display mock mode indicator
        if self.mock_mode:
            print("\n[MOCK MODE ACTIVE - Data is simulated]")
        
        # Display component status
        print("\nCOMPONENT STATUS:")
        print("-" * 40)
        for component, status in self.component_status.items():
            status_text = "ONLINE " if status else "OFFLINE"
            
            # Use ASCII characters instead of Unicode for cross-platform compatibility
            status_symbol = "+" if status else "X"
            
            # Convert snake_case to Title Case for display
            component_name = " ".join(word.capitalize() for word in component.split("_"))
            
            # Print status with ASCII symbols
            print(f"{component_name:20}: [{status_symbol}] {status_text}")
        
        # Display system metrics
        print("\nSYSTEM METRICS:")
        print("-" * 40)
        print(f"Consciousness Level: {self.system_metrics['consciousness_level']:.2f}")
        print(f"Neural Activity:     {self.system_metrics['neural_activity']:.2f}")
        print(f"Memory Usage:        {self.system_metrics['memory_usage']:.2f}")
        print(f"Exchange Count:      {self.system_metrics['exchange_count']}")
        print(f"Active Topics:       {self.system_metrics['topic_count']}")
        
        # Display neural activity chart (ASCII)
        print("\nNEURAL ACTIVITY (Last 20 seconds):")
        print("-" * 60)
        
        # Create a simple ASCII chart
        chart_height = 5
        for h in range(chart_height, 0, -1):
            line = ""
            for val in self.neural_history:
                threshold = h / chart_height
                if val >= threshold:
                    line += "#"  # Use # instead of █ for compatibility
                else:
                    line += " "
            print(f"{(h/chart_height):.1f} |{line}")
        
        # Chart baseline
        print("    +" + "-" * len(self.neural_history))
        print("      " + "TIME" + " " * (len(self.neural_history) - 6) + "NOW")
        
        # Display footer
        print("\n" + "=" * 80)
        print(f"Update Interval: {self.update_interval}s               Press Ctrl+C to exit")
        print("=" * 80)
    
    def run(self):
        """Run the system monitor until stopped"""
        logger.info(f"Starting system monitor (Mock Mode: {self.mock_mode})")
        
        try:
            while not self.stop_event.is_set():
                self.update()
                self.display_console()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            logger.info("System monitor stopped by user")
        finally:
            logger.info("System monitor shutdown")
    
    def start(self):
        """Start the system monitor in a separate thread"""
        self.monitor_thread = Thread(target=self.run)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return self.monitor_thread
    
    def stop(self):
        """Stop the system monitor"""
        self.stop_event.set()
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LUMINA v7.5 System Monitor")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    args = parser.parse_args()
    
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Create and start monitor
        monitor = SystemMonitor(mock_mode=args.mock, update_interval=args.interval)
        monitor.run()
    except Exception as e:
        logger.error(f"Error running system monitor: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 