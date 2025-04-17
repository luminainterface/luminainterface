#!/usr/bin/env python3
"""
V7-V8 Bridge Controller

This module provides a unified controller for managing both the v7 to v8 
and v8 to v7 bridges. It ensures proper startup, shutdown, and monitoring
of bridge processes.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/bridge_controller_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bridge_controller")

class BridgeController:
    """
    Controller for managing bidirectional bridges between v7 and v8 systems.
    """
    
    def __init__(self, v7_port=5678, v8_port=8765, sync_interval=30):
        """Initialize the controller with configuration parameters"""
        self.v7_port = v7_port
        self.v8_port = v8_port
        self.sync_interval = sync_interval
        self.v7_to_v8_bridge = None
        self.v8_to_v7_bridge = None
        self.running = False
        self.monitor_thread = None
        self.status_check_interval = 60  # Check bridge status every minute
        
        try:
            # Import bridge modules
            from src.bridge.v7_to_v8_bridge import V7ToV8Bridge
            from src.bridge.v8_to_v7_knowledge_bridge import V8ToV7KnowledgeBridge
            
            # Create bridge instances
            self.v7_to_v8_bridge = V7ToV8Bridge(
                v7_connection_port=v7_port,
                v8_health_port=v8_port
            )
            self.v7_to_v8_bridge.sync_interval = sync_interval
            
            self.v8_to_v7_bridge = V8ToV7KnowledgeBridge(
                v8_health_port=v8_port
            )
            self.v8_to_v7_bridge.sync_interval = sync_interval
            
            logger.info("Bridge controller initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import bridge modules: {e}")
            logger.error("Make sure v7_to_v8_bridge.py and v8_to_v7_knowledge_bridge.py exist")
            raise
    
    def start_bridges(self):
        """Start both bridge processes"""
        if self.running:
            logger.info("Bridges are already running")
            return
            
        # Start v7 to v8 bridge
        if self.v7_to_v8_bridge:
            try:
                self.v7_to_v8_bridge.start()
                logger.info("V7 to V8 bridge started")
            except Exception as e:
                logger.error(f"Failed to start V7 to V8 bridge: {e}")
        
        # Start v8 to v7 bridge
        if self.v8_to_v7_bridge:
            try:
                self.v8_to_v7_bridge.start()
                logger.info("V8 to V7 bridge started")
            except Exception as e:
                logger.error(f"Failed to start V8 to V7 bridge: {e}")
        
        # Start bridge monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_bridges)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Bridge controller started")
    
    def stop_bridges(self):
        """Stop both bridge processes"""
        self.running = False
        
        # Stop monitoring thread
        if self.monitor_thread:
            try:
                self.monitor_thread.join(timeout=2.0)
            except Exception as e:
                logger.error(f"Error stopping monitor thread: {e}")
        
        # Stop v7 to v8 bridge
        if self.v7_to_v8_bridge:
            try:
                self.v7_to_v8_bridge.stop()
                logger.info("V7 to V8 bridge stopped")
            except Exception as e:
                logger.error(f"Error stopping V7 to V8 bridge: {e}")
        
        # Stop v8 to v7 bridge
        if self.v8_to_v7_bridge:
            try:
                self.v8_to_v7_bridge.stop()
                logger.info("V8 to V7 bridge stopped")
            except Exception as e:
                logger.error(f"Error stopping V8 to V7 bridge: {e}")
        
        logger.info("Bridge controller stopped")
    
    def _monitor_bridges(self):
        """Monitor the health of bridge processes and restart if needed"""
        while self.running:
            try:
                # Check v7 to v8 bridge status
                if self.v7_to_v8_bridge:
                    v7_to_v8_status = getattr(self.v7_to_v8_bridge, "running", False)
                    if not v7_to_v8_status and self.running:
                        logger.warning("V7 to V8 bridge is not running, attempting restart")
                        try:
                            self.v7_to_v8_bridge.start()
                            logger.info("V7 to V8 bridge restarted")
                        except Exception as e:
                            logger.error(f"Failed to restart V7 to V8 bridge: {e}")
                
                # Check v8 to v7 bridge status
                if self.v8_to_v7_bridge:
                    v8_to_v7_status = getattr(self.v8_to_v7_bridge, "running", False)
                    if not v8_to_v7_status and self.running:
                        logger.warning("V8 to V7 bridge is not running, attempting restart")
                        try:
                            self.v8_to_v7_bridge.start()
                            logger.info("V8 to V7 bridge restarted")
                        except Exception as e:
                            logger.error(f"Failed to restart V8 to V7 bridge: {e}")
                
                # Log overall bridge status
                if self.running:
                    self._log_status()
                
            except Exception as e:
                logger.error(f"Error in bridge monitor: {e}")
            
            # Wait for next check
            for _ in range(self.status_check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _log_status(self):
        """Log the current status of bridges"""
        try:
            v7_to_v8_status = "Unknown"
            v8_to_v7_status = "Unknown"
            
            # Get v7 to v8 bridge status
            if self.v7_to_v8_bridge:
                try:
                    status = getattr(self.v7_to_v8_bridge, "get_status", lambda: {})()
                    v7_to_v8_status = "Running" if status.get("running", False) else "Stopped"
                except Exception as e:
                    logger.error(f"Error getting V7 to V8 bridge status: {e}")
            
            # Get v8 to v7 bridge status
            if self.v8_to_v7_bridge:
                try:
                    status = getattr(self.v8_to_v7_bridge, "get_status", lambda: {})()
                    v8_to_v7_status = "Running" if status.get("running", False) else "Stopped"
                except Exception as e:
                    logger.error(f"Error getting V8 to V7 bridge status: {e}")
            
            logger.info(f"Bridge Status - V7→V8: {v7_to_v8_status}, V8→V7: {v8_to_v7_status}")
            
            # Save status to file for external monitoring
            status_file = os.path.join(project_root, "logs", "bridge_status.json")
            try:
                with open(status_file, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "controller_running": self.running,
                        "v7_to_v8_bridge": v7_to_v8_status,
                        "v8_to_v7_bridge": v8_to_v7_status
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving bridge status: {e}")
                
        except Exception as e:
            logger.error(f"Error logging bridge status: {e}")
    
    def get_status(self):
        """Get the current status of bridges"""
        v7_to_v8_status = {}
        v8_to_v7_status = {}
        
        # Get v7 to v8 bridge status
        if self.v7_to_v8_bridge:
            try:
                v7_to_v8_status = getattr(self.v7_to_v8_bridge, "get_status", lambda: {})()
            except Exception as e:
                logger.error(f"Error getting V7 to V8 bridge status: {e}")
        
        # Get v8 to v7 bridge status
        if self.v8_to_v7_bridge:
            try:
                v8_to_v7_status = getattr(self.v8_to_v7_bridge, "get_status", lambda: {})()
            except Exception as e:
                logger.error(f"Error getting V8 to V7 bridge status: {e}")
        
        return {
            "controller_running": self.running,
            "v7_port": self.v7_port,
            "v8_port": self.v8_port,
            "sync_interval": self.sync_interval,
            "v7_to_v8_bridge": v7_to_v8_status,
            "v8_to_v7_bridge": v8_to_v7_status,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function to run the bridge controller"""
    parser = argparse.ArgumentParser(description="V7-V8 Bridge Controller")
    parser.add_argument("--v7-port", type=int, default=5678, help="V7 connection port")
    parser.add_argument("--v8-port", type=int, default=8765, help="V8 health check port")
    parser.add_argument("--sync-interval", type=int, default=30, help="Bridge sync interval in seconds")
    parser.add_argument("--status-interval", type=int, default=60, help="Status check interval in seconds")
    args = parser.parse_args()
    
    try:
        # Create and start bridge controller
        controller = BridgeController(
            v7_port=args.v7_port,
            v8_port=args.v8_port,
            sync_interval=args.sync_interval
        )
        controller.status_check_interval = args.status_interval
        controller.start_bridges()
        
        logger.info("Bridge controller running. Press Ctrl+C to stop.")
        
        # Main loop
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping bridges...")
        
    except Exception as e:
        logger.error(f"Error in main controller: {e}")
        
    finally:
        # Stop bridges on exit
        try:
            controller.stop_bridges()
        except:
            pass

if __name__ == "__main__":
    main() 