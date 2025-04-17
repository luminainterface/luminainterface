#!/usr/bin/env python3
"""
V8 Connection Monitor

This module provides monitoring of connections between the different 
components of the v8 seed system. It periodically checks all components
to verify they are running and communicating properly.
"""

import os
import sys
import json
import time
import logging
import argparse
import socket
import threading
import http.client
import urllib.request
import urllib.error
import ssl
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from colorama import Fore, Style
import traceback

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Define default connections directory
DEFAULT_CONNECTIONS_DIR = os.path.join(project_root, "config", "v8", "connections")
DEFAULT_CONNECTIONS_FILE = os.path.join(DEFAULT_CONNECTIONS_DIR, "latest_connections.json")

# Setup logging
log_dir = os.path.join(project_root, "logs", "v8")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"connection_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v8.connection_monitor")

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
        QStatusBar, QProgressBar, QFrame, QGridLayout, QMenu, QAction,
        QDialog, QTextEdit, QLineEdit, QFormLayout, QMessageBox, QCheckBox,
        QScrollArea
    )
    from PySide6.QtCore import Qt, Signal, Slot, QTimer
    from PySide6.QtGui import QColor, QPalette, QFont, QIcon, QPixmap
    HAS_PYSIDE = True
except ImportError:
    logger.warning("PySide6 not available, using console mode only")
    HAS_PYSIDE = False

class SystemComponent:
    """Represents a component in the v8 system"""
    
    def __init__(self, name: str, host: str, port: int):
        self.name = name
        self.host = host
        self.port = port
        self.status = "unknown"  # "online", "offline", "unknown", "error", "degraded"
        self.last_checked = None
        self.connected_to = set()  # Names of components this is connected to
        self.response_time = None  # Response time in ms
        self.error_message = None  # Last error message
        self.health_checks = {
            "port_open": False,
            "http_api": False,
            "connection_verified": False
        }
        self.check_count = 0
        self.success_count = 0
        self.api_endpoints = {
            "health": "/health",
            "status": "/status",
            "info": "/info",
            "ping": "/ping",
            "heartbeat": "/heartbeat",
            "": "/"  # Try the root endpoint too
        }
        self.last_successful_check = None
        self.consecutive_failures = 0
        
    def check_status(self) -> str:
        """Check if the component is online and update its status"""
        # Record that we checked
        self.last_checked = datetime.now()
        self.check_count += 1
        start_time = time.time()
        self.error_message = None
        
        try:
            # 1. Try to connect to the port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.5)  # Shorter timeout for faster checking
            logger.debug(f"Checking if port is open for {self.name} at {self.host}:{self.port}")
            result = s.connect_ex((self.host, self.port))
            s.close()
            
            if result == 0:
                # Port is open
                self.health_checks["port_open"] = True
                self.response_time = int((time.time() - start_time) * 1000)
                
                # 2. Try to check the HTTP API if port is open
                try:
                    api_check = self._check_api_endpoint()
                    if api_check:
                        # API is accessible
                        self.health_checks["http_api"] = True
                        self.last_successful_check = datetime.now()
                        self.consecutive_failures = 0
                        self.status = "online"
                        self.success_count += 1
                    else:
                        # Port is open but API not responding correctly
                        self.health_checks["http_api"] = False
                        self.consecutive_failures += 1
                        self.status = "degraded"
                        self.error_message = "API endpoints not responding"
                except Exception as e:
                    # Error checking API
                    logger.debug(f"API check for {self.name} failed: {str(e)}")
                    self.health_checks["http_api"] = False
                    self.consecutive_failures += 1
                    self.status = "degraded"
                    self.error_message = f"API Error: {str(e)}"
            else:
                # Port is closed
                self.status = "offline"
                self.health_checks["port_open"] = False
                self.health_checks["http_api"] = False
                self.response_time = None
                self.consecutive_failures += 1
                self.error_message = f"Connection refused (port {self.port})"
                
        except socket.timeout:
            # Connection timeout
            logger.debug(f"Connection timeout checking {self.name}")
            self.status = "offline"
            self.error_message = "Connection timeout"
            self.response_time = None
            self.consecutive_failures += 1
            self.health_checks["port_open"] = False
            self.health_checks["http_api"] = False
        except socket.error as e:
            # Socket error
            logger.debug(f"Socket error checking {self.name}: {e}")
            self.status = "error"
            self.error_message = f"Socket error: {str(e)}"
            self.response_time = None
            self.consecutive_failures += 1
            self.health_checks["port_open"] = False
            self.health_checks["http_api"] = False
        except Exception as e:
            # Unexpected error
            logger.error(f"Error checking {self.name}: {e}")
            self.status = "error"
            self.error_message = str(e)
            self.response_time = None
            self.consecutive_failures += 1
            self.health_checks["port_open"] = False
            self.health_checks["http_api"] = False
            
        # If component has failed too many times in a row, mark as error
        if self.consecutive_failures > 5 and self.status != "error":
            logger.warning(f"Component {self.name} has failed {self.consecutive_failures} consecutive checks")
            self.status = "error"
            
        logger.debug(f"Component {self.name} status: {self.status}")
        return self.status
    
    def _check_api_endpoint(self) -> bool:
        """Try to check multiple API endpoints for health status"""
        # Set a shorter timeout for each request
        timeout = 1.0
        
        # Try all endpoints until one works
        for endpoint_name, path in self.api_endpoints.items():
            try:
                url = f"http://{self.host}:{self.port}{path}"
                logger.debug(f"Trying API endpoint {url}")
                
                # Don't verify SSL for internal components
                response = requests.get(
                    url, 
                    timeout=timeout, 
                    verify=False,
                    headers={"Connection": "close"}  # Avoid keeping connections open
                )
                
                # Check if response is successful
                if response.status_code == 200:
                    logger.debug(f"API endpoint {endpoint_name} for {self.name} is available")
                    
                    # Try to parse response as JSON to verify it's returning valid data
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            # Look for common status indicators
                            status_keys = ["status", "state", "health", "running"]
                            status_values = ["ok", "running", "available", "online", "healthy", "true", "success"]
                            
                            for key in status_keys:
                                if key in data and str(data[key]).lower() in status_values:
                                    logger.debug(f"Component {self.name} reported '{key}': '{data[key]}'")
                                    return True
                            
                            # If we found status data but it doesn't explicitly say it's good,
                            # still treat a valid JSON response as success
                            return True
                    except (ValueError, json.JSONDecodeError):
                        # Not JSON or invalid JSON, but still a 200 response
                        # Let's consider this a success for non-JSON endpoints
                        return True
                        
            except requests.exceptions.Timeout:
                logger.debug(f"API endpoint {endpoint_name} for {self.name} timed out")
                continue
            except requests.exceptions.ConnectionError:
                logger.debug(f"API endpoint {endpoint_name} for {self.name} connection error")
                continue
            except requests.exceptions.SSLError:
                logger.debug(f"API endpoint {endpoint_name} for {self.name} SSL error")
                # Try again without SSL verification (in case it's using a self-signed cert)
                try:
                    response = requests.get(url, timeout=timeout, verify=False)
                    if response.status_code == 200:
                        return True
                except Exception:
                    continue
            except Exception as e:
                logger.debug(f"API check error for {self.name} on {endpoint_name}: {e}")
                continue
                
        # If we reached here, no endpoint worked
        return False
    
    def verify_connection_to(self, target_component: 'SystemComponent') -> bool:
        """Verify that this component can connect to the target component"""
        # Quick check - both components need to have open ports
        if not self.health_checks["port_open"] or not target_component.health_checks["port_open"]:
            return False
            
        # Try to verify actual connectivity between components
        try:
            # First check if both components are at least responding
            if self.status not in ["online", "degraded"] or target_component.status not in ["online", "degraded"]:
                return False
            
            # Perform more thorough connection verification
            # For now, we'll check if both endpoints are available
            # A more sophisticated implementation would check actual communication
            # between the components via their APIs
            
            # Check if both components can be reached
            if self.health_checks["http_api"] and target_component.health_checks["http_api"]:
                self.health_checks["connection_verified"] = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying connection from {self.name} to {target_component.name}: {e}")
            return False
        
    def get_uptime_percentage(self) -> float:
        """Calculate the uptime percentage based on successful checks"""
        if self.check_count == 0:
            return 0.0
        return (self.success_count / self.check_count) * 100.0
    
    def get_time_since_last_success(self) -> Optional[float]:
        """Get time in seconds since last successful check"""
        if self.last_successful_check is None:
            return None
        return (datetime.now() - self.last_successful_check).total_seconds()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert component status to dictionary"""
        time_since_success = self.get_time_since_last_success()
        
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_successful": self.last_successful_check.isoformat() if self.last_successful_check else None,
            "time_since_success": time_since_success,
            "consecutive_failures": self.consecutive_failures,
            "connected_to": list(self.connected_to),
            "response_time": self.response_time,
            "error_message": self.error_message,
            "health_checks": self.health_checks,
            "check_count": self.check_count,
            "success_count": self.success_count,
            "uptime_percentage": self.get_uptime_percentage()
        }

class ConnectionMonitor:
    """Monitors connections between v8 system components"""
    
    def __init__(self, connections_file: str, system_key: Optional[str] = None):
        self.connections_file = connections_file
        self.system_key = system_key
        self.components: Dict[str, SystemComponent] = {}
        self.connection_map: Dict[str, List[str]] = {}
        self.running = False
        self.monitor_thread = None
        self.check_interval = 5.0  # seconds
        self.status_changed_callback = None
        self.history: List[Dict[str, Any]] = []  # History of system status
        self.history_max_size = 100  # Maximum number of history entries to keep
        self.last_check_time = None
        self.config_loaded = False
        self.startup_time = datetime.now()
        self.config_file_mtime = None  # Last modification time of config file
        self.monitor_mode = "active"  # "active", "passive", "debug"
        self._lock = threading.RLock()  # For thread safety
        
        # Try to load connections with error handling
        try:
            self.load_connections()
            self.config_loaded = True
        except Exception as e:
            logger.error(f"Failed to load connections file: {e}")
            self.config_loaded = False
        
    def load_connections(self):
        """Load connection information from config file"""
        with self._lock:
            try:
                if not self.connections_file:
                    raise ValueError("No connections file specified")
                
                # Check if file exists
                if not os.path.exists(self.connections_file):
                    logger.error(f"Connections file not found: {self.connections_file}")
                    if os.path.exists(os.path.join(project_root, "data", "v8", "connections", "latest_connections.json")):
                        # Try to use the latest connections file instead
                        latest_file = os.path.join(project_root, "data", "v8", "connections", "latest_connections.json")
                        logger.info(f"Trying latest connections file: {latest_file}")
                        self.connections_file = latest_file
                    else:
                        raise FileNotFoundError(f"Connections file not found: {self.connections_file}")
                    
                # Store the last modification time    
                self.config_file_mtime = os.path.getmtime(self.connections_file)
                logger.debug(f"Configuration file mtime: {self.config_file_mtime}")
                
                logger.debug(f"Loading connections from: {self.connections_file}")
                with open(self.connections_file, 'r', encoding='utf-8-sig') as f:
                    config_data = f.read()
                    
                # Check for empty file
                if not config_data.strip():
                    raise ValueError(f"Empty connections file: {self.connections_file}")
                
                # Remove any BOM or other non-standard characters
                config_data = config_data.strip().lstrip('\ufeff')
                
                # Parse JSON
                try:
                    config = json.loads(config_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in connections file: {e}")
                    # Try to fix common JSON issues
                    logger.info("Attempting to fix JSON formatting...")
                    try:
                        # Sometimes the JSON might have comments or trailing commas
                        # This is a quick fix attempt before giving up
                        import re
                        # Remove comments
                        config_data = re.sub(r'//.*?\n', '\n', config_data)
                        config_data = re.sub(r'/\*.*?\*/', '', config_data, flags=re.DOTALL)
                        # Fix trailing commas
                        config_data = re.sub(r',(\s*[\]}])', r'\1', config_data)
                        
                        # Print the processed JSON for debugging
                        logger.debug(f"Processed JSON: {config_data}")
                        
                        try:
                            config = json.loads(config_data)
                            logger.info("Successfully fixed JSON formatting")
                        except json.JSONDecodeError as e2:
                            logger.error(f"Still invalid JSON after fixing: {e2}")
                            # Try one more approach - dump the content to a temporary file and reload it
                            temp_file = os.path.join(os.path.dirname(self.connections_file), "temp_fixed.json")
                            with open(temp_file, 'w', encoding='utf-8') as tf:
                                tf.write(config_data)
                            logger.debug(f"Wrote fixed JSON to temporary file: {temp_file}")
                            
                            with open(temp_file, 'r', encoding='utf-8') as tf:
                                config_data = tf.read()
                            config = json.loads(config_data)
                            logger.info("Successfully loaded JSON from temporary file")
                    except Exception as fix_error:
                        logger.error(f"Error fixing JSON: {fix_error}")
                        # If that didn't work, re-raise the original error
                        raise e
                    
                # Validate config has required fields
                if "components" not in config:
                    raise ValueError("Invalid connections file: 'components' section missing")
                    
                if "connections" not in config:
                    raise ValueError("Invalid connections file: 'connections' section missing")
                    
                # Clear existing components if we're reloading
                previous_components = self.components.copy()
                self.components.clear()
                    
                # Create components
                for name, info in config.get("components", {}).items():
                    if not isinstance(info, dict) or "port" not in info:
                        logger.warning(f"Invalid component configuration for {name}")
                        continue
                        
                    try:
                        port = int(info.get("port", 0))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid port for component {name}: {info.get('port')}")
                        port = 0
                        
                    component = SystemComponent(
                        name=name,
                        host=info.get("host", "localhost"),
                        port=port
                    )
                    
                    # Copy status from previous component if available
                    if name in previous_components:
                        prev = previous_components[name]
                        if prev.host == component.host and prev.port == component.port:
                            component.status = prev.status
                            component.last_checked = prev.last_checked
                            component.check_count = prev.check_count
                            component.success_count = prev.success_count
                            component.health_checks = prev.health_checks.copy()
                            component.error_message = prev.error_message
                            component.response_time = prev.response_time
                            component.last_successful_check = prev.last_successful_check
                            component.consecutive_failures = prev.consecutive_failures
                    
                    self.components[name] = component
                    
                # Set up connection map
                self.connection_map = config.get("connections", {})
                
                # Initialize expected connections for each component
                for name, component in self.components.items():
                    if name in self.connection_map:
                        component.connected_to = set(self.connection_map[name])
                        
                logger.info(f"Loaded {len(self.components)} components from {self.connections_file}")
                
                # Record last load time
                self.last_check_time = datetime.now()
                
                # Clear old history when loading a new config
                if not previous_components:
                    self.history.clear()
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in connections file: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading connections: {e}")
                raise
                
    def check_for_config_changes(self):
        """Check if the config file has changed and reload if necessary"""
        try:
            if not os.path.exists(self.connections_file):
                logger.warning(f"Config file no longer exists: {self.connections_file}")
                return False
                
            current_mtime = os.path.getmtime(self.connections_file)
            if self.config_file_mtime is None:
                self.config_file_mtime = current_mtime
                return False
                
            if current_mtime != self.config_file_mtime:
                logger.info(f"Config file has changed (mtime: {current_mtime} vs {self.config_file_mtime}), reloading")
                self.load_connections()
                return True
        except Exception as e:
            logger.error(f"Error checking for config changes: {e}")
            
        return False
            
    def check_all_components(self):
        """Check status of all components"""
        with self._lock:
            results = {}
            
            # Check if config file has changed
            self.check_for_config_changes()
            
            if not self.components:
                logger.warning("No components to check")
                return {}
            
            # Check each component
            for name, component in self.components.items():
                try:
                    logger.debug(f"Checking component: {name}")
                    status = component.check_status()
                    results[name] = status
                except Exception as e:
                    logger.error(f"Error checking component {name}: {e}")
                    results[name] = "error"
                    component.status = "error"
                    component.error_message = str(e)
            
            # Call the callback if registered
            if self.status_changed_callback:
                try:
                    self.status_changed_callback(self.components)
                except Exception as e:
                    logger.error(f"Error in status_changed_callback: {e}")
                
            # Update last check time
            self.last_check_time = datetime.now()
            
            # Add to history
            history_entry = {
                "timestamp": self.last_check_time.isoformat(),
                "components": {name: comp.to_dict() for name, comp in self.components.items()},
                "system_health": self.calculate_system_health()
            }
            self.history.append(history_entry)
            
            # Trim history if needed
            if len(self.history) > self.history_max_size:
                self.history = self.history[-self.history_max_size:]
                
            return results
        
    def verify_connections(self):
        """Verify connections between components"""
        with self._lock:
            # Check that components can communicate with their expected connections
            components_to_check = {
                name: component for name, component in self.components.items() 
                if component.status in ["online", "degraded"]
            }
            
            connection_status = {}
            
            for name, component in components_to_check.items():
                if name not in self.connection_map:
                    continue
                    
                connection_status[name] = {}
                
                for target_name in self.connection_map[name]:
                    if target_name in components_to_check:
                        target_component = components_to_check[target_name]
                        
                        # Verify the connection
                        try:
                            is_connected = component.verify_connection_to(target_component)
                            status = "connected" if is_connected else "disconnected"
                        except Exception as e:
                            logger.error(f"Error verifying connection {name} -> {target_name}: {e}")
                            status = "error"
                    else:
                        status = "disconnected"
                        
                    connection_status[name][target_name] = status
                    
            return connection_status
            
    def calculate_system_health(self) -> float:
        """Calculate overall system health percentage"""
        with self._lock:
            if not self.components:
                return 0.0
                
            # Give different weights to different status types
            status_weights = {
                "online": 1.0,
                "degraded": 0.5,
                "unknown": 0.0,
                "offline": 0.0,
                "error": 0.0
            }
            
            total_weight = sum(status_weights.get(c.status, 0.0) for c in self.components.values())
            max_possible = len(self.components)
            
            return (total_weight / max_possible) * 100.0 if max_possible > 0 else 0.0
        
    def start_monitoring(self):
        """Start continuous monitoring in a separate thread"""
        with self._lock:
            if self.running:
                logger.info("Monitor already running")
                return
                
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, 
                                                name="connection_monitor_thread")
            self.monitor_thread.start()
            logger.info("Started connection monitoring")
        
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        with self._lock:
            logger.info("Stopping connection monitoring...")
            self.running = False
            
        if self.monitor_thread and self.monitor_thread.is_alive():
            try:
                self.monitor_thread.join(timeout=3.0)
                logger.info("Monitoring thread stopped")
            except Exception as e:
                logger.error(f"Error stopping monitor thread: {e}")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        consecutive_errors = 0
        check_interval = self.check_interval
        
        logger.info(f"Starting monitor loop with interval {check_interval}s")
        
        while self.running:
            try:
                # Check all components
                start_time = time.time()
                results = self.check_all_components()
                
                # Verify connections
                connections = self.verify_connections()
                
                # Calculate health
                health = self.calculate_system_health()
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Save monitoring data periodically
                if len(self.history) % 10 == 0:
                    self._save_monitoring_data()
                    
                # Dynamic sleep interval based on system health
                if health < 50.0:
                    # Check more frequently if system health is poor
                    check_interval = max(1.0, self.check_interval / 2)
                    logger.debug(f"System health below 50%, checking more frequently: {check_interval}s")
                else:
                    check_interval = self.check_interval
                
                # Calculate how long the checks took
                elapsed = time.time() - start_time
                
                # Sleep for the remaining interval time, or at least 0.5 seconds
                sleep_time = max(0.5, check_interval - elapsed)
                logger.debug(f"Checks took {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                
                # Use small sleep increments so we can respond quickly to stop requests
                for _ in range(int(sleep_time * 2)):
                    if not self.running:
                        break
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                consecutive_errors += 1
                
                # If we've had too many consecutive errors, back off
                if consecutive_errors > 5:
                    logger.warning(f"Too many consecutive errors ({consecutive_errors}), backing off monitoring frequency")
                    time.sleep(self.check_interval * 2)
                else:
                    time.sleep(self.check_interval)
        
        logger.info("Monitor loop exiting")
    
    def _save_monitoring_data(self):
        """Save monitoring data to disk"""
        try:
            data_dir = os.path.join(project_root, "data", "v8", "monitoring")
            os.makedirs(data_dir, exist_ok=True)
            
            # Use a more unique filename with date and time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitor_data_{timestamp}.json"
            filepath = os.path.join(data_dir, filename)
            
            # Create a summary with the most important data
            summary = {
                "timestamp": datetime.now().isoformat(),
                "monitor_uptime": (datetime.now() - self.startup_time).total_seconds(),
                "config_file": self.connections_file,
                "components": {name: comp.to_dict() for name, comp in self.components.items()},
                "connection_map": self.connection_map,
                "system_health": self.calculate_system_health(),
                "history_summary": {
                    "entries": len(self.history),
                    "oldest": self.history[0]["timestamp"] if self.history else None,
                    "newest": self.history[-1]["timestamp"] if self.history else None,
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Also save a "latest" version for quick access
            latest_path = os.path.join(data_dir, "latest_monitor_data.json")
            with open(latest_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.debug(f"Saved monitoring data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
                
    def set_status_changed_callback(self, callback):
        """Set callback for when status changes"""
        self.status_changed_callback = callback
        
    def reload_connections(self) -> bool:
        """Reload the connections configuration file"""
        try:
            self.load_connections()
            return True
        except Exception as e:
            logger.error(f"Failed to reload connections: {e}")
            return False
            
    def restart_component(self, component_name: str) -> bool:
        """Attempt to restart a component (placeholder - would need implementation)"""
        logger.info(f"Restart requested for component: {component_name}")
        
        # This would typically integrate with the system's process management
        # For now, we'll just log the request and return False
        return False
        
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the system status"""
        total = len(self.components)
        online = sum(1 for c in self.components.values() if c.status == "online")
        degraded = sum(1 for c in self.components.values() if c.status == "degraded")
        offline = sum(1 for c in self.components.values() if c.status == "offline")
        unknown = sum(1 for c in self.components.values() if c.status == "unknown")
        error = sum(1 for c in self.components.values() if c.status == "error")
        
        system_health = self.calculate_system_health()
        
        # Get connection success rate
        total_connections = 0
        successful_connections = 0
        
        connection_status = self.verify_connections()
        for source, targets in connection_status.items():
            for target, status in targets.items():
                total_connections += 1
                if status == "connected":
                    successful_connections += 1
                    
        connection_rate = (successful_connections / total_connections * 100) if total_connections > 0 else 0
        
        # Calculate uptime
        monitor_uptime_seconds = (datetime.now() - self.startup_time).total_seconds()
        hours, remainder = divmod(monitor_uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        monitor_uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        return {
            "total_components": total,
            "online": online,
            "degraded": degraded,
            "offline": offline,
            "unknown": unknown,
            "error": error,
            "health_percentage": system_health,
            "connection_success_rate": connection_rate,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "last_checked": self.last_check_time.isoformat() if self.last_check_time else None,
            "connections_file": self.connections_file,
            "monitor_uptime_seconds": monitor_uptime_seconds,
            "monitor_uptime": monitor_uptime,
            "uptime": {name: comp.get_uptime_percentage() for name, comp in self.components.items()}
        }

if HAS_PYSIDE:
    class ComponentStatusWidget(QFrame):
        """Widget to display a single component's status"""
        
        def __init__(self, component: SystemComponent, parent=None):
            super().__init__(parent)
            self.component = component
            self.setup_ui()
            
        def setup_ui(self):
            self.setFrameShape(QFrame.StyledPanel)
            self.setFrameShadow(QFrame.Raised)
            self.setMinimumWidth(200)
            self.setMinimumHeight(150)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)
            
            # Component name with bold font
            self.name_label = QLabel(f"<b>{self.component.name}</b>")
            self.name_label.setAlignment(Qt.AlignCenter)
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            self.name_label.setFont(font)
            
            # Status indicator
            self.status_label = QLabel(f"Status: {self.component.status}")
            
            # Address
            self.address_label = QLabel(f"{self.component.host}:{self.component.port}")
            self.address_label.setAlignment(Qt.AlignCenter)
            
            # Response time
            self.response_label = QLabel("Response: N/A")
            
            # Health indicators in a grid layout
            health_layout = QGridLayout()
            health_layout.setHorizontalSpacing(8)
            health_layout.setVerticalSpacing(2)
            
            self.port_indicator = QLabel("◯")
            self.api_indicator = QLabel("◯")
            self.conn_indicator = QLabel("◯")
            
            health_layout.addWidget(QLabel("Port:"), 0, 0)
            health_layout.addWidget(self.port_indicator, 0, 1)
            health_layout.addWidget(QLabel("API:"), 1, 0)
            health_layout.addWidget(self.api_indicator, 1, 1)
            health_layout.addWidget(QLabel("Conn:"), 2, 0)
            health_layout.addWidget(self.conn_indicator, 2, 1)
            
            # Actions
            actions_layout = QHBoxLayout()
            self.check_button = QPushButton("Check")
            self.check_button.setFixedHeight(24)
            self.check_button.clicked.connect(self.check_component)
            actions_layout.addWidget(self.check_button)
            
            # Add to layout
            layout.addWidget(self.name_label)
            layout.addWidget(self.status_label)
            layout.addWidget(self.address_label)
            layout.addWidget(self.response_label)
            layout.addLayout(health_layout)
            layout.addStretch(1)
            layout.addLayout(actions_layout)
            
            # Set initial colors
            self.update_ui()
            
        def update_ui(self):
            """Update the UI based on component status"""
            # Set colors based on status
            if self.component.status == "online":
                self.setStyleSheet("QFrame { background-color: #d4edda; border: 1px solid #c3e6cb; }")
                self.status_label.setText(f"Status: <span style='color: green;'>{self.component.status}</span>")
            elif self.component.status == "degraded":
                self.setStyleSheet("QFrame { background-color: #fff3cd; border: 1px solid #ffeeba; }")
                self.status_label.setText(f"Status: <span style='color: #856404;'>{self.component.status}</span>")
            elif self.component.status == "offline":
                self.setStyleSheet("QFrame { background-color: #f8d7da; border: 1px solid #f5c6cb; }")
                self.status_label.setText(f"Status: <span style='color: red;'>{self.component.status}</span>")
            elif self.component.status == "error":
                self.setStyleSheet("QFrame { background-color: #f8d7da; border: 1px solid #f5c6cb; }")
                self.status_label.setText(f"Status: <span style='color: #721c24;'>{self.component.status}</span>")
            else:  # unknown
                self.setStyleSheet("QFrame { background-color: #e2e3e5; border: 1px solid #d6d8db; }")
                self.status_label.setText(f"Status: <span style='color: #383d41;'>{self.component.status}</span>")
                
            # Update response time
            if self.component.response_time is not None:
                self.response_label.setText(f"Response: {self.component.response_time} ms")
            else:
                self.response_label.setText("Response: N/A")
                
            # Update health indicators
            self.port_indicator.setText("●" if self.component.health_checks["port_open"] else "◯")
            self.port_indicator.setStyleSheet("color: green;" if self.component.health_checks["port_open"] else "color: red;")
            
            self.api_indicator.setText("●" if self.component.health_checks["http_api"] else "◯")
            self.api_indicator.setStyleSheet("color: green;" if self.component.health_checks["http_api"] else "color: red;")
            
            self.conn_indicator.setText("●" if self.component.health_checks["connection_verified"] else "◯")
            self.conn_indicator.setStyleSheet("color: green;" if self.component.health_checks["connection_verified"] else "color: red;")
            
            # Show error message if any
            if self.component.error_message:
                self.setToolTip(f"Error: {self.component.error_message}")
            else:
                self.setToolTip("")
            
        def check_component(self):
            """Check this component's status when button is clicked"""
            self.check_button.setEnabled(False)
            self.check_button.setText("Checking...")
            
            try:
                # Update component status in a separate thread to avoid freezing UI
                def check_and_update():
                    old_status = self.component.status
                    self.component.check_status()
                    # Signal UI thread to update
                    Qt.QMetaObject.invokeMethod(
                        self, 
                        "update_after_check", 
                        Qt.Qt.QueuedConnection
                    )
                
                threading.Thread(target=check_and_update, daemon=True).start()
            except Exception as e:
                logger.error(f"Error during component check: {e}")
                self.check_button.setText("Check")
                self.check_button.setEnabled(True)
                self.setToolTip(f"Check error: {str(e)}")
                
        @Slot()
        def update_after_check(self):
            """Called after the check completes to update UI"""
            self.update_ui()
            self.check_button.setText("Check")
            self.check_button.setEnabled(True)

    class ConnectionMapWidget(QWidget):
        """Widget to display and visualize connections between components"""
        
        def __init__(self, monitor: 'ConnectionMonitor', parent=None):
            super().__init__(parent)
            self.monitor = monitor
            self.setup_ui()
            
        def setup_ui(self):
            self.setMinimumHeight(200)
            layout = QVBoxLayout(self)
            
            # Connection table
            self.connection_table = QTableWidget()
            self.connection_table.setColumnCount(3)
            self.connection_table.setHorizontalHeaderLabels(["Source", "Target", "Status"])
            self.connection_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            # Refresh button
            self.refresh_button = QPushButton("Refresh Connections")
            self.refresh_button.clicked.connect(self.refresh_connections)
            
            layout.addWidget(QLabel("<b>Component Connections</b>"))
            layout.addWidget(self.connection_table)
            layout.addWidget(self.refresh_button)
            
            self.refresh_connections()
            
        def refresh_connections(self):
            """Refresh the connection map display"""
            try:
                self.connection_table.setRowCount(0)
                
                connections = self.monitor.verify_connections()
                row = 0
                
                for source, targets in connections.items():
                    for target, status in targets.items():
                        self.connection_table.insertRow(row)
                        
                        source_item = QTableWidgetItem(source)
                        target_item = QTableWidgetItem(target)
                        status_item = QTableWidgetItem(status)
                        
                        # Set colors based on status
                        if status == "connected":
                            status_item.setBackground(QColor("#d4edda"))
                        elif status == "disconnected":
                            status_item.setBackground(QColor("#f8d7da"))
                        else:  # error
                            status_item.setBackground(QColor("#f8d7da"))
                        
                        self.connection_table.setItem(row, 0, source_item)
                        self.connection_table.setItem(row, 1, target_item)
                        self.connection_table.setItem(row, 2, status_item)
                        
                        row += 1
            except Exception as e:
                logger.error(f"Error refreshing connections: {e}")

    class MonitorConnectionsWindow(QMainWindow):
        """Main window for the connections monitor"""
        
        refresh_triggered = Signal()
        
        def __init__(self, monitor: ConnectionMonitor):
            super().__init__()
            self.monitor = monitor
            self.component_widgets = {}
            
            # Set up auto-refresh timer but don't start it until UI is ready
            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self.refresh_ui)
            
            # Connect signals
            self.refresh_triggered.connect(self.refresh_ui)
            
            # Set monitor callback
            self.monitor.set_status_changed_callback(self.on_status_changed)
            
            # Create the UI
            self.setup_ui()
            
            # Start monitoring if not started
            if not self.monitor.running:
                self.monitor.start_monitoring()
                
            # Initial UI update and start timer
            self.refresh_ui()
            self.refresh_timer.start(5000)  # Refresh every 5 seconds
            
        def setup_ui(self):
            # Set window properties
            self.setWindowTitle("V8 Component Connection Monitor")
            self.setMinimumSize(900, 600)
            
            # Create central widget
            central_widget = QWidget()
            main_layout = QVBoxLayout(central_widget)
            self.setCentralWidget(central_widget)
            
            # Add status bar
            self.statusBar().showMessage("Initializing...")
            
            # Create header
            header_layout = QHBoxLayout()
            header_label = QLabel("<h2>V8 Component Connection Monitor</h2>")
            header_label.setAlignment(Qt.AlignCenter)
            refresh_button = QPushButton("Refresh All")
            refresh_button.clicked.connect(self.refresh_ui)
            refresh_button.setFixedWidth(100)
            
            self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
            self.auto_refresh_checkbox.setChecked(True)
            self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)
            
            reload_config_button = QPushButton("Reload Config")
            reload_config_button.clicked.connect(self.reload_config)
            reload_config_button.setFixedWidth(100)
            
            header_layout.addWidget(header_label)
            header_layout.addWidget(refresh_button)
            header_layout.addWidget(self.auto_refresh_checkbox)
            header_layout.addWidget(reload_config_button)
            
            # Create system summary section
            summary_frame = QFrame()
            summary_frame.setFrameShape(QFrame.StyledPanel)
            summary_frame.setFrameShadow(QFrame.Raised)
            summary_layout = QGridLayout(summary_frame)
            
            self.summary_labels = {
                "health": QLabel("System Health: 0%"),
                "online": QLabel("Online: 0"),
                "degraded": QLabel("Degraded: 0"),
                "offline": QLabel("Offline: 0"),
                "error": QLabel("Error: 0"),
                "connections": QLabel("Connections: 0"),
                "last_check": QLabel("Last Check: Never"),
                "config_file": QLabel("Config: Unknown"),
                "uptime": QLabel("Monitor Uptime: 0s")
            }
            
            # Setup progress bar for health
            self.health_progress = QProgressBar()
            self.health_progress.setRange(0, 100)
            self.health_progress.setValue(0)
            self.health_progress.setTextVisible(True)
            self.health_progress.setFormat("System Health: %p%")
            
            # Add summary widgets to layout
            row = 0
            summary_layout.addWidget(self.health_progress, row, 0, 1, 2)
            row += 1
            
            # Add labels in a 2-column grid
            col = 0
            for label in self.summary_labels.values():
                if col > 1:
                    col = 0
                    row += 1
                summary_layout.addWidget(label, row, col)
                col += 1
                
            # Component grid section - put in a scroll area
            self.components_frame = QFrame()
            self.components_layout = QGridLayout(self.components_frame)
            self.components_layout.setSpacing(10)  # Add spacing between components
            
            # Create scrollable area for components
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.components_frame)
            
            # Connection map
            self.connection_map_widget = ConnectionMapWidget(self.monitor)
            
            # Add all sections to main layout
            main_layout.addLayout(header_layout)
            main_layout.addWidget(summary_frame)
            main_layout.addWidget(scroll)
            main_layout.addWidget(self.connection_map_widget)
            
            # Create component widgets
            self.create_component_widgets()
            
        def create_component_widgets(self):
            """Create widgets for each component"""
            try:
                # Clear existing widgets
                self.component_widgets.clear()
                
                # Remove existing widgets from layout
                while self.components_layout.count():
                    item = self.components_layout.takeAt(0)
                    if item and item.widget():
                        item.widget().deleteLater()
                
                # Create new widgets
                if not self.monitor.components:
                    label = QLabel("No components found in configuration")
                    label.setAlignment(Qt.AlignCenter)
                    self.components_layout.addWidget(label, 0, 0)
                    return
                    
                # Sort components by name
                component_names = sorted(self.monitor.components.keys())
                
                # Add to grid layout
                max_columns = 3  # Reduce columns for better visibility
                for i, name in enumerate(component_names):
                    component = self.monitor.components[name]
                    widget = ComponentStatusWidget(component)
                    
                    row = i // max_columns
                    col = i % max_columns
                    
                    self.components_layout.addWidget(widget, row, col)
                    self.component_widgets[name] = widget
                    
                logger.debug(f"Created {len(self.component_widgets)} component widgets")
            except Exception as e:
                logger.error(f"Error creating component widgets: {e}")
                self.statusBar().showMessage(f"Error creating component widgets: {str(e)}")
                
        def refresh_ui(self):
            """Refresh all UI elements"""
            try:
                # Check all components
                self.monitor.check_all_components()
                
                # Update system summary
                summary = self.monitor.get_system_summary()
                
                self.health_progress.setValue(int(summary["health_percentage"]))
                
                # Update color based on health
                if summary["health_percentage"] >= 80:
                    self.health_progress.setStyleSheet("QProgressBar::chunk { background-color: #28a745; }")
                elif summary["health_percentage"] >= 60:
                    self.health_progress.setStyleSheet("QProgressBar::chunk { background-color: #17a2b8; }")
                elif summary["health_percentage"] >= 40:
                    self.health_progress.setStyleSheet("QProgressBar::chunk { background-color: #ffc107; }")
                elif summary["health_percentage"] >= 20:
                    self.health_progress.setStyleSheet("QProgressBar::chunk { background-color: #fd7e14; }")
                else:
                    self.health_progress.setStyleSheet("QProgressBar::chunk { background-color: #dc3545; }")
                
                # Update summary labels
                self.summary_labels["online"].setText(f"Online: {summary['online']}")
                self.summary_labels["degraded"].setText(f"Degraded: {summary['degraded']}")
                self.summary_labels["offline"].setText(f"Offline: {summary['offline']}")
                self.summary_labels["error"].setText(f"Error: {summary['error']}")
                self.summary_labels["last_check"].setText(
                    f"Last Check: {datetime.now().strftime('%H:%M:%S')}"
                )
                
                # Truncate long paths for display
                config_path = summary.get('connections_file', 'Unknown')
                if len(config_path) > 40:
                    config_path = "..." + config_path[-37:]
                self.summary_labels["config_file"].setText(f"Config: {os.path.basename(config_path)}")
                
                self.summary_labels["uptime"].setText(
                    f"Monitor Uptime: {summary.get('monitor_uptime', 'N/A')}"
                )
                
                # Update component widgets
                for name, widget in self.component_widgets.items():
                    widget.update_ui()
                    
                # Refresh connection map
                self.connection_map_widget.refresh_connections()
                
                # Update status bar
                self.statusBar().showMessage(
                    f"System Health: {summary['health_percentage']:.1f}% | Online: {summary['online']} | "
                    f"Last Check: {datetime.now().strftime('%H:%M:%S')}"
                )
                
            except Exception as e:
                logger.error(f"Error refreshing UI: {e}")
                self.statusBar().showMessage(f"Error refreshing UI: {str(e)}")
                
        def reload_config(self):
            """Reload the connection configuration"""
            try:
                self.statusBar().showMessage("Reloading configuration...")
                if self.monitor.reload_connections():
                    self.create_component_widgets()
                    self.refresh_ui()
                    self.statusBar().showMessage("Configuration reloaded successfully")
                else:
                    self.statusBar().showMessage("Failed to reload configuration")
            except Exception as e:
                logger.error(f"Error reloading config: {e}")
                self.statusBar().showMessage(f"Error reloading config: {str(e)}")
                
        def toggle_auto_refresh(self, enabled):
            """Toggle auto-refresh"""
            if enabled:
                self.refresh_timer.start(5000)
                self.statusBar().showMessage("Auto-refresh enabled (5s)")
            else:
                self.refresh_timer.stop()
                self.statusBar().showMessage("Auto-refresh disabled")
                
        def on_status_changed(self, components):
            """Called when the status of any component changes"""
            # This is called from a different thread, so use signals
            self.refresh_triggered.emit()
            
        def closeEvent(self, event):
            """Handle window close event"""
            try:
                self.monitor.stop_monitoring()
                # Wait a moment for threads to clean up
                QTimer.singleShot(500, self.close)
                event.accept()
            except Exception as e:
                logger.error(f"Error during window close: {e}")
                event.accept()

def launch_gui(connections_file, system_key=None):
    """Launch the GUI version of the connection monitor"""
    try:
        app = QApplication([])
        
        # Set application style
        app.setStyle("Fusion")
        
        # Set dark palette for better visibility
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        # Uncomment to use dark theme
        # app.setPalette(palette)
        
        # Create monitor
        logger.info(f"Creating monitor with connections file: {connections_file}")
        monitor = ConnectionMonitor(connections_file, system_key)
        
        # Check if configuration was loaded successfully
        if not monitor.config_loaded:
            logger.error("Failed to load connections configuration")
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Configuration Error")
            error_dialog.setText("Failed to load connections configuration")
            error_dialog.setInformativeText(f"Could not load configuration from {connections_file}.\n"
                                            "Please check that the file exists and is valid JSON.")
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
            return 1
        
        # Create and show window
        window = MonitorConnectionsWindow(monitor)
        window.show()
        
        # Run event loop
        return app.exec_()
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        print(f"Error launching GUI: {e}")
        return 1

def console_mode(connections_file, system_key=None, continuous=False, interval=5):
    """Run the connection monitor in console mode"""
    try:
        print(f"Loading connections from: {connections_file}")
        monitor = ConnectionMonitor(connections_file, system_key)
        
        if not monitor.config_loaded:
            print("Failed to load connections file. Check log for details.")
            return 1
            
        if continuous:
            # Start monitoring and wait for keyboard interrupt
            print(f"Starting continuous monitoring (Ctrl+C to exit). Checking every {interval} seconds.")
            monitor.check_interval = interval
            monitor.start_monitoring()
            
            try:
                # Define safe status symbols for Windows
                # Windows console might not support Unicode symbols
                if os.name == 'nt':
                    status_symbols = {
                        "online": "+", 
                        "degraded": "!",
                        "offline": "X",
                        "error": "E",
                        "unknown": "?"
                    }
                    connection_symbols = {
                        "connected": "+",
                        "disconnected": "X"
                    }
                else:
                    status_symbols = {
                        "online": "✓", 
                        "degraded": "⚠",
                        "offline": "✗",
                        "error": "!",
                        "unknown": "?"
                    }
                    connection_symbols = {
                        "connected": "✓",
                        "disconnected": "✗"
                    }
                
                # Main display loop
                while True:
                    # Print current status
                    system_summary = monitor.get_system_summary()
                    
                    # Clear screen on Windows systems
                    if os.name == 'nt':
                        os.system('cls')
                    else:
                        # Clear screen on Unix/Linux systems
                        os.system('clear')
                    
                    # Header
                    print("\n" + "=" * 80)
                    print(f"LUMINA V8 CONNECTION MONITOR")
                    print(f"Connections file: {os.path.basename(monitor.connections_file)}")
                    print("=" * 80)
                    
                    # System health
                    health = system_summary['health_percentage']
                    # Use simple characters for health bar on Windows
                    if os.name == 'nt':
                        health_bar = "#" * int(health / 10) + "-" * (10 - int(health / 10))
                    else:
                        health_bar = "█" * int(health / 10) + "░" * (10 - int(health / 10))
                        
                    # Only use colors on systems that support them
                    if os.name != 'nt':
                        health_color = ""
                        if health >= 80:
                            health_color = "\033[92m"  # Green
                        elif health >= 60:
                            health_color = "\033[94m"  # Blue
                        elif health >= 40:
                            health_color = "\033[93m"  # Yellow
                        elif health >= 20:
                            health_color = "\033[91m"  # Red
                        else:
                            health_color = "\033[95m"  # Magenta
                            
                        reset = "\033[0m"
                        print(f"SYSTEM HEALTH: {health_color}{health:.1f}% [{health_bar}]{reset}")
                    else:
                        print(f"SYSTEM HEALTH: {health:.1f}% [{health_bar}]")
                    
                    # Component counts
                    print(f"COMPONENTS: {system_summary['online']} online, {system_summary.get('degraded', 0)} degraded, " +
                          f"{system_summary['offline']} offline, {system_summary['error']} error")
                    print(f"MONITOR UPTIME: {system_summary.get('monitor_uptime', 'N/A')}")
                    print(f"LAST CHECKED: {system_summary['last_checked']}")
                    print("=" * 80)
                    
                    # Table header
                    print(f"{'NAME':<20} {'HOST:PORT':<20} {'STATUS':<10} {'RESPONSE':<10} {'UPTIME %':<10}")
                    print("-" * 80)
                    
                    # Sort components by status (online first, then degraded, etc.)
                    def sort_key(comp_item):
                        name, comp = comp_item
                        status_order = {"online": 0, "degraded": 1, "offline": 2, "error": 3, "unknown": 4}
                        return status_order.get(comp["status"], 5), name
                    
                    sorted_components = sorted(system_summary["components"].items(), key=sort_key)
                    
                    for name, comp in sorted_components:
                        status_symbol = status_symbols.get(comp["status"], "?")
                        
                        host_port = f"{comp['host']}:{comp['port']}"
                        response = f"{comp['response_time']} ms" if comp['response_time'] is not None else "N/A"
                        uptime = f"{comp.get('uptime_percentage', 0):.1f}%"
                        
                        # Color status text on Unix/Linux
                        if os.name != 'nt':
                            status_color = {
                                "online": "\033[92m",  # Green
                                "degraded": "\033[93m",  # Yellow
                                "offline": "\033[91m",  # Red
                                "error": "\033[95m",  # Magenta
                                "unknown": "\033[90m"  # Gray
                            }.get(comp["status"], "")
                            
                            status_text = f"{status_color}{comp['status']}{reset}"
                        else:
                            status_text = comp['status']
                        
                        print(f"{status_symbol} {name:<18} {host_port:<20} {status_text:<10} {response:<10} {uptime:<10}")
                    
                    # Connection status
                    if system_summary.get("online", 0) > 1:
                        print("\nCONNECTIONS:")
                        print("-" * 80)
                        connections = monitor.verify_connections()
                        
                        if not connections:
                            print("No active connections found.")
                        else:
                            for source, targets in connections.items():
                                for target, status in targets.items():
                                    status_icon = connection_symbols.get(status, "?")
                                    print(f"{source} -> {target}: {status_icon} {status}")
                    
                    # Wait for next check interval with countdown
                    remaining = interval
                    while remaining > 0 and monitor.running:
                        sys.stdout.write(f"\rNext update in {remaining}s (Ctrl+C to exit)")
                        sys.stdout.flush()
                        time.sleep(1)
                        remaining -= 1
                    print("\r" + " " * 40 + "\r", end="")  # Clear the countdown line
                    
            except KeyboardInterrupt:
                print("\nStopping monitoring.")
                monitor.stop_monitoring()
        else:
            # Run once mode
            print("Checking component status...")
            status = monitor.check_all_components()
            connections = monitor.verify_connections()
            
            # Print results
            system_summary = monitor.get_system_summary()
            print("\nSystem Health:", f"{system_summary['health_percentage']:.1f}%")
            print("\nComponent Status:")
            print("=" * 80)
            
            print(f"{'NAME':<20} {'HOST:PORT':<20} {'STATUS':<10} {'RESPONSE':<10}")
            print("-" * 80)
            
            # Define safe status symbols for Windows
            if os.name == 'nt':
                status_icons = {
                    "online": "+", 
                    "degraded": "!",
                    "offline": "X",
                    "error": "E",
                    "unknown": "?"
                }
                connection_icons = {
                    "connected": "+",
                    "disconnected": "X"
                }
            else:
                status_icons = {
                    "online": "✓", 
                    "degraded": "⚠",
                    "offline": "✗",
                    "error": "!",
                    "unknown": "?"
                }
                connection_icons = {
                    "connected": "✓",
                    "disconnected": "✗"
                }
            
            # Sort components by status
            components_sorted = sorted(
                status.items(), 
                key=lambda x: (
                    {"online": 0, "degraded": 1, "offline": 2, "error": 3, "unknown": 4}.get(x[1], 5),
                    x[0]
                )
            )
            
            for name, status_value in components_sorted:
                component = monitor.components[name]
                status_icon = status_icons.get(status_value, "?")
                host_port = f"{component.host}:{component.port}"
                response = f"{component.response_time} ms" if component.response_time is not None else "N/A"
                print(f"{status_icon} {name:<18} {host_port:<20} {status_value:<10} {response:<10}")
                
                # Print error message if there is one
                if component.error_message:
                    print(f"  Error: {component.error_message}")
                
            if connections:
                print("\nConnection Status:")
                print("=" * 80)
                for source, targets in connections.items():
                    print(f"From {source}:")
                    for target, conn_status in targets.items():
                        icon = connection_icons.get(conn_status, "?")
                        print(f"  {icon} -> {target}: {conn_status}")
            else:
                print("\nNo connection information available.")
        
        return 0
    except Exception as e:
        logger.error(f"Error in console mode: {e}")
        print(f"Error: {e}")
        return 1

def main():
    """Main entry point for the program"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="LUMINA V8 Connection Monitor")
    parser.add_argument("-f", "--connections-file", "--connections", "--connection-file", dest="connections_file", 
                        help="Path to connections JSON file")
    parser.add_argument("-s", "--system", "--system-key", dest="system",
                        help="System key to monitor (if not provided, monitors all)")
    parser.add_argument("-t", "--continuous", action="store_true", 
                        help="Run monitoring continuously")
    parser.add_argument("-i", "--interval", type=int, default=5, 
                        help="Check interval in seconds (default: 5)")
    parser.add_argument("-g", "--gui", action="store_true", 
                        help="Use GUI mode (requires PySide6)")
    parser.add_argument("-d", "--debug", action="store_true", 
                        help="Enable debug logging")
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return 1
    
    # Setup logging based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Find the connections file
    connections_file = args.connections_file
    if not connections_file:
        # Try the default connections file
        if os.path.exists(DEFAULT_CONNECTIONS_FILE):
            connections_file = DEFAULT_CONNECTIONS_FILE
            logger.info(f"Using default connections file: {connections_file}")
        else:
            # Try to find any JSON file in the default connections directory
            try:
                if os.path.exists(DEFAULT_CONNECTIONS_DIR):
                    json_files = [f for f in os.listdir(DEFAULT_CONNECTIONS_DIR) if f.endswith('.json')]
                    if json_files:
                        connections_file = os.path.join(DEFAULT_CONNECTIONS_DIR, json_files[0])
                        logger.info(f"Using found connections file: {connections_file}")
                    else:
                        # Try the parent directory
                        parent_dir = os.path.dirname(DEFAULT_CONNECTIONS_DIR)
                        if os.path.exists(parent_dir):
                            json_files = [f for f in os.listdir(parent_dir) if f.endswith('.json')]
                            if json_files:
                                connections_file = os.path.join(parent_dir, json_files[0])
                                logger.info(f"Using found connections file in parent dir: {connections_file}")
                            else:
                                logger.error(f"No JSON files found in {DEFAULT_CONNECTIONS_DIR} or parent directory")
                                print(f"Error: No connections file found. Please provide a path with --connections-file")
                                return 1
                        else:
                            logger.error(f"No JSON files found in {DEFAULT_CONNECTIONS_DIR}")
                            print(f"Error: No connections file found. Please provide a path with --connections-file")
                            print(f"Please provide a connections file path with --connections-file")
                            return 1
                else:
                    logger.error(f"Default connections directory not found: {DEFAULT_CONNECTIONS_DIR}")
                    print(f"Error: Default connections directory not found: {DEFAULT_CONNECTIONS_DIR}")
                    print(f"Please provide a connections file path with --connections-file")
                    return 1
            except Exception as e:
                logger.error(f"Error finding connections file: {e}")
                print(f"Error: Could not find connections file: {e}")
                print(f"Please provide a connections file path with --connections-file")
                return 1
    
    if not os.path.exists(connections_file):
        logger.error(f"Connections file not found: {connections_file}")
        print(f"Error: Connections file not found: {connections_file}")
        return 1
        
    # Choose mode based on arguments
    if args.gui:
        return launch_gui(connections_file, args.system)
    else:
        return console_mode(connections_file, args.system, args.continuous, args.interval)

if __name__ == "__main__":
    sys.exit(main()) 