from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

class VersionBridge:
    def __init__(self):
        self.version_components = {}
        self.initialized = False
    
    def initialize(self):
        """Initialize all version bridges"""
        if self.initialized:
            return
            
        # Add src directory to Python path
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.append(str(src_path))
            
        # Initialize each version bridge
        for version in range(1, 6):
            try:
                module_name = f"src.v{version}.bridge"
                bridge_module = __import__(module_name, fromlist=['get_bridge'])
                bridge = bridge_module.get_bridge()
                self.version_components[f"v{version}"] = bridge
                print(f"Successfully initialized v{version} bridge")
            except ImportError as e:
                print(f"Warning: Could not initialize v{version} bridge: {e}")
        
        self.initialized = True
    
    def get_component(self, version: str, component_name: str) -> Optional[Any]:
        """Get a specific component from a version"""
        if not self.initialized:
            self.initialize()
            
        if version not in self.version_components:
            return None
            
        return self.version_components[version].get_component(component_name)
    
    def get_visualization_data(self, version: str) -> Dict[str, Any]:
        """Get visualization data from a specific version"""
        if not self.initialized:
            self.initialize()
            
        if version not in self.version_components:
            return {}
            
        return self.version_components[version].get_visualization_data()
    
    def send_command(self, version: str, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to a specific version"""
        if not self.initialized:
            self.initialize()
            
        if version not in self.version_components:
            return {"status": "error", "message": f"Version {version} not available"}
            
        return self.version_components[version].process_command(command, data)

# Singleton instance
_bridge_instance = None

def get_bridge() -> VersionBridge:
    """Get the singleton bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = VersionBridge()
    return _bridge_instance 