"""
Version Bridge Integration with Spiderweb Architecture

This module handles the integration of different versions of the Lumina Neural Network
using the Spiderweb architecture for decentralized communication and data flow.
"""

from typing import Dict, Any, Optional, List
import logging
from src.spiderweb_architecture import SpiderwebArchitecture, SpiderwebNode
from src.central_node import CentralNode

class VersionBridgeIntegration:
    """Handles version compatibility and integration using Spiderweb architecture"""
    
    def __init__(self, central_node: Optional[CentralNode] = None):
        self.central_node = central_node
        self.spiderweb = SpiderwebArchitecture()
        self.version_capabilities = self._initialize_version_capabilities()
        self.logger = logging.getLogger('VersionBridgeIntegration')
        
    def _initialize_version_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize version capabilities dictionary"""
        return {
            '1': {
                'features': ['basic_processing', 'text_input'],
                'outputs': ['text_response']
            },
            '2': {
                'features': ['basic_processing', 'text_input', 'image_input'],
                'outputs': ['text_response', 'image_response']
            },
            # Add more versions as needed
        }
        
    def add_version(self, version_id: str, system: CentralNode) -> bool:
        """Add a new version to the Spiderweb architecture"""
        if version_id not in self.version_capabilities:
            self.logger.error(f"Version {version_id} not supported")
            return False
            
        success = self.spiderweb.add_node(version_id, system)
        if success:
            # Register default message handlers
            self.spiderweb.register_handler(version_id, 'data_request', 
                                          self._handle_data_request)
            self.spiderweb.register_handler(version_id, 'version_query',
                                          self._handle_version_query)
        return success
        
    def remove_version(self, version_id: str) -> bool:
        """Remove a version from the Spiderweb architecture"""
        return self.spiderweb.remove_node(version_id)
        
    def connect_versions(self, version_id1: str, version_id2: str) -> bool:
        """Connect two versions in the Spiderweb architecture"""
        return self.spiderweb.connect_nodes(version_id1, version_id2)
        
    def process_with_version(self, data: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """Process data using the specified version"""
        if target_version not in self.spiderweb.nodes:
            raise ValueError(f"Version {target_version} not available")
            
        # Validate data compatibility
        if not self._validate_data_compatibility(data, target_version):
            raise ValueError(f"Data not compatible with version {target_version}")
            
        # Get the system for the target version
        system = self.spiderweb.nodes[target_version].system
        
        # Process the data
        try:
            result = system.process_data(data)
            return {
                'success': True,
                'data': result,
                'version': target_version
            }
        except Exception as e:
            self.logger.error(f"Error processing data with version {target_version}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'version': target_version
            }
            
    def _validate_data_compatibility(self, data: Dict[str, Any], target_version: str) -> bool:
        """Validate if data is compatible with target version"""
        required_features = self.version_capabilities[target_version]['features']
        return all(feature in data for feature in required_features)
        
    def _handle_data_request(self, message: Dict[str, Any]):
        """Handle data request messages"""
        version_id = message['source']
        data = message['data']
        
        try:
            result = self.process_with_version(data, version_id)
            response = {
                'type': 'data_response',
                'data': result,
                'timestamp': message['timestamp']
            }
            self.spiderweb.broadcast_message(version_id, response)
        except Exception as e:
            self.logger.error(f"Error handling data request: {str(e)}")
            
    def _handle_version_query(self, message: Dict[str, Any]):
        """Handle version query messages"""
        version_id = message['source']
        response = {
            'type': 'version_response',
            'capabilities': self.version_capabilities.get(version_id, {}),
            'timestamp': message['timestamp']
        }
        self.spiderweb.broadcast_message(version_id, response)
        
    def get_version_status(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific version"""
        return self.spiderweb.get_node_status(version_id)
        
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get status of the entire version bridge"""
        return {
            'architecture': self.spiderweb.get_architecture_status(),
            'supported_versions': list(self.version_capabilities.keys())
        }

# Example usage
if __name__ == "__main__":
    from central_node import CentralNode
    
    # Initialize central node
    central = CentralNode()
    
    # Create version bridge integration
    bridge_integration = VersionBridgeIntegration(central)
    
    # Test data
    test_data = {
        'symbol': 'infinity',
        'emotion': 'wonder',
        'breath': 'deep',
        'paradox': 'existence'
    }
    
    # Process with all versions
    for version in bridge_integration.version_capabilities.keys():
        try:
            result = bridge_integration.process_with_version(test_data, version)
            print(f"Version {version} result:", result)
        except Exception as e:
            print(f"Error processing with version {version}: {str(e)}")
        
    # Get version information
    version_info = bridge_integration.get_bridge_status()
    print("\nVersion Information:")
    for key, value in version_info.items():
        print(f"{key}: {value}")
        
    # Get specific version capabilities
    for version in bridge_integration.version_capabilities.keys():
        try:
            capabilities = bridge_integration.version_capabilities[version]
            print(f"\nVersion {version} Capabilities:")
            for key, value in capabilities.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Error getting version {version} capabilities: {str(e)}") 