#!/usr/bin/env python3
"""
Spatial Temple Connector

This module provides a connector to interface between the Knowledge CI/CD system
and the Spatial Temple visualization.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode, TempleZone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.spatial_connector")

class SpatialTempleConnector:
    """
    Connects the Knowledge CI/CD system to the Spatial Temple visualization,
    allowing updates to flow between systems.
    """
    
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None):
        """Initialize with an optional temple mapper"""
        self.temple_mapper = temple_mapper
        self.connection_active = False
        self.last_update = None
        self.update_count = 0
        
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper"""
        self.temple_mapper = temple_mapper
        
    def connect(self) -> bool:
        """Establish connection to the temple visualization"""
        if not self.temple_mapper:
            logger.warning("Cannot connect: No temple mapper available")
            return False
            
        self.connection_active = True
        logger.info("Connected to Spatial Temple visualization")
        return True
        
    def disconnect(self) -> bool:
        """Disconnect from the temple visualization"""
        self.connection_active = False
        logger.info("Disconnected from Spatial Temple visualization")
        return True
        
    def update_temple(self) -> int:
        """Update the temple visualization with current mapper data"""
        if not self.temple_mapper or not self.connection_active:
            logger.warning("Cannot update: Not connected or no temple mapper")
            return 0
            
        try:
            # In a real implementation, this would update a live visualization
            # Here we'll simulate by saving the current state to a file
            
            # Create a serializable view of the temple data
            temple_data = {
                "nodes": {},
                "zones": {},
                "updated_at": datetime.now().isoformat()
            }
            
            # Add nodes
            for node_id, node in self.temple_mapper.nodes.items():
                temple_data["nodes"][node_id] = {
                    "id": node_id,
                    "concept": node.concept,
                    "position": node.position,
                    "node_type": node.node_type,
                    "weight": node.weight,
                    "connections": list(node.connections),
                    "attributes": node.attributes
                }
                
            # Add zones
            for zone_id, zone in self.temple_mapper.zones.items():
                temple_data["zones"][zone_id] = {
                    "id": zone_id,
                    "name": zone.name,
                    "type": zone.zone_type,
                    "position": zone.position,
                    "nodes": list(zone.nodes)
                }
                
            # Save to file
            temple_dir = os.path.join(project_root, "data", "temple")
            os.makedirs(temple_dir, exist_ok=True)
            
            temple_path = os.path.join(temple_dir, f"temple_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(temple_path, 'w') as f:
                json.dump(temple_data, f, indent=2)
                
            # Also save to latest state file for visualization tools
            latest_path = os.path.join(temple_dir, "temple_state_latest.json")
            with open(latest_path, 'w') as f:
                json.dump(temple_data, f, indent=2)
                
            # Update status
            self.last_update = datetime.now()
            self.update_count += 1
            
            logger.info(f"Updated Spatial Temple visualization with {len(temple_data['nodes'])} nodes and {len(temple_data['zones'])} zones")
            return len(temple_data["nodes"])
                
        except Exception as e:
            logger.error(f"Error updating Spatial Temple visualization: {e}")
            return 0
            
    def notify(self, message: str, level: str = "info") -> bool:
        """Send a notification to the temple visualization"""
        if not self.connection_active:
            logger.warning("Cannot notify: Not connected")
            return False
            
        try:
            # In a real implementation, this would notify a live visualization
            # Here we'll simulate by saving to a notifications file
            
            notification = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "level": level
            }
            
            notifications_dir = os.path.join(project_root, "data", "notifications")
            os.makedirs(notifications_dir, exist_ok=True)
            
            notifications_path = os.path.join(notifications_dir, "temple_notifications.jsonl")
            
            # Append to file
            with open(notifications_path, 'a') as f:
                f.write(json.dumps(notification) + "\n")
                
            logger.info(f"Sent notification to Spatial Temple: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the connector"""
        return {
            "connected": self.connection_active,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "update_count": self.update_count,
            "nodes_count": len(self.temple_mapper.nodes) if self.temple_mapper else 0,
            "zones_count": len(self.temple_mapper.zones) if self.temple_mapper else 0
        }

def create_test_connector():
    """Create a test connector with a basic temple mapper"""
    from src.v8.demo_data_generator import generate_demo_nodes
    
    # Create temple mapper
    mapper = SpatialTempleMapper()
    
    # Generate demo nodes
    demo_nodes = generate_demo_nodes(10)
    for node in demo_nodes:
        mapper.nodes[node.id] = node
        
    # Create connector
    connector = SpatialTempleConnector(mapper)
    connector.connect()
    
    return connector
    
if __name__ == "__main__":
    # Create test connector
    connector = create_test_connector()
    
    # Update temple
    connector.update_temple()
    
    # Send notification
    connector.notify("Spatial Temple Connector test complete", "info") 