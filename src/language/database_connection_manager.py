#!/usr/bin/env python3
'''
Database Connection Manager

This module provides centralized management of database connections
across the system, allowing components to register and share connections.
'''

import os
import sys
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton instance
_manager_instance = None

class DatabaseConnectionManager:
    '''
    Database Connection Manager for centralized connection management.
    
    This class provides:
    1. Registration of database components
    2. Centralized connection management
    3. Connection status monitoring
    '''
    
    def __init__(self):
        '''Initialize the Database Connection Manager'''
        self.components = {}
        self.status = {
            "initialized": True,
            "component_count": 0,
            "connected_components": 0
        }
        
        logger.info("Database Connection Manager initialized")
    
    @property
    def initialized(self) -> bool:
        '''
        Check if the database connection manager is initialized
        
        Returns:
            bool: True if initialized, False otherwise
        '''
        return self.status.get("initialized", False)
    
    def register_component(self, name: str, component: Any) -> bool:
        '''
        Register a database component
        
        Args:
            name: Name of the component
            component: The component instance
            
        Returns:
            bool: True if registered, False otherwise
        '''
        if name in self.components:
            logger.warning(f"Component {name} already registered")
            return False
            
        self.components[name] = {
            "instance": component,
            "connected": True
        }
        
        self.status["component_count"] += 1
        self.status["connected_components"] += 1
        
        logger.info(f"Registered database component: {name}")
        return True
    
    def get_component(self, name: str) -> Optional[Any]:
        '''
        Get a registered database component
        
        Args:
            name: Name of the component
            
        Returns:
            Optional[Any]: The component if found, None otherwise
        '''
        component_info = self.components.get(name)
        
        if component_info:
            return component_info["instance"]
            
        return None
    
    def get_status(self) -> Dict[str, Any]:
        '''
        Get the current status of the connection manager
        
        Returns:
            Dict[str, Any]: Status information
        '''
        return {
            "status": self.status,
            "components": [name for name in self.components.keys()]
        }
    
    def verify_connections(self) -> Dict[str, bool]:
        '''
        Verify all database connections
        
        Returns:
            Dict[str, bool]: Connection status for each component
        '''
        connection_status = {}
        
        for name, component_info in self.components.items():
            component = component_info["instance"]
            
            # Check if the component has a verify method
            if hasattr(component, "verify_connection"):
                connection_status[name] = component.verify_connection()
            elif hasattr(component, "get_status"):
                status = component.get_status()
                connection_status[name] = status.get("running", False)
            else:
                connection_status[name] = True  # Assume connected if no way to check
        
        return connection_status
        
    def shutdown(self) -> bool:
        '''
        Shutdown all database connections
        
        Returns:
            bool: True if shutdown successful, False otherwise
        '''
        success = True
        for name, component_info in self.components.items():
            component = component_info["instance"]
            
            try:
                # Try to shutdown the component
                if hasattr(component, "shutdown"):
                    component.shutdown()
                elif hasattr(component, "close"):
                    component.close()
                    
                component_info["connected"] = False
                self.status["connected_components"] -= 1
                
            except Exception as e:
                logger.error(f"Error shutting down component {name}: {e}")
                success = False
                
        self.status["initialized"] = False
        logger.info("Database Connection Manager shutdown complete")
        return success

def get_database_connection_manager() -> DatabaseConnectionManager:
    '''
    Get the singleton instance of the Database Connection Manager
    
    Returns:
        DatabaseConnectionManager: The singleton instance
    '''
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = DatabaseConnectionManager()
        
    return _manager_instance
