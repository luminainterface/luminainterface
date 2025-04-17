"""
Plugin Interface for LUMINA V7

This module provides the base interface that all plugins should implement.
"""

class PluginInterface:
    """Base interface for all LUMINA V7 plugins"""
    
    def __init__(self, plugin_id=None):
        """
        Initialize the plugin
        
        Args:
            plugin_id: Unique identifier for this plugin instance
        """
        self.plugin_id = plugin_id or self.__class__.__name__
        self.enabled = True
    
    def get_plugin_id(self):
        """Get the plugin's unique identifier"""
        return self.plugin_id
    
    def get_plugin_name(self):
        """Get the human-readable name of the plugin"""
        return self.__class__.__name__
    
    def get_plugin_description(self):
        """Get a description of what the plugin does"""
        return "No description provided"
    
    def initialize(self, context=None):
        """
        Initialize the plugin with the given context
        
        Args:
            context: Application context or settings
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        return True
    
    def process_message(self, message, context=None):
        """
        Process a message (override in subclasses)
        
        Args:
            message: The message to process
            context: Additional context for processing
            
        Returns:
            dict: Processing result
        """
        return {"processed": False, "plugin": self.get_plugin_id()}
    
    def get_status(self):
        """
        Get the current status of the plugin
        
        Returns:
            dict: Status information
        """
        return {
            "plugin_id": self.get_plugin_id(),
            "enabled": self.enabled,
            "status": "active" if self.enabled else "disabled"
        }
    
    def shutdown(self):
        """Perform any cleanup when shutting down"""
        self.enabled = False
        return True 