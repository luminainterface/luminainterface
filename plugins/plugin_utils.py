"""
Plugin utilities for LUMINA V7 Template UI
This module provides helper functions for loading plugins and integrating with the template UI
"""

import os
import sys
import json
import logging
import importlib.util
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("PluginUtils")
logger.setLevel(logging.INFO)

def discover_plugins(plugin_dir: str = "plugins") -> List[Dict[str, Any]]:
    """
    Discover all available plugins in the plugins directory
    
    Args:
        plugin_dir: The directory to search for plugins
        
    Returns:
        A list of plugin information dictionaries
    """
    plugins = []
    plugin_path = Path(plugin_dir)
    
    if not plugin_path.exists() or not plugin_path.is_dir():
        logger.warning(f"Plugin directory {plugin_dir} does not exist or is not a directory")
        return plugins
        
    # Look for Python files that might be plugins
    for file_path in plugin_path.glob("*.py"):
        # Skip __init__.py and utility files
        if file_path.name.startswith("__") or file_path.name == "plugin_utils.py":
            continue
            
        try:
            # Load the module
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {file_path}")
                continue
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if it has a Plugin class
            if hasattr(module, "Plugin"):
                plugin_class = getattr(module, "Plugin")
                # Create temporary instance to get metadata
                temp_instance = plugin_class()
                
                plugins.append({
                    "name": getattr(temp_instance, "name", module_name),
                    "version": getattr(temp_instance, "version", "0.1.0"),
                    "description": getattr(temp_instance, "description", ""),
                    "module_path": str(file_path),
                    "module_name": module_name,
                    "class_name": "Plugin"
                })
                logger.info(f"Discovered plugin: {getattr(temp_instance, 'name', module_name)} v{getattr(temp_instance, 'version', '0.1.0')}")
        except Exception as e:
            logger.error(f"Error loading plugin {file_path}: {e}")
            
    return plugins
    
def load_plugin(plugin_info: Dict[str, Any], parent=None) -> Optional[Any]:
    """
    Load a specific plugin based on its information
    
    Args:
        plugin_info: Dictionary with plugin information
        parent: The parent object to pass to the plugin constructor
        
    Returns:
        The plugin instance or None if loading failed
    """
    try:
        module_path = plugin_info["module_path"]
        module_name = plugin_info["module_name"]
        class_name = plugin_info["class_name"]
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not load spec for {module_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        plugin_class = getattr(module, class_name)
        plugin_instance = plugin_class(parent)
        
        logger.info(f"Loaded plugin: {plugin_info['name']} v{plugin_info['version']}")
        return plugin_instance
    except Exception as e:
        logger.error(f"Error instantiating plugin {plugin_info['name']}: {e}")
        return None
        
def setup_plugin_ui(plugin, main_ui):
    """
    Set up the plugin UI by calling its setup_ui method
    
    Args:
        plugin: The plugin instance
        main_ui: The main UI instance
    """
    if hasattr(plugin, "setup_ui") and callable(plugin.setup_ui):
        try:
            plugin.setup_ui(main_ui)
            logger.info(f"Set up UI for plugin: {plugin.name}")
        except Exception as e:
            logger.error(f"Error setting up UI for plugin {plugin.name}: {e}")
    else:
        logger.warning(f"Plugin {getattr(plugin, 'name', 'Unknown')} does not have a setup_ui method")

# Wiki auto-reading utilities

def fetch_wiki_article(topic: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch a Wikipedia article summary and content
    
    Args:
        topic: The topic to search for
        
    Returns:
        A tuple of (summary, content) or (None, None) if not found
    """
    WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
    
    try:
        # Search for the topic
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": topic,
            "srprop": "snippet",
            "srlimit": 1
        }
        
        search_response = requests.get(WIKI_API_ENDPOINT, params=search_params)
        search_data = search_response.json()
        
        if "query" in search_data and "search" in search_data["query"] and search_data["query"]["search"]:
            page_id = search_data["query"]["search"][0]["pageid"]
            
            # Get summary
            summary_params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "pageids": page_id
            }
            
            summary_response = requests.get(WIKI_API_ENDPOINT, params=summary_params)
            summary_data = summary_response.json()
            
            # Get full content
            content_params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "pageids": page_id
            }
            
            content_response = requests.get(WIKI_API_ENDPOINT, params=content_params)
            content_data = content_response.json()
            
            if "query" in summary_data and "pages" in summary_data["query"]:
                summary = summary_data["query"]["pages"][str(page_id)]["extract"]
                content = content_data["query"]["pages"][str(page_id)]["extract"]
                
                return summary, content
        
        return None, None
    except Exception as e:
        logger.error(f"Error fetching wiki article: {e}")
        return None, None

def add_to_knowledge_base(topic: str, content: str, memory=None, mistral=None):
    """
    Add knowledge to the memory system and Mistral autowiki
    
    Args:
        topic: The topic of the knowledge
        content: The content to add
        memory: Optional onsite memory instance
        mistral: Optional mistral integration instance
        
    Returns:
        True if added to at least one system, False otherwise
    """
    success = False
    
    # Add to memory
    if memory:
        try:
            memory.add_knowledge(topic, {"content": content, "source": "wikipedia"})
            logger.info(f"Added knowledge to memory: {topic}")
            success = True
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
    
    # Add to mistral autowiki
    if mistral and hasattr(mistral, "add_autowiki_entry"):
        try:
            mistral.add_autowiki_entry(topic, content)
            logger.info(f"Added to Mistral autowiki: {topic}")
            success = True
        except Exception as e:
            logger.error(f"Error adding to autowiki: {e}")
            
    return success

def batch_process_wiki_topics(topics: List[str], memory=None, mistral=None, 
                              callback=None, max_topics=10):
    """
    Process a batch of wiki topics and add them to knowledge systems
    
    Args:
        topics: List of topics to process
        memory: Optional onsite memory instance  
        mistral: Optional mistral integration instance
        callback: Optional callback function(topic, status, current, total)
        max_topics: Maximum number of topics to process
        
    Returns:
        Number of successfully processed topics
    """
    processed = 0
    total = min(len(topics), max_topics)
    
    for i, topic in enumerate(topics[:max_topics]):
        if callback:
            callback(topic, "Processing", i, total)
            
        summary, content = fetch_wiki_article(topic)
        if summary and content:
            success = add_to_knowledge_base(topic, content, memory, mistral)
            if success:
                processed += 1
                if callback:
                    callback(topic, "Success", i, total)
            else:
                if callback:
                    callback(topic, "Failed to add", i, total)
        else:
            if callback:
                callback(topic, "Not found", i, total)
    
    return processed 