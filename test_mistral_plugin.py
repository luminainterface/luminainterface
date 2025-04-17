#!/usr/bin/env python3
"""
Test script for the Mistral Neural Chat Plugin

This script tests the Mistral Neural Chat plugin functionality
without requiring the full Template UI framework.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PluginTester")

# Ensure we can import from current directory
sys.path.insert(0, str(Path(__file__).parent.absolute()))

def test_plugin_import():
    """Test importing the plugin module"""
    logger.info("Testing plugin import...")
    try:
        sys.path.append("plugins")
        import mistral_neural_chat_plugin
        logger.info("✓ Successfully imported plugin module")
        return mistral_neural_chat_plugin
    except Exception as e:
        logger.error(f"✗ Failed to import plugin module: {e}")
        return None

def test_plugin_creation(module):
    """Test creating the plugin instance"""
    logger.info("Testing plugin creation...")
    try:
        plugin = module.Plugin()
        logger.info(f"✓ Successfully created plugin instance: {plugin.name} {plugin.version}")
        logger.info(f"  Description: {plugin.description}")
        return plugin
    except Exception as e:
        logger.error(f"✗ Failed to create plugin instance: {e}")
        return None

def test_mistral_integration(plugin):
    """Test Mistral integration component"""
    logger.info("Testing Mistral integration...")
    
    if not hasattr(plugin, "_initialize_components"):
        logger.error("✗ Plugin does not have _initialize_components method")
        return False
        
    try:
        # Try to load API key
        api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key:
            plugin.api_key = api_key
            logger.info("Using API key from environment variable")
            
            result = plugin._initialize_components()
            if result:
                logger.info("✓ Successfully initialized Mistral integration")
                return True
            else:
                logger.warning("⚠ Initialized components but returned False")
                return False
        else:
            logger.warning("⚠ No Mistral API key found. Set MISTRAL_API_KEY environment variable")
            return False
    except Exception as e:
        logger.error(f"✗ Error in Mistral integration: {e}")
        return False

def test_memory_integration(plugin):
    """Test memory integration component"""
    logger.info("Testing memory integration...")
    
    if not hasattr(plugin, "memory"):
        logger.error("✗ Plugin does not have memory attribute")
        return False
        
    try:
        if plugin.memory:
            logger.info(f"✓ Memory initialized: {type(plugin.memory).__name__}")
            
            # Test adding a conversation
            plugin.memory.add_conversation({"role": "user", "content": "Test message"})
            plugin.memory.add_conversation({"role": "assistant", "content": "Test response"})
            
            logger.info("✓ Successfully added test conversation to memory")
            return True
        else:
            logger.warning("⚠ Memory component is None")
            return False
    except Exception as e:
        logger.error(f"✗ Error in memory integration: {e}")
        return False

def test_message_processing(plugin):
    """Test message processing capability"""
    logger.info("Testing message processing...")
    
    if not plugin.mistral:
        logger.warning("⚠ Mistral integration not available. Skipping message test.")
        return False
        
    try:
        test_message = "Hello, this is a test message."
        logger.info(f"Sending test message: '{test_message}'")
        
        # Process message manually
        response = plugin.mistral.process_message(test_message, nn_weight=plugin.nn_weight)
        
        if response:
            logger.info(f"✓ Received response: '{response[:50]}...'")
            return True
        else:
            logger.warning("⚠ No response received")
            return False
    except Exception as e:
        logger.error(f"✗ Error processing message: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Mistral Neural Chat Plugin tests...")
    
    # Test plugin import
    module = test_plugin_import()
    if not module:
        return 1
    
    # Test plugin creation
    plugin = test_plugin_creation(module)
    if not plugin:
        return 1
    
    # Test Mistral integration
    mistral_ok = test_mistral_integration(plugin)
    
    # Test memory integration
    memory_ok = test_memory_integration(plugin)
    
    # Test message processing if Mistral integration is working
    if mistral_ok:
        message_ok = test_message_processing(plugin)
    else:
        message_ok = False
        logger.warning("⚠ Skipping message processing test due to Mistral integration failure")
    
    # Print summary
    logger.info("\n----- TEST SUMMARY -----")
    logger.info(f"Plugin Import:        {'✓ PASS' if module else '✗ FAIL'}")
    logger.info(f"Plugin Creation:      {'✓ PASS' if plugin else '✗ FAIL'}")
    logger.info(f"Mistral Integration:  {'✓ PASS' if mistral_ok else '⚠ WARNING' if plugin.api_key is None else '✗ FAIL'}")
    logger.info(f"Memory Integration:   {'✓ PASS' if memory_ok else '✗ FAIL'}")
    logger.info(f"Message Processing:   {'✓ PASS' if message_ok else '⚠ SKIPPED' if not mistral_ok else '✗ FAIL'}")
    
    if module and plugin and (memory_ok or mistral_ok):
        logger.info("Overall result: PASS (with some tests potentially skipped)")
        return 0
    else:
        logger.info("Overall result: FAIL")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 