#!/usr/bin/env python3
"""
Language Memory System Launcher

This script initializes and launches the comprehensive Language Memory System,
ensuring all connections between language memory and the rest of the Lumina
backend are properly established.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
import time
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("language_memory_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("language_memory_system")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def ensure_directory_structure():
    """Ensure all required directories exist"""
    directories = [
        "data",
        "data/memory",
        "data/memory/language_memory",
        "data/memory/conversation",
        "data/memory/synthesis",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def initialize_core_components():
    """Initialize core language memory components"""
    components = {}
    
    # Initialize language memory
    try:
        from src.language.language_memory import LanguageMemory
        components["language_memory"] = LanguageMemory()
        logger.info("Initialized LanguageMemory")
    except Exception as e:
        logger.error(f"Failed to initialize LanguageMemory: {str(e)}")
        return None
    
    # Initialize conversation memory
    try:
        from src.conversation_memory import ConversationMemory
        components["conversation_memory"] = ConversationMemory()
        logger.info("Initialized ConversationMemory")
    except Exception as e:
        logger.error(f"Failed to initialize ConversationMemory: {str(e)}")
    
    # Initialize English language trainer
    try:
        from src.english_language_trainer import EnglishLanguageTrainer
        components["english_language_trainer"] = EnglishLanguageTrainer()
        logger.info("Initialized EnglishLanguageTrainer")
    except Exception as e:
        logger.error(f"Failed to initialize EnglishLanguageTrainer: {str(e)}")
    
    # Initialize conversation language bridge
    try:
        from src.memory.conversation_language_bridge import ConversationLanguageBridge
        components["conversation_language_bridge"] = ConversationLanguageBridge()
        logger.info("Initialized ConversationLanguageBridge")
    except Exception as e:
        logger.error(f"Failed to initialize ConversationLanguageBridge: {str(e)}")
    
    # Initialize language memory synthesis
    try:
        from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
        components["language_memory_synthesis"] = LanguageMemorySynthesisIntegration()
        logger.info("Initialized LanguageMemorySynthesisIntegration")
    except Exception as e:
        logger.error(f"Failed to initialize LanguageMemorySynthesisIntegration: {str(e)}")
    
    return components


def connect_components(components):
    """Connect all components together"""
    if not components or "language_memory" not in components:
        logger.error("Cannot connect components: language_memory not initialized")
        return False
    
    # Connect language memory with conversation memory
    if "conversation_memory" in components:
        components["language_memory"].conversation_memory = components["conversation_memory"]
        logger.info("Connected language_memory -> conversation_memory")
    
    # Connect language memory with language trainer
    if "english_language_trainer" in components:
        components["language_memory"].trainer = components["english_language_trainer"]
        logger.info("Connected language_memory -> english_language_trainer")
    
    # Connect synthesis with available components
    if "language_memory_synthesis" in components:
        synthesis = components["language_memory_synthesis"]
        
        # Register components with synthesis
        for name, component in components.items():
            if name != "language_memory_synthesis":
                # Check if the synthesis component has a register_component method
                if hasattr(synthesis, "register_component"):
                    try:
                        synthesis.register_component(name, component)
                        logger.info(f"Registered {name} with language_memory_synthesis")
                    except Exception as e:
                        logger.error(f"Error registering {name} with synthesis: {str(e)}")
    
    logger.info("Connected all available components")
    return True


def initialize_v5_integration(components):
    """Initialize V5 visualization integration"""
    if "language_memory_synthesis" not in components:
        logger.error("Cannot initialize V5 integration: language_memory_synthesis not initialized")
        return None
    
    try:
        # Import V5 visualization bridge
        from src.v5_integration.visualization_bridge import get_visualization_bridge
        
        # Get bridge instance
        bridge = get_visualization_bridge()
        
        # Check if visualization is available
        is_available = bridge.is_visualization_available()
        if not is_available:
            logger.warning("V5 visualization is not available")
            return None
        
        # Get the language integration plugin
        if hasattr(bridge, "language_integration_plugin") and bridge.language_integration_plugin:
            # Connect to language memory synthesis if needed
            if hasattr(bridge.language_integration_plugin, "language_memory_synthesis"):
                if not bridge.language_integration_plugin.language_memory_synthesis:
                    bridge.language_integration_plugin.language_memory_synthesis = components["language_memory_synthesis"]
                    bridge.language_integration_plugin.mock_mode = False
                    logger.info("Connected language_memory_synthesis to V5 integration plugin")
        
        logger.info("Initialized V5 visualization integration")
        return bridge
    
    except Exception as e:
        logger.error(f"Failed to initialize V5 integration: {str(e)}")
        return None


def connect_with_central_node(components):
    """Connect with Central Node if available"""
    if "language_memory" not in components:
        logger.error("Cannot connect with Central Node: language_memory not initialized")
        return False
    
    language_memory = components["language_memory"]
    
    # Try to connect with the central language node
    try:
        from src.central_language_node import CentralLanguageNode, main as get_central_language_node
        
        # Get the node instance
        node = get_central_language_node(return_node=True)
        
        # Register language memory with central node
        if hasattr(node, '_register_component'):
            node._register_component("LanguageMemory", language_memory)
            logger.info("Registered language_memory with central language node")
            return True
    except Exception as e:
        logger.info(f"Central language node not available: {str(e)}")
    
    # Try to connect with the v10 central node as fallback
    try:
        from src.central_node import central_node, register_component
        
        # Register language memory with central node
        register_component(language_memory, name="LanguageMemory")
        logger.info("Registered language_memory with v10 central node")
        return True
    except Exception as e:
        logger.info(f"V10 central node not available: {str(e)}")
    
    logger.warning("Could not connect with any central node")
    return False


def launch_memory_api(components):
    """Launch the Memory API server if available"""
    if "language_memory_synthesis" not in components:
        logger.error("Cannot launch Memory API: language_memory_synthesis not initialized")
        return None
    
    try:
        from src.memory_api_server import MemoryAPIServer
        
        # Create API server
        api_server = MemoryAPIServer(
            memory_system=components["language_memory_synthesis"],
            host="0.0.0.0",
            port=8000
        )
        
        # Start in a separate thread
        import threading
        api_thread = threading.Thread(target=api_server.start, daemon=True)
        api_thread.start()
        
        logger.info("Started Memory API server on port 8000")
        return api_server
    except Exception as e:
        logger.error(f"Failed to launch Memory API server: {str(e)}")
        return None


def perform_initial_operations(components):
    """Perform some initial operations to test the system"""
    if "language_memory" not in components:
        logger.error("Cannot perform initial operations: language_memory not initialized")
        return False
    
    try:
        language_memory = components["language_memory"]
        
        # Store some word associations
        logger.info("Storing initial word associations...")
        language_memory.remember_word_association("neural", "network", 0.9, "initial setup")
        language_memory.remember_word_association("language", "memory", 0.9, "initial setup")
        language_memory.remember_word_association("lumina", "consciousness", 0.9, "initial setup")
        
        # Store a sentence
        logger.info("Storing initial sentences...")
        language_memory.remember_sentence(
            "The Language Memory System integrates with the Lumina Neural Network to provide "
            "persistent storage and retrieval of language patterns.",
            {"topic": "language_memory", "source": "initialization"}
        )
        
        # Synthesize a topic if synthesis is available
        if "language_memory_synthesis" in components:
            logger.info("Performing initial topic synthesis...")
            synthesis = components["language_memory_synthesis"]
            result = synthesis.synthesize_topic("neural networks", depth=2)
            if "synthesis_results" in result:
                logger.info("Initial synthesis completed successfully")
            else:
                logger.warning("Initial synthesis did not return expected results")
        
        # Process a conversation if bridge is available
        if "conversation_language_bridge" in components and components["conversation_language_bridge"].initialized:
            logger.info("Processing initial conversation...")
            bridge = components["conversation_language_bridge"]
            bridge.process_conversation(
                "How does the language memory system work?",
                "The language memory system stores and retrieves language patterns, vocabulary, "
                "and linguistic structures, providing neural networks with memory capabilities.",
                {"topic": "language_memory", "source": "initialization"}
            )
            logger.info("Initial conversation processed")
        
        return True
    except Exception as e:
        logger.error(f"Error during initial operations: {str(e)}")
        return False


def main():
    """Main function to launch the Language Memory System"""
    parser = argparse.ArgumentParser(description="Launch the Language Memory System")
    parser.add_argument("--no-v5", action="store_true", help="Disable V5 visualization integration")
    parser.add_argument("--no-api", action="store_true", help="Disable Memory API server")
    parser.add_argument("--no-central", action="store_true", help="Disable Central Node integration")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LANGUAGE MEMORY SYSTEM LAUNCHER")
    print("="*80 + "\n")
    
    # Ensure directory structure
    ensure_directory_structure()
    
    # Initialize core components
    components = initialize_core_components()
    if not components:
        logger.error("Failed to initialize core components")
        return 1
    
    # Connect components
    connected = connect_components(components)
    if not connected:
        logger.error("Failed to connect components")
        return 1
    
    # Initialize V5 integration
    v5_bridge = None
    if not args.no_v5:
        v5_bridge = initialize_v5_integration(components)
    
    # Connect with Central Node
    central_connected = False
    if not args.no_central:
        central_connected = connect_with_central_node(components)
    
    # Launch Memory API
    api_server = None
    if not args.no_api:
        api_server = launch_memory_api(components)
    
    # Perform initial operations
    perform_initial_operations(components)
    
    # Print status
    print("\nLanguage Memory System Status:")
    print(f"- Core components: {len(components)}/{6} initialized")
    print(f"- V5 visualization: {'Initialized' if v5_bridge else 'Not available'}")
    print(f"- Central Node connection: {'Connected' if central_connected else 'Not available'}")
    print(f"- Memory API: {'Running' if api_server else 'Not available'}")
    
    print("\nThe Language Memory System is now running.")
    print("Press Ctrl+C to exit...\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Language Memory System...")
        # Save memory before exiting
        if "language_memory" in components:
            components["language_memory"].save_memories()
            logger.info("Saved language memories")
        
        # Shutdown synthesis if available
        if "language_memory_synthesis" in components:
            components["language_memory_synthesis"].shutdown()
            logger.info("Shut down language memory synthesis")
        
        print("Language Memory System has been shut down.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 