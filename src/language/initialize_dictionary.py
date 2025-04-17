#!/usr/bin/env python3
"""
Initialize Dictionary System

This script initializes the Dictionary Manager with socket support for AutoWiki
integration. It sets up the necessary database tables, socket connections,
and UI adapters.

Usage:
    python src/language/initialize_dictionary.py [--data-dir DATA_DIR]
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/dictionary_system.log')
    ]
)

logger = logging.getLogger("dictionary_system")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Initialize Dictionary System for AutoWiki integration")
    parser.add_argument('--data-dir', type=str, default='data/dictionary',
                        help='Directory for dictionary data (default: data/dictionary)')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to existing database (default: None, creates a new one)')
    parser.add_argument('--socket-id', type=str, default=None,
                        help='Socket ID for the dictionary plugin (default: auto-generated)')
    return parser.parse_args()

def main():
    """Main initialization function"""
    # Parse command line arguments
    args = parse_args()
    data_dir = args.data_dir
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info(f"Initializing Dictionary System with data directory: {data_dir}")
    
    try:
        # Import required modules
        from src.language.database_manager import DatabaseManager
        from src.language.dictionary_manager import DictionaryManager
        from src.language.dictionary_socket import DictionarySocketPlugin
        
        # Initialize database manager
        if args.db_path:
            db_dir = os.path.dirname(args.db_path)
            db_name = os.path.basename(args.db_path)
            db_manager = DatabaseManager(data_dir=db_dir, db_name=db_name)
            logger.info(f"Using existing database at {args.db_path}")
        else:
            db_manager = DatabaseManager()
            logger.info("Created new database manager")
        
        # Initialize dictionary manager
        dictionary_manager = DictionaryManager(data_dir=data_dir, db_manager=db_manager)
        logger.info("Dictionary Manager initialized")
        
        # Initialize dictionary socket plugin
        socket_plugin = DictionarySocketPlugin(dictionary_manager, plugin_id=args.socket_id)
        logger.info(f"Dictionary Socket Plugin initialized with ID: {socket_plugin.node_id}")
        
        # Check for V5 socket manager
        try:
            from src.v5.frontend_socket_manager import FrontendSocketManager
            try:
                socket_manager = FrontendSocketManager()
                socket_manager.register_plugin(socket_plugin)
                logger.info("Registered Dictionary Socket Plugin with Frontend Socket Manager")
            except Exception as e:
                logger.warning(f"Could not register with Frontend Socket Manager: {e}")
        except ImportError:
            logger.warning("Frontend Socket Manager not available")
        
        # Check for V7 AutoWiki components
        try:
            from src.v7.auto_wiki.auto_wiki_plugin import AutoWikiPlugin
            try:
                auto_wiki = AutoWikiPlugin()
                if socket_plugin.socket:
                    socket_plugin.socket.connect_to(auto_wiki.socket)
                    logger.info("Connected Dictionary Socket to AutoWiki Plugin")
            except Exception as e:
                logger.warning(f"Could not connect to AutoWiki Plugin: {e}")
        except ImportError:
            logger.warning("AutoWiki Plugin not available")
        
        # Create some sample dictionary entries
        logger.info("Creating sample dictionary entries...")
        
        try:
            # Create sample entries
            neural_id = dictionary_manager.add_entry(
                term="Neural Network",
                definition="A computing system inspired by biological neural networks that can learn to perform tasks by considering examples.",
                source="initialization",
                confidence=0.9,
                verified=True
            )
            
            language_id = dictionary_manager.add_entry(
                term="Language Model",
                definition="A statistical model that predicts the probability of a sequence of words or tokens.",
                source="initialization",
                confidence=0.9,
                verified=True
            )
            
            # Add relationships
            dictionary_manager.add_relationship(
                source_id=neural_id,
                target_id=language_id,
                relationship_type="related",
                strength=0.8
            )
            
            logger.info("Sample dictionary entries created")
            
        except Exception as e:
            logger.warning(f"Error creating sample entries: {e}")
        
        # Print success message with socket ID for reference
        print("\n" + "="*80)
        print("Dictionary System initialized successfully")
        print(f"Socket ID: {socket_plugin.node_id}")
        print(f"Data directory: {os.path.abspath(data_dir)}")
        print("Ready to receive AutoWiki data")
        print("="*80 + "\n")
        
        # Keep running to maintain socket connections
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down Dictionary System...")
            logger.info("Dictionary System shutting down")
            
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        print(f"Error: {e}")
        print("Make sure you have all required modules installed.")
        return 1
    except Exception as e:
        logger.error(f"Error initializing Dictionary System: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 