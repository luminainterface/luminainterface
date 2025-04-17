#!/usr/bin/env python3
"""
LUMINA V7 System Launcher with Mistral and AutoWiki Integration

This script initializes and launches the V7 system with enhanced
language capabilities through Mistral AI integration and automatic
knowledge acquisition via the AutoWiki system.

Usage:
    python v7_launcher.py [options]

Options:
    --mistral-key KEY     Mistral API key
    --mistral-model MODEL Mistral model to use
    --no-gui              Run without GUI
    --no-autowiki         Disable autowiki system
    --debug               Enable debug logging
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/v7_launcher.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LUMINA V7 System Launcher")
    parser.add_argument("--mistral-key", help="Mistral API key")
    parser.add_argument("--mistral-model", default="mistral-small-latest", 
                      help="Mistral model to use (default: mistral-small-latest)")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--no-autowiki", action="store_true", help="Disable autowiki system")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--topics", help="File with topics to add to autowiki queue")
    return parser.parse_args()

def show_banner():
    """Show the V7 banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   ██╗     ██╗   ██╗███╗   ███╗██╗███╗   ██╗ █████╗      ║
    ║   ██║     ██║   ██║████╗ ████║██║████╗  ██║██╔══██╗     ║
    ║   ██║     ██║   ██║██╔████╔██║██║██╔██╗ ██║███████║     ║
    ║   ██║     ██║   ██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║     ║
    ║   ███████╗╚██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██║  ██║     ║
    ║   ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝     ║
    ║                                                          ║
    ║   V7.1.0.0 - Advanced Neural Network with Mistral       ║
    ║   © 2023 LUMINA Labs                                     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("LUMINA V7.1.0.0 with Mistral integration starting...")

def main():
    """Main entry point"""
    # Parse arguments
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Show banner
    show_banner()
    
    # Check for required dependencies
    try:
        # Import V7 components
        from src.v7.run_v7 import initialize_system
        has_v7 = True
        logger.info("V7 core system available")
    except ImportError:
        has_v7 = False
        logger.warning("V7 core system not available")
    
    try:
        # Import Mistral integration
        from src.v7.mistral_integration import MistralEnhancedSystem, MISTRAL_AVAILABLE
        has_mistral_integration = True
        logger.info("Mistral integration available")
    except ImportError:
        has_mistral_integration = False
        logger.warning("Mistral integration not available")
    
    try:
        # Import AutoWiki
        from src.v7.autowiki import AutoWiki
        has_autowiki = True
        logger.info("AutoWiki system available")
    except ImportError:
        has_autowiki = False
        logger.warning("AutoWiki system not available")
    
    # Check if GUI is available
    try:
        if args.no_gui:
            has_gui = False
        else:
            try:
                from PySide6.QtWidgets import QApplication
                has_gui = True
                logger.info("Using PySide6 for UI")
            except ImportError:
                try:
                    from PyQt5.QtWidgets import QApplication
                    has_gui = True
                    logger.info("Using PyQt5 for UI")
                except ImportError:
                    has_gui = False
                    logger.warning("Neither PySide6 nor PyQt5 is available. Running without GUI.")
    except Exception as e:
        has_gui = False
        logger.error(f"Error checking GUI availability: {e}")
    
    # Initialize V7 system if available
    system_manager = None
    if has_v7:
        try:
            system_manager = initialize_system()
            if not system_manager:
                logger.error("Failed to initialize V7 system")
        except Exception as e:
            logger.error(f"Error initializing V7 system: {e}")
    
    # Initialize Mistral if available
    mistral_system = None
    if has_mistral_integration:
        try:
            # Get API key from argument or environment
            api_key = args.mistral_key or os.environ.get("MISTRAL_API_KEY")
            if not api_key and MISTRAL_AVAILABLE:
                logger.warning("No Mistral API key provided. Set with --mistral-key or MISTRAL_API_KEY environment variable.")
            
            # Create Mistral system
            mistral_system = MistralEnhancedSystem(
                api_key=api_key,
                model=args.mistral_model,
                system_manager=system_manager
            )
            logger.info(f"Mistral Enhanced System initialized with model {args.mistral_model}")
        except Exception as e:
            logger.error(f"Error initializing Mistral system: {e}")
    
    # Initialize AutoWiki if available and not disabled
    autowiki = None
    if has_autowiki and not args.no_autowiki:
        try:
            # Create AutoWiki system
            autowiki = AutoWiki(
                mistral_system=mistral_system,
                auto_fetch=True
            )
            logger.info("AutoWiki system initialized")
            
            # Add topics from file if specified
            if args.topics:
                try:
                    with open(args.topics, 'r') as f:
                        topics = [line.strip() for line in f if line.strip()]
                    count = autowiki.add_topics(topics)
                    logger.info(f"Added {count} topics to the AutoWiki queue")
                except Exception as e:
                    logger.error(f"Error adding topics from file: {e}")
        except Exception as e:
            logger.error(f"Error initializing AutoWiki system: {e}")
    
    # Run with or without GUI
    if has_gui and not args.no_gui:
        try:
            # Import GUI components
            from src.v7.run_v7 import run_with_gui
            
            # Create application
            app = QApplication(sys.argv)
            
            # Run with GUI
            exit_code = run_with_gui(system_manager)
            
            # Clean up
            if mistral_system:
                mistral_system.close()
            
            return exit_code
        except Exception as e:
            logger.error(f"Error running with GUI: {e}")
            return 1
    else:
        try:
            # Run without GUI
            logger.info("Running without GUI")
            
            # Create interactive console
            print("\nLUMINA V7 Interactive Console")
            print("Type 'help' for available commands, 'exit' to quit")
            
            # Command handling loop
            while True:
                try:
                    command = input("\n> ").strip().lower()
                    
                    if command in ('exit', 'quit', 'q'):
                        break
                    
                    elif command == 'help':
                        print("\nAvailable commands:")
                        print("  help       - Show this help")
                        print("  status     - Show system status")
                        print("  mistral    - Enter Mistral chat mode")
                        print("  autowiki   - Show AutoWiki status")
                        print("  fetch      - Process AutoWiki queue")
                        print("  add TOPIC  - Add topic to AutoWiki queue")
                        print("  exit       - Exit the system")
                    
                    elif command == 'status':
                        print("\n=== System Status ===")
                        if system_manager:
                            print(f"V7 System: Active")
                        else:
                            print(f"V7 System: Not available")
                            
                        if mistral_system:
                            stats = mistral_system.get_system_stats()
                            print(f"Mistral: Active (Model: {args.mistral_model})")
                            print(f"  Exchanges: {stats['total_exchanges']}")
                            print(f"  Dictionary entries: {stats['dictionary_entries']}")
                            print(f"  Avg. consciousness: {stats['avg_consciousness_level']:.2f}")
                        else:
                            print(f"Mistral: Not available")
                            
                        if autowiki:
                            status = autowiki.get_status()
                            print(f"AutoWiki: Active")
                            print(f"  Queue size: {status['queue_size']}")
                            print(f"  Topics fetched: {status['topics_fetched']}")
                            print(f"  Success rate: {status['successful_fetches']}/{status['total_fetches']}")
                        else:
                            print(f"AutoWiki: Not available")
                    
                    elif command == 'mistral':
                        if mistral_system:
                            print("\n=== Mistral Chat Mode ===")
                            print("Type 'exit' to return to main menu")
                            
                            while True:
                                chat_input = input("\nYou: ").strip()
                                if chat_input.lower() in ('exit', 'back'):
                                    break
                                    
                                result = mistral_system.process_message(chat_input)
                                print(f"\nAssistant: {result['response']}")
                                print(f"[CL: {result.get('consciousness_level', 0.0):.2f}, " +
                                     f"NLS: {result.get('neural_linguistic_score', 0.0):.2f}, " +
                                     f"RPD: {result.get('recursive_pattern_depth', 0)}]")
                        else:
                            print("Mistral system not available")
                    
                    elif command == 'autowiki':
                        if autowiki:
                            status = autowiki.get_status()
                            print("\n=== AutoWiki Status ===")
                            for key, value in status.items():
                                print(f"{key}: {value}")
                        else:
                            print("AutoWiki system not available")
                    
                    elif command == 'fetch':
                        if autowiki:
                            print("Processing AutoWiki queue...")
                            count = autowiki.process_queue(max_items=3)
                            print(f"Successfully processed {count} topics")
                        else:
                            print("AutoWiki system not available")
                    
                    elif command.startswith('add '):
                        if autowiki:
                            topic = command[4:].strip()
                            if topic:
                                if autowiki.add_topic(topic):
                                    print(f"Added '{topic}' to the AutoWiki queue")
                                else:
                                    print(f"Topic '{topic}' is already in the queue or has been fetched")
                            else:
                                print("Please specify a topic to add")
                        else:
                            print("AutoWiki system not available")
                    
                    else:
                        print("Unknown command. Type 'help' for available commands.")
                
    except KeyboardInterrupt:
                    print("\nUse 'exit' to quit")
                except Exception as e:
                    print(f"Error: {e}")
            
            # Clean up
            print("Shutting down...")
            if mistral_system:
                mistral_system.close()
            if system_manager:
                system_manager.shutdown()
            
            return 0
        
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt, shutting down...")
            
            # Clean up
            if mistral_system:
                mistral_system.close()
            if system_manager:
                system_manager.shutdown()
            
            return 0
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 
