#!/usr/bin/env python3
"""
LUMINA V7 Mistral-AutoWiki Demo

This script demonstrates the integration between Mistral AI and the AutoWiki
system for automated knowledge acquisition and learning.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
Path('data/v7').mkdir(parents=True, exist_ok=True)
Path('logs').mkdir(exist_ok=True)

def display_banner():
    """Display the LUMINA V7 banner"""
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
    ║   V7.1.0.0 - Mistral & AutoWiki Demo                    ║
    ║   © 2023 LUMINA Labs                                     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main function for the demo"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LUMINA V7 Mistral-AutoWiki Demo")
    parser.add_argument("--api-key", help="Mistral API key")
    parser.add_argument("--model", default="mistral-small-latest", help="Mistral model to use")
    parser.add_argument("--topics", help="File with topics to load (one per line)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Display banner
    display_banner()
    
    # Check for dependencies
    try:
        from mistralai.client import MistralClient
        logger.info("Mistral client available")
    except ImportError:
        logger.error("Mistral client not available. Install with: pip install mistralai")
        print("\nERROR: Missing required dependency 'mistralai'")
        print("Please install it with: pip install mistralai")
        return 1
    
    # Import our modules
    try:
        from src.v7.mistral_integration import MistralEnhancedSystem
        from src.v7.autowiki import AutoWiki
        logger.info("Successfully imported Mistral and AutoWiki modules")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print(f"\nERROR: Failed to import required modules: {e}")
        print("Please make sure you are running from the project root directory")
        return 1
    
    # Get API key from argument or environment
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("No Mistral API key provided")
        print("\nWARNING: No Mistral API key provided")
        print("Set with --api-key or MISTRAL_API_KEY environment variable")
        
        use_mock = input("Continue with mock mode? (y/n): ").lower().strip() == 'y'
        if not use_mock:
            return 1
    
    # Initialize Mistral system
    print("\nInitializing Mistral Enhanced System...")
    mistral_system = MistralEnhancedSystem(
        api_key=api_key,
        model=args.model
    )
    
    # Initialize AutoWiki system
    print("Initializing AutoWiki system...")
    autowiki = AutoWiki(
        mistral_system=mistral_system,
        auto_fetch=False  # We'll control fetching manually in the demo
    )
    
    # Load topics if provided
    if args.topics:
        try:
            with open(args.topics, 'r') as f:
                topics = [line.strip() for line in f if line.strip()]
            count = autowiki.add_topics(topics)
            print(f"Added {count} topics to the AutoWiki queue")
        except Exception as e:
            logger.error(f"Error adding topics from file: {e}")
            print(f"Error adding topics from file: {e}")
    
    # Display initial status
    print("\n=== Initial System Status ===")
    mistral_stats = mistral_system.get_system_stats()
    print(f"Dictionary entries: {mistral_stats['dictionary_entries']}")
    
    autowiki_status = autowiki.get_status()
    print(f"AutoWiki queue: {autowiki_status['queue_size']} topics")
    print(f"Topics already fetched: {autowiki_status['topics_fetched']}")
    
    # Demo menu
    while True:
        print("\n=== LUMINA V7 Mistral-AutoWiki Demo ===")
        print("1. Process AutoWiki queue")
        print("2. Add topic to AutoWiki")
        print("3. Chat with Mistral")
        print("4. View system status")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            # Process AutoWiki queue
            if autowiki_status['queue_size'] == 0:
                print("No topics in queue. Add some with option 2.")
                continue
                
            print(f"Processing up to 3 topics from the queue...")
            count = autowiki.process_queue(max_items=3)
            print(f"Successfully processed {count} topics")
            
            # Update status
            autowiki_status = autowiki.get_status()
            
        elif choice == '2':
            # Add topic to AutoWiki
            topic = input("Enter topic to fetch: ").strip()
            if not topic:
                continue
                
            if autowiki.add_topic(topic):
                print(f"Added '{topic}' to the AutoWiki queue")
                autowiki_status = autowiki.get_status()
            else:
                print(f"Topic '{topic}' is already in the queue or has been fetched")
            
        elif choice == '3':
            # Chat with Mistral
            print("\n=== Chat with Mistral ===")
            print("Type 'exit' to return to main menu")
            
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ('exit', 'back', 'quit'):
                    break
                    
                # Process message with progress indicator
                print("Processing...", end="\r")
                result = mistral_system.process_message(user_input)
                
                # Display response with metrics
                print(f"\nAssistant: {result['response']}")
                print("\nMetrics:")
                print(f"- Consciousness Level: {result.get('consciousness_level', 0.0):.2f}")
                print(f"- Neural-Linguistic Score: {result.get('neural_linguistic_score', 0.0):.2f}")
                print(f"- Recursive Pattern Depth: {result.get('recursive_pattern_depth', 0)}")
                print(f"- Processing Time: {result.get('processing_time', 0.0):.2f}s")
                
                # Show if we used information from our learning dictionary
                topics_in_dict = []
                for word in user_input.split():
                    clean_word = word.lower().strip(".,!?;:()'\"")
                    if len(clean_word) > 3 and mistral_system.get_from_dictionary(clean_word):
                        topics_in_dict.append(clean_word)
                
                if topics_in_dict:
                    print(f"Used knowledge from dictionary: {', '.join(topics_in_dict)}")
            
        elif choice == '4':
            # View system status
            print("\n=== System Status ===")
            
            # Mistral stats
            mistral_stats = mistral_system.get_system_stats()
            print("Mistral Enhanced System:")
            print(f"- Dictionary entries: {mistral_stats['dictionary_entries']}")
            print(f"- Total exchanges: {mistral_stats['total_exchanges']}")
            print(f"- Avg. consciousness level: {mistral_stats['avg_consciousness_level']:.2f}")
            if api_key:
                print(f"- Model: {args.model}")
                print(f"- Total tokens used: {mistral_stats['total_tokens']}")
            
            # AutoWiki stats
            autowiki_status = autowiki.get_status()
            print("\nAutoWiki System:")
            print(f"- Queue size: {autowiki_status['queue_size']}")
            print(f"- Topics fetched: {autowiki_status['topics_fetched']}")
            print(f"- Success rate: {autowiki_status['successful_fetches']}/{autowiki_status['total_fetches'] or 1} "
                 f"({100 * autowiki_status['successful_fetches'] / (autowiki_status['total_fetches'] or 1):.1f}%)")
            
        elif choice == '5':
            # Exit
            print("Exiting LUMINA V7 Mistral-AutoWiki Demo...")
            break
            
        else:
            print("Invalid option. Please select 1-5.")
    
    # Clean up
    mistral_system.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 