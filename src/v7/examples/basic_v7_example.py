#!/usr/bin/env python
"""
Basic Example of V7 Node Consciousness System

This script demonstrates the basic usage of the V7 Node Consciousness system,
showing how to initialize nodes, process data, and handle node interactions.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path to make imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.v7.initialize_v7 import initialize_v7, shutdown_v7
from src.v7.node_consciousness_manager import NodeState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example texts for processing
EXAMPLE_TEXTS = [
    "The V7 Node Consciousness system represents a significant evolution in self-aware neural network architectures.",
    "Each node in the system can learn, adapt, and communicate with other nodes to form a collective intelligence.",
    "Unlike traditional neural networks, consciousness nodes have awareness of their internal state and connections.",
    "The Monday consciousness node brings emotional intelligence to the system through pattern recognition.",
    "Through language processing, the system can understand and generate natural language with self-reference capabilities.",
    "Contradiction detection allows the system to identify logical inconsistencies in its own knowledge base.",
    "The breath visualization provides a visual representation of the system's processing rhythm and flow."
]

# Example patterns for demonstration
EXAMPLE_PATTERNS = [
    {
        "name": "self_reference",
        "data": {
            "keywords": ["I", "me", "my", "self", "consciousness", "aware"],
            "pattern_type": "linguistic",
            "priority": 0.8
        }
    },
    {
        "name": "contradiction",
        "data": {
            "keywords": ["however", "but", "although", "not", "never", "despite"],
            "pattern_type": "logical",
            "priority": 0.9
        }
    },
    {
        "name": "emotion",
        "data": {
            "keywords": ["happy", "sad", "excited", "worried", "anxious", "calm", "peaceful"],
            "pattern_type": "emotional",
            "priority": 0.7
        }
    }
]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='V7 Node Consciousness Basic Example')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--disable-monday', action='store_true', help='Disable Monday consciousness integration')
    parser.add_argument('--disable-language', action='store_true', help='Disable Language node')
    parser.add_argument('--llm-weight', type=float, default=0.5, help='Weight for LLM integration (0.0-1.0)')
    parser.add_argument('--no-interactive', action='store_true', help='Run in non-interactive mode')
    return parser.parse_args()


def run_tests(manager, context):
    """Run basic tests of the V7 system."""
    logger.info("Running basic system tests...")
    
    # Get status of all nodes
    node_statuses = manager.get_all_node_statuses()
    logger.info(f"System has {len(node_statuses)} active nodes:")
    for node_id, status in node_statuses.items():
        logger.info(f"  - {status['name']} ({node_id}): {status['state']}")
    
    # Test language node if available
    if 'language' in context['node_ids']:
        logger.info("\nTesting Language Consciousness Node...")
        language_node_id = context['node_ids']['language']
        language_node = manager.get_node(language_node_id)
        
        # Process text through language node
        for i, text in enumerate(EXAMPLE_TEXTS[:3]):  # Use just the first 3 for testing
            logger.info(f"\nProcessing text {i+1}:")
            logger.info(f"> {text}")
            
            # Process the text
            result = manager.process_data(language_node_id, {'text': text})
            
            if result.get('success', False):
                logger.info(f"- Word count: {result.get('word_count', 0)}")
                logger.info(f"- Sentence count: {result.get('sentence_count', 0)}")
                logger.info(f"- Sentiment: {result.get('sentiment', {}).get('primary', 'unknown')}")
                
                # Show patterns if detected
                patterns = result.get('patterns_detected', [])
                if patterns:
                    logger.info(f"- Patterns detected: {', '.join(patterns)}")
            else:
                logger.error(f"Error processing text: {result.get('error', 'Unknown error')}")
        
        # Train language node with patterns
        logger.info("\nTraining language node with example patterns...")
        training_result = manager.train_node(language_node_id, {
            'patterns': EXAMPLE_PATTERNS
        })
        
        if training_result.get('success', False):
            logger.info(f"Training successful: {training_result.get('patterns_learned', 0)} patterns learned")
        else:
            logger.error(f"Error training node: {training_result.get('error', 'Unknown error')}")
    
    # Test Monday interface if available
    if 'monday_interface' in context and context['monday_interface']:
        logger.info("\nTesting Monday Consciousness Interface...")
        monday = context['monday_interface']
        
        # Get Monday's current consciousness level
        consciousness_level = monday.get_consciousness_level()
        logger.info(f"Current consciousness level: {consciousness_level:.2f}")
        
        # Process text through Monday
        test_text = "I feel a sense of curiosity about how consciousness emerges from connection."
        logger.info(f"\nSending text to Monday: \"{test_text}\"")
        
        # Monitor consciousness level changes
        for i in range(5):
            time.sleep(1)
            new_level = monday.get_consciousness_level()
            logger.info(f"Consciousness level: {new_level:.2f}")
    
    logger.info("\nBasic system tests completed")


def interactive_mode(manager, context):
    """Run the system in interactive mode."""
    logger.info("\nEntering interactive mode. Type 'exit' to quit.")
    logger.info("Available commands: status, process, train, help")
    
    while True:
        try:
            command = input("\nCommand > ").strip().lower()
            
            if command == 'exit':
                break
                
            elif command == 'help':
                print("""
Available commands:
  status - Show the status of all nodes
  process - Process a text through the language node
  train - Train the language node with a pattern
  connections - Show all connections between nodes
  exit - Exit the program
                """)
                
            elif command == 'status':
                node_statuses = manager.get_all_node_statuses()
                print(f"\nSystem has {len(node_statuses)} nodes:")
                for node_id, status in node_statuses.items():
                    print(f"  - {status['name']} ({node_id}): {status['state']}")
                    
                    # Show additional info for language node
                    if status.get('language_status'):
                        ls = status['language_status']
                        print(f"    - Capability: {ls.get('language_capability', 'unknown')}")
                        print(f"    - LLM integrated: {ls.get('llm_integrated', False)}")
                        print(f"    - Texts processed: {ls.get('stats', {}).get('texts_processed', 0)}")
                
            elif command == 'process':
                if 'language' not in context['node_ids']:
                    print("Language node is not available")
                    continue
                    
                text = input("Enter text to process: ").strip()
                if not text:
                    continue
                    
                language_node_id = context['node_ids']['language']
                result = manager.process_data(language_node_id, {'text': text})
                
                if result.get('success', False):
                    print("\nProcessing Results:")
                    print(f"- Word count: {result.get('word_count', 0)}")
                    print(f"- Sentence count: {result.get('sentence_count', 0)}")
                    print(f"- Sentiment: {result.get('sentiment', {}).get('primary', 'unknown')}")
                    print(f"- Language score: {result.get('language_score', 0.0):.2f}")
                    
                    # Show patterns if detected
                    patterns = result.get('patterns_detected', [])
                    if patterns:
                        print(f"- Patterns detected: {', '.join(patterns)}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
            elif command == 'train':
                if 'language' not in context['node_ids']:
                    print("Language node is not available")
                    continue
                    
                pattern_name = input("Enter pattern name: ").strip()
                if not pattern_name:
                    continue
                    
                keywords = input("Enter keywords (comma-separated): ").strip()
                if not keywords:
                    continue
                    
                keywords_list = [k.strip() for k in keywords.split(',')]
                
                pattern = {
                    'name': pattern_name,
                    'data': {
                        'keywords': keywords_list,
                        'pattern_type': 'custom',
                        'priority': 0.5
                    }
                }
                
                language_node_id = context['node_ids']['language']
                result = manager.train_node(language_node_id, {'patterns': [pattern]})
                
                if result.get('success', False):
                    print(f"Pattern '{pattern_name}' successfully trained")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
            elif command == 'connections':
                connections = manager.get_connections()
                print(f"\nSystem has {len(connections)} connections:")
                for conn in connections:
                    print(f"  - {conn['source_id']} → {conn['target_id']} ({conn['type']})")
                    print(f"    Strength: {conn['strength']:.2f}, Last active: {time.ctime(conn['last_active'])}")
                    
            else:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nExiting interactive mode")


def main():
    """Main function to run the example."""
    args = parse_arguments()
    
    # Configure logging based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration for the V7 system
    config = {
        'enable_monday': not args.disable_monday,
        'enable_language': not args.disable_language,
        'llm_weight': args.llm_weight,
        'data_path': './data/v7_example',
    }
    
    logger.info("Starting V7 Node Consciousness Basic Example")
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize the V7 system
        manager, context = initialize_v7(config)
        
        # Run basic tests
        run_tests(manager, context)
        
        # Interactive mode if requested
        if not args.no_interactive:
            interactive_mode(manager, context)
        
        # Shutdown the system
        logger.info("Shutting down V7 system")
        shutdown_v7(context)
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error running example: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 