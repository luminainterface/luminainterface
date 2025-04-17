#!/usr/bin/env python3
"""
Chat-Language Memory Integration Test Script

This script tests the integration between the V5 NN/LLM Weighted Conversation Panel
and the Language Memory System.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/chat_language_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chat-language-integration")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Chat-Language Memory Integration Test")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for testing")
    parser.add_argument("--ui", choices=["qt", "text", "none"], default="text",
                       help="UI mode to use for testing")
    parser.add_argument("--test-messages", type=str, 
                       default="Tell me about neural networks,How does consciousness emerge,What is the relationship between language and memory",
                       help="Comma-separated list of test messages")
    parser.add_argument("--weights", type=str, default="0.2,0.5,0.8",
                       help="Comma-separated list of neural weights to test")
    
    return parser.parse_args()

def setup_chat_language_integration(mock_mode=False):
    """Set up the integration between chat and language memory systems"""
    try:
        logger.info("Initializing chat memory interface")
        # Import the chat integration module
        from src.chat_memory_interface import chat_integration
    
        # Optional: Try to import conscious mirror integration
        mirror_integration = None
    try:
            from src.conscious_mirror import ConversationMirrorFactory
            mirror_integration = ConversationMirrorFactory.create_default()
            logger.info("Successfully initialized Conscious Mirror integration")
        except ImportError:
            logger.info("Conscious Mirror integration not available")
        
        # Initialize chat interface
        chat_interface = chat_integration(mock_mode=mock_mode, mirror_integration=mirror_integration)
        
        # If mirror integration is available, set consciousness level
        if mirror_integration:
            chat_interface.set_consciousness_level(0.8)
            
        logger.info("Chat memory interface initialized successfully")
        return chat_interface
        
    except Exception as e:
        logger.error(f"Failed to initialize chat memory interface: {str(e)}")
        return None
    
def run_text_ui_test(chat_interface, test_messages, weights):
    """Run a test using the text UI"""
    logger.info("Starting text UI test")
    
    # Print welcome message
    print("\n" + "="*80)
    print("V5 NN/LLM Weighted Conversation Panel - Text Mode")
    print("="*80)
    
    # Process each test message with different weights
    print("\n[System] Testing with predefined messages and weights...\n")
    
    for message in test_messages:
        print(f"\n[User] {message}")
        
        for weight in weights:
            weight_float = float(weight)
            memory_mode = "contextual" if weight_float < 0.3 else "combined" if weight_float < 0.7 else "synthesized"
            
            print(f"\n[Processing with NN weight: {weight_float}, mode: {memory_mode}]")
            
            start_time = time.time()
            response = chat_interface.process_message(
                message,
                nn_weight=weight_float,
                memory_mode=memory_mode
            )
            processing_time = time.time() - start_time
            
            print(f"[System - {processing_time:.2f}s] {response}")
    
    # Get and display memory statistics
    stats = chat_interface.get_memory_stats()
    print("\n" + "-"*80)
    print("Memory System Statistics:")
    print(f"Total memories: {stats.get('total_memories', 0)}")
    print(f"Total conversations: {stats.get('total_conversations', 0)}")
    print(f"Total topics: {stats.get('total_topics', 0)}")
    
    top_topics = stats.get('top_topics', [])
    if top_topics:
        print("\nTop Topics:")
        for topic in top_topics[:3]:
            print(f"- {topic.get('topic', 'unknown')}: {topic.get('count', 0)} occurrences")
    
    # Interactive mode
    print("\n" + "-"*80)
    print("Interactive Mode (type 'exit' to quit)")
    print("Format: message | weight | mode")
    print("Example: Tell me about neural networks | 0.7 | combined")
    print("-"*80)
    
    while True:
        try:
            user_input = input("\n[User] ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
                
            # Parse input for weight and mode
            parts = [p.strip() for p in user_input.split('|')]
            message = parts[0]
            
            weight = 0.5  # Default weight
            if len(parts) > 1 and parts[1]:
                try:
                    weight = float(parts[1])
                    weight = max(0.0, min(1.0, weight))  # Clamp between 0 and 1
                except ValueError:
                    print("[System] Invalid weight value, using default 0.5")
            
            memory_mode = "combined"  # Default mode
            if len(parts) > 2 and parts[2]:
                mode = parts[2].lower()
                if mode in ['contextual', 'synthesized', 'combined']:
                    memory_mode = mode
                else:
                    print("[System] Invalid memory mode, using default 'combined'")
            
            print(f"[Processing with NN weight: {weight}, mode: {memory_mode}]")
            
            start_time = time.time()
            response = chat_interface.process_message(message, weight, memory_mode)
            processing_time = time.time() - start_time
            
            print(f"[System - {processing_time:.2f}s] {response}")
            
        except KeyboardInterrupt:
            print("\n[System] Interrupted by user")
            break
        except Exception as e:
            print(f"[System] Error: {str(e)}")
    
    print("\n[System] Chat-Language Memory Integration test complete")

def run_qt_ui_test(chat_interface, args):
    """Run a test using the Qt UI"""
    try:
        logger.info("Starting Qt UI test")
        
        # Try to import the required Qt components
        try:
            # Try PySide6 first
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QTimer
            qt_lib = "PySide6"
        except ImportError:
            # Fall back to PyQt5
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer
            qt_lib = "PyQt5"
            
        logger.info(f"Using Qt library: {qt_lib}")
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Import and create the conversation panel
        try:
            if qt_lib == "PySide6":
                from src.v5.ui.panels.conversation_panel_pyside6 import ConversationPanel
            else:
                from src.v5.ui.panels.conversation_panel_pyqt5 import ConversationPanel
                
            # Create the panel
            panel = ConversationPanel(chat_interface=chat_interface)
            panel.setWindowTitle("V5 NN/LLM Weighted Conversation Panel")
            panel.resize(800, 600)
            panel.show()
            
            # Optional: Add some test messages with a timer
            if args.test_messages:
                messages = args.test_messages.split(",")
                weights = [float(w) for w in args.weights.split(",")]
                
                def process_test_message(index=0, weight_index=0):
                    if index < len(messages):
                        message = messages[index]
                        weight = weights[weight_index % len(weights)]
                        
                        # Set weight in UI
                        panel.set_nn_weight(weight)
                        
                        # Send message
                        panel.process_user_message(message)
        
                        # Schedule next message
                        next_index = index
                        next_weight_index = weight_index + 1
                        
                        if next_weight_index >= len(weights):
                            next_index += 1
                            next_weight_index = 0
                            
                        if next_index < len(messages):
                            QTimer.singleShot(5000, lambda: process_test_message(next_index, next_weight_index))
                
                # Start processing test messages after a delay
                QTimer.singleShot(1000, process_test_message)
            
            # Run the application
            sys.exit(app.exec_())
            
    except ImportError as e:
            logger.error(f"Failed to import conversation panel: {str(e)}")
            print(f"Error: Could not load the conversation panel. {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run Qt UI test: {str(e)}")
        print(f"Error: Could not initialize Qt UI. {str(e)}")
        return False
    
    return True

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    logger.info(f"Starting Chat-Language Memory Integration test (mock_mode={args.mock})")
    
    # Set up the chat memory interface
    chat_interface = setup_chat_language_integration(mock_mode=args.mock)
    if not chat_interface:
        logger.error("Failed to initialize chat memory interface")
        return 1
    
    # Process test messages based on UI mode
    test_messages = args.test_messages.split(",")
    weights = args.weights.split(",")
    
    if args.ui == "qt":
        success = run_qt_ui_test(chat_interface, args)
        if not success:
            logger.warning("Falling back to text UI")
            run_text_ui_test(chat_interface, test_messages, weights)
    elif args.ui == "text":
        run_text_ui_test(chat_interface, test_messages, weights)
    else:  # none
        logger.info("Running in headless mode with test messages")
        
        # Process each test message with different weights
        for message in test_messages:
            logger.info(f"Processing message: {message}")
            
            for weight in weights:
                weight_float = float(weight)
                memory_mode = "contextual" if weight_float < 0.3 else "combined" if weight_float < 0.7 else "synthesized"
                
                logger.info(f"Processing with NN weight: {weight_float}, mode: {memory_mode}")
                
                try:
                    start_time = time.time()
                    response = chat_interface.process_message(
                        message,
                        nn_weight=weight_float,
                        memory_mode=memory_mode
                    )
                    processing_time = time.time() - start_time
                    
                    logger.info(f"Response received in {processing_time:.2f}s")
                    logger.info(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        # Get memory statistics
        try:
            stats = chat_interface.get_memory_stats()
            logger.info(f"Memory stats: {stats}")
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
    
    logger.info("Chat-Language Memory Integration test complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 