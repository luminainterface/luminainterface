#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Mistral Chat Application with Enhanced Language System Integration

This script launches the Mistral Chat application with the Enhanced Language
System integration, providing both the onsite memory system and language
processing capabilities.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mistral_language_app")

# Check for PySide6
try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    logger.error("PySide6 is not installed. Please install it using 'pip install pyside6'")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mistral Chat with Enhanced Language System")
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without actual language components"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for data storage"
    )
    
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.7,
        help="Initial LLM weight (0.0-1.0)"
    )
    
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.3,
        help="Initial neural network weight (0.0-1.0)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Mistral API key (will use MISTRAL_API_KEY env var if not provided)"
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable onsite memory integration"
    )
    
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Disable enhanced language system integration"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate weights
    if not 0.0 <= args.llm_weight <= 1.0:
        logger.error(f"Invalid LLM weight: {args.llm_weight}. Must be between 0.0 and 1.0.")
        return 1
    
    if not 0.0 <= args.nn_weight <= 1.0:
        logger.error(f"Invalid NN weight: {args.nn_weight}. Must be between 0.0 and 1.0.")
        return 1
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
    
    # Create configuration
    config = {
        "data_dir": args.data_dir,
        "llm_weight": args.llm_weight,
        "nn_weight": args.nn_weight,
        "mistral_api_key": api_key,
        "use_memory": not args.no_memory,
        "use_language": not args.no_language
    }
    
    # Import the required modules
    try:
        # Import based on configuration
        if not args.no_language:
            try:
                from src.language.mistral_language_bridge import get_mistral_language_bridge
                logger.info("Enhanced Language System integration enabled")
                has_language = True
            except ImportError:
                logger.warning("Enhanced Language System not available")
                has_language = False
        else:
            has_language = False
            
        # Import based on memory configuration
        if not args.no_memory:
            try:
                from src.v7.ui.onsite_memory_integration import OnsiteMemoryIntegration
                logger.info("Onsite Memory integration enabled")
                has_memory = True
            except ImportError:
                logger.warning("Onsite Memory not available")
                has_memory = False
        else:
            has_memory = False
            
        # Import the main window class
        from src.v7.ui.mistral_pyside_app import MistralChatWindow
        logger.info("Mistral PySide6 interface imported")
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.error("Make sure the project is properly installed")
        return 1
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Initialize language bridge if enabled
    language_bridge = None
    if has_language:
        try:
            language_bridge = get_mistral_language_bridge(
                data_dir=args.data_dir,
                llm_weight=args.llm_weight,
                nn_weight=args.nn_weight,
                config=config,
                mock_mode=args.mock
            )
            logger.info("Language bridge initialized")
        except Exception as e:
            logger.error(f"Error initializing language bridge: {e}")
            language_bridge = None
    
    # Initialize onsite memory if enabled
    memory_integration = None
    if has_memory:
        try:
            memory_integration = OnsiteMemoryIntegration(
                data_dir=os.path.join(args.data_dir, "onsite_memory"),
                memory_file="mistral_memory.json"
            )
            logger.info("Onsite memory initialized")
        except Exception as e:
            logger.error(f"Error initializing onsite memory: {e}")
            memory_integration = None
    
    # Create custom MistralChatWindow subclass to integrate language system
    class EnhancedMistralChatWindow(MistralChatWindow):
        def __init__(self, language_bridge=None, memory_integration=None):
            # Store language bridge
            self.language_bridge = language_bridge
            self.memory_integration = memory_integration
            
            # Call parent constructor
            super().__init__()
            
            # Replace memory integration if provided
            if self.memory_integration:
                self.memory_integration = memory_integration
            
            # Modify UI if language system is available
            if self.language_bridge:
                # Add info to status bar
                self.setStatusTip("Enhanced with Language System")
            
        def _send_message(self):
            """Override send message to use language bridge if available"""
            message = self.message_input.text().strip()
            if not message:
                return
            
            # Store message for potential knowledge extraction
            self.current_user_message = message
            
            # Add message to chat
            self._add_message_to_chat("user", message)
            
            # Check if we have language bridge
            if self.language_bridge:
                # Process with language bridge
                use_memory = getattr(self, "use_memory", True)
                self.language_bridge.process_message_async(message, enhance_with_language=True)
                
                # Clear input field
                self.message_input.clear()
                return
                
            # Use default method if no language bridge
            # Check if we should use memory context
            if hasattr(self, "use_memory") and self.use_memory and self.memory_integration:
                # Get relevant context from memory
                context = self.memory_integration.search_context_for_query(message)
                
                if context:
                    # Add context to the message
                    enhanced_message = f"{context}\n\nWith that in mind, please answer: {message}"
                    
                    # Add memory context note to chat
                    self._add_system_message("Using memory context to enhance your query...")
                    
                    # Send enhanced message
                    self.integration.process_message_async(enhanced_message)
                else:
                    # No relevant context found, send original message
                    self.integration.process_message_async(message)
            else:
                # Send original message without memory context
                self.integration.process_message_async(message)
            
            # Clear input field
            self.message_input.clear()
    
    # Create window
    window = EnhancedMistralChatWindow(
        language_bridge=language_bridge,
        memory_integration=memory_integration
    )
    window.show()
    
    # Run application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 