#!/usr/bin/env python3
"""
LUMINA v7.5 Frontend
Integrates with Mistral AI for enhanced conversational capabilities
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Mistral integration
from src.api.mistral_integration_fixed import MistralIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"lumina_v7.5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("LUMINA")

class LuminaFrontend:
    def __init__(self):
        """Initialize LUMINA frontend with Mistral integration"""
        # Get parameters from environment or use defaults
        self.model = os.getenv("LLM_MODEL", "mistral-medium")
        self.llm_weight = float(os.getenv("LLM_WEIGHT", "0.7"))
        self.nn_weight = float(os.getenv("NN_WEIGHT", "0.3"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.9"))
        
        # Initialize Mistral integration
        self.mistral = MistralIntegration(
            model=self.model,
            llm_weight=self.llm_weight,
            nn_weight=self.nn_weight,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        # Initialize conversation history
        self.conversation = [
            {"role": "system", "content": "You are LUMINA v7.5, an advanced AI assistant with integrated neural processing capabilities."}
        ]
        
        # Create conversations directory if it doesn't exist
        os.makedirs("data/conversations", exist_ok=True)
        
    def save_conversation(self):
        """Save the current conversation to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("data/conversations", f"conversation_{timestamp}.txt")
        
        with open(filename, "w", encoding="utf-8") as f:
            for message in self.conversation:
                f.write(f"{message['role'].upper()}: {message['content']}\n\n")
                
        logger.info(f"Conversation saved to {filename}")
        
    def run(self):
        """Run the interactive chat session"""
        print("\nLUMINA v7.5 Initialized")
        print("======================")
        print(f"Model: {self.model}")
        print(f"LLM Weight: {self.llm_weight}")
        print(f"Neural Weight: {self.nn_weight}")
        print("Type 'exit' to end the session, 'save' to save the conversation")
        print("===============================================")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for commands
                if user_input.lower() == "exit":
                    print("\nEnding session...")
                    self.save_conversation()
                    break
                elif user_input.lower() == "save":
                    self.save_conversation()
                    print("Conversation saved!")
                    continue
                elif not user_input:
                    continue
                
                # Add user message to conversation
                self.conversation.append({"role": "user", "content": user_input})
                
                # Get response from Mistral integration
                response = self.mistral.process_message(self.conversation)
                
                # Add assistant response to conversation
                self.conversation.append({"role": "assistant", "content": response})
                
                # Print response
                print("\nLUMINA:", response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving conversation...")
                self.save_conversation()
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\nError: {e}")
                print("Please try again or type 'exit' to end the session")

def main():
    """Main entry point for LUMINA frontend"""
    try:
        frontend = LuminaFrontend()
        frontend.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 