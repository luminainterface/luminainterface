#!/usr/bin/env python3
"""
Simple CLI Demo for LUMINA v7.5
This standalone version doesn't require any external dependencies
"""

import os
import json
import random
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class SimpleCentralNode:
    """Simplified Central Node with mock responses"""
    
    def __init__(self):
        logger.info("Initializing SimpleCentralNode")
        
        # State management
        self.active_conversations = {}
        self.conversation_history = {}
        
        # Neural state
        self.neural_state = {
            'temperature': 0.7,
            'top_p': 0.9,
            'llm_weight': 0.7,
            'neural_weight': 0.3,
            'resonance': 0.6,
            'coherence': 0.8,
            'engagement': 0.7,
            'complexity': 0.5
        }
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def start_conversation(self):
        """Start a new conversation"""
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_conversations[conversation_id] = {
            'start_time': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'message_count': 0
        }
        self.conversation_history[conversation_id] = []
        logger.info(f"Started conversation: {conversation_id}")
        return conversation_id
    
    def process_message(self, conversation_id, message):
        """Process a message in a conversation using mock responses"""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Update conversation state
        self.active_conversations[conversation_id]['last_activity'] = datetime.now().isoformat()
        self.active_conversations[conversation_id]['message_count'] += 1
        
        # Get current neural state
        neural_state = self.get_neural_state()
        
        # Generate a mock response based on the message
        response = self._generate_mock_response(message)
        
        # Update neural state with random fluctuations
        self._update_neural_state()
        
        # Store in conversation history
        self.conversation_history[conversation_id].append({
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'system_response': response,
            'neural_state': neural_state
        })
        
        return {
            'response': response,
            'conversation_id': conversation_id,
            'neural_state': neural_state
        }
    
    def _generate_mock_response(self, message):
        """Generate a mock response based on the user's message"""
        responses = [
            f"I understand your message about '{message[:20]}...' and I'm processing it with my neural architecture.",
            f"That's an interesting point about '{message[:20]}...'. My neural networks are processing this information.",
            f"I've analyzed your input regarding '{message[:20]}...'. My response is being generated with a neural weighting of {self.neural_state['neural_weight']:.2f}.",
            f"Thank you for sharing that information about '{message[:20]}...'. I'm using my neural architecture to formulate a response.",
            f"I'm considering multiple factors as I process your message about '{message[:20]}...'. My system is using both LLM and neural weights to generate this response."
        ]
        return random.choice(responses)
    
    def _update_neural_state(self):
        """Update neural state with small random fluctuations"""
        for key in self.neural_state:
            change = random.uniform(-0.05, 0.05)
            self.neural_state[key] = min(1.0, max(0.0, self.neural_state[key] + change))
    
    def end_conversation(self, conversation_id):
        """End a conversation"""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Save conversation history
        self._save_conversation_history(conversation_id)
        
        # Remove from active conversations
        del self.active_conversations[conversation_id]
        logger.info(f"Ended conversation: {conversation_id}")
    
    def get_conversation_history(self, conversation_id):
        """Get conversation history"""
        if conversation_id not in self.conversation_history:
            raise ValueError(f"Conversation {conversation_id} not found")
        return self.conversation_history[conversation_id]
    
    def get_neural_state(self):
        """Get current neural state"""
        return self.neural_state.copy()
    
    def save_state(self):
        """Save neural state and conversations"""
        try:
            # Save neural state
            with open("data/neural_state.json", "w") as f:
                json.dump(self.neural_state, f)
            
            # Save all conversations
            for conv_id in self.conversation_history:
                self._save_conversation_history(conv_id)
            
            logger.info("System state saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self):
        """Load neural state and conversations"""
        try:
            # Load neural state
            if os.path.exists("data/neural_state.json"):
                with open("data/neural_state.json", "r") as f:
                    self.neural_state = json.load(f)
            
            # Load conversations
            for filename in os.listdir("data"):
                if filename.startswith("conversation_") and filename.endswith(".json"):
                    conv_id = filename.replace("conversation_", "").replace(".json", "")
                    with open(os.path.join("data", filename), "r") as f:
                        self.conversation_history[conv_id] = json.load(f)
            
            logger.info("System state loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def _save_conversation_history(self, conversation_id):
        """Save a specific conversation history"""
        try:
            filename = f"data/conversation_{conversation_id}.json"
            with open(filename, "w") as f:
                json.dump(self.conversation_history[conversation_id], f)
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
            return False

def main():
    """Main function for the CLI interface"""
    try:
        # Initialize central node
        node = SimpleCentralNode()
        
        # Start conversation
        conv_id = node.start_conversation()
        
        print("\nWelcome to the LUMINA v7.5 Central Node CLI (Demo Version)!")
        print("This is a demonstration using simulated responses.")
        print("Type 'exit' to end the conversation")
        print("Type 'history' to view conversation history")
        print("Type 'state' to view current neural state")
        print("Type 'save' to save the current state")
        print("Type 'load' to load previous state")
        print("\nStart chatting:")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                node.end_conversation(conv_id)
                print("Conversation ended. Goodbye!")
                break
                
            elif user_input.lower() == 'history':
                history = node.get_conversation_history(conv_id)
                print("\nConversation History:")
                for i, msg in enumerate(history, 1):
                    print(f"\nExchange {i}:")
                    print(f"User: {msg['user_message']}")
                    print(f"System: {msg['system_response']}")
                    print(f"Neural State: Temperature={msg['neural_state']['temperature']:.2f}, Neural Weight={msg['neural_state']['neural_weight']:.2f}")
                    
            elif user_input.lower() == 'state':
                state = node.get_neural_state()
                print("\nCurrent Neural State:")
                print(f"Temperature: {state['temperature']:.2f}")
                print(f"Top P: {state['top_p']:.2f}")
                print(f"LLM Weight: {state['llm_weight']:.2f}")
                print(f"Neural Weight: {state['neural_weight']:.2f}")
                print(f"Resonance: {state['resonance']:.2f}")
                print(f"Coherence: {state['coherence']:.2f}")
                print(f"Engagement: {state['engagement']:.2f}")
                print(f"Complexity: {state['complexity']:.2f}")
                
            elif user_input.lower() == 'save':
                node.save_state()
                print("\nState saved successfully")
                
            elif user_input.lower() == 'load':
                node.load_state()
                print("\nState loaded successfully")
                
            else:
                # Process message
                response = node.process_message(conv_id, user_input)
                print(f"\nSystem: {response['response']}")
                print(f"Neural State: Temperature={response['neural_state']['temperature']:.2f}, Neural Weight={response['neural_state']['neural_weight']:.2f}")
                
    except Exception as e:
        logger.error(f"Error in CLI: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 