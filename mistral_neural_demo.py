#!/usr/bin/env python3
"""
Mistral Neural Network Integration Demonstration

This script demonstrates the integration of the neural network module
with the Mistral weighted chat system, showing how the neural network
can enhance responses based on the weights.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mistral_neural_demo.log", mode='a')
    ]
)
logger = logging.getLogger("MistralNeuralDemo")

# Import our neural integration
try:
    from neural_integration import NeuralIntegration
    NEURAL_AVAILABLE = True
except ImportError:
    logger.warning("Neural integration module not available. Running in limited mode.")
    NEURAL_AVAILABLE = False

# Try to import Mistral AI client
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
    logger.info("Mistral AI client imported successfully")
except ImportError:
    logger.warning("Mistral AI client not available. Running in mock mode.")
    MISTRAL_AVAILABLE = False


class MistralNeuralDemo:
    """
    Demonstration of Mistral integration with Neural Network
    
    This class provides a simple demo of how to use the neural network
    module with the Mistral weighted chat system.
    """
    
    def __init__(
        self, 
        model="mistral-small",
        api_key=None,
        llm_weight=0.7,
        nn_weight=0.3,
        model_dir=None,
        mock_mode=False
    ):
        """
        Initialize the demo
        
        Args:
            model: Mistral model to use
            api_key: Mistral API key
            llm_weight: Weight for language model (0.0-1.0)
            nn_weight: Weight for neural network (0.0-1.0)
            model_dir: Directory for neural network models
            mock_mode: Whether to run in mock mode without Mistral API
        """
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        self.model_dir = model_dir or os.path.join("data", "neural_models")
        self.mock_mode = mock_mode or not MISTRAL_AVAILABLE or not self.api_key
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize neural integration
        if NEURAL_AVAILABLE:
            try:
                self.neural_integration = NeuralIntegration(
                    model_dir=self.model_dir,
                    llm_weight=self.llm_weight,
                    nn_weight=self.nn_weight
                )
                logger.info("Neural integration initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing neural integration: {e}")
                self.neural_integration = None
        else:
            logger.warning("Neural integration not available")
            self.neural_integration = None
        
        # Initialize Mistral client
        if not self.mock_mode and MISTRAL_AVAILABLE and self.api_key:
            try:
                self.client = MistralClient(api_key=self.api_key)
                logger.info(f"Mistral client initialized with model {self.model}")
            except Exception as e:
                logger.error(f"Error initializing Mistral client: {e}")
                self.client = None
                self.mock_mode = True
        else:
            logger.info("Running in mock mode without Mistral API")
            self.client = None
            self.mock_mode = True
        
        # Conversation history
        self.conversation_history = []
    
    def process_message(self, message, context=None):
        """
        Process a message using neural network and Mistral LLM
        
        Args:
            message: User message
            context: Optional context from previous conversations
            
        Returns:
            Dict with response and processing information
        """
        logger.info(f"Processing message: {message}")
        
        # Process with neural network
        neural_data = None
        if self.neural_integration:
            neural_data = self.neural_integration.process_message(message, context)
            logger.info(f"Neural processing complete. Score: {neural_data.get('neural_score', 0.0)}")
        
        # Get response from Mistral
        if not self.mock_mode and self.client:
            try:
                # Prepare messages for Mistral
                messages = []
                
                # Add system message with context and weights
                system_message = "You are an AI assistant that incorporates neural network analysis."
                if neural_data and neural_data.get("neural_enhanced", False):
                    system_message += f" Neural network weight: {self.nn_weight:.2f}, LLM weight: {self.llm_weight:.2f}."
                
                if context:
                    system_message += f"\n\nContext information: {context}"
                
                messages.append(ChatMessage(role="system", content=system_message))
                
                # Add user message
                messages.append(ChatMessage(role="user", content=message))
                
                # Get chat completion
                chat_response = self.client.chat(
                    model=self.model,
                    messages=messages
                )
                
                # Extract response
                response_text = chat_response.choices[0].message.content
                logger.info("Received response from Mistral API")
            except Exception as e:
                logger.error(f"Error getting Mistral response: {e}")
                response_text = f"Error: Failed to get response from Mistral API - {e}"
        else:
            # Mock response
            logger.info("Generating mock response")
            if "hello" in message.lower():
                response_text = "Hello! I'm the Neural-Enhanced Mistral Assistant. How can I help you today?"
            elif "how" in message.lower() or "?" in message:
                response_text = "That's an interesting question. In a real implementation, I would provide a detailed response using both Mistral's language model and neural network analysis to enhance understanding."
            else:
                response_text = f"I've received your message. This is a demonstration of the Neural-Enhanced Mistral Chat system, using LLM weight of {self.llm_weight:.2f} and neural network weight of {self.nn_weight:.2f}."
        
        # Enhance response with neural data if available
        enhanced_response = response_text
        if self.neural_integration and neural_data and neural_data.get("neural_enhanced", False):
            enhanced_response = self.neural_integration.enhance_response(
                message,
                response_text,
                neural_data
            )
            logger.info("Response enhanced with neural insights")
        
        # Save to conversation history
        self.conversation_history.append({
            "user": message,
            "assistant": enhanced_response,
            "neural_data": neural_data
        })
        
        # Prepare result
        result = {
            "original_message": message,
            "response": enhanced_response,
            "neural_enhanced": neural_data is not None and neural_data.get("neural_enhanced", False),
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight
        }
        
        # Add neural data if available
        if neural_data:
            result["neural_data"] = neural_data
        
        return result
    
    def adjust_weights(self, llm_weight=None, nn_weight=None):
        """
        Adjust the weights for LLM and neural network
        
        Args:
            llm_weight: New LLM weight (0.0-1.0)
            nn_weight: New neural network weight (0.0-1.0)
            
        Returns:
            Dict with the new weights
        """
        # Update weights
        if llm_weight is not None:
            self.llm_weight = max(0.0, min(1.0, llm_weight))
        
        if nn_weight is not None:
            self.nn_weight = max(0.0, min(1.0, nn_weight))
        
        # Update weights in neural integration
        if self.neural_integration:
            self.neural_integration.adjust_weights(self.llm_weight, self.nn_weight)
        
        logger.info(f"Weights adjusted - LLM: {self.llm_weight}, NN: {self.nn_weight}")
        
        return {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight
        }
    
    def get_status(self):
        """
        Get the status of the demo system
        
        Returns:
            Dict with status information
        """
        status = {
            "model": self.model,
            "mock_mode": self.mock_mode,
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "conversation_count": len(self.conversation_history),
            "mistral_available": MISTRAL_AVAILABLE and not self.mock_mode and self.client is not None,
            "neural_available": NEURAL_AVAILABLE and self.neural_integration is not None
        }
        
        # Add neural integration status if available
        if self.neural_integration:
            status["neural_integration"] = self.neural_integration.get_status()
        
        return status


def interactive_demo():
    """Run an interactive demo of the Mistral Neural integration"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Mistral Neural Integration Demo")
    parser.add_argument("--api-key", help="Mistral API key")
    parser.add_argument("--model", default="mistral-small", help="Mistral model to use")
    parser.add_argument("--llm-weight", type=float, default=0.7, help="Weight for LLM (0.0-1.0)")
    parser.add_argument("--nn-weight", type=float, default=0.3, help="Weight for Neural Network (0.0-1.0)")
    parser.add_argument("--model-dir", help="Directory for neural network models")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without Mistral API")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/neural_models", exist_ok=True)
    
    # Initialize demo
    demo = MistralNeuralDemo(
        model=args.model,
        api_key=args.api_key,
        llm_weight=args.llm_weight,
        nn_weight=args.nn_weight,
        model_dir=args.model_dir,
        mock_mode=args.mock
    )
    
    # Show initial status
    status = demo.get_status()
    print("\n===== Mistral Neural Integration Demo =====")
    print(f"Model: {status['model']} (Mock mode: {status['mock_mode']})")
    print(f"Weights - LLM: {status['llm_weight']:.1f}, Neural Network: {status['nn_weight']:.1f}")
    print(f"Neural available: {status['neural_available']}")
    print(f"Mistral available: {status['mistral_available']}")
    print("===========================================\n")
    
    # Interactive loop
    print("Enter your messages below. Type 'exit' to quit, 'status' to see status,")
    print("or 'weights X.X Y.Y' to adjust LLM and NN weights.\n")
    
    while True:
        try:
            user_input = input("> ")
            
            # Check for commands
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            elif user_input.lower() == "status":
                status = demo.get_status()
                print("\n----- System Status -----")
                print(f"Model: {status['model']} (Mock mode: {status['mock_mode']})")
                print(f"Weights - LLM: {status['llm_weight']:.1f}, Neural Network: {status['nn_weight']:.1f}")
                print(f"Conversations: {status['conversation_count']}")
                
                if 'neural_integration' in status:
                    neural_stats = status['neural_integration']['stats']
                    print(f"\nNeural stats:")
                    print(f"- Messages processed: {neural_stats['messages_processed']}")
                    print(f"- Neural calls: {neural_stats['neural_calls']}")
                    print(f"- Average neural score: {neural_stats['average_neural_score']:.2f}")
                    print(f"- Average processing time: {neural_stats['average_processing_time']:.2f}s")
                
                print("-----------------------\n")
                continue
            elif user_input.lower().startswith("weights "):
                try:
                    parts = user_input.split()
                    if len(parts) >= 3:
                        llm = float(parts[1])
                        nn = float(parts[2])
                        new_weights = demo.adjust_weights(llm, nn)
                        print(f"Weights updated: LLM={new_weights['llm_weight']:.1f}, NN={new_weights['nn_weight']:.1f}")
                    else:
                        print("Usage: weights <llm_weight> <nn_weight>")
                except ValueError:
                    print("Invalid weights format. Use: weights 0.7 0.3")
                continue
            
            # Get context from conversation history
            context = None
            if demo.conversation_history:
                last_conversation = demo.conversation_history[-1]
                context = f"Previous message: {last_conversation['user']}\nPrevious response: {last_conversation['assistant']}"
            
            # Process message
            result = demo.process_message(user_input, context)
            
            # Print response
            print(f"\n{result['response']}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Mistral Neural Integration Demo!")


if __name__ == "__main__":
    interactive_demo()