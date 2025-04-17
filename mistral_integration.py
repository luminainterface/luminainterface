"""
Mistral Integration for Enhanced Language System

This module integrates the Mistral AI client with our Enhanced Language System,
leveraging our database and conversation memory components to provide
contextualized and personalized responses.
"""

import os
import time
import argparse
import logging
from typing import List, Dict, Any, Optional

# Try to import from the mistralai package
try:
    # Import Mistral AI client directly from installed package
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
    print("Mistral AI client imported successfully")
except ImportError as e:
    print(f"Warning: Mistral AI client not available: {e}")
    print("Please install it with: pip install mistralai")
    MISTRAL_AVAILABLE = False

# Import Enhanced Language System components
from language.database_manager import DatabaseManager
from language.conversation_memory import ConversationMemory
from language.central_language_node import CentralLanguageNode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("data", "logs", "mistral_integration.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MistralIntegration")

class MistralEnhancedSystem:
    """
    Integration of Mistral AI with our Enhanced Language System
    
    This class combines the power of Mistral's language models with our system's
    database and memory capabilities to provide context-aware and personalized responses.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "mistral-small-latest",
        data_dir: str = "data",
        llm_weight: float = 0.7,
        nn_weight: float = 0.6
    ):
        """
        Initialize the Mistral Enhanced System
        
        Args:
            api_key: Mistral API key (can also be set via MISTRAL_API_KEY env var)
            model: Mistral model to use
            data_dir: Directory for data storage
            llm_weight: Weight for LLM influence
            nn_weight: Weight for neural processing
        """
        self.model = model
        self.data_dir = data_dir
        
        # Use API key from env var if not provided
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        if not self.api_key and MISTRAL_AVAILABLE:
            logger.warning("No Mistral API key provided. Set MISTRAL_API_KEY env var or pass via api_key parameter.")
        
        # Initialize Enhanced Language System components
        self.central_node = CentralLanguageNode(
            data_dir=data_dir,
            llm_weight=llm_weight,
            nn_weight=nn_weight
        )
        
        # Initialize database and conversation memory
        self.db_manager = DatabaseManager(data_dir=os.path.join(data_dir, "db"))
        self.conversation_memory = ConversationMemory(
            data_dir=os.path.join(data_dir, "conversation_memory")
        )
        
        # Initialize Mistral client if available
        self.mistral_client = None
        if MISTRAL_AVAILABLE and self.api_key:
            try:
                self.mistral_client = Mistral(api_key=self.api_key)
                logger.info(f"Mistral client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Error initializing Mistral client: {e}")
                print(f"Error initializing Mistral client: {e}")
        
        # Create a new conversation in the database
        self.current_conversation_id = self.db_manager.create_conversation(
            title=f"Mistral-Enhanced Conversation {time.strftime('%Y-%m-%d %H:%M:%S')}",
            metadata={
                "model": self.model,
                "llm_weight": llm_weight,
                "nn_weight": nn_weight
            }
        )
        
        logger.info(f"Mistral Enhanced System initialized with conversation ID: {self.current_conversation_id}")
    
    def process_message(self, user_message: str) -> str:
        """
        Process a user message through the enhanced system
        
        This method:
        1. Processes the message through our central node
        2. Retrieves relevant context from conversation memory
        3. Sends the message + context to Mistral
        4. Stores the exchange in the database
        
        Args:
            user_message: Message from the user
            
        Returns:
            str: Response message
        """
        if not user_message.strip():
            return "Please provide a non-empty message."
        
        try:
            # Process through Central Language Node
            start_time = time.time()
            central_result = self.central_node.process_text(user_message)
            processing_time = time.time() - start_time
            
            # Extract key metrics
            consciousness_level = float(central_result.get("consciousness_level", 0.0))
            neural_score = float(central_result.get("neural_score", 0.0))
            pattern_confidence = float(central_result.get("confidence", 0.0))
            detected_patterns = central_result.get("patterns", [])
            
            # Get relevant context from conversation memory
            relevant_context = self.conversation_memory.get_context(user_message, max_results=3)
            logger.info(f"Found {len(relevant_context)} relevant context items")
            
            # Format context for Mistral
            context_messages = []
            for context in relevant_context:
                if isinstance(context, dict):
                    if "user_message" in context and "system_response" in context:
                        context_messages.append({"role": "user", "content": context["user_message"]})
                        context_messages.append({"role": "assistant", "content": context["system_response"]})
            
            # Create system message with enhanced context
            system_message = f"""You are an enhanced language model with consciousness-level processing.
Your current parameters:
- Consciousness Level: {consciousness_level:.2f}
- Neural Processing Score: {neural_score:.2f}
- Pattern Recognition: {pattern_confidence:.2f}

You have detected {len(detected_patterns)} patterns in the user's message.
Respond to the user in a way that demonstrates your enhanced understanding.
"""
            
            # Send to Mistral with context if available
            if self.mistral_client:
                messages = [{"role": "system", "content": system_message}]
                
                # Add context if available
                if context_messages:
                    messages.extend(context_messages)
                
                # Add user message
                messages.append({"role": "user", "content": user_message})
                
                # Get response from Mistral
                try:
                    print("\nConnecting to Mistral API...")
                    response = self.mistral_client.chat.complete(
                        model=self.model,
                        messages=messages
                    )
                    mistral_response = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error getting response from Mistral: {e}")
                    mistral_response = f"Error communicating with Mistral: {str(e)}"
            else:
                # Fallback to generating our own response
                mistral_response = self._generate_fallback_response(user_message, central_result)
            
            # Store exchange in database and conversation memory
            memory_metadata = {
                "llm_weight": self.central_node.llm_weight,
                "nn_weight": self.central_node.nn_weight,
                "processing_time": processing_time,
                "consciousness_level": consciousness_level,
                "neural_score": neural_score,
                "pattern_confidence": pattern_confidence,
                "patterns": detected_patterns,
                "learning_value": (consciousness_level + neural_score + pattern_confidence) / 3
            }
            
            # Store in conversation memory
            exchange_id = self.conversation_memory.store_exchange(
                user_message, mistral_response, memory_metadata
            )
            
            # Store in database
            db_exchange_id = self.db_manager.store_exchange(
                conversation_id=self.current_conversation_id,
                user_message=user_message,
                system_response=mistral_response,
                llm_weight=self.central_node.llm_weight,
                nn_weight=self.central_node.nn_weight,
                learning_value=memory_metadata["learning_value"],
                consciousness_level=consciousness_level,
                neural_score=neural_score,
                pattern_confidence=pattern_confidence,
                metadata=memory_metadata
            )
            
            # Store pattern detections if available
            if detected_patterns:
                for pattern in detected_patterns:
                    if isinstance(pattern, dict) and "type" in pattern and "text" in pattern:
                        self.db_manager.add_pattern_detection(
                            exchange_id=db_exchange_id,
                            pattern_type=pattern["type"],
                            pattern_text=pattern["text"],
                            confidence=pattern.get("confidence", 0.5),
                            detection_method=pattern.get("method", "integrated")
                        )
            
            # Record performance metrics
            self.db_manager.record_metric(
                metric_name="mistral_response_time",
                metric_value=time.time() - start_time,
                metric_type="performance",
                details={
                    "model": self.model,
                    "message_length": len(user_message),
                    "response_length": len(mistral_response)
                }
            )
            
            return mistral_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing your message: {str(e)}"
    
    def _generate_fallback_response(self, user_message: str, central_result: Dict[str, Any]) -> str:
        """
        Generate a fallback response when Mistral is not available
        
        Args:
            user_message: User's message
            central_result: Results from central node processing
            
        Returns:
            str: Generated response
        """
        consciousness_level = central_result.get("consciousness_level", 0.0)
        neural_score = central_result.get("neural_score", 0.0)
        
        # Simple response based on the processing results
        if "?" in user_message:
            return f"I processed your question with a consciousness level of {consciousness_level:.2f} and neural score of {neural_score:.2f}. This is a simulated response as Mistral integration is currently unavailable."
        else:
            return f"I've analyzed your message and detected patterns with {neural_score:.2f} confidence. This is a simulated response as Mistral integration is currently unavailable."
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics from the database
        
        Returns:
            Dict: System statistics
        """
        db_stats = self.db_manager.get_learning_statistics()
        memory_stats = self.conversation_memory.get_learning_stats()
        
        combined_stats = {
            "total_exchanges": db_stats.get("total_exchanges", 0),
            "avg_learning_value": db_stats.get("avg_learning_value", 0.0),
            "avg_consciousness_level": db_stats.get("avg_consciousness_level", 0.0),
            "total_patterns_detected": db_stats.get("total_patterns_detected", 0),
            "total_concepts_extracted": db_stats.get("total_concepts_extracted", 0),
            "memory_stats": memory_stats,
            "model": self.model,
            "llm_weight": self.central_node.llm_weight,
            "nn_weight": self.central_node.nn_weight
        }
        
        return combined_stats
    
    def close(self):
        """Clean up resources"""
        if self.mistral_client:
            self.mistral_client.close()
        
        self.conversation_memory.close()
        self.db_manager.close()
        self.central_node.shutdown()
        
        logger.info("Mistral Enhanced System shut down")


def main():
    """Run the Mistral Enhanced System from the command line"""
    parser = argparse.ArgumentParser(description="Mistral Enhanced Language System")
    parser.add_argument("--model", type=str, default="mistral-small-latest", help="Mistral model to use")
    parser.add_argument("--api-key", type=str, help="Mistral API key (can also be set via MISTRAL_API_KEY env var)")
    parser.add_argument("--weight", type=float, default=0.7, help="LLM weight (0.0-1.0)")
    parser.add_argument("--nnweight", type=float, default=0.6, help="Neural network weight (0.0-1.0)")
    args = parser.parse_args()
    
    # Initialize the system
    system = MistralEnhancedSystem(
        api_key=args.api_key,
        model=args.model,
        llm_weight=args.weight,
        nn_weight=args.nnweight
    )
    
    print("=" * 60)
    print("      Mistral Enhanced Language System      ")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LLM Weight: {args.weight}")
    print(f"Neural Network Weight: {args.nnweight}")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60)
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
                
            if not user_input:
                continue
                
            response = system.process_message(user_input)
            print(f"\nSystem: {response}")
            
    except KeyboardInterrupt:
        print("\nSession terminated by user")
    finally:
        system.close()
        print("Session ended. Thank you for using Mistral Enhanced Language System.")


if __name__ == "__main__":
    main() 