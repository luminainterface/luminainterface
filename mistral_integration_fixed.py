#!/usr/bin/env python3
"""
Fixed Mistral Integration for Enhanced Language System

This is a patched version that fixes the NeuralLinguisticProcessor initialization issue.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure path to ensure imports work correctly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MistralIntegration")

# Try to import from mistralai package
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
    logger.info("Mistral AI client imported successfully")
except ImportError as e:
    logger.warning(f"Warning: Mistral AI client not available: {e}")
    logger.warning("Please install it with: pip install mistralai")
    MISTRAL_AVAILABLE = False

# Import database and conversation memory components from the correct location
try:
    from src.language.database_manager import DatabaseManager
    from src.language.conversation_memory import ConversationMemory
except ImportError as e:
    logger.error(f"Error importing database components: {e}")
    
    # Try alternative import paths
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from language.database_manager import DatabaseManager
        from language.conversation_memory import ConversationMemory
        logger.info("Imported database components using alternate path")
    except ImportError as e:
        logger.error(f"Failed to import database components: {e}")
        DatabaseManager = None
        ConversationMemory = None

# This is our patched version of CentralLanguageNode initialization
class CentralLanguageNodeFixed:
    """
    Patched version of CentralLanguageNode that correctly initializes NeuralLinguisticProcessor
    """
    
    def __init__(self, data_dir: str = "data", llm_weight: float = 0.5, nn_weight: float = 0.5):
        """
        Initialize the Central Language Node with fixed NeuralLinguisticProcessor handling
        
        Args:
            data_dir: Root data directory
            llm_weight: Weight for LLM influence (0.0-1.0)
            nn_weight: Weight for neural network vs symbolic processing (0.0-1.0)
        """
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        
        # Component paths
        self.language_memory_dir = os.path.join(data_dir, "memory/language_memory")
        self.conscious_mirror_dir = os.path.join(data_dir, "v10")
        self.neural_processor_dir = os.path.join(data_dir, "neural_linguistic")
        self.recursive_patterns_dir = os.path.join(data_dir, "recursive_patterns")
        
        # Create necessary directories
        for directory in [
            self.language_memory_dir,
            self.conscious_mirror_dir,
            self.neural_processor_dir,
            self.recursive_patterns_dir
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.language_memory = None
        self.conscious_mirror = None
        self.neural_processor = None
        self.pattern_analyzer = None
        
        # Initialize components with proper handling of parameters
        self._initialize_components()
        
        # Synchronize weights after initialization
        self._synchronize_weights()
        
        logger.info(f"Central Language Node initialized with LLM weight: {llm_weight}, NN weight: {nn_weight}")
    
    def _initialize_components(self):
        """Initialize language components with proper parameter handling"""
        try:
            # Try different import paths
            try:
                # First try import from src.language
                from src.language.language_memory import LanguageMemory
                from src.language.conscious_mirror_language import ConsciousMirrorLanguage
                from src.language.neural_linguistic_processor import NeuralLinguisticProcessor
                from src.language.recursive_pattern_analyzer import RecursivePatternAnalyzer
                logger.info("Imported language components from src.language")
            except ImportError:
                # Try direct import from language
                from language.language_memory import LanguageMemory
                from language.conscious_mirror_language import ConsciousMirrorLanguage
                from language.neural_linguistic_processor import NeuralLinguisticProcessor
                from language.recursive_pattern_analyzer import RecursivePatternAnalyzer
                logger.info("Imported language components from language")

            # Initialize components with appropriate arguments
            self.language_memory = LanguageMemory(
                data_dir=self.language_memory_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            self.conscious_mirror = ConsciousMirrorLanguage(
                data_dir=self.conscious_mirror_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            # KEY FIX: Initialize NeuralLinguisticProcessor without data_dir
            # as the class doesn't accept this parameter
            self.neural_processor = NeuralLinguisticProcessor(
                language_memory=self.language_memory,
                conscious_mirror_language=self.conscious_mirror
            )
            # Set weights after initialization
            if hasattr(self.neural_processor, 'set_llm_weight'):
                self.neural_processor.set_llm_weight(self.llm_weight)
            if hasattr(self.neural_processor, 'set_nn_weight'):
                self.neural_processor.set_nn_weight(self.nn_weight)
            
            self.pattern_analyzer = RecursivePatternAnalyzer(
                data_dir=self.recursive_patterns_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            logger.info("All components initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import components: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _synchronize_weights(self):
        """Ensure all components have the same LLM and NN weights"""
        # Update component weights
        if self.language_memory:
            if hasattr(self.language_memory, 'set_llm_weight'):
                self.language_memory.set_llm_weight(self.llm_weight)
            if hasattr(self.language_memory, 'set_nn_weight'):
                self.language_memory.set_nn_weight(self.nn_weight)
        
        if self.conscious_mirror:
            if hasattr(self.conscious_mirror, 'set_llm_weight'):
                self.conscious_mirror.set_llm_weight(self.llm_weight)
            if hasattr(self.conscious_mirror, 'set_nn_weight'):
                self.conscious_mirror.set_nn_weight(self.nn_weight)
        
        if self.neural_processor:
            if hasattr(self.neural_processor, 'set_llm_weight'):
                self.neural_processor.set_llm_weight(self.llm_weight)
            if hasattr(self.neural_processor, 'set_nn_weight'):
                self.neural_processor.set_nn_weight(self.nn_weight)
        
        if self.pattern_analyzer:
            if hasattr(self.pattern_analyzer, 'set_llm_weight'):
                self.pattern_analyzer.set_llm_weight(self.llm_weight)
            if hasattr(self.pattern_analyzer, 'set_nn_weight'):
                self.pattern_analyzer.set_nn_weight(self.nn_weight)
        
        logger.info(f"Synchronized weights across all components: LLM={self.llm_weight}, NN={self.nn_weight}")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text through all language components"""
        result = {
            "text": text,
            "length": len(text),
            "timestamp": time.time(),
            "consciousness_level": 0.5,  # Default value
            "neural_score": 0.6,  # Default value
            "confidence": 0.7,  # Default value
            "patterns": []
        }
        
        # In a real implementation, we would process the text through each component
        # For now, we'll just return simulated results
        if self.conscious_mirror:
            try:
                conscious_result = self.conscious_mirror.process_text(text)
                result["consciousness_level"] = conscious_result.get("consciousness_level", 0.5)
            except Exception as e:
                logger.error(f"Error in consciousness processing: {e}")
        
        if self.neural_processor:
            try:
                neural_result = self.neural_processor.process_text(text)
                result["neural_score"] = neural_result.get("neural_linguistic_score", 0.6)
            except Exception as e:
                logger.error(f"Error in neural processing: {e}")
        
        if self.pattern_analyzer:
            try:
                pattern_result = self.pattern_analyzer.analyze_text(text)
                result["patterns"] = pattern_result.get("patterns", [])
                result["confidence"] = pattern_result.get("confidence", 0.7)
            except Exception as e:
                logger.error(f"Error in pattern analysis: {e}")
        
        return result
    
    def set_llm_weight(self, weight: float) -> None:
        """Set the LLM weight for all components"""
        self.llm_weight = max(0.0, min(1.0, weight))
        self._synchronize_weights()
    
    def set_nn_weight(self, weight: float) -> None:
        """Set the neural network weight for all components"""
        self.nn_weight = max(0.0, min(1.0, weight))
        self._synchronize_weights()
    
    def shutdown(self):
        """Clean up resources"""
        logger.info("Shutting down Central Language Node")
        # Add cleanup code here if needed

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
        
        # Initialize Enhanced Language System components with fixed CentralLanguageNode
        self.central_node = CentralLanguageNodeFixed(
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
                self.mistral_client = MistralClient(api_key=self.api_key)
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
                    # Convert messages to ChatMessage format
                    chat_messages = []
                    for msg in messages:
                        chat_messages.append(
                            ChatMessage(role=msg["role"], content=msg["content"])
                        )
                    
                    # Use the compatible API format
                    response = self.mistral_client.chat(
                        model=self.model,
                        messages=chat_messages
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
            pass  # Close Mistral client if needed
        
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