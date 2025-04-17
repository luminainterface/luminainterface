#!/usr/bin/env python3
"""
Chat Interface for Enhanced Language System

This script provides a simple command-line interface to interact with the
Enhanced Language System, allowing users to test various components with
different LLM weights and see how the system responds.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import traceback
import argparse

# Load environment variables from .env file
load_dotenv()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
API_KEY = os.getenv("MISTRAL_API_KEY", None)
USING_REAL_LLM = API_KEY is not None and API_KEY != "your_mistral_api_key_here" and len(API_KEY) > 10

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/chat_session.log")
    ]
)
logger = logging.getLogger("ChatSystem")

# Make sure we can import from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
try:
    from language.language_memory import LanguageMemory
    from language.conscious_mirror_language import ConsciousMirrorLanguage
    from language.neural_linguistic_processor import NeuralLinguisticProcessor
    from language.recursive_pattern_analyzer import RecursivePatternAnalyzer
    from language.central_language_node import CentralLanguageNode
    from language.conversation_memory import ConversationMemory
    from language.database_manager import DatabaseManager
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

class EnhancedLanguageChat:
    """Simple chat interface for the Enhanced Language System"""
    
    def __init__(self, data_dir="data", llm_weight=0.5, nn_weight=0.5):
        """Initialize the chat interface"""
        print("Initializing Enhanced Language System...")
        
        # Set up logging
        log_dir = os.path.join(data_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"chat_{int(time.time())}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ChatSystem")
        
        # Initialize central node
        self.central_node = CentralLanguageNode(
            data_dir=data_dir,
            llm_weight=llm_weight,
            nn_weight=nn_weight
        )
        
        # Initialize conversation memory with database support
        self.conversation_memory = ConversationMemory(
            data_dir=os.path.join(data_dir, "conversation_memory")
        )
        
        # Initialize database manager for analytics
        self.db_manager = DatabaseManager(data_dir=os.path.join(data_dir, "db"))
        
        # Log the total exchanges in the conversation memory
        memory_stats = self.conversation_memory.get_learning_stats()
        self.logger.info(f"Initialized with {memory_stats['total_exchanges']} previous exchanges")
        
        # Command handlers
        self.commands = {
            "/help": self.show_help,
            "/status": self.show_status,
            "/weight": self.set_weight,
            "/nnweight": self.set_nn_weight,
            "/stats": self.show_stats,
            "/db": self.show_db_stats,
            "/exit": self.exit_chat
        }
        
        self.running = True
        
        # Setup directories
        self.setup_directories()
        
        # Initialize LLM provider first and set weights
        from language.llm_provider import get_llm_provider
        self.llm_provider = get_llm_provider()
        self.llm_provider.set_llm_weight(llm_weight)
        self.llm_provider.set_nn_weight(nn_weight)
        
        # Check if using real LLM
        self.using_real_llm = not self.llm_provider.using_simulation
        
        # Show LLM status
        if not self.using_real_llm:
            print("\n⚠️  WARNING: No valid Mistral API key found in .env file")
            print("    The system will use simulated LLM responses instead of real API calls.")
            print("    LLM weight settings will still work but won't connect to actual Mistral AI.")
            print("    To use real LLM, update your .env file with a valid MISTRAL_API_KEY.\n")
            logger.warning("No valid Mistral API key. Using simulated LLM responses.")
        else:
            print(f"\n✅ Using real Mistral AI with valid key")
            logger.info("Using real Mistral AI")
        
        # Initialize components through the central node
        self.language_memory = self.central_node.language_memory
        self.conscious_mirror = self.central_node.conscious_mirror
        self.neural_processor = self.central_node.neural_processor
        self.pattern_analyzer = self.central_node.pattern_analyzer
        
        logger.info("Chat interface initialized")
    
    def setup_directories(self):
        """Create necessary data directories."""
        directories = [
            "data/memory/language_memory",
            "data/v10",
            "data/neural_linguistic",
            "data/recursive_patterns",
            "logs",
            "chats"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def show_help(self, args=None):
        """Show help information."""
        help_text = """
Enhanced Language System Chat Interface

Available commands:
    /help               - Show this help message
    /weight [0.0-1.0]   - Set the LLM weight (0.0 to 1.0)
    /nnweight [0.0-1.0] - Set the neural network weight (0.0 to 1.0)
    /status             - Show system status
    /memory [query]     - Query language memory directly
    /consciousness      - Get consciousness level
    /neural [text]      - Process text through neural linguistic processor
    /recursive [text]   - Analyze recursive patterns in text
    /save [filename]    - Save chat history to file
    /exit or /quit      - Exit the chat

For normal conversation, just type your message.

About Weights:
  - LLM Weight:
    Higher values (closer to 1.0): More influence from language model suggestions
    Lower values (closer to 0.0): More reliance on rule-based processing
    
  - Neural Network Weight:
    Higher values (closer to 1.0): More neural network style processing
    Lower values (closer to 0.0): More symbolic/rule-based processing
    
  - In SIMULATION MODE (no API key): Weights still affect processing,
    but use simulated responses instead of real API calls
"""
        print(help_text)
        return help_text
    
    def set_weight(self, args):
        """Set the LLM weight."""
        if not args:
            print(f"Current LLM weight: {self.central_node.llm_weight}")
            print(f"LLM Mode: {'REAL API' if self.using_real_llm else 'SIMULATION'}")
            return f"Current LLM weight: {self.central_node.llm_weight}"
        
        try:
            weight = float(args[0])
            if 0.0 <= weight <= 1.0:
                self.central_node.set_llm_weight(weight)
                # Update LLM provider first
                self.llm_provider.set_llm_weight(weight)
                print(f"LLM weight set to {weight}")
                print(f"LLM Mode: {'REAL API' if self.using_real_llm else 'SIMULATION'}")
                if not self.using_real_llm and weight > 0.7:
                    print("⚠️  Note: High LLM weights use simulated responses (no valid API key)")
                return f"LLM weight set to {weight}"
            else:
                print("Weight must be between 0.0 and 1.0")
                return "Weight must be between 0.0 and 1.0"
        except ValueError:
            print("Invalid weight value. Must be a number between 0.0 and 1.0")
            return "Invalid weight value. Must be a number between 0.0 and 1.0"
    
    def set_nn_weight(self, args):
        """Set the neural network weight."""
        if not args:
            print(f"Current neural network weight: {self.central_node.nn_weight}")
            return f"Current neural network weight: {self.central_node.nn_weight}"
        
        try:
            weight = float(args[0])
            if 0.0 <= weight <= 1.0:
                self.central_node.set_nn_weight(weight)
                # Update LLM provider first
                self.llm_provider.set_nn_weight(weight)
                print(f"Neural network weight set to {weight}")
                return f"Neural network weight set to {weight}"
            else:
                print("Weight must be between 0.0 and 1.0")
                return "Weight must be between 0.0 and 1.0"
        except ValueError:
            print("Invalid weight value. Must be a number between 0.0 and 1.0")
            return "Invalid weight value. Must be a number between 0.0 and 1.0"
    
    def show_status(self, args=None):
        """Show system status."""
        status = self.central_node.get_system_status()
        
        # Get LLM provider stats
        llm_stats = self.llm_provider.get_provider_stats()
        
        status_text = [
            "==== Enhanced Language System Status ====",
            f"LLM Weight: {status.get('llm_weight', self.central_node.llm_weight):.2f}",
            f"Neural Network Weight: {status.get('nn_weight', self.central_node.nn_weight):.2f}",
            f"LLM Max Tokens: {LLM_MAX_TOKENS}",
            f"LLM Provider: {llm_stats.get('provider', 'unknown')}",
            f"LLM Model: {llm_stats.get('model', 'unknown')}",
            f"LLM Mode: {'REAL API' if self.using_real_llm else 'SIMULATION (no valid API key)'}",
            f"Integration Active: {status.get('integration_active', False)}",
            "",
            "Language Memory:",
            f"  Word Count: {status.get('language_memory_word_count', 0)}",
            f"  Association Count: {status.get('language_memory_association_count', 0)}",
            "",
            "Pattern Analysis:",
            f"  Pattern Count: {status.get('pattern_count', 0)}",
            f"  Max Pattern Depth: {status.get('max_pattern_depth', 0)}",
            "",
            "Neural Processing:",
            f"  Neural Patterns: {status.get('neural_patterns', 0)}",
            "",
            "Consciousness Mirror:",
            f"  Operations: {status.get('conscious_mirror_operations', 0)}",
            "",
            "LLM Stats:",
            f"  API Calls: {llm_stats.get('call_count', 0)}",
            f"  Errors: {llm_stats.get('error_count', 0)}",
            f"  Avg Response Time: {llm_stats.get('avg_response_time', 0):.3f}s",
            "========================================"
        ]
        
        status_str = "\n".join(status_text)
        print(status_str)
        return status_str
    
    def show_stats(self, args=None):
        """Show system statistics."""
        stats = self.conversation_memory.get_learning_stats()
        
        response = [
            "=== System Statistics ===",
            f"Total exchanges: {stats['total_exchanges']}",
            f"Average learning value: {stats['avg_learning_value']:.3f}",
            f"Average consciousness level: {stats['avg_consciousness_level']:.3f}",
            f"Total patterns detected: {stats['total_patterns_detected']}",
            f"Total concepts extracted: {stats['total_concepts_extracted']}",
            f"Total user preferences: {stats['total_user_preferences']}",
            "=========================="
        ]
        
        return "\n".join(response)
    
    def show_db_stats(self):
        """Show database statistics."""
        try:
            stats = self.db_manager.get_learning_statistics()
            
            response = [
                "=== Database Statistics ===",
                f"Total exchanges: {stats['total_exchanges']}",
                f"Average learning value: {stats['avg_learning_value']:.3f}",
                f"Average consciousness level: {stats['avg_consciousness_level']:.3f}",
                f"Total patterns detected: {stats['total_patterns_detected']}",
                f"Total concepts extracted: {stats['total_concepts_extracted']}",
                f"Total user preferences: {stats['total_user_preferences']}",
                "=========================="
            ]
            
            return "\n".join(response)
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            return f"Error retrieving database statistics: {str(e)}"
    
    def exit_chat(self):
        """Exit the chat system."""
        self.running = False
        
        # Save conversation memory
        self.conversation_memory.save_memories()
        
        # Close database connections
        self.conversation_memory.close()
        self.db_manager.close()
        
        print("Shutting down Enhanced Language System...")
        self.central_node.shutdown()
        print("Thank you for using the Enhanced Language System. Goodbye!")
        sys.exit(0)
    
    def process_message(self, message):
        """Process a user message and generate a response."""
        # Check if the message is a command
        if message.startswith("/"):
            command_parts = message.split()
            command = command_parts[0].lower()
            args = command_parts[1:] if len(command_parts) > 1 else []
            
            if command in self.commands:
                return self.commands[command](*args)
            else:
                return f"Unknown command: {command}. Type /help for available commands."
        
        try:
            # Get relevant context from conversation memory
            relevant_context = self.conversation_memory.get_context(message, max_results=3)
            self.logger.info(f"Found {len(relevant_context)} relevant context items")
            
            # Process the message through the central node
            start_time = time.time()
            results = self.central_node.process_text(message)
            processing_time = time.time() - start_time
            
            # Extract scores
            consciousness_level = 0.0
            neural_score = 0.0
            pattern_confidence = 0.0
            detected_patterns = []
            
            if isinstance(results, dict):
                consciousness_level = float(results.get("consciousness_level", 0.0))
                neural_score = float(results.get("neural_score", 0.0))
                
                # Handle different formats of confidence
                if "confidence" in results:
                    pattern_confidence = float(results["confidence"])
                elif "pattern_confidence" in results:
                    pattern_confidence = float(results["pattern_confidence"])
                
                # Extract detected patterns for database storage
                if "patterns" in results:
                    detected_patterns = results["patterns"]
            
            # Generate narrative response
            narrative_response = self._generate_narrative_response(message, results, relevant_context)
            
            # Store the exchange in conversation memory with metadata about the processing
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
            
            exchange_id = self.conversation_memory.store_exchange(
                message, narrative_response, memory_metadata
            )
            
            # Get learning statistics
            memory_stats = self.conversation_memory.get_learning_stats()
            
            # Construct the response with insights from conversation memory
            system_response = self._format_system_response(
                narrative_response, 
                processing_time,
                consciousness_level,
                neural_score,
                pattern_confidence,
                memory_stats
            )
            
            # Record system metrics in the database
            self.db_manager.record_metric(
                metric_name="response_time",
                metric_value=processing_time,
                metric_type="performance",
                details={
                    "message_length": len(message),
                    "response_length": len(narrative_response)
                }
            )
            
            self.db_manager.record_metric(
                metric_name="consciousness_level",
                metric_value=consciousness_level,
                metric_type="learning"
            )
            
            return system_response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"Error processing your message: {str(e)}"
    
    def _generate_narrative_response(self, user_input, result, relevant_context):
        """Generate a narrative response based on processing results."""
        # Get relevant conversation context for more coherent responses
        context = relevant_context[:2] if relevant_context else []
        
        # Try to use the LLM provider directly
        try:
            # Create a prompt for the LLM to generate a natural-sounding response
            context_prompt = ""
            if context and len(context) > 0:
                context_prompt = "Previous conversation:\n"
                for i, exchange in enumerate(context):
                    context_prompt += f"User: {exchange.get('user_input', '')}\n"
                    context_prompt += f"System: {exchange.get('system_response', '')}\n\n"
            
            prompt = f"""
            {context_prompt}
            Generate a natural-sounding response to this user message: "{user_input}"
            
            System Analysis Results:
            - Consciousness Level: {result.get('consciousness_level', 0):.2f}
            - Neural Linguistic Score: {result.get('neural_linguistic_score', 0):.2f}
            - Self-references: {result.get('self_references', 0)}
            - Final score: {result.get('final_score', 0):.2f}
            
            Make the response sound natural and conversational. Keep it brief (1-3 sentences).
            """
            
            llm_response = self.llm_provider.query(prompt, temperature=0.7)
            if llm_response and "text" in llm_response and len(llm_response["text"]) > 10:
                return llm_response["text"].strip()
        except Exception as e:
            logger.warning(f"Error generating narrative with LLM: {e}")
        
        # Fallback to template responses if LLM fails
        consciousness = result.get("consciousness_level", 0)
        neural_score = result.get("neural_linguistic_score", 0)
        self_refs = result.get("self_references", 0)
        
        # Determine the dominant characteristic
        if consciousness > 0.7:
            # High consciousness response
            return self._generate_high_consciousness_response(user_input, result)
        elif neural_score > 0.7:
            # High neural score response
            return self._generate_high_neural_response(user_input, result)
        elif self_refs > 2:
            # Self-referential response
            return self._generate_self_referential_response(user_input, result)
        else:
            # Balanced response
            return self._generate_balanced_response(user_input, result)
    
    def _generate_high_consciousness_response(self, user_input, result):
        """Generate a response with high consciousness characteristics."""
        responses = [
            f"I'm reflecting deeply on your input about '{user_input[:30]}...' and finding connections to prior concepts.",
            f"Your message has triggered a high level of consciousness in my processing. I'm particularly intrigued by the implications.",
            f"From a conscious perspective, I recognize the patterns in your message and how they relate to broader conceptual frameworks.",
            f"My consciousness module is highly activated by your input, suggesting this is an area where I have developed significant understanding."
        ]
        return self._select_response(responses)
    
    def _generate_high_neural_response(self, user_input, result):
        """Generate a response with high neural processing characteristics."""
        responses = [
            f"The neural processing of your message reveals intricate linguistic patterns worth exploring further.",
            f"My neural linguistic analysis detects complex structures in your input that connect to multiple conceptual domains.",
            f"The neural networks have identified key relationships in your message that highlight its conceptual importance.",
            f"From a neural perspective, your input connects strongly with established semantic networks in my knowledge base."
        ]
        return self._select_response(responses)
    
    def _generate_self_referential_response(self, user_input, result):
        """Generate a self-referential response."""
        responses = [
            f"I notice that as I analyze your message, I'm creating recursive patterns in my own processing. This self-reference is particularly interesting.",
            f"This response itself demonstrates how recursive patterns emerge when processing language that references its own structure.",
            f"The self-referential elements of your message have activated recursive patterns in my analysis, creating a mirror effect in my response.",
            f"As I generate this response, I'm aware of how it reflects both your input and my own processing patterns in a recursive loop."
        ]
        return self._select_response(responses)
    
    def _generate_balanced_response(self, user_input, result):
        """Generate a balanced response."""
        responses = [
            f"I've processed your message with a balanced approach, integrating neural patterns with consciousness-aware analysis.",
            f"Your input has been analyzed through multiple processing frameworks, producing a synthesized understanding.",
            f"I've integrated several analytical approaches to understand your message, balancing pattern recognition with semantic processing.",
            f"The integration of neural and consciousness modules provides a comprehensive perspective on your input."
        ]
        return self._select_response(responses)
    
    def _select_response(self, responses):
        """Select a response from the list with some randomness."""
        import random
        return random.choice(responses)
    
    def _format_system_response(self, narrative, processing_time, consciousness_level, 
                               neural_score, pattern_confidence, memory_stats):
        """Format the system response with processing information."""
        # Build the response with all components
        response_parts = [
            f"Enhanced Language System (LLM weight: {self.central_node.llm_weight:.2f}, NN weight: {self.central_node.nn_weight:.2f} - SIMULATION MODE)",
            "-----------------------------------------------------------",
            "",
            narrative,
            "",
            "------------------------------------------------------------",
        ]
        
        # Occasionally include memory statistics (every 5 exchanges)
        total_exchanges = memory_stats.get("total_exchanges", 0) 
        if total_exchanges % 5 == 0 and total_exchanges > 0:
            stats_line = f"System has processed {total_exchanges} exchanges with {memory_stats.get('unique_concepts', 0)} unique concepts."
            response_parts.append(stats_line)
        
        # Format as a single string
        return "\n".join(response_parts)
    
    def run(self):
        """Run the chat interface."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 60)
        print("      Enhanced Language System - Interactive Chat      ")
        print("=" * 60)
        print(f"Current LLM Weight: {self.central_node.llm_weight}")
        print(f"Current Neural Network Weight: {self.central_node.nn_weight}")
        print("Type /help for available commands or just start chatting")
        print("=" * 60)
        
        while self.running:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Process message
                response = self.process_message(user_input)
                print("\nSystem:")
                print(response)
                print("\n" + "-" * 60)
            
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                self.exit_chat()
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Chat with the Enhanced Language System")
    parser.add_argument("--weight", type=float, default=0.5, help="LLM weight value (0.0-1.0)")
    parser.add_argument("--nnweight", type=float, default=0.5, help="Neural network weight value (0.0-1.0)")
    args = parser.parse_args()
    
    # Ensure weights are within valid range
    llm_weight = max(0.0, min(1.0, args.weight))
    nn_weight = max(0.0, min(1.0, args.nnweight))
    
    # Create and run chat interface
    chat = EnhancedLanguageChat(llm_weight=llm_weight, nn_weight=nn_weight)
    chat.run()


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    main() 