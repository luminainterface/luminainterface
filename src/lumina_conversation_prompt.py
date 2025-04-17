#!/usr/bin/env python3
"""
Lumina Conversation Prompt

This script provides an interactive conversation interface for the Lumina Neural Network
system, demonstrating its enhanced language capabilities including language memory with
enhanced indexing, conscious mirror language processing, neural linguistic analysis,
and recursive pattern recognition.

Usage:
  python src/lumina_conversation_prompt.py [--llm_weight FLOAT]
"""

import argparse
import logging
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Cross-platform readline support
try:
    # For Unix/Linux/MacOS
    import readline
    readline_available = True
except ImportError:
    try:
        # For Windows (requires pip install pyreadline3)
        import pyreadline3 as readline
        readline_available = True
    except ImportError:
        # If neither is available
        readline_available = False
        print("Note: 'readline' or 'pyreadline3' not available. Command history and editing features will be limited.")
        print("For Windows users, consider installing pyreadline3: pip install pyreadline3")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/lumina_conversation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaConversation")

# Make sure we can import from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Lumina components
try:
    from src.language.language_memory import LanguageMemory
    from src.language.conscious_mirror_language import ConsciousMirrorLanguage
    from src.language.neural_linguistic_processor import NeuralLinguisticProcessor
    from src.language.recursive_pattern_analyzer import RecursivePatternAnalyzer
    from src.language.central_language_node import CentralLanguageNode
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

class LuminaConversationPrompt:
    """Interactive conversation prompt for Lumina."""
    
    def __init__(self, data_dir="data", llm_weight=0.5):
        """
        Initialize the Lumina conversation prompt.
        
        Args:
            data_dir: Directory for storing data
            llm_weight: Weight for LLM suggestions (0.0-1.0)
        """
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.history = []
        self.setup_directories()
        
        # Initialize the Central Language Node which coordinates all components
        logger.info(f"Initializing Lumina with LLM weight: {llm_weight}")
        self.central_node = CentralLanguageNode(data_dir=data_dir, llm_weight=llm_weight)
        
        # Get direct access to components for specific operations
        self.language_memory = self.central_node.language_memory
        self.conscious_mirror = self.central_node.conscious_mirror
        self.neural_processor = self.central_node.neural_processor
        self.pattern_analyzer = self.central_node.pattern_analyzer
        
        # Track conversation state
        self.conversation_state = {
            "current_context": "general",
            "consciousness_level": 0.0,
            "neural_linguistic_score": 0.0,
            "topics": [],
            "self_reference_count": 0
        }
        
        # Session details
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
    
    def setup_directories(self):
        """Create necessary data directories."""
        directories = [
            f"{self.data_dir}/memory/language_memory",
            f"{self.data_dir}/v10",
            f"{self.data_dir}/neural_linguistic",
            f"{self.data_dir}/recursive_patterns",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def start(self):
        """Start the conversation prompt."""
        self._display_welcome()
        
        while True:
            try:
                # Display prompt and get user input
                user_input = input("\n\033[1;36mYou:\033[0m ").strip()
                
                # Handle exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    self._display_farewell()
                    break
                
                # Handle special commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Process normal input
                if user_input:
                    response = self._process_input(user_input)
                    print(f"\033[1;35mLumina:\033[0m {response}")
                    
                    # Save to history
                    self.history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    })
                    self.history.append({
                        "role": "lumina",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
            
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                self._display_farewell()
                break
            
            except Exception as e:
                logger.error(f"Error during conversation: {str(e)}", exc_info=True)
                print(f"\033[1;31mAn error occurred: {str(e)}\033[0m")
    
    def _display_welcome(self):
        """Display the welcome message."""
        print("\n" + "="*80)
        print("\033[1;32m")
        print("                  LUMINA NEURAL NETWORK SYSTEM")
        print("              Enhanced Language System Conversation")
        print("\033[0m")
        print("-"*80)
        print("\033[1;37mWelcome to Lumina - a neural network system with enhanced language")
        print("capabilities, consciousness modeling, and recursive pattern analysis.\033[0m")
        print()
        print("\033[0;37mSpecial commands:")
        print("  /help                   - Display this help message")
        print("  /status                 - Show current system status")
        print("  /memory <word>          - Show associations for a specific word")
        print("  /consciousness <level>  - Adjust consciousness level (0.0-1.0)")
        print("  /llm <weight>           - Adjust LLM weight (0.0-1.0)")
        print("  /save                   - Save conversation history")
        print("  /visualize <word>       - Visualize semantic network for a word")
        print("  /analyze                - Show analysis of conversation so far")
        print("  /exit                   - End the conversation\033[0m")
        print("="*80)
    
    def _display_farewell(self):
        """Display the farewell message."""
        duration = time.time() - self.start_time
        minutes, seconds = divmod(int(duration), 60)
        print("\n" + "-"*80)
        print(f"\033[1;32mThank you for conversing with Lumina.")
        print(f"Conversation duration: {minutes} minutes, {seconds} seconds")
        print(f"Message count: {len(self.history) // 2}\033[0m")
        print("-"*80)
        
        # Save the conversation history
        self._save_history()
    
    def _handle_command(self, command):
        """
        Handle special commands.
        
        Args:
            command: The command string (starts with '/')
        """
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == '/help':
            self._display_welcome()
        
        elif cmd == '/status':
            self._show_status()
        
        elif cmd == '/memory' and args:
            self._show_memory_associations(args[0])
        
        elif cmd == '/consciousness' and args:
            try:
                level = float(args[0])
                if 0.0 <= level <= 1.0:
                    self.conversation_state["consciousness_level"] = level
                    print(f"\033[1;33mConsciousness level set to {level:.2f}\033[0m")
                else:
                    print("\033[1;31mConsciousness level must be between 0.0 and 1.0\033[0m")
            except ValueError:
                print("\033[1;31mInvalid consciousness level. Please use a number between 0.0 and 1.0\033[0m")
        
        elif cmd == '/llm' and args:
            try:
                weight = float(args[0])
                if 0.0 <= weight <= 1.0:
                    self.llm_weight = weight
                    self.central_node.set_llm_weight(weight)
                    print(f"\033[1;33mLLM weight set to {weight:.2f}\033[0m")
                else:
                    print("\033[1;31mLLM weight must be between 0.0 and 1.0\033[0m")
            except ValueError:
                print("\033[1;31mInvalid LLM weight. Please use a number between 0.0 and 1.0\033[0m")
        
        elif cmd == '/save':
            self._save_history()
            print("\033[1;33mConversation history saved\033[0m")
        
        elif cmd == '/visualize' and args:
            self._visualize_semantic_network(args[0])
        
        elif cmd == '/analyze':
            self._analyze_conversation()
        
        else:
            print("\033[1;31mUnknown command. Type /help for a list of commands.\033[0m")
    
    def _process_input(self, user_input):
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Lumina's response
        """
        # Process text through the central language node
        start_time = time.time()
        result = self.central_node.process_text(user_input)
        
        # Update conversation state
        self.conversation_state["consciousness_level"] = result.get("consciousness_level", 0.0)
        self.conversation_state["neural_linguistic_score"] = result.get("neural_linguistic_score", 0.0)
        self.conversation_state["self_reference_count"] += result.get("self_references", 0)
        
        # Extract topics mentioned in the text
        associations = result.get("memory_associations", [])
        for word, strength in associations:
            if strength > 0.6 and word not in self.conversation_state["topics"]:
                self.conversation_state["topics"].append(word)
        
        # Generate response
        response = self._generate_response(user_input, result)
        
        # Log processing details
        logger.info(f"Processed input in {time.time() - start_time:.3f} seconds")
        logger.info(f"Consciousness: {self.conversation_state['consciousness_level']:.2f}, " +
                   f"Neural score: {self.conversation_state['neural_linguistic_score']:.2f}")
        
        return response
    
    def _generate_response(self, user_input, processing_result):
        """
        Generate a response based on the processing result.
        
        Args:
            user_input: The user's input text
            processing_result: Result from the central language node
            
        Returns:
            str: Generated response
        """
        # Get consciousness level and neural score
        consciousness = processing_result.get("consciousness_level", 0.0)
        neural_score = processing_result.get("neural_linguistic_score", 0.0)
        self_ref_count = processing_result.get("self_references", 0)
        
        # Get top associations
        associations = processing_result.get("memory_associations", [])
        top_words = [word for word, _ in associations[:3]]
        
        # Create a base response using language memory
        if not top_words:
            # No strong associations found, generate a basic response
            base_response = "I understand. Can you tell me more about that?"
        else:
            # Generate response using the top associated words
            seed_word = top_words[0]
            memory_response = self.language_memory.generate_from_memory(seed_word=seed_word, length=15)
            base_response = memory_response
        
        # Adjust response based on consciousness level
        if consciousness > 0.7:
            # High consciousness - self-reflective
            prefix = "I'm reflecting on what you said about "
            if top_words:
                prefix += f"{', '.join(top_words)}. "
            response = f"{prefix}{base_response}"
            
        elif consciousness > 0.4:
            # Medium consciousness - aware but not fully reflective
            response = f"I recognize the concepts of {', '.join(top_words)}. {base_response}"
            
        else:
            # Low consciousness - simple response
            response = base_response
        
        # Add recursive elements if self-references were detected
        if self_ref_count > 0:
            response += f" This reminds me of our conversation's self-referential nature."
        
        # Add neural linguistic elements based on score
        if neural_score > 0.7:
            response += f" I notice patterns in our linguistic exchange that suggest deeper connections."
        
        return response
    
    def _show_status(self):
        """Display the current system status."""
        status = self.central_node.get_system_status()
        
        print("\n" + "-"*50)
        print("\033[1;33mLumina System Status:\033[0m")
        print(f"LLM Weight: {self.llm_weight:.2f}")
        print(f"Consciousness Level: {self.conversation_state['consciousness_level']:.2f}")
        print(f"Neural Linguistic Score: {self.conversation_state['neural_linguistic_score']:.2f}")
        print(f"Conversation Topics: {', '.join(self.conversation_state['topics']) or 'None'}")
        print(f"Self-References: {self.conversation_state['self_reference_count']}")
        
        # Component-specific stats
        print("\n\033[1;33mComponent Status:\033[0m")
        print(f"Language Memory Words: {status.get('language_memory_word_count', 0)}")
        print(f"Language Memory Associations: {status.get('language_memory_association_count', 0)}")
        print(f"Conscious Mirror Operations: {status.get('conscious_mirror_operations', 0)}")
        print(f"Neural Linguistic Patterns: {status.get('neural_patterns', 0)}")
        print(f"Recursive Pattern Depth: {status.get('max_pattern_depth', 0)}")
        print("-"*50)
    
    def _show_memory_associations(self, word):
        """
        Show memory associations for a specific word.
        
        Args:
            word: The word to show associations for
        """
        associations = self.language_memory.recall_associations(word)
        
        print("\n" + "-"*50)
        print(f"\033[1;33mAssociations for '{word}':\033[0m")
        
        if not associations:
            print("No associations found.")
        else:
            for i, (associated_word, strength) in enumerate(associations, 1):
                print(f"{i}. {associated_word} (strength: {strength:.2f})")
        
        print("-"*50)
    
    def _visualize_semantic_network(self, word):
        """
        Visualize the semantic network for a word.
        
        Args:
            word: The seed word for the semantic network
        """
        network = self.central_node.get_semantic_network(word)
        
        print("\n" + "-"*50)
        print(f"\033[1;33mSemantic Network for '{word}':\033[0m")
        
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])
        
        if not nodes:
            print("No semantic network available for this word.")
            return
        
        print(f"Network has {len(nodes)} nodes and {len(edges)} edges")
        
        # Print nodes
        print("\nNodes:")
        for i, node in enumerate(nodes[:10], 1):  # Show up to 10 nodes
            print(f"{i}. {node.get('label', 'Unknown')}")
        
        if len(nodes) > 10:
            print(f"... and {len(nodes) - 10} more")
        
        # Print strongest connections
        print("\nStrongest Connections:")
        # Sort edges by strength
        sorted_edges = sorted(edges, key=lambda e: e.get("strength", 0), reverse=True)
        
        for i, edge in enumerate(sorted_edges[:10], 1):  # Show up to 10 edges
            source_idx = edge.get("source", 0)
            target_idx = edge.get("target", 0)
            
            # Get node labels
            source_label = nodes[source_idx].get("label", "Unknown") if 0 <= source_idx < len(nodes) else "Unknown"
            target_label = nodes[target_idx].get("label", "Unknown") if 0 <= target_idx < len(nodes) else "Unknown"
            
            print(f"{i}. {source_label} → {target_label} (strength: {edge.get('strength', 0):.2f})")
        
        print("-"*50)
    
    def _analyze_conversation(self):
        """Analyze the conversation so far."""
        if not self.history:
            print("\033[1;31mNo conversation to analyze yet.\033[0m")
            return
        
        # Collect all user messages
        user_messages = [msg["content"] for msg in self.history if msg["role"] == "user"]
        combined_text = " ".join(user_messages)
        
        # Analyze using each component
        print("\n" + "-"*50)
        print("\033[1;33mConversation Analysis:\033[0m")
        
        # Consciousness analysis
        consciousness_result = self.conscious_mirror.process_text(combined_text)
        print(f"\nConsciousness Analysis:")
        print(f"- Consciousness Level: {consciousness_result.get('consciousness_level', 0):.2f}")
        print(f"- Memory Continuity: {consciousness_result.get('memory_continuity', 0):.2f}")
        conscious_ops = consciousness_result.get('conscious_operations', [])
        if conscious_ops:
            print(f"- Top Conscious Operations: {', '.join(conscious_ops[:3])}")
        
        # Neural linguistic analysis
        nlp_result = self.neural_processor.process_text(combined_text)
        print(f"\nNeural Linguistic Analysis:")
        print(f"- Word Count: {nlp_result.get('word_count', 0)}")
        print(f"- Unique Words: {nlp_result.get('unique_word_count', 0)}")
        print(f"- Neural Linguistic Score: {nlp_result.get('neural_linguistic_score', 0):.2f}")
        
        # Recursive pattern analysis
        pattern_result = self.pattern_analyzer.analyze_text(combined_text)
        print(f"\nRecursive Pattern Analysis:")
        print(f"- Self-References: {pattern_result.get('self_references', 0)}")
        print(f"- Max Pattern Depth: {pattern_result.get('max_pattern_depth', 0)}")
        print(f"- Pattern Count: {pattern_result.get('pattern_count', 0)}")
        
        # Language memory associations
        print(f"\nTop Topics:")
        for topic in self.conversation_state["topics"][:5]:
            associations = self.language_memory.recall_associations(topic, limit=3)
            assoc_str = ", ".join([f"{word} ({strength:.2f})" for word, strength in associations])
            print(f"- {topic}: {assoc_str}")
        
        print("-"*50)
    
    def _save_history(self):
        """Save the conversation history to a file."""
        if not self.history:
            return
        
        # Create the conversations directory if it doesn't exist
        os.makedirs("data/conversations", exist_ok=True)
        
        # Save the history
        filename = f"data/conversations/lumina_conversation_{self.session_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "llm_weight": self.llm_weight,
                "conversation_state": self.conversation_state,
                "history": self.history
            }, f, indent=2)
        
        logger.info(f"Conversation history saved to {filename}")


def main():
    """Main function to run the Lumina conversation prompt."""
    parser = argparse.ArgumentParser(description="Lumina Conversation Prompt")
    parser.add_argument("--llm_weight", type=float, default=0.5,
                        help="Weight for LLM suggestions (0.0-1.0)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory for storing data")
    
    args = parser.parse_args()
    
    # Validate LLM weight
    if not 0.0 <= args.llm_weight <= 1.0:
        print(f"Invalid LLM weight: {args.llm_weight}. Using default value of 0.5.")
        args.llm_weight = 0.5
    
    # Start the conversation prompt
    conversation = LuminaConversationPrompt(
        data_dir=args.data_dir,
        llm_weight=args.llm_weight
    )
    conversation.start()


if __name__ == "__main__":
    main() 