#!/usr/bin/env python3
"""
Language Memory System Demo

This script demonstrates the integration of the Language Memory system
with the Lumina GUI and other components of the neural network project.
"""

import logging
import json
import os
import sys
import time
from pathlib import Path
import random
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('language_memory_demo.log')
    ]
)
logger = logging.getLogger('language_memory_demo')

# Import memory components
try:
    from language_memory import LanguageMemory
    from memory_manager import memory_manager, MemoryManager
    from conversation_memory import ConversationMemory
    memory_system_available = True
    logger.info("Memory system components successfully imported")
except ImportError as e:
    logger.warning(f"Memory system import error: {e}")
    memory_system_available = False

# Try to import Lumina GUI components
try:
    from internal_language import InternalLanguageSystem
    from english_language_trainer import EnglishLanguageTrainer
    lumina_components_available = True
    logger.info("Lumina components successfully imported")
except ImportError as e:
    logger.warning(f"Lumina components import error: {e}")
    lumina_components_available = False

# Sample language patterns for demo
SAMPLE_QUESTIONS = [
    "What is the theory of neural networks?",
    "How does deep learning work?",
    "Can you explain the concept of backpropagation?",
    "What's the difference between supervised and unsupervised learning?",
    "How are transformers used in language processing?",
    "What are recurrent neural networks?",
    "Can you describe how convolutional networks process images?",
    "What is reinforcement learning?",
    "How do GANs generate realistic images?",
    "What are the ethical implications of AI systems?"
]

SAMPLE_RESPONSES = [
    "Neural networks are computational models inspired by the brain's structure. They consist of interconnected nodes (neurons) organized in layers that process information.",
    "Deep learning uses multi-layered neural networks to learn hierarchical representations of data. Each layer extracts progressively more abstract features.",
    "Backpropagation is an algorithm that calculates gradients in neural networks by applying the chain rule, propagating error backwards through the network to update weights.",
    "Supervised learning uses labeled data for training, while unsupervised learning finds patterns in unlabeled data without specific guidance.",
    "Transformers use self-attention mechanisms to weigh the importance of different words in a sequence, enabling parallel processing and better handling of long-range dependencies.",
    "Recurrent Neural Networks (RNNs) process sequential data by maintaining an internal memory state, making them suited for tasks like language modeling and time series analysis.",
    "Convolutional networks use filters that slide across the input image, detecting features like edges, textures, and shapes, which are combined in deeper layers to recognize complex objects.",
    "Reinforcement learning is a training approach where agents learn optimal behavior through trial and error, receiving rewards or penalties based on their actions in an environment.",
    "Generative Adversarial Networks (GANs) consist of two neural networks: a generator that creates content and a discriminator that evaluates it, competing to improve the realism of generated outputs.",
    "AI ethics encompasses issues like bias in algorithms, privacy concerns, job displacement, autonomous decision-making, and ensuring AI development benefits humanity as a whole."
]

SAMPLE_GLYPHS = [
    "circle", "square", "triangle", "spiral", "wave", 
    "grid", "star", "hexagon", "infinity", "network"
]

SAMPLE_EMOTIONS = [
    "curious", "focused", "inspired", "analytical", "thoughtful",
    "enthusiastic", "contemplative", "fascinated", "interested", "reflective"
]

class LanguageMemoryDemo:
    """Demo application for Language Memory system integration with Lumina GUI"""
    
    def __init__(self):
        """Initialize the demo application"""
        self.memory_enabled = memory_system_available
        self.lumina_enabled = lumina_components_available
        
        # Init memory components
        if self.memory_enabled:
            # Create a memory directory if it doesn't exist
            Path("data/memory").mkdir(parents=True, exist_ok=True)
            
            # Initialize memory manager if not already initialized
            self.memory_manager = memory_manager
            
            # Initialize language memory
            self.language_memory = LanguageMemory(
                component_name="language_memory_demo",
                memory_file="data/memory/language_memory_demo.jsonl"
            )
            
            # Initialize conversation memory for comparison
            self.conversation_memory = ConversationMemory(
                component_name="conversation_memory_demo",
                memory_file="data/memory/conversation_memory_demo.jsonl"
            )
            
            logger.info("Memory components initialized successfully")
        else:
            logger.warning("Running demo without memory components")
        
        # Init Lumina components
        if self.lumina_enabled:
            # Initialize the internal language system
            self.internal_language = InternalLanguageSystem()
            
            # Initialize the English language trainer
            self.english_trainer = EnglishLanguageTrainer()
            
            logger.info("Lumina components initialized successfully")
        else:
            logger.warning("Running demo without Lumina components")
    
    def generate_demo_conversation(self, count=5):
        """Generate sample conversations for the demo"""
        conversations = []
        
        for i in range(count):
            # Select random question and response
            question = random.choice(SAMPLE_QUESTIONS)
            response = random.choice(SAMPLE_RESPONSES)
            
            # Generate metadata
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "emotion": random.choice(SAMPLE_EMOTIONS),
                "glyph": random.choice(SAMPLE_GLYPHS),
                "demo_id": f"demo_{i}",
                "topics": ["ai", "neural networks", "machine learning"],
            }
            
            conversations.append({
                "user_input": question,
                "system_response": response,
                "metadata": metadata
            })
            
            # Add a small delay to simulate real-time conversation
            time.sleep(0.5)
        
        return conversations
    
    def store_demo_conversations(self, conversations):
        """Store demo conversations in memory systems"""
        if not self.memory_enabled:
            logger.warning("Memory system not available, skipping storage")
            return
        
        # Store in both memory systems for comparison
        language_memory_ids = []
        conversation_memory_ids = []
        
        for conv in conversations:
            user_input = conv["user_input"]
            system_response = conv["system_response"]
            metadata = conv["metadata"]
            
            # Store in language memory
            lang_memory_id = self.language_memory.store(
                user_input=user_input,
                system_response=system_response,
                metadata=metadata
            )
            language_memory_ids.append(lang_memory_id)
            
            # Store in conversation memory
            conv_memory_id = self.conversation_memory.store(
                user_input=user_input,
                system_response=system_response,
                metadata=metadata
            )
            conversation_memory_ids.append(conv_memory_id)
        
        logger.info(f"Stored {len(conversations)} conversations in memory systems")
        return language_memory_ids, conversation_memory_ids
    
    def demonstrate_language_memory_features(self):
        """Show the unique features of language memory compared to conversation memory"""
        if not self.memory_enabled:
            logger.warning("Memory system not available, skipping demo")
            return
        
        # 1. Demonstrate concept retrieval
        print("\n✨ LANGUAGE MEMORY DEMO - CONCEPT RETRIEVAL ✨")
        print("=" * 60)
        
        concepts = ["neural networks", "learning", "algorithm", "ethics"]
        for concept in concepts:
            memories = self.language_memory.retrieve_by_concept(concept)
            print(f"\nMemories related to concept '{concept}':")
            print("-" * 40)
            
            if memories:
                for i, memory in enumerate(memories):
                    print(f"Memory {i+1}:")
                    print(f"  User: {memory.get('user_input')}")
                    print(f"  System: {memory.get('system_response')[:80]}...")
            else:
                print("No memories found for this concept")
        
        # 2. Demonstrate pattern retrieval
        print("\n✨ LANGUAGE MEMORY DEMO - PATTERN RETRIEVAL ✨")
        print("=" * 60)
        
        patterns = ["question", "definition", "comparison"]
        for pattern in patterns:
            memories = self.language_memory.retrieve_by_pattern(pattern)
            print(f"\nMemories with pattern '{pattern}':")
            print("-" * 40)
            
            if memories:
                for i, memory in enumerate(memories):
                    print(f"Memory {i+1}:")
                    print(f"  User: {memory.get('user_input')}")
                    print(f"  System: {memory.get('system_response')[:80]}...")
            else:
                print("No memories found for this pattern")
        
        # 3. Demonstrate vocabulary statistics
        print("\n✨ LANGUAGE MEMORY DEMO - VOCABULARY STATISTICS ✨")
        print("=" * 60)
        
        vocab_stats = self.language_memory.get_vocabulary_stats()
        print(f"Total vocabulary size: {vocab_stats.get('vocabulary_size', 0)} words")
        
        print("\nTop words by frequency:")
        for word, count in vocab_stats.get('top_words', [])[:10]:
            print(f"  {word}: {count} occurrences")
        
        # 4. Generate comprehensive language report
        print("\n✨ LANGUAGE MEMORY DEMO - COMPREHENSIVE REPORT ✨")
        print("=" * 60)
        
        report = self.language_memory.generate_language_report()
        
        print(f"Total concepts: {report.get('concepts', {}).get('total_concepts', 0)}")
        print("\nTop concepts:")
        for concept, count in report.get('concepts', {}).get('top_concepts', [])[:5]:
            print(f"  {concept}: {count} memories")
        
        print("\nLanguage patterns detected:")
        for pattern, count in report.get('patterns', {}).items():
            print(f"  {pattern}: {count} memories")
        
        print("\nSymbolic associations:")
        for symbol, data in report.get('symbols', {}).items():
            print(f"  Symbol '{symbol}':")
            print(f"    Recent contexts: {data.get('contexts', ['None'])[-1]}")
            print(f"    Top associated words: {', '.join(word for word, _ in data.get('top_words', []))}")
    
    def demonstrate_lumina_integration(self):
        """Demonstrate integration with Lumina components"""
        if not self.lumina_enabled or not self.memory_enabled:
            logger.warning("Lumina components or memory system not available, skipping integration demo")
            return
        
        print("\n✨ LANGUAGE MEMORY INTEGRATION WITH LUMINA ✨")
        print("=" * 60)
        
        # 1. Store English Trainer data in language memory
        print("\nStoring English Trainer data in language memory...")
        
        # Generate sample training data
        for i in range(3):
            # Train vocabulary at random difficulty
            difficulty = random.randint(1, 5)
            result = self.english_trainer.train_vocabulary(difficulty=difficulty)
            
            # Store in language memory
            metadata = {
                "type": "vocabulary_training",
                "difficulty": difficulty,
                "performance": result,
                "glyph": "book"
            }
            
            self.language_memory.store(
                user_input=f"Train vocabulary at difficulty {difficulty}",
                system_response=f"Training complete with accuracy {result.get('accuracy', 0)}%",
                metadata=metadata
            )
        
        print("Training data stored in language memory")
        
        # 2. Process text with internal language system and store result
        print("\nProcessing text with Internal Language System...")
        
        # Sample text to process
        sample_text = "Neural networks can learn complex patterns from data."
        
        # Process with internal language system
        processed_result = self.internal_language.process_input(sample_text)
        
        # Store in language memory
        metadata = {
            "type": "text_processing",
            "modules_used": ["encoder", "decoder"],
            "glyph": "network"
        }
        
        self.language_memory.store(
            user_input=sample_text,
            system_response=processed_result,
            metadata=metadata
        )
        
        print(f"Processed text: '{sample_text}'")
        print(f"Result: '{processed_result}'")
        print("Processing result stored in language memory")
        
        # 3. Generate a Lumina GUI report with language memory stats
        print("\nGenerating Lumina GUI Language Report...")
        
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "language_memory_stats": self.language_memory.generate_language_report(),
            "gui_state": {
                "active_modules": ["language_memory", "internal_language", "english_trainer"],
                "current_view": "language_analysis"
            }
        }
        
        print(f"Report generated at {report['timestamp']}")
        print(f"Active modules: {report['gui_state']['active_modules']}")
        print(f"Memory count: {report['language_memory_stats'].get('memory_count', 0)}")
        
        # Save report to file
        with open("language_memory_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("Report saved to language_memory_report.json")

def main():
    """Main function to run the demo"""
    print("\n" + "=" * 80)
    print(" " * 30 + "LANGUAGE MEMORY DEMO")
    print("=" * 80 + "\n")
    
    # Initialize demo
    demo = LanguageMemoryDemo()
    
    # Generate sample conversations
    print("Generating sample conversations...")
    conversations = demo.generate_demo_conversation(count=8)
    print(f"Generated {len(conversations)} sample conversations")
    
    # Store conversations in memory
    print("\nStoring conversations in memory systems...")
    demo.store_demo_conversations(conversations)
    
    # Demonstrate language memory features
    demo.demonstrate_language_memory_features()
    
    # Demonstrate Lumina integration
    demo.demonstrate_lumina_integration()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "DEMO COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main() 