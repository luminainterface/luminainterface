#!/usr/bin/env python3
"""
Test script for the Onsite Memory System

This script demonstrates the core functionality of the Onsite Memory System
including conversation storage, knowledge management, and context retrieval.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_onsite_memory")

# Import the OnsiteMemory class - use the absolute import
try:
    from src.v7.onsite_memory import OnsiteMemory
    logger.info("Successfully imported OnsiteMemory")
except ImportError as e:
    logger.error(f"Failed to import OnsiteMemory: {e}")
    sys.exit(1)

def divider(title):
    """Print a section divider"""
    width = 80
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")

def test_basic_operations():
    """Test basic memory operations"""
    divider("BASIC MEMORY OPERATIONS")
    
    # Create memory system with a test file
    memory_file = "test_memory.json"
    memory = OnsiteMemory(
        data_dir="data/test_memory",
        memory_file=memory_file,
        auto_save=True
    )
    
    print(f"Memory system initialized with file: {memory.memory_file}")
    
    # Add test conversations
    print("\nAdding test conversations...")
    
    c1_id = memory.add_conversation(
        "Tell me about neural networks",
        "Neural networks are computational models inspired by the human brain that consist of layers of interconnected nodes or 'neurons'. They are a fundamental component of deep learning and can be trained to recognize patterns in data.",
        {"topic": "AI", "importance": 0.8}
    )
    
    c2_id = memory.add_conversation(
        "How do transformers work?",
        "Transformer models use a self-attention mechanism that helps them understand context and relationships between words in a sequence. Unlike RNNs, they process the entire input at once rather than sequentially, which allows for more efficient parallel processing.",
        {"topic": "AI", "importance": 0.9}
    )
    
    c3_id = memory.add_conversation(
        "What is the weather like today?",
        "I don't have access to real-time weather data. To get the current weather, you should check a weather service or app that provides up-to-date information for your location.",
        {"topic": "Weather", "importance": 0.3}
    )
    
    print(f"Added 3 conversations with IDs: {c1_id}, {c2_id}, {c3_id}")
    
    # Add test knowledge
    print("\nAdding test knowledge entries...")
    
    memory.add_knowledge(
        "Neural Networks",
        "Artificial neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. They are designed to recognize patterns and interpret data through a machine learning process.",
        "AI textbook"
    )
    
    memory.add_knowledge(
        "Transformer Architecture",
        "Transformer is a deep learning model architecture that adopts the mechanism of self-attention. Unlike RNNs, it doesn't process data sequentially but instead uses attention to weight the significance of different parts of the input data.",
        "Research paper"
    )
    
    memory.add_knowledge(
        "Python Programming",
        "Python is a high-level, interpreted programming language known for its readability and simplicity. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "Documentation"
    )
    
    print(f"Added 3 knowledge entries")
    
    # Save memory
    print("\nSaving memory...")
    memory.save_memory()
    
    # Get stats
    stats = memory.get_stats()
    print("\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up
    memory.stop()
    print("\nMemory system stopped")
    
    return memory.memory_file

def test_search_and_retrieval(memory_file):
    """Test search and retrieval capabilities"""
    divider("SEARCH AND RETRIEVAL")
    
    # Load the memory file
    memory = OnsiteMemory(
        data_dir="data/test_memory",
        memory_file=memory_file,
        auto_save=False
    )
    
    print(f"Loaded memory from {memory.memory_file}")
    
    # Get all knowledge topics
    topics = memory.get_all_topics()
    print("\nAll knowledge topics:")
    for topic in topics:
        print(f"  - {topic}")
    
    # Search for neural networks
    print("\nSearching for 'neural'...")
    results = memory.search_knowledge("neural")
    
    print(f"Found {len(results)} knowledge results:")
    for idx, result in enumerate(results):
        topic = result.get("topic", "")
        content_preview = result.get("entry", {}).get("content", "")[:100] + "..."
        relevance = result.get("relevance", 0)
        print(f"  {idx+1}. {topic} (relevance: {relevance:.1f})")
        print(f"     {content_preview}")
    
    # Search conversations
    print("\nSearching conversations for 'transformer'...")
    conv_results = memory.search_conversations("transformer")
    
    print(f"Found {len(conv_results)} conversation results:")
    for idx, conv in enumerate(conv_results):
        user_msg = conv.get("user_message", "")
        assistant_preview = conv.get("assistant_response", "")[:100] + "..."
        print(f"  {idx+1}. User: {user_msg}")
        print(f"     Assistant: {assistant_preview}")
    
    # Clean up
    memory.stop()
    print("\nMemory system stopped")

def test_context_enhancement():
    """Test context enhancement for queries"""
    divider("CONTEXT ENHANCEMENT")
    
    # Create memory with test data
    memory = OnsiteMemory(
        data_dir="data/test_memory",
        memory_file="context_test.json",
        auto_save=True
    )
    
    # Add knowledge entries
    memory.add_knowledge(
        "GPT Models",
        "GPT (Generative Pre-trained Transformer) models are a family of large language models developed by OpenAI. They use transformer architecture and are trained on vast amounts of text data to generate human-like text.",
        "AI documentation"
    )
    
    memory.add_knowledge(
        "Mistral AI",
        "Mistral AI is a company that develops advanced language models. Their models are known for efficient performance and high-quality outputs compared to their parameter count.",
        "Tech news"
    )
    
    # Add a conversation
    memory.add_conversation(
        "What's the difference between GPT-3 and GPT-4?",
        "GPT-4 is more advanced than GPT-3, with improved capabilities in reasoning, creativity, and handling nuanced instructions. It has a larger parameter count and was trained on more diverse data, resulting in better performance across various tasks.",
        {"topic": "AI Models"}
    )
    
    # Simulate context enhancement
    print("Simulating context enhancement for a query...")
    
    query = "Tell me about GPT models"
    print(f"\nUser query: '{query}'")
    
    # Search for relevant context
    knowledge_results = memory.search_knowledge(query, limit=2)
    
    if knowledge_results:
        print("\nRelevant knowledge context found:")
        context = "Based on my memory:\n\n"
        
        for idx, result in enumerate(knowledge_results):
            topic = result.get("topic", "")
            content = result.get("entry", {}).get("content", "")
            context += f"{idx+1}. {topic}: {content}\n\n"
            print(f"  - {topic}")
        
        print("\nEnhanced query would include this context:")
        print(f"\n{context}\n")
        print(f"With that in mind, please answer: {query}")
    else:
        print("No relevant context found")
    
    # Try a different query that should match the conversation
    query2 = "What do you know about the differences between GPT versions?"
    print(f"\nUser query: '{query2}'")
    
    # Search for relevant conversations
    conv_results = memory.search_conversations(query2, limit=1)
    
    if conv_results:
        print("\nRelevant conversation found:")
        context = "From previous conversations:\n\n"
        
        for idx, conv in enumerate(conv_results):
            user_msg = conv.get("user_message", "")
            assistant_msg = conv.get("assistant_response", "")
            
            # Truncate long messages
            if len(assistant_msg) > 200:
                assistant_msg = assistant_msg[:200] + "..."
            
            context += f"{idx+1}. You asked: {user_msg}\n"
            context += f"   I answered: {assistant_msg}\n\n"
            
            print(f"  - Previous Q: {user_msg}")
        
        print("\nEnhanced query would include this conversation context:")
        print(f"\n{context}\n")
        print(f"With that in mind, please answer: {query2}")
    else:
        print("No relevant conversation found")
    
    # Clean up
    memory.stop()
    print("\nMemory system stopped")

def test_preference_storage():
    """Test preference storage"""
    divider("PREFERENCE STORAGE")
    
    # Create memory
    memory = OnsiteMemory(
        data_dir="data/test_memory",
        memory_file="prefs_test.json",
        auto_save=True
    )
    
    # Store preferences
    print("Storing user preferences...")
    
    memory.set_preference("theme", "dark")
    memory.set_preference("font_size", 14)
    memory.set_preference("language", "en-US")
    memory.set_preference("notification_enabled", True)
    memory.set_preference("chat_history_length", 50)
    
    # Retrieve preferences
    print("\nRetrieving preferences:")
    print(f"  theme: {memory.get_preference('theme')}")
    print(f"  font_size: {memory.get_preference('font_size')}")
    print(f"  language: {memory.get_preference('language')}")
    print(f"  notifications: {memory.get_preference('notification_enabled')}")
    print(f"  chat_history_length: {memory.get_preference('chat_history_length')}")
    
    # Test default value
    print(f"\nNon-existent preference with default: {memory.get_preference('auto_save', True)}")
    
    # Save and clean up
    memory.save_memory()
    memory.stop()
    print("\nMemory system stopped")

def main():
    """Main test function"""
    print("ONSITE MEMORY SYSTEM TEST")
    print("========================\n")
    
    # Run tests
    memory_file = test_basic_operations()
    test_search_and_retrieval(memory_file)
    test_context_enhancement()
    test_preference_storage()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 