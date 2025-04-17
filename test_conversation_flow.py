#!/usr/bin/env python3
"""
Test script for ConversationFlow 

This script demonstrates the ConversationFlow module's capabilities in a simulated conversation.
"""

import sys
import os
from pathlib import Path

# Ensure the src directory is in the path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import the ConversationFlow class
try:
    from src.v7_5.conversation_flow import ConversationFlow
except ImportError:
    try:
        from conversation_flow import ConversationFlow
    except ImportError:
        print("Could not import ConversationFlow module.")
        sys.exit(1)

def simulate_conversation():
    """Simulate a conversation to test the conversation flow system"""
    print("Initializing ConversationFlow...")
    flow = ConversationFlow(data_dir="data/test_conversation", context_window_size=3)
    
    # Define some sample exchanges
    exchanges = [
        ("What is machine learning?", 
         "Machine learning is a field of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data."),
        
        ("How does supervised learning work?", 
         "Supervised learning involves training a model on labeled data. The model learns to map inputs to outputs based on example input-output pairs."),
        
        ("What's the difference between classification and regression?", 
         "Classification predicts discrete categories (like spam/not spam), while regression predicts continuous values (like house prices)."),
        
        ("Can you explain neural networks?", 
         "Neural networks are computing systems inspired by the human brain. They consist of layers of interconnected nodes or 'neurons' that process information."),
        
        ("How does backpropagation work in neural networks?", 
         "Backpropagation is an algorithm that calculates gradients of the loss function with respect to the weights, allowing the network to learn by updating weights to minimize errors."),
        
        ("What's the weather like today?", 
         "I don't have access to real-time weather data. You would need to check a weather service for current conditions."),
        
        ("Tell me about natural language processing.", 
         "Natural language processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language."),
        
        ("How are transformers used in NLP?", 
         "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data like text. They've revolutionized NLP tasks like translation and text generation.")
    ]
    
    # Process exchanges and show the evolving context
    for i, (user_message, system_response) in enumerate(exchanges):
        print(f"\n=== Exchange {i+1} ===")
        print(f"User: {user_message}")
        print(f"System: {system_response}")
        
        # Process the exchange
        context = flow.process_exchange(user_message, system_response)
        
        # Show conversation context
        print("\nUpdated Conversation Context:")
        print(f"- Active topics: {', '.join(context['active_topics'][:5])}")
        print(f"- Exchange count: {context['exchange_count']}")
        print(f"- Current thought process: {context['current_thought_process']}")
        
        # Check if we're starting a new thought process
        if i > 0 and context['current_thought_process'] != previous_thought_process:
            print("➡️ Detected TOPIC CHANGE or NEW THOUGHT PROCESS")
        
        # Remember the current thought process for the next iteration
        previous_thought_process = context['current_thought_process']
    
    # Show final stats
    print("\n=== Conversation Flow Stats ===")
    stats = flow.get_stats()
    print(f"- Total exchanges: {stats['exchange_count']}")
    print(f"- Active topics: {stats['active_topics_count']}")
    print(f"- Thought processes: {stats['thought_process_count']}")
    
    # Save state
    flow.save_state()
    print("\nConversation state saved.")
    
    # Demonstrate topic search
    print("\n=== Searching for 'neural' topic ===")
    neural_exchanges = flow.get_topic_history("neural")
    for ex in neural_exchanges:
        print(f"- User: {ex['user_message'][:30]}...")
        print(f"  System: {ex['system_response'][:30]}...")
    
    # Demonstrate thought process retrieval
    thought_process_id = stats.get('current_thought_process')
    if thought_process_id:
        print(f"\n=== Current Thought Process ({thought_process_id}) ===")
        thought_exchanges = flow.get_thought_process(thought_process_id)
        for ex in thought_exchanges:
            print(f"- Exchange: {ex['user_message'][:30]}... -> {ex['system_response'][:30]}...")

if __name__ == "__main__":
    simulate_conversation() 