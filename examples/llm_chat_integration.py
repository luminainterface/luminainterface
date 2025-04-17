#!/usr/bin/env python3
"""
LLM Chat Integration Example

This script demonstrates how to integrate the Language Memory System with an LLM chat interface.
It simulates a chat session where memory is used to enhance LLM responses.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import memory API
from src.memory_api import MemoryAPI

# Simulation of LLM response generation
# In a real implementation, this would call an actual LLM API
def simulate_llm_response(prompt, system_message="You are a helpful assistant."):
    """
    Simulate an LLM response for demonstration purposes.
    
    In a real implementation, this would call an API like OpenAI, Anthropic, etc.
    
    Args:
        prompt: The prompt to send to the LLM
        system_message: Optional system message for the LLM
        
    Returns:
        Simulated LLM response
    """
    print(f"\n[SIMULATED LLM CALL]")
    print(f"System: {system_message}")
    print(f"Prompt: {prompt}")
    
    # Simple response generation based on keywords in the prompt
    # In a real implementation, this would be the actual LLM response
    
    if "neural networks" in prompt.lower():
        return ("Neural networks learn patterns through a process called training. "
                "They adjust their weights based on examples and feedback, gradually "
                "improving their ability to recognize and generalize from patterns in data.")
    
    elif "memory" in prompt.lower():
        return ("Memory systems allow for the storage and retrieval of information over time. "
                "In AI systems, memory can be implemented through various mechanisms like "
                "conversation histories, knowledge bases, or more sophisticated memory modules "
                "that can synthesize information across interactions.")
    
    elif "language" in prompt.lower():
        return ("Language processing in AI systems involves understanding, generating, "
                "and manipulating human language. This encompasses natural language processing, "
                "understanding semantics, syntax, and pragmatics of communication.")
    
    else:
        return ("I don't have specific information about that topic in my memory systems. "
                "Could you ask about neural networks, memory systems, or language processing?")


class MemoryChatInterface:
    """
    Chat interface that uses the Memory API to enhance LLM interactions.
    
    This class demonstrates how to integrate the memory system with a chat interface,
    including:
    - Storing conversation history
    - Retrieving relevant memories
    - Enhancing LLM prompts with memory context
    - Synthesizing information across conversations
    """
    
    def __init__(self):
        """Initialize the memory-enhanced chat interface"""
        # Initialize the Memory API
        self.memory_api = MemoryAPI()
        
        # Chat session information
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []
        
        # Memory enhancement settings
        self.memory_mode = "combined"  # Options: contextual, synthesized, combined
        self.auto_synthesize = True   # Automatically synthesize topics
        
        print("Memory-Enhanced Chat Interface initialized")
        print(f"Session ID: {self.session_id}")
    
    def process_user_message(self, user_message):
        """
        Process a user message through the memory-enhanced chat system.
        
        Args:
            user_message: The message from the user
            
        Returns:
            The assistant's response
        """
        # 1. Store the user message in memory
        store_result = self.memory_api.store_conversation(
            message=user_message,
            metadata={
                "session_id": self.session_id,
                "role": "user",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # 2. Enhance the message with memory
        enhanced = self.memory_api.enhance_message_with_memory(
            message=user_message,
            enhance_mode=self.memory_mode
        )
        
        # 3. Extract topics for potential synthesis
        topics_to_synthesize = []
        if self.auto_synthesize and enhanced["status"] == "success":
            topics_to_synthesize = enhanced.get("mode_contexts", {}).get("topics", [])
        
        # 4. Create the LLM prompt with enhanced context
        memory_context = enhanced.get("enhanced_context", "")
        
        llm_prompt = f"User: {user_message}\n\n"
        
        if memory_context:
            llm_prompt += f"Relevant memory context:\n{memory_context}\n\n"
        
        llm_prompt += "Please respond to the user based on both their current message and any relevant context from memory."
        
        # 5. Get response from LLM
        system_message = "You are an AI assistant with memory capabilities. Use the provided memory context to give informed and consistent responses."
        llm_response = simulate_llm_response(llm_prompt, system_message)
        
        # 6. Store the assistant's response in memory
        response_store_result = self.memory_api.store_conversation(
            message=llm_response,
            metadata={
                "session_id": self.session_id,
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "in_response_to": store_result.get("memory_id")
            }
        )
        
        # 7. Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "memory_id": store_result.get("memory_id")
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": llm_response,
            "memory_id": response_store_result.get("memory_id")
        })
        
        # 8. Perform synthesis for extracted topics in the background
        if self.auto_synthesize and topics_to_synthesize:
            for topic in topics_to_synthesize:
                print(f"\nSynthesizing topic: {topic}")
                synthesis = self.memory_api.synthesize_topic(topic)
                if synthesis["status"] == "success" and "synthesis_results" in synthesis:
                    memory = synthesis["synthesis_results"]["synthesized_memory"]
                    print(f"Synthesis ID: {memory.get('id')}")
                    print(f"Core understanding: {memory.get('core_understanding')}")
        
        return llm_response
    
    def get_conversation_history(self):
        """
        Get the conversation history for this session.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history
    
    def get_memory_summary(self):
        """
        Get a summary of the memory system state.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = self.memory_api.get_memory_stats()
        
        memory_summary = {
            "conversation_turns": len(self.conversation_history) // 2,
            "memory_mode": self.memory_mode,
            "auto_synthesize": self.auto_synthesize,
            "synthesis_count": stats.get("synthesis_stats", {}).get("synthesis_count", 0),
            "topics_synthesized": stats.get("synthesis_stats", {}).get("topics_synthesized", []),
            "memory_count": stats.get("language_memory_stats", {}).get("memory_count", 0)
        }
        
        return memory_summary


def main():
    """Run a demonstration of the memory-enhanced chat interface"""
    print("=" * 80)
    print("MEMORY-ENHANCED LLM CHAT DEMO")
    print("=" * 80)
    print("\nThis demo shows how the Language Memory System can enhance LLM chat by")
    print("providing relevant memory context and synthesizing information across conversations.")
    print("\nNote: This demo simulates LLM responses. In a real implementation,")
    print("you would connect to an actual LLM API like OpenAI, Anthropic, etc.")
    
    # Initialize the chat interface
    chat = MemoryChatInterface()
    
    # Sample conversation flow
    sample_messages = [
        "Tell me about neural networks and how they learn patterns.",
        "How is memory implemented in AI systems?",
        "Can you remind me what we discussed about neural networks earlier?",
        "How do memory and neural networks work together in modern AI?",
        "What are the main topics we've discussed in our conversation so far?"
    ]
    
    # Process each message
    for i, message in enumerate(sample_messages):
        print("\n" + "=" * 80)
        print(f"TURN {i+1}")
        print("=" * 80)
        
        print(f"\nUser: {message}")
        
        # Small delay for better readability
        time.sleep(1)
        
        # Process the message
        response = chat.process_user_message(message)
        
        print(f"\nAssistant: {response}")
        
        # Small delay between turns
        time.sleep(1.5)
    
    # Print final memory summary
    print("\n" + "=" * 80)
    print("MEMORY SYSTEM SUMMARY")
    print("=" * 80)
    
    memory_summary = chat.get_memory_summary()
    print(f"\nConversation Turns: {memory_summary['conversation_turns']}")
    print(f"Memory Mode: {memory_summary['memory_mode']}")
    print(f"Auto-Synthesize: {memory_summary['auto_synthesize']}")
    print(f"Synthesis Count: {memory_summary['synthesis_count']}")
    print(f"Topics Synthesized: {memory_summary['topics_synthesized']}")
    print(f"Memory Count: {memory_summary['memory_count']}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main() 