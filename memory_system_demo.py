"""
Memory System Demo - Demonstrates integration of all memory components

This script shows how the Memory Manager, ConversationMemory, and MemoryAgent
components work together to create a more intelligent and context-aware
conversational system.
"""

import logging
import time
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create data directories
data_dir = Path("data/memory")
data_dir.mkdir(parents=True, exist_ok=True)

try:
    # Import memory components
    from memory_manager import memory_manager
    from conversation_memory import ConversationMemory
    from memory_agent import MemoryAgent
    
    logger.info("Successfully imported all memory components")
except ImportError as e:
    logger.error(f"Error importing memory components: {str(e)}")
    raise

def print_divider(title):
    """Print a section divider with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")

def print_memory_stats():
    """Print statistics about the memory system"""
    print_divider("MEMORY SYSTEM STATS")
    
    # Get memory manager stats
    manager_stats = memory_manager.get_stats()
    print(f"Memory Manager Stats:")
    print(f"  - Total shares: {manager_stats.get('total_shares', 0)}")
    print(f"  - Total broadcasts: {manager_stats.get('total_broadcasts', 0)}")
    print(f"  - Total memories shared: {manager_stats.get('total_memories_shared', 0)}")
    print(f"  - Registered components: {memory_manager.get_component_names()}")
    
    # Get activity log
    activity_log = memory_manager.get_activity_log(limit=5)
    print("\nRecent Memory Activity:")
    for entry in activity_log:
        action = entry.get("action", "")
        source = entry.get("source", "")
        target = entry.get("target", "")
        if target:
            print(f"  - {action}: {source} -> {target}")
        else:
            print(f"  - {action}: {source}")

def simulate_multi_agent_conversation():
    """Simulate a conversation between multiple agents with memory sharing"""
    print_divider("MULTI-AGENT CONVERSATION DEMO")
    
    # Create agents
    agent1 = MemoryAgent("weather_agent")
    agent2 = MemoryAgent("travel_agent")
    
    print("Initialized two agents: 'weather_agent' and 'travel_agent'")
    print("Cross-subscribing agents to share memories...\n")
    
    # Subscribe agents to each other's conversation memory
    weather_memory_name = f"{agent1.agent_name}_conversation_memory"
    travel_memory_name = f"{agent2.agent_name}_conversation_memory"
    
    agent1.conversation_memory.subscribe_to(travel_memory_name)
    agent2.conversation_memory.subscribe_to(weather_memory_name)
    
    print("Starting conversation with weather_agent:")
    
    # User talks to weather agent
    print("\nUser: Hello weather agent, what's the forecast for tomorrow?")
    response = agent1.process_input("Hello weather agent, what's the forecast for tomorrow?")
    print(f"Weather Agent: {response}")
    
    print("\nUser: I'm planning a trip to the beach if it's sunny")
    response = agent1.process_input("I'm planning a trip to the beach if it's sunny")
    print(f"Weather Agent: {response}")
    
    print("\n[Memory sharing occurs between agents...]")
    time.sleep(1)  # Pause for effect
    
    print("\nSwitching to conversation with travel_agent:")
    
    # User talks to travel agent - should have context from weather agent
    print("\nUser: Hello travel agent, can you recommend some activities?")
    response = agent2.process_input("Hello travel agent, can you recommend some activities?")
    print(f"Travel Agent: {response}")
    
    print("\nUser: I'm interested in beach activities specifically")
    response = agent2.process_input("I'm interested in beach activities specifically")
    print(f"Travel Agent: {response}")
    
    # Agent should remember the beach context from the weather agent
    print("\nUser: What should I pack for my trip?")
    response = agent2.process_input("What should I pack for my trip?")
    print(f"Travel Agent: {response}")
    
    # Print memory stats
    print_memory_stats()
    
    # Print agent stats
    print("\nWeather Agent Memory Stats:")
    print(json.dumps(agent1.get_memory_stats(), indent=2))
    
    print("\nTravel Agent Memory Stats:")
    print(json.dumps(agent2.get_memory_stats(), indent=2))

def demonstrate_memory_persistence():
    """Demonstrate how memories persist across sessions"""
    print_divider("MEMORY PERSISTENCE DEMO")
    
    # Create a dedicated memory for this demo
    persistence_memory = ConversationMemory(
        component_name="persistence_demo",
        memory_file="data/memory/persistence_demo.jsonl"
    )
    
    # Store some memories
    print("Storing memories in persistence_demo component...")
    
    persistence_memory.store(
        "Remember this important information for later",
        "I'll remember that you need to call the doctor tomorrow at 9am",
        metadata={
            "topics": ["reminder", "health"],
            "emotion": "neutral",
            "keywords": ["doctor", "appointment"],
            "entities": ["doctor", "9am", "tomorrow"],
            "importance": 0.9
        }
    )
    
    # Display stored memory
    print("\nRetrieving important memories:")
    important_memories = persistence_memory.retrieve_important()
    for memory in important_memories:
        print(f"  - User: {memory.get('user_input')}")
        print(f"    Agent: {memory.get('system_response')}")
        print(f"    Topics: {memory.get('metadata', {}).get('topics')}")
        print(f"    Importance: {memory.get('metadata', {}).get('importance')}")
    
    print("\nMemory has been stored to disk. In a real application, this would persist across restarts.")
    
    # Show file location
    file_path = Path("data/memory/persistence_demo.jsonl").absolute()
    print(f"\nStored at: {file_path}")
    
    # Read and display file contents
    try:
        with open(file_path, "r") as f:
            print("\nFile contents:")
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def interactive_agent_demo():
    """Run an interactive demo with a memory-enabled agent"""
    print_divider("INTERACTIVE AGENT DEMO")
    
    # Create a memory agent
    agent = MemoryAgent("interactive_agent")
    
    print("Interactive Memory Agent Demo")
    print("Type 'exit' to end the conversation")
    print("Type 'stats' to see memory statistics")
    print()
    
    print("Agent: Hello! I'm a memory-enabled agent. How can I help you today?")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nExiting interactive demo...")
            break
            
        if user_input.lower() == 'stats':
            stats = agent.get_memory_stats()
            print("\nMemory Stats:")
            print(json.dumps(stats, indent=2))
            continue
            
        # Process the input
        response = agent.process_input(user_input)
        print(f"Agent: {response}")

def main():
    """Run the memory system demo"""
    print_divider("MEMORY SYSTEM DEMO")
    
    print("This demo showcases the memory system components:")
    print("1. Memory Manager - Central coordination system")
    print("2. Conversation Memory - Stores and retrieves conversation history")
    print("3. Memory Agent - Uses memory for intelligent responses")
    
    options = [
        ("Multi-Agent Conversation", simulate_multi_agent_conversation),
        ("Memory Persistence Demo", demonstrate_memory_persistence),
        ("Interactive Agent Demo", interactive_agent_demo),
        ("Memory System Stats", print_memory_stats),
        ("Exit", None)
    ]
    
    while True:
        print("\nSelect a demo to run:")
        for i, (name, _) in enumerate(options, 1):
            print(f"{i}. {name}")
            
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if choice < 1 or choice > len(options):
                print(f"Please enter a number between 1 and {len(options)}")
                continue
                
            if choice == len(options):  # Exit option
                print("\nExiting demo...")
                break
                
            # Run the selected demo
            demo_func = options[choice-1][1]
            demo_func()
            
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error running demo: {str(e)}")

if __name__ == "__main__":
    main() 