#!/usr/bin/env python3
"""
V7 Memory Node Demonstration
============================

This script demonstrates the capabilities of the V7 Memory Node system,
including memory storage, retrieval, search, and decay functionality.
"""

import os
import sys
import time
import argparse
import logging
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the path so we can import the modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.v7.memory_node import MemoryConsciousnessNode
from src.v7.node_consciousness_manager import NodeConsciousnessManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoryDemo")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='V7 Memory Node Demonstration')
    
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no UI)')
    parser.add_argument('--sqlite', action='store_true', help='Use SQLite database for memory storage')
    parser.add_argument('--csv-export', type=str, help='Export memories to CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--accelerate-decay', action='store_true', help='Accelerate memory decay for demonstration')
    
    return parser.parse_args()

def setup_nodes(args):
    """Initialize the nodes required for the demo"""
    # Create a node consciousness manager
    manager = NodeConsciousnessManager()
    
    # Configure the memory node
    memory_config = {
        'store_type': 'sqlite' if args.sqlite else 'json',
        'memory_path': str(project_root / 'data' / 'memories' / 'demo'),
        'memory_persistence': True,
        'decay_enabled': True,
        'decay_rate': 0.1 if args.accelerate_decay else 0.05,  # Higher decay rate for demo if requested
        'decay_interval': 30 if args.accelerate_decay else 86400,  # 30 seconds vs 24 hours
    }
    
    # Create and initialize the memory node
    memory_node = MemoryConsciousnessNode(
        node_id="demo_memory",
        name="Memory Demo Node", 
        config=memory_config
    )
    
    # Register the node with the manager
    manager.register_node(memory_node)
    
    # Activate the node
    memory_node.activate()
    
    return manager, memory_node

def demonstrate_memory_storage(memory_node):
    """Demonstrate storing memories of different types and importance levels"""
    logger.info("=== Demonstrating Memory Storage ===")
    
    # Store factual memories
    store_result = memory_node.process({
        'store': {
            'content': "The capital of France is Paris.",
            'memory_type': "fact",
            'strength': 0.8,
            'tags': ["geography", "europe", "capital"]
        },
        'metadata': {
            'category': "geography", 
            'confidence': 0.95
        }
    })
    
    store_result = memory_node.process({
        'store': {
            'content': "Water boils at 100 degrees Celsius at sea level.",
            'memory_type': "fact",
            'strength': 0.7,
            'tags': ["science", "physics", "temperature"]
        },
        'metadata': {
            'category': "science", 
            'confidence': 0.99
        }
    })
    
    # Store experiential memory
    store_result = memory_node.process({
        'store': {
            'content': "User showed frustration when the system misunderstood their query about weather data.",
            'memory_type': "experience",
            'strength': 0.6,
            'tags': ["user", "emotion", "interaction"]
        },
        'metadata': {
            'emotion': "frustration", 
            'topic': "weather", 
            'interaction_id': "12345"
        }
    })
    
    # Store relational memory
    store_result = memory_node.process({
        'store': {
            'content': "User John prefers concise answers with examples.",
            'memory_type': "relation",
            'strength': 0.9,
            'tags': ["user", "preference", "communication"]
        },
        'metadata': {
            'user_id': "john_doe", 
            'preference_type': "communication_style"
        }
    })
    
    # Store procedural memory
    store_result = memory_node.process({
        'store': {
            'content': "To process CSV files, first check the delimiter and then use the pandas library.",
            'memory_type': "procedure",
            'strength': 0.75,
            'tags': ["data", "processing", "file"]
        },
        'metadata': {
            'domain': "data_processing", 
            'tool': "pandas"
        }
    })
    
    logger.info(f"✅ Stored 5 memories of different types")
    
    # Show memory counts by type
    result = memory_node.process({'list': {}})
    if result['success']:
        memories = result['memories']
        memory_types = {}
        for memory in memories:
            memory_type = memory['memory_type']
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
        
        for memory_type, count in memory_types.items():
            logger.info(f"  - {memory_type}: {count} memories")
    
    return True

def demonstrate_memory_retrieval(memory_node):
    """Demonstrate retrieving memories by different criteria"""
    logger.info("\n=== Demonstrating Memory Retrieval ===")
    
    # Retrieve by type
    facts_result = memory_node.process({
        'search': {
            'memory_type': 'fact'
        },
        'params': {
            'limit': 10
        }
    })
    
    if facts_result['success']:
        logger.info(f"Retrieved {len(facts_result['memories'])} factual memories:")
        for i, memory in enumerate(facts_result['memories'], 1):
            logger.info(f"  {i}. {memory['content']} (Strength: {memory['strength']:.2f})")
    
    # Retrieve by tags
    geography_result = memory_node.process({
        'search': {
            'tags': ['geography']
        },
        'params': {
            'limit': 10
        }
    })
    
    if geography_result['success']:
        logger.info(f"\nRetrieved {len(geography_result['memories'])} geography-related memories:")
        for i, memory in enumerate(geography_result['memories'], 1):
            logger.info(f"  {i}. {memory['content']} (Tags: {', '.join(memory['tags'])})")
    
    # Retrieve by minimum strength
    strong_memories_result = memory_node.process({
        'search': {
            'min_strength': 0.8
        },
        'params': {
            'limit': 10
        }
    })
    
    if strong_memories_result['success']:
        logger.info(f"\nRetrieved {len(strong_memories_result['memories'])} high-strength memories (>= 0.8):")
        for i, memory in enumerate(strong_memories_result['memories'], 1):
            logger.info(f"  {i}. {memory['content']} (Strength: {memory['strength']:.2f})")
    
    return True

def demonstrate_memory_search(memory_node):
    """Demonstrate search capabilities"""
    logger.info("\n=== Demonstrating Memory Search ===")
    
    # Add more diverse memories for better search demonstration
    memory_node.process({
        'store': {
            'content': "Python is a high-level programming language known for its readability.",
            'memory_type': "fact",
            'strength': 0.75,
            'tags': ["programming", "technology", "language"]
        },
        'metadata': {
            'category': "technology", 
            'subcategory': "programming"
        }
    })
    
    memory_node.process({
        'store': {
            'content': "Machine learning models require careful validation to avoid overfitting.",
            'memory_type': "fact",
            'strength': 0.85,
            'tags': ["ai", "machine_learning", "model"]
        },
        'metadata': {
            'category': "technology", 
            'subcategory': "machine_learning"
        }
    })
    
    memory_node.process({
        'store': {
            'content': "Regular exercise improves cognitive function and mental wellbeing.",
            'memory_type': "fact",
            'strength': 0.7,
            'tags': ["health", "exercise", "wellbeing"]
        },
        'metadata': {
            'category': "health", 
            'subcategory': "fitness"
        }
    })
    
    # Search by content
    search_terms = [
        "programming",
        "machine learning",
        "health",
        "geography",
        "data processing"
    ]
    
    for term in search_terms:
        search_result = memory_node.process({
            'search': {
                'content_contains': term
            },
            'params': {
                'limit': 3
            }
        })
        
        logger.info(f"\nSearch results for content containing '{term}':")
        
        if search_result['success'] and search_result['memories']:
            for i, memory in enumerate(search_result['memories'], 1):
                logger.info(f"  {i}. {memory['content']} (Strength: {memory['strength']:.2f})")
        else:
            logger.info("  No matching memories found.")
    
    return True

def demonstrate_memory_decay(memory_node):
    """Demonstrate memory decay over time"""
    logger.info("\n=== Demonstrating Memory Decay Simulation ===")
    
    # Store a test memory
    mem_id = f"decay_test_{uuid.uuid4().hex[:8]}"
    
    store_result = memory_node.process({
        'store': {
            'id': mem_id,
            'content': "This is a test memory for demonstrating decay.",
            'memory_type': "fact",
            'strength': 0.6,
            'tags': ["test", "decay"]
        },
        'metadata': {
            'purpose': "decay_test"
        }
    })
    
    if not store_result['success']:
        logger.error("Failed to store test memory for decay demonstration")
        return False
    
    # Get the fresh memory
    retrieve_result = memory_node.process({
        'retrieve': mem_id
    })
    
    if not retrieve_result['success']:
        logger.error("Failed to retrieve test memory for decay demonstration")
        return False
    
    fresh_memory = retrieve_result['memory']
    logger.info(f"Fresh memory strength: {fresh_memory['strength']:.4f}")
    
    # Calculate theoretical decay over time
    decay_rate = memory_node.decay_rate
    simulated_days = [1, 7, 14, 30, 60, 90]
    
    for days in simulated_days:
        # Calculate decay based on the rate
        decay_factor = max(0.1, 1.0 - (decay_rate * days))
        decayed_strength = fresh_memory['strength'] * decay_factor
        
        logger.info(f"After {days} days: Strength = {decayed_strength:.4f} " +
                   f"({(1 - decay_factor) * 100:.1f}% decay)")
    
    logger.info("\nNote: Memories below minimum strength may be archived or removed")
    
    if memory_node.decay_enabled and memory_node.decay_interval < 60:
        logger.info("\nAccelerated decay is enabled for this demo.")
        logger.info("Wait a few minutes and then list memories to see decay in action.")
    
    return True

def demonstrate_memory_update(memory_node):
    """Demonstrate updating existing memories"""
    logger.info("\n=== Demonstrating Memory Updates ===")
    
    # First store a memory
    mem_id = f"update_test_{uuid.uuid4().hex[:8]}"
    
    store_result = memory_node.process({
        'store': {
            'id': mem_id,
            'content': "The Earth is the third planet from the Sun.",
            'memory_type': "fact",
            'strength': 0.7,
            'tags': ["astronomy", "planet", "solar_system"]
        },
        'metadata': {
            'category': "astronomy",
            'confidence': 0.9
        }
    })
    
    if not store_result['success']:
        logger.error("Failed to store test memory for update demonstration")
        return False
    
    # Get the original memory
    retrieve_result = memory_node.process({
        'retrieve': mem_id
    })
    
    if not retrieve_result['success']:
        logger.error("Failed to retrieve test memory for update demonstration")
        return False
    
    original_memory = retrieve_result['memory']
    logger.info(f"Original memory: {original_memory['content']}")
    logger.info(f"Original strength: {original_memory['strength']:.2f}")
    logger.info(f"Original tags: {original_memory['tags']}")
    
    # Update the memory
    update_result = memory_node.process({
        'update': mem_id,
        'updates': {
            'content': "The Earth is the third planet from the Sun and the only known planet with life.",
            'strength': 0.9,
            'tags': original_memory['tags'] + ["life"]
        }
    })
    
    if not update_result['success']:
        logger.error("Failed to update test memory")
        return False
    
    # Get the updated memory
    retrieve_result = memory_node.process({
        'retrieve': mem_id
    })
    
    if not retrieve_result['success']:
        logger.error("Failed to retrieve updated test memory")
        return False
    
    updated_memory = retrieve_result['memory']
    logger.info(f"\nUpdated memory: {updated_memory['content']}")
    logger.info(f"Updated strength: {updated_memory['strength']:.2f}")
    logger.info(f"Updated tags: {updated_memory['tags']}")
    
    return True

def export_memories_to_csv(memory_node, output_path):
    """Export all memories to a CSV file"""
    if not output_path:
        return False
    
    # Get all memories
    result = memory_node.process({'list': {}})
    
    if not result['success']:
        logger.error(f"Failed to retrieve memories for export: {result.get('error')}")
        return False
        
    memories = result['memories']
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write to CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("id,created_at,memory_type,strength,content,tags,metadata\n")
        
        # Write data
        for memory in memories:
            created_at = memory.get('created_at', '')
            memory_id = memory.get('id', '')
            memory_type = memory.get('memory_type', '')
            strength = memory.get('strength', 0)
            content = str(memory.get('content', '')).replace('"', '""')
            tags = json.dumps(memory.get('tags', [])).replace('"', '""')
            metadata = json.dumps(memory.get('metadata', {})).replace('"', '""')
            
            f.write(f'"{memory_id}","{created_at}","{memory_type}",{strength},"{content}","{tags}","{metadata}"\n')
    
    logger.info(f"\n✅ Exported {len(memories)} memories to {output_path}")
    return True

def main():
    """Main demonstration function"""
    args = parse_arguments()
    
    # Configure logging based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("Starting V7 Memory Node Demonstration")
    
    try:
        # Setup the nodes
        manager, memory_node = setup_nodes(args)
        
        # Run the demonstrations
        demonstrate_memory_storage(memory_node)
        demonstrate_memory_retrieval(memory_node)
        demonstrate_memory_search(memory_node)
        demonstrate_memory_decay(memory_node)
        demonstrate_memory_update(memory_node)
        
        # Export memories if requested
        if args.csv_export:
            export_memories_to_csv(memory_node, args.csv_export)
        
        logger.info("\n✅ Memory Node Demonstration completed successfully!")
        
        # If accelerated decay is enabled, keep running to observe decay
        if args.accelerate_decay:
            logger.info("\nWaiting to observe memory decay (press Ctrl+C to exit)...")
            try:
                while True:
                    time.sleep(30)
                    result = memory_node.process({'list': {}})
                    if result['success']:
                        logger.info(f"\nMemory strengths after decay ({datetime.now().strftime('%H:%M:%S')}):")
                        for memory in result['memories']:
                            logger.info(f"  - {memory['content'][:50]}... : {memory['strength']:.4f}")
            except KeyboardInterrupt:
                logger.info("\nDecay observation stopped by user")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you have installed all dependencies and are running from the project root.")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed with error: {e}", exc_info=args.debug)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 