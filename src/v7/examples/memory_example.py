#!/usr/bin/env python3
"""
Memory Node Example for V7 Node Consciousness.

This example demonstrates the key capabilities of the Memory Node:
- Creating and initializing a Memory Node
- Storing different types of memories
- Retrieving memories by various criteria
- Observing memory decay over time
- Visualizing memory relationships and strengths
"""

import sys
import os
import time
import logging
import random
import uuid
from typing import Dict, List, Any
from pathlib import Path

# Add the project root to the Python path to allow imports
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryExample")

try:
    # Import PySide6 for UI
    from PySide6.QtWidgets import QApplication
    QT_AVAILABLE = True
except ImportError:
    logger.warning("PySide6 not available, running with limited UI")
    QT_AVAILABLE = False

# Import the memory node module
try:
    from src.v7.memory.memory_node import MemoryNode
    from src.v7.ui.memory_visualization import MemoryVisualizer
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Make sure you are running this script from the project root directory")
    sys.exit(1)

class MemoryExample:
    """Example class demonstrating Memory Node capabilities."""
    
    def __init__(self, storage_type="sqlite", enable_persistence=True):
        """Initialize the example."""
        logger.info("Initializing Memory Node example")
        
        self.memory_node = MemoryNode(
            storage_type=storage_type,
            memory_path=str(root_dir / "data" / "memories"),
            enable_persistence=enable_persistence,
            decay_rate=0.01,  # Memories decay slowly over time
            decay_interval=30,  # Check for decay every 30 seconds
            minimum_strength=0.1  # Memories with strength below 0.1 are removed
        )
        
        # Create a visualizer if Qt is available
        self.visualizer = MemoryVisualizer() if QT_AVAILABLE else None
        
        # Sample data for demonstration
        self.facts = [
            "The sky appears blue due to Rayleigh scattering",
            "Python was created by Guido van Rossum",
            "Water boils at 100 degrees Celsius at sea level",
            "The Earth orbits the Sun once every 365.25 days",
            "The speed of light is approximately 299,792,458 meters per second"
        ]
        
        self.experiences = [
            "I learned to use the memory system effectively",
            "I observed an interesting pattern in user behavior",
            "I processed a complex query involving multiple entities",
            "I helped translate a difficult technical concept",
            "I identified a potential optimization in the system"
        ]
        
        self.relations = [
            {"source": "water", "relation": "boils_at", "target": "100°C"},
            {"source": "Earth", "relation": "orbits", "target": "Sun"},
            {"source": "memory_node", "relation": "component_of", "target": "v7_system"},
            {"source": "language_node", "relation": "connected_to", "target": "memory_node"},
            {"source": "python", "relation": "created_by", "target": "Guido van Rossum"}
        ]
        
        self.procedures = [
            "To store a memory: call memory_node.store_memory() with the content and type",
            "To retrieve a memory: use memory_node.get_memory() with the memory ID",
            "To search memories: use memory_node.search_memories() with search criteria",
            "To update a memory: call memory_node.update_memory() with new data",
            "To delete a memory: call memory_node.delete_memory() with the memory ID"
        ]
    
    def populate_sample_memories(self):
        """Populate the memory node with sample memories."""
        logger.info("Populating memory node with sample memories")
        
        # Store facts
        for fact in self.facts:
            memory_id = str(uuid.uuid4())
            self.memory_node.store_memory(
                memory_id=memory_id,
                content=fact,
                memory_type="fact",
                strength=random.uniform(0.7, 1.0),  # Facts start with high strength
                tags=["fact", "knowledge", "permanent"],
                metadata={"source": "system", "verified": True}
            )
            logger.info(f"Stored fact: {fact[:30]}...")
        
        # Store experiences
        for experience in self.experiences:
            memory_id = str(uuid.uuid4())
            self.memory_node.store_memory(
                memory_id=memory_id,
                content=experience,
                memory_type="experience",
                strength=random.uniform(0.5, 0.9),  # Experiences have varied strength
                tags=["experience", "observation"],
                metadata={"timestamp": time.time(), "context": "example"}
            )
            logger.info(f"Stored experience: {experience[:30]}...")
        
        # Store relations
        for relation in self.relations:
            memory_id = str(uuid.uuid4())
            content = f"{relation['source']} {relation['relation']} {relation['target']}"
            self.memory_node.store_memory(
                memory_id=memory_id,
                content=content,
                memory_type="relation",
                strength=random.uniform(0.6, 0.95),  # Relations are important
                tags=["relation", relation["relation"]],
                metadata=relation
            )
            logger.info(f"Stored relation: {content}")
        
        # Store procedures
        for procedure in self.procedures:
            memory_id = str(uuid.uuid4())
            self.memory_node.store_memory(
                memory_id=memory_id,
                content=procedure,
                memory_type="procedure",
                strength=random.uniform(0.8, 1.0),  # Procedures are critical
                tags=["procedure", "instruction", "how-to"],
                metadata={"usage_count": 0, "last_used": time.time()}
            )
            logger.info(f"Stored procedure: {procedure[:30]}...")
    
    def demonstrate_retrieval(self):
        """Demonstrate memory retrieval capabilities."""
        logger.info("\n--- Memory Retrieval Demonstration ---")
        
        # List all memories
        all_memories = self.memory_node.list_memories()
        logger.info(f"Total memories stored: {len(all_memories)}")
        
        # Get memory by ID (first memory in the list)
        if all_memories:
            first_memory_id = all_memories[0]["id"]
            memory = self.memory_node.get_memory(first_memory_id)
            logger.info(f"Retrieved memory by ID: {memory['content'][:50]}...")
        
        # Search memories by type
        fact_memories = self.memory_node.search_memories(memory_type="fact")
        logger.info(f"Found {len(fact_memories)} fact memories")
        
        # Search memories by tag
        instruction_memories = self.memory_node.search_memories(tags=["instruction"])
        logger.info(f"Found {len(instruction_memories)} instruction memories")
        
        # Search memories by content (simple keyword search)
        python_memories = self.memory_node.search_memories(content="Python")
        logger.info(f"Found {len(python_memories)} memories related to Python")
        
        # Search memories by minimum strength
        strong_memories = self.memory_node.search_memories(min_strength=0.8)
        logger.info(f"Found {len(strong_memories)} memories with strength >= 0.8")
        
        return all_memories
    
    def demonstrate_updates(self):
        """Demonstrate memory update capabilities."""
        logger.info("\n--- Memory Update Demonstration ---")
        
        # Get all procedure memories
        procedure_memories = self.memory_node.search_memories(memory_type="procedure")
        
        # Simulate using a procedure by updating its metadata
        if procedure_memories:
            memory = procedure_memories[0]
            memory_id = memory["id"]
            
            # Get current metadata
            current_metadata = memory.get("metadata", {})
            usage_count = current_metadata.get("usage_count", 0)
            
            # Update the memory
            logger.info(f"Updating memory: {memory['content'][:50]}...")
            logger.info(f"Current usage count: {usage_count}")
            
            self.memory_node.update_memory(
                memory_id=memory_id,
                metadata={
                    **current_metadata,
                    "usage_count": usage_count + 1,
                    "last_used": time.time()
                },
                # Increase strength when used
                strength=min(1.0, memory.get("strength", 0.5) + 0.05)
            )
            
            # Verify the update
            updated_memory = self.memory_node.get_memory(memory_id)
            logger.info(f"Updated usage count: {updated_memory['metadata']['usage_count']}")
            logger.info(f"Updated strength: {updated_memory['strength']:.2f}")
    
    def demonstrate_decay(self, simulation_speed=10):
        """Demonstrate memory decay over time."""
        logger.info("\n--- Memory Decay Demonstration ---")
        logger.info(f"Simulating memory decay at {simulation_speed}x speed")
        
        # Get initial memory stats
        all_memories = self.memory_node.list_memories()
        initial_count = len(all_memories)
        
        # Force memory decay by advancing the system's internal clock
        # Note: This is for demonstration purposes only
        self.memory_node._last_decay_check = time.time() - (simulation_speed * self.memory_node._decay_interval)
        
        # Trigger decay check
        logger.info("Triggering memory decay...")
        self.memory_node._check_memory_decay()
        
        # Get updated memory stats
        remaining_memories = self.memory_node.list_memories()
        final_count = len(remaining_memories)
        
        logger.info(f"Initial memory count: {initial_count}")
        logger.info(f"Remaining memory count: {final_count}")
        logger.info(f"Memories decayed: {initial_count - final_count}")
        
        # Show some example strengths
        if remaining_memories:
            logger.info("Sample of remaining memory strengths:")
            for i, memory in enumerate(remaining_memories[:5]):
                logger.info(f"  Memory {i+1}: {memory['strength']:.2f} - {memory['content'][:30]}...")
    
    def visualize_memories(self):
        """Visualize the memories if visualization is available."""
        if not self.visualizer or not self.visualizer.is_available():
            logger.warning("Memory visualization is not available (PySide6 required)")
            return
        
        logger.info("\n--- Memory Visualization ---")
        logger.info("Launching memory visualization widget")
        
        # Get all memories
        all_memories = self.memory_node.list_memories()
        
        # Update the visualizer with memory data
        self.visualizer.update(all_memories)
        
        # Show the visualization widget
        self.visualizer.show()
        
        logger.info("Memory visualization launched")
        logger.info("Close the visualization window to continue")
    
    def run_demonstration(self):
        """Run the complete memory node demonstration."""
        logger.info("Starting Memory Node demonstration")
        
        # Initialize and populate memories
        self.populate_sample_memories()
        
        # Demonstrate retrieval
        all_memories = self.demonstrate_retrieval()
        
        # Demonstrate updates
        self.demonstrate_updates()
        
        # Visualize memories (if available)
        self.visualize_memories()
        
        # Simulate memory access to affect strength
        logger.info("\nSimulating memory access patterns...")
        for _ in range(5):
            if all_memories:
                # Randomly access some memories to increase their strength
                for _ in range(3):
                    memory = random.choice(all_memories)
                    memory_id = memory["id"]
                    self.memory_node.get_memory(memory_id)  # This accesses the memory
                    logger.info(f"Accessed memory: {memory['content'][:30]}...")
                
                # Wait a bit
                time.sleep(0.5)
        
        # Visualize again after access patterns
        self.visualize_memories()
        
        # Demonstrate decay
        self.demonstrate_decay()
        
        # Final visualization after decay
        self.visualize_memories()
        
        logger.info("\nMemory Node demonstration completed")


def main():
    """Main entry point for the memory example."""
    # Create Qt application if available
    app = QApplication(sys.argv) if QT_AVAILABLE else None
    
    try:
        # Run the demonstration
        example = MemoryExample(
            storage_type="sqlite",
            enable_persistence=True
        )
        example.run_demonstration()
        
        # If we have a Qt app, start its event loop
        if app:
            logger.info("Starting Qt event loop")
            sys.exit(app.exec())
            
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
    finally:
        logger.info("Memory example completed")


if __name__ == "__main__":
    main() 