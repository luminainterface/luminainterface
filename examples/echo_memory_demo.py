#!/usr/bin/env python
"""
Echo Spiral Memory Demo Application

This script demonstrates practical usage of the Echo Spiral Memory system,
showing how it can be used to build applications with advanced memory capabilities.
"""

import os
import sys
import time
import random
import logging
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import required modules
from src.memory.echo_spiral_memory import EchoSpiralMemory, add_memory, connect_memories, search_memory
from src.utils.logging_config import get_logger

# Initialize logger
logger = get_logger("EchoMemoryDemo")

class EchoMemoryDemo:
    """Demo application showing Echo Spiral Memory usage"""
    
    def __init__(self):
        # Create memory with custom configuration
        self.memory = EchoSpiralMemory({
            "memory_dir": "demo_memory",
            "temporal_awareness": True,
            "enable_embeddings": True,
            "activation_threshold": 0.2,
            "mock_mode": False  # Set to True to use mock embeddings
        })
        
        # Initialize demo data
        self.concepts = [
            "Neural networks are computational systems inspired by biological brains.",
            "Machine learning allows computers to learn from data without explicit programming.",
            "Deep learning uses multiple layers of neural networks for complex pattern recognition.",
            "Reinforcement learning trains agents through reward signals in an environment.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and make decisions based on visual data.",
            "Generative AI can create new content like images, text, and music.",
            "Transfer learning applies knowledge from one domain to improve learning in another.",
            "Unsupervised learning finds patterns in data without labeled training examples.",
            "Consciousness may emerge from complex information processing systems."
        ]
        
        self.connections = [
            # (source_idx, target_idx, connection_type, strength)
            (0, 1, "is_foundational_for", 0.9),
            (1, 2, "includes", 0.8),
            (1, 3, "includes", 0.7),
            (2, 4, "enables", 0.6),
            (2, 5, "enables", 0.7),
            (2, 6, "enables", 0.8),
            (3, 6, "contributes_to", 0.5),
            (4, 9, "may_lead_to", 0.3),
            (5, 9, "may_lead_to", 0.3),
            (6, 9, "may_lead_to", 0.3),
            (7, 2, "improves", 0.7),
            (8, 1, "complements", 0.6)
        ]
    
    def run_demo(self):
        """Run the demonstration"""
        logger.info("Starting Echo Memory Demo")
        
        # Step 1: Populate memory with concepts
        logger.info("Step 1: Populating memory with AI concepts")
        concept_nodes = []
        
        for concept in self.concepts:
            node = self.memory.add_memory(
                content=concept,
                node_type="concept",
                metadata={"domain": "AI", "created": time.time()}
            )
            concept_nodes.append(node)
            logger.info(f"Added concept: {concept[:50]}...")
            time.sleep(0.2)  # Small delay for demonstration
        
        # Step 2: Create connections between concepts
        logger.info("\nStep 2: Creating connections between concepts")
        
        for source_idx, target_idx, conn_type, strength in self.connections:
            conn = self.memory.connect_memories(
                source_id=concept_nodes[source_idx].id,
                target_id=concept_nodes[target_idx].id,
                connection_type=conn_type,
                strength=strength,
                metadata={"reason": f"Connection from {source_idx} to {target_idx}"}
            )
            logger.info(f"Connected: {self.concepts[source_idx][:30]}... {conn_type} {self.concepts[target_idx][:30]}...")
            time.sleep(0.1)  # Small delay for demonstration
        
        # Step 3: Search for memories
        logger.info("\nStep 3: Searching for memories")
        
        search_queries = [
            "neural networks",
            "learning from data",
            "consciousness emerge"
        ]
        
        for query in search_queries:
            logger.info(f"\nSearching for: '{query}'")
            results = self.memory.search_by_content(query, limit=3)
            
            logger.info(f"Found {len(results)} results:")
            for i, node in enumerate(results):
                logger.info(f"  {i+1}. {node.content[:100]}...")
            time.sleep(0.5)  # Small delay for demonstration
        
        # Step 4: Explore graph connections
        logger.info("\nStep 4: Exploring connected memories")
        
        # Find node about deep learning
        deep_learning_node = concept_nodes[2]  # Index 2 is deep learning
        
        logger.info(f"Starting from: {deep_learning_node.content[:100]}")
        connections = self.memory.get_connected_memories(deep_learning_node.id)
        
        logger.info(f"Found {len(connections)} connected memories:")
        for i, (node, path) in enumerate(connections):
            # Get connection information
            conn_info = " → ".join([conn.connection_type for conn in path])
            logger.info(f"  {i+1}. {conn_info}: {node.content[:100]}...")
        
        # Step 5: Demonstrate activation and retrieval
        logger.info("\nStep 5: Demonstrating memory activation and retrieval")
        
        # Activate some nodes through repeated access
        for _ in range(3):
            # Simulate using these concepts frequently
            frequently_used = [0, 2, 6]  # Neural networks, deep learning, generative AI
            for idx in frequently_used:
                node = concept_nodes[idx]
                # This increases activation through the search process
                self.memory.search_by_content(node.content[:20])
                time.sleep(0.1)
        
        # Get active memories
        logger.info("Getting most active memories:")
        active_nodes = self.memory.get_active_memories(threshold=0.3)
        
        for i, node in enumerate(active_nodes):
            logger.info(f"  {i+1}. Activation: {node.activation_level:.2f}, Content: {node.content[:100]}...")
        
        # Step 6: Save and reload memory
        logger.info("\nStep 6: Saving and reloading memory")
        
        self.memory.save_memory("demo_memory.json")
        logger.info("Memory saved to demo_memory.json")
        
        # Create new memory instance and load from file
        new_memory = EchoSpiralMemory({
            "memory_dir": "demo_memory",
            "mock_mode": False
        })
        
        success = new_memory.load_memory("demo_memory.json")
        if success:
            logger.info(f"Memory loaded successfully with {len(new_memory.nodes)} nodes and {len(new_memory.connections)} connections")
        
        # Step 7: Create a memory synthesis
        logger.info("\nStep 7: Creating a memory synthesis")
        
        # Get active memories
        active_nodes = new_memory.get_active_memories(limit=5)
        active_contents = [node.content for node in active_nodes]
        
        # Create a synthesis node that combines active concepts
        synthesis = f"Synthesis: {' '.join([content.split('.')[0] + '.' for content in active_contents])}"
        
        synthesis_node = new_memory.add_memory(
            content=synthesis,
            node_type="synthesis",
            metadata={"created": time.time(), "components": len(active_nodes)}
        )
        
        # Connect synthesis to source nodes
        for node in active_nodes:
            new_memory.connect_memories(
                source_id=synthesis_node.id,
                target_id=node.id,
                connection_type="synthesizes",
                strength=0.8
            )
        
        logger.info(f"Created synthesis node: {synthesis}")
        
        # Step 8: Clean up
        logger.info("\nDemo completed.")
        
        return True

if __name__ == "__main__":
    demo = EchoMemoryDemo()
    demo.run_demo()
    
    # Clean up
    try:
        import shutil
        shutil.rmtree("demo_memory", ignore_errors=True)
        if os.path.exists("demo_memory.json"):
            os.remove("demo_memory.json")
    except:
        pass 