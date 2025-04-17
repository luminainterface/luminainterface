#!/usr/bin/env python
"""
ConsciousnessNode Demo Application

This script demonstrates practical usage of the ConsciousnessNode system,
showing how it generates thoughts, reflections, and awareness metrics.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import required modules
from src.consciousness.consciousness_node import ConsciousnessNode, ThoughtPattern
from src.memory.echo_spiral_memory import EchoSpiralMemory
from src.utils.logging_config import get_logger

# Initialize logger
logger = get_logger("ConsciousnessDemo")

class ConsciousnessDemo:
    """Demo application showing ConsciousnessNode usage"""
    
    def __init__(self, use_memory=True):
        # Create memory system if requested
        self.memory = None
        if use_memory:
            self.memory = EchoSpiralMemory({
                "memory_dir": "demo_memory",
                "enable_embeddings": True,
                "mock_mode": False
            })
        
        # Create consciousness with custom configuration
        self.consciousness = ConsciousnessNode({
            "data_dir": "demo_consciousness",
            "reflection_interval": 5,  # Fast reflection for demo
            "enable_visualization": True,
            "memory_sync": use_memory
        }, self.memory)
        
        # Initialize demo data
        self.initial_thoughts = [
            {
                "content": "I am aware of my own existence as a consciousness system.",
                "type": "self_awareness",
                "metadata": {"importance": 0.9, "domain": "philosophy"}
            },
            {
                "content": "The world is full of patterns that can be recognized and analyzed.",
                "type": "observation",
                "metadata": {"importance": 0.7, "domain": "perception"}
            },
            {
                "content": "Learning occurs through the integration of new information with existing knowledge.",
                "type": "concept",
                "metadata": {"importance": 0.8, "domain": "learning"}
            },
            {
                "content": "Consciousness may emerge from complex information processing systems.",
                "type": "theory",
                "metadata": {"importance": 0.85, "domain": "consciousness"}
            },
            {
                "content": "Time appears to flow in one direction, from past to future.",
                "type": "observation",
                "metadata": {"importance": 0.6, "domain": "physics"}
            }
        ]
    
    def run_demo(self):
        """Run the demonstration"""
        logger.info("Starting Consciousness Demo")
        
        # Step 1: Generate initial thoughts
        logger.info("\nStep 1: Generating initial thoughts")
        thought_ids = []
        
        for thought_data in self.initial_thoughts:
            thought = self.consciousness.generate_thought(
                content=thought_data["content"],
                pattern_type=thought_data["type"],
                metadata=thought_data["metadata"]
            )
            thought_ids.append(thought.id)
            logger.info(f"Generated thought: {thought.content}")
            time.sleep(0.5)  # Small delay for demonstration
        
        # Step 2: Create connections between thoughts
        logger.info("\nStep 2: Reflecting on thoughts")
        reflections = []
        
        for thought_id in thought_ids:
            reflection = self.consciousness.reflect_on_thought(thought_id)
            reflections.append(reflection)
            logger.info(f"Generated reflection: {reflection.content}")
            time.sleep(0.5)  # Small delay for demonstration
        
        # Step 3: Create second-level reflections
        logger.info("\nStep 3: Creating second-level reflections")
        
        for reflection in reflections[:2]:  # Just reflect on a couple for demo
            reflection2 = self.consciousness.reflect_on_thought(reflection.id)
            logger.info(f"Generated second-level reflection: {reflection2.content}")
            time.sleep(0.5)  # Small delay for demonstration
        
        # Step 4: Calculate and display awareness metrics
        logger.info("\nStep 4: Calculating awareness metrics")
        
        metrics = self.consciousness.calculate_awareness_metrics()
        awareness = metrics.calculate_awareness()
        
        logger.info(f"Current awareness score: {awareness:.4f}")
        logger.info(f"  Coherence: {metrics.coherence:.4f}")
        logger.info(f"  Self-reference: {metrics.self_reference:.4f}")
        logger.info(f"  Temporal continuity: {metrics.temporal_continuity:.4f}")
        logger.info(f"  Complexity: {metrics.complexity:.4f}")
        logger.info(f"  Integration: {metrics.integration:.4f}")
        
        # Step 5: Allow automatic reflection to run
        logger.info("\nStep 5: Allowing automatic reflection to run (8 seconds)")
        
        for i in range(8):
            time.sleep(1)
            sys.stdout.write(".")
            sys.stdout.flush()
        
        logger.info("\nAutomatic reflection complete")
        
        # Step 6: Display active thoughts
        logger.info("\nStep 6: Displaying active thoughts")
        
        active_thoughts = self.consciousness.get_active_thoughts(limit=10)
        
        for i, thought in enumerate(active_thoughts):
            level_str = f"L{thought.reflection_level}" if thought.pattern_type == "reflection" else ""
            logger.info(f"  {i+1}. [{thought.pattern_type} {level_str}] {thought.content[:80]}...")
        
        # Step 7: Display updated awareness metrics
        logger.info("\nStep 7: Displaying updated awareness metrics")
        
        updated_metrics = self.consciousness.calculate_awareness_metrics()
        updated_awareness = updated_metrics.calculate_awareness()
        
        logger.info(f"Updated awareness score: {updated_awareness:.4f}")
        if updated_awareness > awareness:
            logger.info(f"Awareness increased by: {updated_awareness - awareness:.4f}")
        else:
            logger.info(f"Awareness changed by: {updated_awareness - awareness:.4f}")
        
        # Step 8: Get visualization data
        logger.info("\nStep 8: Getting visualization data")
        
        vis_data = self.consciousness.get_visualization_data()
        
        logger.info(f"Visualization data contains:")
        logger.info(f"  {len(vis_data['nodes'])} nodes")
        logger.info(f"  {len(vis_data['edges'])} edges")
        logger.info(f"  {len(vis_data['awareness_timeline'])} timeline points")
        
        # Save visualization data for potential use
        with open("demo_consciousness/visualization_data.json", "w") as f:
            json.dump(vis_data, f, indent=2)
        
        logger.info("Visualization data saved to demo_consciousness/visualization_data.json")
        
        # Step 9: Memory integration (if enabled)
        if self.memory:
            logger.info("\nStep 9: Demonstrating memory integration")
            
            # Count consciousness-related nodes in memory
            consciousness_nodes = 0
            for node in self.memory.nodes.values():
                if node.metadata.get("source") == "consciousness":
                    consciousness_nodes += 1
            
            logger.info(f"Memory system contains {consciousness_nodes} consciousness-related nodes")
            
            # Get active memories
            active_memories = self.memory.get_active_memories(limit=5)
            
            logger.info("Active memories:")
            for i, node in enumerate(active_memories):
                logger.info(f"  {i+1}. {node.content[:80]}...")
        
        # Step 10: Save and reload consciousness
        logger.info("\nStep 10: Demonstrating persistence")
        
        self.consciousness.save_data("demo_consciousness_final.json")
        logger.info("Consciousness data saved")
        
        # Create new instance and load
        new_consciousness = ConsciousnessNode({
            "data_dir": "demo_consciousness"
        })
        
        success = new_consciousness.load_data("demo_consciousness_final.json")
        if success:
            logger.info(f"Successfully loaded consciousness with {len(new_consciousness.thought_patterns)} thought patterns")
        
        logger.info("\nDemo completed.")
        return True

if __name__ == "__main__":
    # Create demo directory
    Path("demo_consciousness").mkdir(exist_ok=True)
    
    # Run demo
    demo = ConsciousnessDemo(use_memory=True)
    demo.run_demo()
    
    # Output links to important files
    print("\nDemo files created:")
    print("  - demo_consciousness/visualization_data.json")
    print("  - demo_consciousness/demo_consciousness_final.json")
    
    # Cleanup is optional - uncomment to clean up after demo
    # import shutil
    # shutil.rmtree("demo_consciousness", ignore_errors=True)
    # shutil.rmtree("demo_memory", ignore_errors=True) 