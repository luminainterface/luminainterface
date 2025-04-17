#!/usr/bin/env python
"""
V7 Memory Node Example

This script demonstrates how to use the V7 Memory Node with visualization components.
It shows memory creation, retrieval, search, and decay functionality.
"""

import sys
import os
import time
import argparse
import logging
import random
from pathlib import Path

# Ensure the src directory is in the Python path
script_dir = Path(__file__).resolve().parent.parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import required modules
from src.v7.v6v7_connector import V6V7Connector
from src.v7.memory.memory_node import MemoryNode
from src.v7.ui.v7_visualization_connector import V7VisualizationConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("memory_node_example")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="V7 Memory Node Example")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--mock", action="store_true", help="Use mock backend components")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--memory-file", type=str, help="Path to persistent memory file")
    parser.add_argument("--decay-rate", type=float, default=0.1, help="Memory decay rate (0-1)")
    return parser.parse_args()

def configure_logging(debug):
    """Configure logging based on arguments."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

def create_test_memories(memory_node, count=10):
    """Create test memories in the memory node."""
    memory_types = ["fact", "experience", "concept", "belief", "goal"]
    sources = ["user", "system", "observation", "inference"]
    
    logger.info(f"Creating {count} test memories...")
    
    for i in range(count):
        memory_type = random.choice(memory_types)
        source = random.choice(sources)
        importance = random.uniform(0.1, 1.0)
        
        # Create memory content based on type
        if memory_type == "fact":
            content = f"Fact #{i}: The sky is blue because of Rayleigh scattering."
        elif memory_type == "experience":
            content = f"Experience #{i}: User asked about neural networks yesterday."
        elif memory_type == "concept":
            content = f"Concept #{i}: Neural networks are computational models inspired by biological neurons."
        elif memory_type == "belief":
            content = f"Belief #{i}: The system should prioritize user requests over background tasks."
        else:  # goal
            content = f"Goal #{i}: Help the user understand machine learning concepts."
            
        # Add the memory
        memory_id = memory_node.add_memory(
            content=content,
            memory_type=memory_type,
            source=source,
            importance=importance,
            metadata={"example_id": i, "created_at": time.time()}
        )
        
        logger.debug(f"Created memory {memory_id}: {memory_type} - {content[:30]}...")
        
        # Small delay for visualization to show progressive creation
        time.sleep(0.2)
    
    logger.info(f"Created {count} test memories successfully")
    return memory_node.get_all_memories()

def demonstrate_memory_operations(memory_node):
    """Demonstrate various memory operations."""
    logger.info("Demonstrating memory operations...")
    
    # Search for memories
    logger.info("Searching for memories containing 'neural'...")
    neural_memories = memory_node.search_memories("neural")
    logger.info(f"Found {len(neural_memories)} memories about neural networks")
    
    # Get memories by type
    logger.info("Retrieving memories by type...")
    facts = memory_node.get_memories_by_type("fact")
    logger.info(f"Found {len(facts)} fact memories")
    
    # Add a high importance memory
    logger.info("Adding a high importance memory...")
    important_id = memory_node.add_memory(
        content="IMPORTANT: The system needs to be updated by next week",
        memory_type="goal",
        importance=0.95,
        source="user"
    )
    
    # Update a memory
    logger.info("Updating a memory...")
    if facts:
        first_fact = facts[0]
        memory_node.update_memory(
            memory_id=first_fact["id"],
            content=first_fact["content"] + " This fact has been verified.",
            importance=first_fact["importance"] + 0.1
        )
        logger.info(f"Updated memory: {first_fact['id']}")
    
    # Associate memories
    logger.info("Creating memory associations...")
    if len(neural_memories) >= 2:
        memory_node.associate_memories(
            neural_memories[0]["id"], 
            neural_memories[1]["id"],
            association_type="related",
            strength=0.8
        )
        logger.info(f"Associated memories: {neural_memories[0]['id']} and {neural_memories[1]['id']}")
    
    return important_id

def demonstrate_decay(memory_node, iterations=5, highlight_id=None):
    """Demonstrate memory decay over time."""
    logger.info(f"Demonstrating memory decay over {iterations} iterations...")
    
    for i in range(iterations):
        logger.info(f"Decay iteration {i+1}/{iterations}")
        
        # Access some memories to reinforce them
        if i % 2 == 0 and highlight_id:
            memory = memory_node.get_memory(highlight_id)
            logger.info(f"Accessing important memory: {memory['content'][:30]}...")
        
        # Trigger decay process
        memory_node.process_decay()
        
        # Get stats after decay
        stats = memory_node.get_statistics()
        logger.info(f"Memory stats after decay: {stats['total_memories']} memories, "
                   f"avg importance: {stats['average_importance']:.2f}")
        
        # Pause to allow visualization to update
        time.sleep(1.5)
    
    return memory_node.get_all_memories()

def main():
    """Main function to run the memory node example."""
    args = parse_arguments()
    configure_logging(args.debug)
    
    logger.info("Starting V7 Memory Node Example")
    
    # Initialize the V6V7 Connector
    connector = V6V7Connector(mock_mode=args.mock)
    logger.info("Initialized V6V7 Connector")
    
    # Initialize the Memory Node
    memory_file = args.memory_file or "memory_example.json"
    memory_node = MemoryNode(
        persistence_file=memory_file,
        config={
            "decay_rate": args.decay_rate,
            "decay_interval": 1.0,  # seconds
            "importance_threshold": 0.1,
            "enable_persistence": True
        }
    )
    connector.register_component("memory_node", memory_node)
    logger.info(f"Initialized Memory Node with persistence file: {memory_file}")
    
    # Initialize the visualization connector
    viz_connector = V7VisualizationConnector(
        v6v7_connector=connector,
        config={
            "enable_memory_visualization": True,
            "memory_update_interval": 1.0,
            "max_visualized_memories": 50
        }
    )
    logger.info("Initialized Visualization Connector")
    
    # If GUI is enabled, set up the visualization
    if not args.no_gui:
        try:
            # Try to import and initialize the Qt application
            from PySide6 import QtWidgets, QtCore
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            
            # Create a main window
            main_window = QtWidgets.QMainWindow()
            main_window.setWindowTitle("V7 Memory Node Visualization")
            main_window.resize(1200, 800)
            
            # Create central widget with tabs
            central_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central_widget)
            tabs = QtWidgets.QTabWidget()
            layout.addWidget(tabs)
            
            # Get memory visualization widget
            memory_viz_widget = viz_connector.get_memory_visualization_widget()
            if memory_viz_widget:
                tabs.addTab(memory_viz_widget, "Memory Visualization")
                logger.info("Added memory visualization widget to UI")
            else:
                logger.warning("Memory visualization widget not available")
                
            # Add stats display
            stats_widget = QtWidgets.QTextEdit()
            stats_widget.setReadOnly(True)
            tabs.addTab(stats_widget, "Memory Stats")
            
            # Setup update timer for stats
            def update_stats():
                if memory_node:
                    stats = memory_node.get_statistics()
                    memories = memory_node.get_all_memories()
                    text = f"<h2>Memory Statistics</h2>"
                    text += f"<p>Total memories: {stats['total_memories']}</p>"
                    text += f"<p>Average importance: {stats['average_importance']:.2f}</p>"
                    text += f"<p>Memory types: {stats['memory_types']}</p>"
                    text += "<h3>Recent Memories</h3>"
                    
                    # Sort by recency and show top 10
                    recent = sorted(memories, key=lambda m: m.get('last_accessed', 0), reverse=True)[:10]
                    for m in recent:
                        text += f"<p><b>{m['memory_type']}:</b> {m['content'][:50]}... "
                        text += f"(Importance: {m['importance']:.2f})</p>"
                    
                    stats_widget.setHtml(text)
            
            stats_timer = QtCore.QTimer()
            stats_timer.timeout.connect(update_stats)
            stats_timer.start(1000)  # Update every second
            
            # Show main window
            main_window.setCentralWidget(central_widget)
            main_window.show()
            
            logger.info("GUI initialized successfully")
        except ImportError as e:
            logger.error(f"Could not initialize GUI: {e}")
            args.no_gui = True
    
    # Create test memories
    all_memories = create_test_memories(memory_node, count=20)
    logger.info(f"Created {len(all_memories)} test memories")
    
    # Demonstrate memory operations
    important_id = demonstrate_memory_operations(memory_node)
    
    # Demonstrate memory decay
    final_memories = demonstrate_decay(memory_node, iterations=10, highlight_id=important_id)
    
    logger.info(f"Final memory count: {len(final_memories)}")
    for memory in sorted(final_memories, key=lambda m: m["importance"], reverse=True)[:5]:
        logger.info(f"Top memory: {memory['content'][:50]}... (Importance: {memory['importance']:.2f})")
    
    # If GUI is enabled, start the event loop
    if not args.no_gui:
        try:
            logger.info("Starting GUI event loop. Press Ctrl+C to exit.")
            sys.exit(app.exec())
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, exiting...")
        except Exception as e:
            logger.error(f"Error in GUI event loop: {e}")
    
    logger.info("V7 Memory Node Example completed successfully")

if __name__ == "__main__":
    main() 