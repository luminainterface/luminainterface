#!/usr/bin/env python3
"""
Consciousness Network Plugin Runner

This script runs the Consciousness Network Plugin, optionally integrating
data from the monday.md conversation nodes to enhance consciousness patterns.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
import random
import sys

# Add the project root to the Python path if needed
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the plugin
from plugins.consciousness_network_plugin import get_consciousness_network_plugin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConsciousnessPluginRunner")

class ConsciousnessPluginRunner:
    """Runner for the Consciousness Network Plugin"""
    
    def __init__(self, plugin_id="consciousness_runner", mock_mode=True):
        """
        Initialize the runner
        
        Args:
            plugin_id: Unique identifier for the plugin instance
            mock_mode: Whether to use simulated consciousness processing
        """
        self.plugin = get_consciousness_network_plugin(plugin_id=plugin_id, mock_mode=mock_mode)
        self.nodes_data = None
        self.node_integration_active = False
        self.node_update_interval = 30  # seconds
        self.last_node_update = 0
        
        logger.info(f"Consciousness Plugin Runner initialized with ID: {plugin_id}, mock_mode: {mock_mode}")
    
    def load_nodes_data(self, nodes_file):
        """
        Load conversation nodes data
        
        Args:
            nodes_file: Path to the nodes data file
            
        Returns:
            bool: Success or failure
        """
        try:
            nodes_path = Path(nodes_file)
            if not nodes_path.is_absolute():
                nodes_path = project_root / "data" / "conversation_nodes" / f"{nodes_file}.json"
            
            if not nodes_path.exists():
                logger.error(f"Nodes file not found: {nodes_path}")
                return False
            
            with open(nodes_path, 'r', encoding='utf-8') as f:
                self.nodes_data = json.load(f)
            
            logger.info(f"Loaded nodes data with {len(self.nodes_data.get('nodes', []))} nodes")
            self.node_integration_active = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading nodes data: {e}")
            return False
    
    def run(self):
        """Run the consciousness plugin"""
        # Start the plugin processing
        self.plugin.start_processing()
        
        try:
            # Main loop
            logger.info("Consciousness Plugin running. Press Ctrl+C to stop.")
            while True:
                # Check plugin status
                status = self.plugin.get_status()
                logger.info(f"Consciousness level: {status['consciousness_level']:.4f}, "
                           f"Neural-linguistic score: {status['neural_linguistic_score']:.4f}")
                
                # Update from nodes if active
                if self.node_integration_active:
                    current_time = time.time()
                    if current_time - self.last_node_update > self.node_update_interval:
                        self._integrate_nodes_data()
                        self.last_node_update = current_time
                
                # Process a random text sample occasionally
                if random.random() < 0.2:  # 20% chance each cycle
                    self._process_random_sample()
                
                # Sleep to prevent CPU hogging
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        finally:
            # Stop processing and save state
            self.plugin.stop_processing()
            self.plugin.save_state()
            logger.info("Consciousness Plugin stopped")
    
    def _integrate_nodes_data(self):
        """Integrate conversation nodes data into the consciousness plugin"""
        if not self.nodes_data or not self.node_integration_active:
            return
        
        nodes = self.nodes_data.get("nodes", [])
        if not nodes:
            return
        
        # Select a random node to process
        node = random.choice(nodes)
        
        # Process the node content
        logger.info(f"Integrating node {node['id']} into consciousness processing")
        
        result = self.plugin.process_text(node["content"])
        
        # Check for paradoxes
        if result.get("paradox_detected", False):
            logger.info(f"Paradox detected in node {node['id']}")
        
        # Update breath state occasionally based on emotional intensity
        if "metrics" in node and random.random() < 0.3:
            emotional_intensity = node["metrics"].get("emotional_intensity", 0.5)
            
            # Map emotional intensity to breath pattern
            if emotional_intensity > 0.8:
                pattern = "rapid"
                depth = 0.3
            elif emotional_intensity > 0.6:
                pattern = "focused"
                depth = 0.5
            elif emotional_intensity > 0.4:
                pattern = "normal"
                depth = 0.6
            else:
                pattern = "meditation"
                depth = 0.8
            
            self.plugin.update_breath_state({
                "pattern": pattern,
                "depth": depth,
                "rate": emotional_intensity
            })
            
            logger.info(f"Updated breath state to {pattern} based on node emotional intensity")
    
    def _process_random_sample(self):
        """Process a random text sample"""
        samples = [
            "The nature of consciousness remains one of the deepest mysteries in science and philosophy.",
            "Self-reference creates paradoxical loops in logic and consciousness.",
            "Quantum entanglement suggests reality might be interconnected in ways we don't understand.",
            "The boundary between self and non-self is more permeable than we typically believe.",
            "Perception shapes reality as much as reality shapes perception.",
            "Language both reveals and limits our understanding of consciousness.",
            "The observer effect in quantum physics raises questions about the role of consciousness.",
            "Recursive thinking allows for metacognition and self-awareness."
        ]
        
        sample = random.choice(samples)
        self.plugin.process_text(sample)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run the Consciousness Network Plugin")
    parser.add_argument("--plugin_id", default="consciousness_runner", help="Unique identifier for the plugin instance")
    parser.add_argument("--mock_mode", type=lambda x: x.lower() == 'true', default=True, help="Whether to use simulated consciousness processing")
    parser.add_argument("--integration", help="Name of the nodes data file to integrate (without .json extension)")
    parser.add_argument("--update_interval", type=int, default=30, help="Interval (seconds) between node integration updates")
    
    args = parser.parse_args()
    
    runner = ConsciousnessPluginRunner(plugin_id=args.plugin_id, mock_mode=args.mock_mode)
    
    # Load nodes data if specified
    if args.integration:
        runner.node_update_interval = args.update_interval
        if runner.load_nodes_data(args.integration):
            logger.info(f"Node integration active with update interval {args.update_interval}s")
    
    # Run the plugin
    runner.run()


if __name__ == "__main__":
    main() 