#!/usr/bin/env python3
"""
AutoWiki Plugin Runner

This script runs the AutoWiki Plugin, optionally using data from
the monday.md conversation nodes to seed the knowledge base.
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
from plugins.auto_wiki_plugin import get_auto_wiki_plugin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoWikiPluginRunner")

class AutoWikiPluginRunner:
    """Runner for the AutoWiki Plugin"""
    
    def __init__(self, plugin_id="autowiki_runner", mock_mode=True):
        """
        Initialize the runner
        
        Args:
            plugin_id: Unique identifier for the plugin instance
            mock_mode: Whether to use simulated knowledge acquisition
        """
        self.plugin = get_auto_wiki_plugin(plugin_id=plugin_id, mock_mode=mock_mode)
        self.nodes_data = None
        self.seeded = False
        
        logger.info(f"AutoWiki Plugin Runner initialized with ID: {plugin_id}, mock_mode: {mock_mode}")
    
    def load_seed_data(self, nodes_file):
        """
        Load conversation nodes data for seeding the knowledge base
        
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
            
            logger.info(f"Loaded nodes data with {len(self.nodes_data.get('nodes', []))} nodes for seeding")
            return True
            
        except Exception as e:
            logger.error(f"Error loading nodes data: {e}")
            return False
    
    def seed_knowledge_base(self):
        """Seed the knowledge base with data from conversation nodes"""
        if not self.nodes_data or self.seeded:
            return False
        
        nodes = self.nodes_data.get("nodes", [])
        if not nodes:
            return False
        
        # Extract significant nodes for knowledge acquisition
        significant_nodes = [
            node for node in nodes 
            if node["metrics"]["consciousness_level"] > 0.7 or 
               node["metrics"]["neural_linguistic_score"] > 0.7
        ]
        
        logger.info(f"Found {len(significant_nodes)} significant nodes for knowledge seeding")
        
        # Process insights from the nodes data
        insights = self.nodes_data.get("insights", [])
        for insight in insights:
            topic = f"insight_{insight['type']}_{random.randint(1000, 9999)}"
            logger.info(f"Adding insight topic: {topic}")
            self.plugin.add_topic(topic, priority=3)
        
        # Process patterns from the nodes data
        patterns = self.nodes_data.get("patterns", [])
        for pattern in patterns:
            topic = f"pattern_{pattern['tag']}_{random.randint(1000, 9999)}"
            logger.info(f"Adding pattern topic: {topic}")
            self.plugin.add_topic(topic, priority=2)
        
        # Add significant nodes as topics
        for node in significant_nodes:
            # Create a topic from the node tags
            tags = node.get("tags", [])
            if tags:
                tag = tags[0]  # Use the first tag as a topic identifier
                topic = f"{tag}_{random.randint(1000, 9999)}"
                logger.info(f"Adding node topic: {topic}")
                self.plugin.add_topic(topic, priority=1)
        
        self.seeded = True
        return True
    
    def run(self):
        """Run the autowiki plugin"""
        # Start the acquisition process
        self.plugin.start_acquisition()
        
        # Seed the knowledge base if we have nodes data
        if self.nodes_data and not self.seeded:
            self.seed_knowledge_base()
        
        try:
            # Main loop
            logger.info("AutoWiki Plugin running. Press Ctrl+C to stop.")
            while True:
                # Check plugin status
                status = self.plugin.get_status()
                logger.info(f"Queue size: {status['queue_size']}, "
                           f"Knowledge base size: {status['knowledge_base_size']}")
                
                # Get a report of recent acquisitions
                recent = status.get("recent_acquisitions", [])
                if recent:
                    logger.info("Recent knowledge acquisitions:")
                    for item in recent:
                        logger.info(f"  - {item['topic']} (verification: {item['verification']:.2f})")
                
                # Sleep to prevent CPU hogging
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        finally:
            # Stop acquisition
            self.plugin.stop_acquisition()
            logger.info("AutoWiki Plugin stopped")
    
    def query_knowledge(self, topic):
        """
        Query the knowledge base for information on a topic
        
        Args:
            topic: Topic to query
            
        Returns:
            dict: Knowledge data
        """
        return self.plugin.get_knowledge(topic)
    
    def generate_learning_pathway(self, domain=None):
        """
        Generate a learning pathway for a knowledge domain
        
        Args:
            domain: Knowledge domain to create pathway for
            
        Returns:
            dict: Learning pathway data
        """
        return self.plugin.get_learning_pathway(domain)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run the AutoWiki Plugin")
    parser.add_argument("--plugin_id", default="autowiki_runner", help="Unique identifier for the plugin instance")
    parser.add_argument("--mock_mode", type=lambda x: x.lower() == 'true', default=True, help="Whether to use simulated knowledge acquisition")
    parser.add_argument("--seed_from", help="Name of the nodes data file to seed the knowledge base (without .json extension)")
    parser.add_argument("--query", help="Topic to query from the knowledge base")
    parser.add_argument("--pathway", help="Knowledge domain to generate a learning pathway for")
    
    args = parser.parse_args()
    
    runner = AutoWikiPluginRunner(plugin_id=args.plugin_id, mock_mode=args.mock_mode)
    
    # Load seed data if specified
    if args.seed_from:
        runner.load_seed_data(args.seed_from)
    
    # Handle a single query if specified
    if args.query:
        result = runner.query_knowledge(args.query)
        print(f"Query result for '{args.query}':")
        print(json.dumps(result, indent=2))
        return
    
    # Generate a learning pathway if specified
    if args.pathway:
        pathway = runner.generate_learning_pathway(args.pathway)
        print(f"Learning pathway for '{args.pathway}':")
        print(json.dumps(pathway, indent=2))
        return
    
    # Run the plugin normally
    runner.run()


if __name__ == "__main__":
    main() 