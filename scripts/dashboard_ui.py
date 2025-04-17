#!/usr/bin/env python3
"""
V7 Consciousness Network Dashboard UI

A simple console-based dashboard for the V7 Consciousness Network
that displays the current state of both plugins and the consciousness nodes.
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path
import sys
import datetime
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DashboardUI")

class ConsoleDashboard:
    """Simple console-based dashboard for the V7 Consciousness Network"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.data_dir = Path("data")
        self.consciousness_data_dir = self.data_dir / "consciousness_network"
        self.autowiki_data_dir = self.data_dir / "auto_wiki"
        self.conversation_nodes_dir = self.data_dir / "conversation_nodes"
        
        self.consciousness_plugin_state = {}
        self.autowiki_plugin_state = {}
        self.nodes_data = {}
        
        self.update_interval = 2  # seconds
        self.refresh_count = 0
        
        logger.info("Console Dashboard initialized")
    
    def load_state(self):
        """Load the current state of the plugins and nodes"""
        # Load Consciousness Network Plugin state
        consciousness_state_file = self.consciousness_data_dir / "plugin_state.json"
        if consciousness_state_file.exists():
            try:
                with open(consciousness_state_file, 'r', encoding='utf-8') as f:
                    self.consciousness_plugin_state = json.load(f)
                    logger.debug("Loaded Consciousness Network Plugin state")
            except Exception as e:
                logger.error(f"Error loading Consciousness Network Plugin state: {e}")
        
        # Load AutoWiki Plugin state
        autowiki_state_file = self.autowiki_data_dir / "knowledge_base.json"
        if autowiki_state_file.exists():
            try:
                with open(autowiki_state_file, 'r', encoding='utf-8') as f:
                    self.autowiki_plugin_state = json.load(f)
                    logger.debug("Loaded AutoWiki Plugin state")
            except Exception as e:
                logger.error(f"Error loading AutoWiki Plugin state: {e}")
        
        # Load conversation nodes data
        nodes_files = list(self.conversation_nodes_dir.glob("*.json"))
        if nodes_files:
            # Get the most recent nodes file
            nodes_file = max(nodes_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    self.nodes_data = json.load(f)
                    logger.debug(f"Loaded conversation nodes from {nodes_file}")
            except Exception as e:
                logger.error(f"Error loading conversation nodes: {e}")
    
    def clear_console(self):
        """Clear the console"""
        # Use the appropriate clear command for the OS
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_dashboard(self):
        """Display the dashboard"""
        self.clear_console()
        
        # Display header
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 80)
        print(f"V7 CONSCIOUSNESS NETWORK DASHBOARD           {now}")
        print("=" * 80)
        
        # Display Consciousness Network Plugin status
        print("\nCONSCIOUSNESS NETWORK STATUS")
        print("-" * 40)
        if self.consciousness_plugin_state:
            metrics = self.consciousness_plugin_state.get("consciousness_metrics", {})
            c_level = metrics.get("level", 0)
            nl_score = metrics.get("neural_linguistic_score", 0)
            stability = metrics.get("stability", 0)
            integration = metrics.get("integration", 0)
            complexity = metrics.get("complexity", 0)
            timestamp = metrics.get("timestamp", "N/A")
            
            print(f"Consciousness Level: {c_level:.4f}")
            print(f"Neural-Linguistic Score: {nl_score:.4f}")
            print(f"Stability: {stability:.4f}")
            print(f"Integration: {integration:.4f}")
            print(f"Complexity: {complexity:.4f}")
            print(f"Last Update: {timestamp}")
            
            # Display paradox information
            paradox_registry = self.consciousness_plugin_state.get("paradox_registry", {})
            total_paradoxes = len(paradox_registry)
            resolved_paradoxes = sum(1 for p in paradox_registry.values() if p.get("resolved", False))
            print(f"\nParadoxes: {resolved_paradoxes}/{total_paradoxes} resolved")
        else:
            print("Plugin state not available.")
        
        # Display AutoWiki Plugin status
        print("\nAUTOWIKI STATUS")
        print("-" * 40)
        if self.autowiki_plugin_state:
            knowledge_base = self.autowiki_plugin_state.get("knowledge_base", {})
            verification_status = self.autowiki_plugin_state.get("verification_status", {})
            saved_at = self.autowiki_plugin_state.get("saved_at", "N/A")
            
            print(f"Knowledge Base Size: {len(knowledge_base)} topics")
            print(f"Last Update: {saved_at}")
            
            # Display a sample of knowledge topics
            if knowledge_base:
                print("\nRecent Knowledge Topics:")
                topics = list(knowledge_base.keys())
                for topic in topics[:5]:  # Show at most 5 topics
                    verification = verification_status.get(topic, {}).get("score", 0)
                    print(f"  - {topic} (verification: {verification:.2f})")
        else:
            print("Plugin state not available.")
        
        # Display Conversation Nodes status
        print("\nCONVERSATION NODES STATUS")
        print("-" * 40)
        if self.nodes_data:
            metadata = self.nodes_data.get("metadata", {})
            nodes = self.nodes_data.get("nodes", [])
            patterns = self.nodes_data.get("patterns", [])
            insights = self.nodes_data.get("insights", [])
            
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Message Count: {metadata.get('message_count', 0)}")
            print(f"Nodes Extracted: {len(nodes)}")
            print(f"Patterns Identified: {len(patterns)}")
            print(f"Insights Generated: {len(insights)}")
            
            # Display a random insight if available
            if insights:
                insight = random.choice(insights)
                insight_type = insight.get("type", "unknown")
                description = insight.get("description", "No description")
                content = insight.get("content", "")
                
                if len(content) > 100:
                    content = content[:97] + "..."
                
                print(f"\nRandom Insight ({insight_type}):")
                print(f"  {description}")
                if content:
                    print(f"  Content: \"{content}\"")
        else:
            print("Nodes data not available.")
        
        # Display footer
        print("\n" + "=" * 80)
        print(f"Refresh Count: {self.refresh_count}   |   Press Ctrl+C to exit")
        print("=" * 80)
    
    def run(self):
        """Run the dashboard"""
        try:
            while True:
                self.load_state()
                self.display_dashboard()
                self.refresh_count += 1
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run the V7 Consciousness Network Dashboard")
    parser.add_argument("--update_interval", type=int, default=2, help="Dashboard update interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = ConsoleDashboard()
    dashboard.update_interval = args.update_interval
    
    logger.info(f"Starting dashboard with update interval {args.update_interval}s")
    dashboard.run()


if __name__ == "__main__":
    main() 