#!/usr/bin/env python
"""
AutoWiki Plugin for LUMINA V7
Provides auto-learning wiki functionality for the V7 system
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import random
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autowiki-plugin")

class AutoWikiPlugin:
    """AutoWiki plugin for V7 systems"""
    
    SAMPLE_TOPICS = [
        "Neural Networks", "Consciousness", "Artificial Intelligence",
        "Machine Learning", "Holographic Interfaces", "Natural Language Processing",
        "Knowledge Graphs", "Memory Systems", "Self-Awareness", "Learning Algorithms"
    ]
    
    def __init__(self, data_path="data/autowiki", auto_fetch=True):
        """Initialize the AutoWiki plugin"""
        self.data_path = Path(data_path)
        self.queue_file = self.data_path / "autowiki_queue.json"
        self.history_file = self.data_path / "autowiki_history.json"
        self.auto_fetch = auto_fetch
        self.running = False
        self.thread = None
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize queue and history
        self.topic_queue = self._load_queue()
        self.topic_history = self._load_history()
        
        logger.info(f"AutoWiki plugin initialized with data path: {data_path}")
        logger.info(f"Auto-fetch: {auto_fetch}")
    
    def start(self):
        """Start the AutoWiki plugin"""
        if self.running:
            logger.warning("AutoWiki plugin already running")
            return
            
        self.running = True
        
        # Only start the thread if auto-fetch is enabled
        if self.auto_fetch:
            self.thread = threading.Thread(
                target=self._run_auto_fetch,
                daemon=True,
                name="AutoWikiThread"
            )
            self.thread.start()
            logger.info("AutoWiki auto-fetching thread started")
        
        logger.info("AutoWiki plugin started")
    
    def stop(self):
        """Stop the AutoWiki plugin"""
        if not self.running:
            logger.warning("AutoWiki plugin not running")
            return
            
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.info("AutoWiki plugin stopped")
    
    def add_topic(self, topic):
        """Add a topic to the queue"""
        # Skip if topic is already in queue or history
        if topic in self.topic_queue or topic in self.topic_history:
            logger.debug(f"Topic already processed or queued: {topic}")
            return False
            
        # Add to queue
        self.topic_queue.append(topic)
        
        # Save queue
        self._save_queue()
        
        logger.info(f"Topic added to queue: {topic}")
        return True
    
    def fetch_topic(self, topic):
        """Fetch information about a topic"""
        # Skip if topic is already in history
        if topic in self.topic_history:
            logger.debug(f"Topic already in history: {topic}")
            return self.topic_history[topic]
        
        # Simulate fetching information
        logger.info(f"Fetching information about: {topic}")
        
        # Mock data - in a real implementation, this would fetch from external sources
        topic_info = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "summary": f"Information about {topic}. This is a simulated summary that would normally contain actual information fetched from reliable sources.",
            "success": True,
            "source": "simulated",
            "categories": random.sample(["Science", "Technology", "AI", "Computing", "Philosophy", "Consciousness"], k=2)
        }
        
        # Add to history
        self.topic_history[topic] = topic_info
        
        # Remove from queue if present
        if topic in self.topic_queue:
            self.topic_queue.remove(topic)
        
        # Save history and queue
        self._save_history()
        self._save_queue()
        
        return topic_info
    
    def _run_auto_fetch(self):
        """Run the auto-fetch thread"""
        logger.info("AutoWiki auto-fetch thread running")
        
        # Add some sample topics to start with
        for topic in self.SAMPLE_TOPICS:
            self.add_topic(topic)
        
        while self.running:
            try:
                # Process queue if there are topics
                if self.topic_queue:
                    # Get the next topic
                    topic = self.topic_queue[0]
                    
                    # Fetch information
                    self.fetch_topic(topic)
                    
                    # Sleep briefly to avoid hammering resources
                    time.sleep(2)
                else:
                    # No topics in queue, sleep longer
                    time.sleep(10)
            except Exception as e:
                logger.error(f"Error in auto-fetch thread: {e}")
                time.sleep(30)  # Wait longer after error
    
    def _load_queue(self):
        """Load the topic queue from file"""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error loading queue file {self.queue_file}, creating new")
        
        # Return empty list if file doesn't exist or couldn't be loaded
        return []
    
    def _save_queue(self):
        """Save the topic queue to file"""
        try:
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(self.topic_queue, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving queue: {e}")
    
    def _load_history(self):
        """Load the topic history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error loading history file {self.history_file}, creating new")
        
        # Return empty dict if file doesn't exist or couldn't be loaded
        return {}
    
    def _save_history(self):
        """Save the topic history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.topic_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V7 AutoWiki Plugin")
    parser.add_argument(
        "--data-path", 
        default="data/autowiki",
        help="Path to the AutoWiki data directory"
    )
    parser.add_argument(
        "--no-auto-fetch", 
        action="store_true",
        help="Disable automatic fetching of information"
    )
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create AutoWiki plugin
    autowiki = AutoWikiPlugin(
        data_path=args.data_path,
        auto_fetch=not args.no_auto_fetch
    )
    
    try:
        # Start plugin
        autowiki.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("AutoWiki plugin interrupted by user")
    finally:
        autowiki.stop()

if __name__ == "__main__":
    main() 