#!/usr/bin/env python3
"""
AutoWiki System for LUMINA V7

This module provides an automated wiki content fetcher that
populates the system's learning dictionary with information
from external sources.
"""

import os
import sys
import time
import json
import logging
import requests
import threading
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from urllib.parse import quote
from bs4 import BeautifulSoup
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import our Mistral integration
try:
    from src.v7.mistral_integration import MistralEnhancedSystem
    MISTRAL_INTEGRATION_AVAILABLE = True
except ImportError:
    MISTRAL_INTEGRATION_AVAILABLE = False
    logger.warning("MistralEnhancedSystem not available")

class AutoWiki:
    """
    Automatic wiki content fetcher that feeds the learning dictionary.
    
    This class provides methods to fetch information from various sources
    and add it to the system's learning dictionary.
    """
    
    QUEUE_FILE = "data/v7/autowiki_queue.json"
    HISTORY_FILE = "data/v7/autowiki_history.json"
    
    def __init__(
        self, 
        mistral_system=None,
        data_dir: str = "data/v7/autowiki",
        auto_fetch: bool = True,
        mock_mode: bool = False
    ):
        """
        Initialize the AutoWiki system.
        
        Args:
            mistral_system: MistralEnhancedSystem instance to feed data to
            data_dir: Directory for storing fetched data
            auto_fetch: Whether to automatically fetch daily content
            mock_mode: Whether to operate in mock mode (no external requests)
        """
        self.mistral_system = mistral_system
        self.data_dir = data_dir
        self.auto_fetch = auto_fetch
        self.mock_mode = mock_mode or not MISTRAL_INTEGRATION_AVAILABLE
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing data
        self._load_queue()
        self._load_history()
        
        # Topics queue
        self.topics_queue = []
        self.topics_file = os.path.join(self.data_dir, "topics_queue.json")
        self._load_topics_queue()
        
        # Start automatic fetching if enabled
        if self.auto_fetch:
            self.start_auto_fetching()
    
    def _load_queue(self) -> None:
        """Load the queue from the queue file."""
        try:
            if os.path.exists(self.QUEUE_FILE):
                with open(self.QUEUE_FILE, 'r', encoding='utf-8') as f:
                    self.topics_queue = json.load(f)
                logger.info(f"Loaded {len(self.topics_queue)} topics from queue file")
        except Exception as e:
            logger.error(f"Error loading queue: {str(e)}")
            self.topics_queue = []
    
    def _save_queue(self) -> None:
        """Save the queue to the queue file."""
        try:
            with open(self.QUEUE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.topics_queue, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving queue: {str(e)}")
    
    def _load_history(self) -> None:
        """Load fetch history from the history file."""
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded history with {len(self.history)} topics")
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            self.history = {}
    
    def _save_history(self) -> None:
        """Save fetch history to the history file."""
        try:
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
    
    def _load_topics_queue(self):
        """Load topics queue from file"""
        try:
            if os.path.exists(self.topics_file):
                with open(self.topics_file, 'r') as f:
                    self.topics_queue = json.load(f)
                logger.info(f"Loaded {len(self.topics_queue)} topics in queue")
        except Exception as e:
            logger.error(f"Error loading topics queue: {e}")
    
    def _save_topics_queue(self):
        """Save topics queue to file"""
        try:
            with open(self.topics_file, 'w') as f:
                json.dump(self.topics_queue, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving topics queue: {e}")
    
    def add_topic(self, topic: str) -> bool:
        """
        Add a topic to the fetch queue
        
        Args:
            topic: The topic to add
            
        Returns:
            Success status
        """
        topic = topic.strip()
        if not topic:
            return False
            
        # Don't add duplicates
        if topic.lower() in [t.lower() for t in self.topics_queue]:
            logger.info(f"Topic '{topic}' already in queue")
            return False
            
        self.topics_queue.append(topic)
        self._save_topics_queue()
        logger.info(f"Added topic '{topic}' to queue")
        return True
    
    def add_topics(self, topics: List[str]) -> int:
        """
        Add multiple topics to the fetch queue
        
        Args:
            topics: List of topics to add
            
        Returns:
            Number of topics added
        """
        added = 0
        for topic in topics:
            if self.add_topic(topic):
                added += 1
        return added
    
    def load_topics_from_file(self, filename: str) -> int:
        """
        Load topics from a file, one per line.
        
        Args:
            filename: Path to the file containing topics.
            
        Returns:
            int: Number of topics successfully added.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                topics = [line.strip() for line in f if line.strip()]
            return self.add_topics(topics)
        except Exception as e:
            logger.error(f"Error loading topics from {filename}: {str(e)}")
            return 0
    
    def fetch_topic(self, topic: str) -> Dict[str, Any]:
        """
        Fetch information about a topic from available sources
        
        Args:
            topic: The topic to fetch information about
            
        Returns:
            Dictionary with fetched information
        """
        result = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "content": "",
            "source": "",
            "size": 0
        }
        
        # Increment fetch counter
        self.history[topic] = {
            "last_fetch": datetime.now().isoformat(),
            "success": False
        }
        
        try:
            # Try Wikipedia first
            wiki_content = self._fetch_from_wikipedia(topic)
            if wiki_content:
                result["content"] = wiki_content
                result["source"] = "wikipedia"
                result["size"] = len(wiki_content)
                result["success"] = True
                
                # Save to file
                self._save_topic_content(topic, wiki_content, "wikipedia")
                
                # Update history
                self.history[topic]["success"] = True
                self._save_history()
                
                # Add to learning dictionary if we have a Mistral system
                if self.mistral_system and hasattr(self.mistral_system, 'process_autowiki'):
                    self.mistral_system.process_autowiki(topic, wiki_content)
                
                return result
            
            # If Wikipedia fails, try alternative sources
            # For now, we'll just log the failure
            logger.warning(f"Failed to fetch information about {topic} from Wikipedia")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching topic {topic}: {e}")
            return result
    
    def _fetch_from_wikipedia(self, topic: str) -> str:
        """
        Fetch information from Wikipedia
        
        Args:
            topic: The topic to fetch
            
        Returns:
            Content as text, or empty string if failed
        """
        if self.mock_mode:
            # Generate mock content in mock mode
            return self._generate_mock_content(topic)
        
        try:
            # Encode topic for URL
            encoded_topic = quote(topic)
            
            # Use Wikipedia's API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
            response = requests.get(url, headers={
                'User-Agent': 'LUMINA_V7/1.0 (Learning system; contact@example.com)'
            })
            
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    # For more detailed content, we could make a second request to:
                    # https://en.wikipedia.org/api/rest_v1/page/html/{encoded_topic}
                    return data['extract']
            
            return ""
            
        except Exception as e:
            logger.error(f"Error fetching from Wikipedia: {e}")
            return ""
    
    def _generate_mock_content(self, topic: str) -> Dict[str, Any]:
        """
        Generate mock content for a topic when in mock mode.
        
        Args:
            topic: The topic to generate mock content for.
            
        Returns:
            dict: The generated mock content and metadata.
        """
        mock_summaries = {
            "artificial intelligence": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
            "neural network": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
            "machine learning": "Machine learning is a field of study in artificial intelligence concerned with the development of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead.",
            "deep learning": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "natural language processing": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "computer vision": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos."
        }
        
        # Generate a random summary if the topic is not in mock_summaries
        if topic.lower() not in mock_summaries:
            words = ["system", "network", "intelligence", "learning", "algorithm", "computing", 
                    "data", "processing", "analysis", "knowledge", "information", "technology"]
            summary = f"{topic.title()} is a concept related to {random.choice(words)} and {random.choice(words)}."
        else:
            summary = mock_summaries[topic.lower()]
            
        return {
            "success": True,
            "topic": topic,
            "title": topic.title(),
            "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            "summary": summary,
            "full_text": f"{summary}\n\nThis is mock content for {topic} generated because the AutoWiki system is running in mock mode.",
            "timestamp": datetime.now().isoformat(),
            "mock": True
        }
    
    def _save_topic_content(self, topic: str, content: str, source: str):
        """Save fetched content to file"""
        try:
            # Clean topic name for filename
            clean_topic = ''.join(c if c.isalnum() else '_' for c in topic)
            filename = os.path.join(self.data_dir, f"{clean_topic}_{source}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved content for '{topic}' to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving topic content: {e}")
    
    def process_queue(self, max_items: int = 5) -> Dict[str, Any]:
        """
        Process items in the fetch queue
        
        Args:
            max_items: Maximum number of items to process
            
        Returns:
            Dictionary with processing results
        """
        if not self.topics_queue:
            logger.info("No topics in queue to process")
            return {"processed": 0, "success": 0, "failed": 0, "topics": []}
            
        processed = 0
        success_count = 0
        failed_count = 0
        processed_topics = []
        
        # Process up to max_items
        topics_to_process = self.topics_queue[:max_items]
        for topic in topics_to_process:
            logger.info(f"Processing topic: {topic}")
            processed += 1
            
            # Fetch content
            content = self.fetch_topic(topic)
            
            # Update history
            self.history[topic] = {
                "last_fetch": datetime.now().isoformat(),
                "success": content.get("success", False)
            }
            
            # Try to add to dictionary
            if content.get("success", False):
                self._add_to_learning_dictionary(content)
                success_count += 1
                processed_topics.append({
                    "topic": topic,
                    "success": True,
                    "title": content.get("title", topic)
                })
            else:
                failed_count += 1
                processed_topics.append({
                    "topic": topic, 
                    "success": False,
                    "error": content.get("error", "Unknown error")
                })
            
            # Remove from queue
            self.topics_queue.remove(topic)
            self._save_topics_queue()
            
            # Be nice to APIs with a short delay
            time.sleep(1)
        
        # Save updated queue and history
        self._save_queue()
        self._save_history()
        
        return {
            "processed": processed,
            "success": success_count,
            "failed": failed_count,
            "topics": processed_topics
        }
    
    def _add_to_learning_dictionary(self, content: Dict[str, Any]) -> bool:
        """
        Add content to the learning dictionary via the Mistral system.
        
        Args:
            content: The content to add to the learning dictionary.
            
        Returns:
            bool: True if the content was added, False otherwise.
        """
        if not self.mistral_system:
            logger.warning("No Mistral system available for learning")
            return False
            
        if not content.get("success", False):
            logger.warning(f"Cannot add unsuccessful content to dictionary: {content.get('topic')}")
            return False
            
        try:
            # Add the topic and summary to the dictionary
            topic = content["topic"]
            definition = content["summary"]
            
            # Use the Mistral system to add to dictionary
            self.mistral_system.add_to_dictionary(topic, definition)
            
            # If we have full text, process it for deeper learning
            if full_text := content.get("full_text"):
                # Process the full text in chunks to avoid context limits
                chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
                for chunk in chunks[:5]:  # Limit to first 5 chunks
                    learning_prompt = f"Learn key information about {topic} from this text: {chunk}"
                    self.mistral_system.process_message(learning_prompt)
                    
            logger.info(f"Added '{topic}' to learning dictionary")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to learning dictionary: {str(e)}")
            return False
    
    def start_auto_fetching(self):
        """Start automatic fetching in background thread"""
        def auto_fetch_worker():
            while self.auto_fetch:
                try:
                    # Process queue if items available
                    if self.topics_queue:
                        logger.info(f"Auto-fetching topics from queue ({len(self.topics_queue)} remaining)")
                        self.process_queue(max_items=3)
                    
                    # Sleep for a while before checking again
                    time.sleep(3600)  # 1 hour
                    
                except Exception as e:
                    logger.error(f"Error in auto-fetch worker: {e}")
                    time.sleep(3600)  # 1 hour even on error
        
        # Start background thread
        thread = threading.Thread(target=auto_fetch_worker)
        thread.daemon = True
        thread.start()
        logger.info("Started auto-fetching thread")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the AutoWiki system"""
        status = {
            "queue_size": 0,
            "total_fetches": 0,
            "successful_fetches": 0,
            "topics_fetched": 0,
            "last_fetch": 0,
            "auto_fetch": self.auto_fetch,
            "mock_mode": self.mock_mode
        }
        
        # Safely access attributes that might not exist in some versions
        if hasattr(self, 'topics_queue'):
            status["queue_size"] = len(self.topics_queue)
        elif hasattr(self, 'queue'):
            status["queue_size"] = len(self.queue)
        
        if hasattr(self, 'history') and self.history:
            status["total_fetches"] = len(self.history)
            status["successful_fetches"] = sum(1 for _, data in self.history.items() if data.get("success", False))
            status["topics_fetched"] = len(self.history)
            
            # Get last fetch time if history exists and has entries
            if self.history and list(self.history.keys()):
                last_topic = list(self.history.keys())[-1]
                status["last_fetch"] = self.history.get(last_topic, {}).get("last_fetch", 0)
        elif hasattr(self, 'fetched_count'):
            status["total_fetches"] = self.fetched_count
            status["topics_fetched"] = self.fetched_count
            if hasattr(self, 'successful_fetches'):
                status["successful_fetches"] = self.successful_fetches
        
        # Add topics_available if applicable
        if hasattr(self, 'topics'):
            status["topics_available"] = len(self.topics)
        
        return status


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AutoWiki System for V7")
    parser.add_argument("--topic", help="Topic to fetch information about")
    parser.add_argument("--queue", help="File with topics to add to queue (one per line)")
    parser.add_argument("--process", action="store_true", help="Process the queue")
    parser.add_argument("--max", type=int, default=5, help="Maximum items to process")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()
    
    # Create AutoWiki system
    autowiki = AutoWiki(auto_fetch=False)
    
    # Show status if requested
    if args.status:
        status = autowiki.get_status()
        print("\n=== AutoWiki Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        sys.exit(0)
    
    # Add topics from file if specified
    if args.queue:
        try:
            with open(args.queue, 'r') as f:
                topics = [line.strip() for line in f if line.strip()]
            count = autowiki.add_topics(topics)
            print(f"Added {count} topics to the queue")
        except Exception as e:
            print(f"Error adding topics from file: {e}")
    
    # Add single topic if specified
    if args.topic:
        if autowiki.add_topic(args.topic):
            print(f"Added '{args.topic}' to the queue")
        else:
            print(f"Topic '{args.topic}' is already in the queue or has been fetched")
    
    # Process queue if requested
    if args.process:
        print(f"Processing up to {args.max} topics...")
        count = autowiki.process_queue(max_items=args.max)
        print(f"Successfully processed {count} topics")
        
    # If no arguments, show help
    if not (args.topic or args.queue or args.process or args.status):
        parser.print_help() 