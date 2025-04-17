#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LUMINA V7.5 AutoWiki Module
===========================
Automatic wiki information fetcher and processor
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import requests
from datetime import datetime
from collections import deque

class AutoWiki:
    """
    AutoWiki module for the LUMINA system, responsible for fetching
    and processing information from Wikipedia and other knowledge sources.
    """
    
    def __init__(self, data_dir="data/autowiki", log_dir="logs/autowiki", 
                 mistral_system=None, auto_fetch=True, mock_mode=False):
        """
        Initialize the AutoWiki module.
        
        Args:
            data_dir (str): Directory for storing wiki data
            log_dir (str): Directory for storing logs
            mistral_system: Reference to the Mistral system (optional)
            auto_fetch (bool): Whether to automatically fetch topics
            mock_mode (bool): Whether to run in mock mode
        """
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.mistral_system = mistral_system
        self.auto_fetch = auto_fetch
        self.mock_mode = mock_mode
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        self.logger = self._setup_logging()
        self.logger.info("Initializing AutoWiki module")
        
        # File paths
        self.queue_file = os.path.join(data_dir, "autowiki_queue.json")
        self.history_file = os.path.join(data_dir, "autowiki_history.json")
        
        # Initialize data structures
        self.queue = self._load_queue()
        self.history = self._load_history()
        
        # Status tracking
        self.status = {
            "active": True,
            "queue_size": len(self.queue),
            "processed_count": len(self.history),
            "last_fetch_time": None,
            "current_topic": None
        }
        
        # Component status update
        self._update_component_status("active")
        
        # Start background processing if auto_fetch is enabled
        if auto_fetch:
            self.process_thread = threading.Thread(target=self._background_processor)
            self.process_thread.daemon = True
            self.process_thread.start()
            self.logger.info("Started background processing thread")
    
    def _setup_logging(self):
        """Set up the logger for the AutoWiki module."""
        logger = logging.getLogger("autowiki")
        logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = os.path.join(self.log_dir, f"autowiki_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_queue(self):
        """Load the queue from file or create a new one."""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    queue = json.load(f)
                self.logger.info(f"Loaded {len(queue)} topics from queue")
                return queue
            except Exception as e:
                self.logger.error(f"Error loading queue: {e}")
                return []
        else:
            self.logger.info("Queue file doesn't exist, creating new queue")
            return []
    
    def _save_queue(self):
        """Save the queue to file."""
        try:
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(self.queue, f, indent=2)
            self.logger.debug(f"Saved {len(self.queue)} topics to queue")
        except Exception as e:
            self.logger.error(f"Error saving queue: {e}")
    
    def _load_history(self):
        """Load the history from file or create a new one."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.logger.info(f"Loaded {len(history)} topics from history")
                return history
            except Exception as e:
                self.logger.error(f"Error loading history: {e}")
                return {}
        else:
            self.logger.info("History file doesn't exist, creating new history")
            return {}
    
    def _save_history(self):
        """Save the history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
            self.logger.debug(f"Saved {len(self.history)} topics to history")
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")
    
    def _update_component_status(self, status):
        """Update the component status in the component_status.json file."""
        status_file = os.path.join(os.getcwd(), "data", "component_status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                status_data["Autowiki"] = status
                with open(status_file, 'w') as f:
                    json.dump(status_data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error updating component status: {e}")
    
    def _background_processor(self):
        """Background processor to fetch topics from the queue."""
        while True:
            try:
                if len(self.queue) > 0:
                    # Process one topic from the queue every 10 minutes
                    self.process_next_topic()
                # Sleep for 10 minutes
                time.sleep(600)
            except Exception as e:
                self.logger.error(f"Error in background processor: {e}")
                time.sleep(60)  # Shorter sleep on error
    
    def add_topic(self, topic):
        """
        Add a topic to the queue.
        
        Args:
            topic (str): The topic to add
            
        Returns:
            bool: True if the topic was added, False otherwise
        """
        # Normalize topic
        topic = topic.strip()
        
        # Check if topic is already in queue or history
        if topic.lower() in [t.lower() for t in self.queue]:
            self.logger.info(f"Topic '{topic}' already in queue")
            return False
        
        if topic.lower() in [t.lower() for t in self.history]:
            self.logger.info(f"Topic '{topic}' already in history")
            return False
        
        # Add topic to queue
        self.queue.append(topic)
        self._save_queue()
        
        # Update status
        self.status["queue_size"] = len(self.queue)
        
        self.logger.info(f"Added topic '{topic}' to queue")
        return True
    
    def get_queue(self):
        """
        Get the current queue.
        
        Returns:
            list: The queue
        """
        return self.queue
    
    def get_history(self):
        """
        Get the processing history.
        
        Returns:
            dict: The history
        """
        return self.history
    
    def process_next_topic(self):
        """
        Process the next topic in the queue.
        
        Returns:
            dict: The processed topic information or None if no topic was processed
        """
        if not self.queue:
            self.logger.info("Queue is empty, nothing to process")
            return None
        
        # Get the next topic
        topic = self.queue.pop(0)
        self._save_queue()
        
        # Update status
        self.status["current_topic"] = topic
        self.status["queue_size"] = len(self.queue)
        
        self.logger.info(f"Processing topic: {topic}")
        
        # Fetch topic information
        topic_info = self.fetch_topic(topic)
        
        if topic_info:
            # Add to history
            self.history[topic] = topic_info
            self._save_history()
            
            # Update status
            self.status["processed_count"] = len(self.history)
            self.status["last_fetch_time"] = datetime.now().isoformat()
            
            # Integrate with Mistral system if available
            if self.mistral_system:
                try:
                    self.mistral_system.add_knowledge(topic, topic_info["summary"])
                    self.logger.info(f"Added knowledge about '{topic}' to Mistral system")
                except Exception as e:
                    self.logger.error(f"Error adding knowledge to Mistral system: {e}")
            
            self.logger.info(f"Successfully processed topic: {topic}")
            return topic_info
        else:
            self.logger.warning(f"Failed to process topic: {topic}")
            return None
    
    def fetch_topic(self, topic):
        """
        Fetch information about a topic.
        
        Args:
            topic (str): The topic to fetch
            
        Returns:
            dict: The topic information or None if fetching failed
        """
        if self.mock_mode:
            self.logger.info(f"Mock mode: Generating mock data for topic '{topic}'")
            return self._generate_mock_data(topic)
        
        try:
            # Try to fetch from Wikipedia
            self.logger.info(f"Fetching information about '{topic}' from Wikipedia")
            return self._fetch_from_wikipedia(topic)
        except Exception as e:
            self.logger.error(f"Error fetching topic '{topic}': {e}")
            return None
    
    def _fetch_from_wikipedia(self, topic):
        """
        Fetch topic information from Wikipedia.
        
        Args:
            topic (str): The topic to fetch
            
        Returns:
            dict: The topic information
        """
        # Use Wikipedia API to get summary
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(topic)
        response = requests.get(url)
        
        if response.status_code != 200:
            self.logger.warning(f"Wikipedia API returned status code {response.status_code}")
            return None
        
        data = response.json()
        
        # Get additional content
        content_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1
        }
        
        content_response = requests.get(content_url, params=params)
        if content_response.status_code == 200:
            content_data = content_response.json()
            pages = content_data.get("query", {}).get("pages", {})
            page_id = next(iter(pages))
            extract = pages[page_id].get("extract", "")
        else:
            extract = ""
            
        # Create topic information
        topic_info = {
            "title": data.get("title", topic),
            "summary": data.get("extract", "No summary available"),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "thumbnail": data.get("thumbnail", {}).get("source", ""),
            "full_content": extract,
            "categories": [],  # Would require another API call
            "fetch_time": datetime.now().isoformat()
        }
        
        return topic_info
    
    def _generate_mock_data(self, topic):
        """
        Generate mock data for a topic.
        
        Args:
            topic (str): The topic to generate mock data for
            
        Returns:
            dict: The mock topic information
        """
        return {
            "title": topic,
            "summary": f"This is a mock summary for the topic '{topic}'. In a real system, this would contain actual information from Wikipedia or another knowledge source.",
            "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            "thumbnail": "",
            "full_content": f"This is mock full content for '{topic}'. It would normally be a longer extract from the Wikipedia article.",
            "categories": ["Mock Category 1", "Mock Category 2"],
            "fetch_time": datetime.now().isoformat()
        }
    
    def get_status(self):
        """
        Get the current status of the AutoWiki module.
        
        Returns:
            dict: The status information
        """
        return self.status
    
    def shutdown(self):
        """Shut down the AutoWiki module."""
        self.logger.info("Shutting down AutoWiki module")
        self.status["active"] = False
        self._update_component_status("offline")
        # Save any unsaved data
        self._save_queue()
        self._save_history()

def main():
    """Main entry point for the AutoWiki module."""
    parser = argparse.ArgumentParser(description="LUMINA V7.5 AutoWiki Module")
    parser.add_argument("--data-dir", default="data/autowiki", help="Directory for storing wiki data")
    parser.add_argument("--log-dir", default="logs/autowiki", help="Directory for storing logs")
    parser.add_argument("--port", type=int, default=7525, help="Port for the AutoWiki service")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--no-auto-fetch", action="store_true", help="Disable automatic fetching")
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 40)
    print("LUMINA V7.5 AutoWiki Module")
    print("=" * 40 + "\n")
    
    # Initialize AutoWiki
    autowiki = AutoWiki(
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        auto_fetch=not args.no_auto_fetch,
        mock_mode=args.mock
    )
    
    print(f"AutoWiki initialized with:")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Log directory: {args.log_dir}")
    print(f"  - Auto-fetch: {not args.no_auto_fetch}")
    print(f"  - Mock mode: {args.mock}")
    print(f"  - Service port: {args.port}")
    print("\nCurrent queue size:", len(autowiki.get_queue()))
    print("Topics in history:", len(autowiki.get_history()))
    print("\nType 'help' for available commands or 'exit' to quit.")
    
    # Simple command loop for testing
    try:
        while True:
            cmd = input("\n> ").strip()
            if cmd.lower() == "exit":
                break
            elif cmd.lower() == "help":
                print("\nAvailable commands:")
                print("  help - Display this help message")
                print("  status - Display current AutoWiki status")
                print("  queue - Display current queue")
                print("  history - Display processing history")
                print("  add <topic> - Add a topic to the queue")
                print("  process - Process the next topic in the queue")
                print("  fetch <topic> - Fetch information about a topic without queuing")
                print("  exit - Exit the AutoWiki module")
            elif cmd.lower() == "status":
                status = autowiki.get_status()
                print("\nAutoWiki Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
            elif cmd.lower() == "queue":
                queue = autowiki.get_queue()
                print("\nCurrent Queue:")
                if queue:
                    for i, topic in enumerate(queue):
                        print(f"  {i+1}. {topic}")
                else:
                    print("  Queue is empty")
            elif cmd.lower() == "history":
                history = autowiki.get_history()
                print("\nProcessing History:")
                if history:
                    for i, (topic, info) in enumerate(history.items()):
                        print(f"  {i+1}. {topic} - {info.get('fetch_time', 'unknown')}")
                else:
                    print("  History is empty")
            elif cmd.lower().startswith("add "):
                topic = cmd[4:].strip()
                if topic:
                    if autowiki.add_topic(topic):
                        print(f"Added topic '{topic}' to queue")
                    else:
                        print(f"Topic '{topic}' already in queue or history")
                else:
                    print("Please specify a topic to add")
            elif cmd.lower() == "process":
                result = autowiki.process_next_topic()
                if result:
                    print(f"Processed topic: {result['title']}")
                    print(f"Summary: {result['summary'][:100]}...")
                else:
                    print("No topic to process or processing failed")
            elif cmd.lower().startswith("fetch "):
                topic = cmd[6:].strip()
                if topic:
                    print(f"Fetching information about '{topic}'...")
                    result = autowiki.fetch_topic(topic)
                    if result:
                        print(f"Title: {result['title']}")
                        print(f"Summary: {result['summary'][:150]}...")
                        print(f"URL: {result['url']}")
                    else:
                        print(f"Failed to fetch information about '{topic}'")
                else:
                    print("Please specify a topic to fetch")
            else:
                print("Unknown command. Type 'help' for available commands.")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nShutting down AutoWiki module...")
        autowiki.shutdown()
        print("Shutdown complete")

if __name__ == "__main__":
    main() 