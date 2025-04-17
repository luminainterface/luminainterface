#!/usr/bin/env python
"""
Wikipedia Training Module

This module automatically fetches Wikipedia articles and uses them for RSEN training.
It can crawl articles from specific categories, follow links, or fetch random articles.
"""

import os
import re
import json
import logging
import random
import time
import threading
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaTrainingModule:
    """Fetches Wikipedia articles and processes them for RSEN training."""
    
    def __init__(self, 
                 training_script: str = 'train_english.py',
                 min_articles: int = 5,
                 check_interval: int = 600,  # 10 minutes
                 auto_start: bool = False,
                 max_articles_per_run: int = 10,
                 categories: List[str] = None,
                 language: str = 'en'):
        """
        Initialize the Wikipedia Training Module.
        
        Args:
            training_script: Path to the training script
            min_articles: Minimum number of articles required to start training
            check_interval: Interval in seconds to check for new articles
            auto_start: Whether to automatically start the collector thread
            max_articles_per_run: Maximum number of articles to fetch in a single run
            categories: List of Wikipedia categories to focus on (optional)
            language: Wikipedia language code (default: 'en' for English)
        """
        self.training_script = training_script
        self.min_articles = min_articles
        self.check_interval = check_interval
        self.max_articles_per_run = max_articles_per_run
        self.categories = categories or [
            "Physics", "Mathematics", "Philosophy", "Computer_science",
            "Linguistics", "Artificial_intelligence", "Cognitive_science"
        ]
        self.language = language
        
        # State tracking
        self.running = False
        self.article_buffer = []  # List of article content
        self.processed_urls = set()  # URLs already processed
        self.wiki_thread = None
        self.training_lock = threading.Lock()
        self.last_training_time = None
        self.last_fetch_time = None
        self.articles_fetched = 0
        
        # API endpoints
        self.wiki_api_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join('.', 'training_data', 'wikipedia')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Load previously processed URLs
        self._load_processed_urls()
        
        # Start wiki thread if auto_start is True
        if auto_start:
            self.start()
    
    def start(self):
        """Start the Wikipedia fetcher thread."""
        if not self.running:
            self.running = True
            self.wiki_thread = threading.Thread(target=self._collection_loop)
            self.wiki_thread.daemon = True
            self.wiki_thread.start()
            logger.info("Wikipedia Training Module started")
    
    def stop(self):
        """Stop the Wikipedia fetcher thread."""
        self.running = False
        if self.wiki_thread:
            self.wiki_thread.join(timeout=5.0)
            logger.info("Wikipedia Training Module stopped")
    
    def _collection_loop(self):
        """Main collection loop that runs in a background thread."""
        while self.running:
            try:
                # Check if we need to fetch more articles
                if len(self.article_buffer) < self.min_articles:
                    self._fetch_articles()
                
                # Check if we have enough articles for training
                if len(self.article_buffer) >= self.min_articles:
                    self._trigger_training()
                
                # Sleep for the specified interval
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in Wikipedia collection loop: {str(e)}")
                time.sleep(60)  # Sleep for a minute on error
    
    def _fetch_articles(self):
        """Fetch new Wikipedia articles."""
        articles_to_fetch = self.max_articles_per_run
        articles_fetched = 0
        
        # First try to fetch from categories
        for category in random.sample(self.categories, min(3, len(self.categories))):
            if articles_fetched >= articles_to_fetch:
                break
                
            try:
                category_articles = self._fetch_category_articles(category)
                for article in category_articles[:articles_to_fetch - articles_fetched]:
                    if self._process_article(article):
                        articles_fetched += 1
            except Exception as e:
                logger.error(f"Error fetching category '{category}': {str(e)}")
        
        # If we still need more, fetch random articles
        if articles_fetched < articles_to_fetch:
            try:
                random_articles = self._fetch_random_articles(articles_to_fetch - articles_fetched)
                for article in random_articles:
                    if self._process_article(article):
                        articles_fetched += 1
            except Exception as e:
                logger.error(f"Error fetching random articles: {str(e)}")
        
        # Update fetch time
        self.last_fetch_time = datetime.now()
        self.articles_fetched += articles_fetched
        
        logger.info(f"Fetched {articles_fetched} new Wikipedia articles. Buffer now has {len(self.article_buffer)} articles.")
        
        # Save processed URLs
        self._save_processed_urls()
    
    def _fetch_category_articles(self, category: str) -> List[Dict]:
        """Fetch articles from a specific Wikipedia category."""
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "50",
            "cmtype": "page"
        }
        
        response = requests.get(self.wiki_api_url, params=params)
        data = response.json()
        
        if "query" in data and "categorymembers" in data["query"]:
            return [
                {
                    "title": page["title"],
                    "pageid": page["pageid"]
                }
                for page in data["query"]["categorymembers"]
            ]
        
        return []
    
    def _fetch_random_articles(self, count: int) -> List[Dict]:
        """Fetch random Wikipedia articles."""
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnlimit": str(count),
            "rnnamespace": "0"  # Main namespace
        }
        
        response = requests.get(self.wiki_api_url, params=params)
        data = response.json()
        
        if "query" in data and "random" in data["query"]:
            return [
                {
                    "title": page["title"],
                    "pageid": page["pageid"]
                }
                for page in data["query"]["random"]
            ]
        
        return []
    
    def _process_article(self, article: Dict) -> bool:
        """Process a Wikipedia article and add its content to the buffer."""
        title = article.get("title", "")
        pageid = article.get("pageid", 0)
        
        # Skip if already processed
        article_url = f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        if article_url in self.processed_urls:
            return False
        
        try:
            # Get article content
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": "1",  # Just the introduction
                "explaintext": "1",  # Plain text, not HTML
                "pageids": str(pageid)
            }
            
            response = requests.get(self.wiki_api_url, params=params)
            data = response.json()
            
            if "query" in data and "pages" in data["query"] and str(pageid) in data["query"]["pages"]:
                page = data["query"]["pages"][str(pageid)]
                content = page.get("extract", "")
                
                # Skip if content is too short
                if len(content) < 100:
                    return False
                
                # Create article data
                article_data = {
                    "title": title,
                    "content": content,
                    "url": article_url,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to buffer
                self.article_buffer.append(article_data)
                self.processed_urls.add(article_url)
                
                return True
        except Exception as e:
            logger.error(f"Error processing article '{title}': {str(e)}")
        
        return False
    
    def _create_training_file(self) -> Optional[str]:
        """
        Create a training file from the collected articles.
        
        Returns:
            Path to the created file, or None if no file was created
        """
        if not self.article_buffer:
            return None
            
        # Create a filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.data_dir, f"wikipedia_articles_{timestamp}.txt")
        
        # Write the articles to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            for article in self.article_buffer:
                # Write title and content
                f.write(f"TITLE: {article['title']}\n\n")
                f.write(f"{article['content']}\n\n")
                f.write("=" * 80 + "\n\n")  # Separator
        
        # Create a metadata file
        meta_file_path = os.path.join(self.data_dir, f"wikipedia_meta_{timestamp}.json")
        with open(meta_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "articles": [{
                    "title": article["title"],
                    "url": article["url"],
                    "timestamp": article["timestamp"]
                } for article in self.article_buffer],
                "timestamp": datetime.now().isoformat(),
                "count": len(self.article_buffer)
            }, f, indent=2)
        
        logger.info(f"Created training file with {len(self.article_buffer)} articles: {file_path}")
        
        # Clear the article buffer
        self.article_buffer = []
        
        return file_path
    
    def _trigger_training(self):
        """Trigger the training process with the collected articles."""
        with self.training_lock:
            # Check if we're already training
            if self.last_training_time and (datetime.now() - self.last_training_time).total_seconds() < 3600:
                logger.info("Skipping training - last training was less than an hour ago")
                return
                
            # Create the training file
            training_file = self._create_training_file()
            if not training_file:
                logger.info("No training file created - skipping training")
                return
                
            # Set the last training time
            self.last_training_time = datetime.now()
            
            # Start the training process
            try:
                # Build the command for the training script
                cmd = [
                    'python', 
                    self.training_script, 
                    '--data', training_file,
                    '--epochs', '3',
                    '--batch_size', '8',
                    '--output_metrics', os.path.join(self.data_dir, f"metrics_{os.path.basename(training_file).replace('.txt', '.json')}")
                ]
                
                # Run the command
                logger.info(f"Starting training process: {' '.join(cmd)}")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Stream the output
                for line in process.stdout:
                    logger.info(f"Training: {line.strip()}")
                
                # Wait for the process to complete
                process.wait()
                
                # Check if the process was successful
                if process.returncode == 0:
                    logger.info("Training process completed successfully")
                else:
                    logger.error(f"Training process failed with return code {process.returncode}")
                    
                    # Get the error output
                    error = process.stderr.read()
                    logger.error(f"Training error: {error}")
                    
                    # Try with fallback option if it failed
                    logger.info("Retrying with fallback option")
                    fallback_cmd = cmd + ['--force_fallback']
                    subprocess.run(fallback_cmd)
                
            except Exception as e:
                logger.error(f"Error starting training process: {str(e)}")
    
    def _load_processed_urls(self):
        """Load previously processed URLs from file."""
        filepath = os.path.join(self.data_dir, "processed_urls.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_urls = set(data["urls"])
                logger.info(f"Loaded {len(self.processed_urls)} previously processed URLs")
            except Exception as e:
                logger.error(f"Error loading processed URLs: {str(e)}")
    
    def _save_processed_urls(self):
        """Save processed URLs to file."""
        filepath = os.path.join(self.data_dir, "processed_urls.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "urls": list(self.processed_urls),
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Saved {len(self.processed_urls)} processed URLs")
        except Exception as e:
            logger.error(f"Error saving processed URLs: {str(e)}")
    
    def force_training(self):
        """Force start a training run with the currently collected articles."""
        if len(self.article_buffer) > 0:
            self._trigger_training()
        else:
            logger.warning("No articles in buffer - cannot force training")
            
            # Fetch some articles first, then train
            self._fetch_articles()
            if len(self.article_buffer) > 0:
                self._trigger_training()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Wikipedia training module."""
        return {
            "running": self.running,
            "article_count": len(self.article_buffer),
            "processed_urls": len(self.processed_urls),
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "last_fetch_time": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            "articles_fetched": self.articles_fetched,
            "training_ready": len(self.article_buffer) >= self.min_articles,
            "categories": self.categories
        }

# If run directly, start the Wikipedia training module
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wikipedia Training Module")
    parser.add_argument("--min_articles", type=int, default=5, help="Minimum number of articles to start training")
    parser.add_argument("--check_interval", type=int, default=600, help="Interval in seconds to check for new articles")
    parser.add_argument("--force", action="store_true", help="Force training immediately with available articles")
    parser.add_argument("--categories", type=str, help="Comma-separated list of Wikipedia categories to focus on")
    args = parser.parse_args()
    
    categories = args.categories.split(",") if args.categories else None
    
    wiki_trainer = WikipediaTrainingModule(
        min_articles=args.min_articles,
        check_interval=args.check_interval,
        categories=categories,
        auto_start=True
    )
    
    if args.force:
        wiki_trainer.force_training()
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        wiki_trainer.stop() 