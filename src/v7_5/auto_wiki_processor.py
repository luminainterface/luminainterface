#!/usr/bin/env python3
"""
AutoWikiProcessor for LUMINA v7.5
Handles wiki integration and information export with monitoring system integration
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from bs4 import BeautifulSoup
import threading
from threading import Thread, Event, Lock
import wikipedia
import random
from queue import Queue
import backoff
from PySide6.QtCore import QObject, Signal, Slot, Property

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"auto_wiki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("AutoWikiProcessor")

# Lumina color scheme
LUMINA_COLORS = {
    'primary': '#000000',      # Black
    'accent': '#C6A962',       # Gold
    'background': '#F5F5F2',   # Off-white
    'text': '#1A1A1A',        # Dark gray
    'success': '#4A5D4F',     # Muted green
    'warning': '#8B7355',     # Muted brown
    'error': '#8B4545'        # Muted red
}

class AutoWikiProcessor(QObject):
    """Processes and exports wiki information with monitoring integration"""
    
    # Qt Signals for monitoring
    statusChanged = Signal(dict)
    articleProcessed = Signal(dict)
    errorOccurred = Signal(str)
    metricsUpdated = Signal(dict)
    
    WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
    BATCH_SIZE = 3
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    def __init__(self):
        super().__init__()
        self.wiki_data_dir = Path("data/wiki")
        self.export_dir = Path("data/exports")
        self.wiki_data_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Task tracking with monitoring
        self._active_tasks = 0
        self._total_tasks = 0
        self._successful_tasks = 0
        self._error_count = 0
        self._performance_metrics = {
            'response_time': [],
            'success_rate': 0.0,
            'throughput': 0.0
        }
        
        # Settings
        self._update_interval = 30  # Default 30 seconds
        self._auto_update_enabled = True
        self._max_concurrent_tasks = 3
        self._last_update = datetime.now()
        
        # Monitoring state
        self._component_state = 'active'
        self._health_score = 1.0
        self._last_error = None
        
        # Queue and processing state
        self.article_queue = Queue()
        self._processing = False
        self.stop_event = Event()
        self.processing_thread = None
        self.last_search = None
        self.queue_lock = Lock()
        
        # Monitoring thread
        self._monitoring_thread = Thread(target=self._monitor_metrics, daemon=True)
        self._monitoring_interval = 1.0  # 1 second monitoring interval
        
        # Initialize Wikipedia API
        wikipedia.set_lang("en")
        
        logger.info("AutoWikiProcessor initialized")
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the monitoring thread"""
        self._monitoring_thread.start()
        logger.info("Monitoring thread started")
    
    def _monitor_metrics(self):
        """Monitor and emit metrics at regular intervals"""
        while not self.stop_event.is_set():
            try:
                metrics = {
                    'component_state': self._component_state,
                    'health_score': self._health_score,
                    'active_tasks': self._active_tasks,
                    'total_tasks': self._total_tasks,
                    'success_rate': self.success_rate,
                    'queue_size': self.article_queue.qsize(),
                    'error_count': self._error_count,
                    'last_error': self._last_error,
                    'performance': self._performance_metrics,
                    'last_update': self._last_update.isoformat()
                }
                
                # Emit metrics for monitoring UI
                self.metricsUpdated.emit(metrics)
                
                # Update component state based on health
                self._update_component_state()
                
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                self.errorOccurred.emit(str(e))
                time.sleep(self.RETRY_DELAY)
    
    def _update_component_state(self):
        """Update component state based on health metrics"""
        if self._health_score >= 0.8:
            new_state = 'active'
        elif self._health_score >= 0.5:
            new_state = 'degraded'
        else:
            new_state = 'error'
            
        if new_state != self._component_state:
            self._component_state = new_state
            self.statusChanged.emit({
                'state': new_state,
                'health_score': self._health_score
            })
    
    @Slot(result=dict)
    def get_monitoring_state(self) -> Dict[str, Any]:
        """Get current monitoring state for UI"""
        return {
            'component_state': self._component_state,
            'health_score': self._health_score,
            'metrics': self._performance_metrics,
            'queue_status': self.get_queue_status()
        }
    
    @Slot(result=float)
    def get_health_score(self) -> float:
        """Get current health score for monitoring"""
        return self._health_score
    
    @property
    def active_tasks(self) -> int:
        """Get number of active tasks"""
        return self._active_tasks
    
    @property
    def total_tasks(self) -> int:
        """Get total number of tasks processed"""
        return self._total_tasks
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self._total_tasks == 0:
            return 0.0
        return (self._successful_tasks / self._total_tasks) * 100
    
    def set_update_interval(self, seconds: int) -> None:
        """Set the update interval in seconds"""
        self._update_interval = max(5, min(300, seconds))  # Clamp between 5-300 seconds
        logger.info(f"Update interval set to {self._update_interval} seconds")
        
    def set_auto_update(self, enabled: bool) -> None:
        """Enable or disable auto-updates"""
        self._auto_update_enabled = enabled
        logger.info(f"Auto-update {'enabled' if enabled else 'disabled'}")
        
    def set_max_tasks(self, max_tasks: int) -> None:
        """Set maximum concurrent tasks"""
        self._max_concurrent_tasks = max(1, min(10, max_tasks))  # Clamp between 1-10 tasks
        logger.info(f"Max concurrent tasks set to {self._max_concurrent_tasks}")
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=MAX_RETRIES)
    def _fetch_random_articles(self, count: int = BATCH_SIZE) -> List[Dict[str, str]]:
        """Fetch multiple random Wikipedia articles using the API"""
        try:
            articles = []
            
            # Get random article titles
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,  # Main namespace
                "rnlimit": count
            }
            
            response = requests.get(self.WIKI_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "query" not in data or "random" not in data["query"]:
                raise ValueError("Invalid API response")
            
            # Get content for each article
            for article in data["query"]["random"]:
                title = article["title"]
                try:
                    # Get article content
                    page = wikipedia.page(title, auto_suggest=False)
                    articles.append({
                        'title': title,
                        'content': page.content[:1000],  # First 1000 chars
                        'url': page.url,
                        'timestamp': datetime.now().isoformat(),
                        'summary': page.summary
                    })
                except wikipedia.exceptions.DisambiguationError:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching random articles: {e}")
            raise
    
    def _process_article_batch(self):
        """Process a batch of articles from the queue with monitoring"""
        try:
            with self.queue_lock:
                if self.article_queue.empty():
                    return
                
                # Get batch of articles
                articles = []
                for _ in range(min(self.BATCH_SIZE, self.article_queue.qsize())):
                    articles.append(self.article_queue.get())
                
            # Process each article
            start_time = time.time()
            successful_articles = 0
            
            for article in articles:
                try:
                    # Save to wiki database
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wiki_file = self.wiki_data_dir / f"update_{timestamp}.json"
                    
                    with open(wiki_file, 'w', encoding='utf-8') as f:
                        json.dump(article, f, indent=2)
                    
                    self._successful_tasks += 1
                    successful_articles += 1
                    
                    # Emit article processed signal
                    self.articleProcessed.emit(article)
                    
                    logger.info(f"Article processed successfully: {article['title']}")
                    
                except Exception as e:
                    self._error_count += 1
                    self._last_error = str(e)
                    self.errorOccurred.emit(str(e))
                    logger.error(f"Error processing article {article['title']}: {e}")
                finally:
                    self.article_queue.task_done()
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._performance_metrics['response_time'].append(processing_time)
            if len(self._performance_metrics['response_time']) > 100:
                self._performance_metrics['response_time'] = self._performance_metrics['response_time'][-100:]
            
            self._performance_metrics['success_rate'] = successful_articles / len(articles)
            self._performance_metrics['throughput'] = len(articles) / processing_time
            
            # Update health score based on performance
            self._update_health_score()
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            self.errorOccurred.emit(str(e))
            logger.error(f"Error processing article batch: {e}")
    
    def _update_health_score(self):
        """Update health score based on performance metrics"""
        try:
            # Calculate health score components
            success_rate_score = self.success_rate / 100.0
            response_time_score = min(1.0, 1.0 / (sum(self._performance_metrics['response_time']) / len(self._performance_metrics['response_time']) if self._performance_metrics['response_time'] else 1.0))
            error_rate_score = max(0.0, 1.0 - (self._error_count / max(1, self._total_tasks)))
            
            # Weighted average for health score
            self._health_score = (
                0.4 * success_rate_score +
                0.3 * response_time_score +
                0.3 * error_rate_score
            )
            
        except Exception as e:
            logger.error(f"Error updating health score: {e}")
            self._health_score = 0.0
    
    def start_processing(self):
        """Start the wiki processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.processing_thread = Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        logger.info("AutoWiki processing started")
    
    def stop_processing(self):
        """Stop the wiki processing thread"""
        if self.processing_thread:
            self.stop_event.set()
            self.processing_thread.join()
            self.processing_thread = None
            logger.info("AutoWiki processing stopped")
    
    def _process_loop(self):
        """Main processing loop that fetches and processes random articles"""
        while not self.stop_event.is_set():
            try:
                # Fetch new batch of articles
                articles = self._fetch_random_articles()
                
                # Add articles to queue
                with self.queue_lock:
                    for article in articles:
                        self.article_queue.put(article)
                
                # Process the batch
                self._process_article_batch()
                
                # Sleep before next batch
                time.sleep(self._update_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(self.RETRY_DELAY)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'queue_size': self.article_queue.qsize(),
            'active_tasks': self._active_tasks,
            'total_tasks': self._total_tasks,
            'success_rate': self.success_rate,
            'last_update': self._last_update.isoformat()
        }
    
    def request_new_articles(self, count: int = BATCH_SIZE) -> bool:
        """Request a new batch of random articles"""
        try:
            articles = self._fetch_random_articles(count)
            
            with self.queue_lock:
                for article in articles:
                    self.article_queue.put(article)
            
            logger.info(f"Added {len(articles)} new articles to queue")
            return True
            
        except Exception as e:
            logger.error(f"Error requesting new articles: {e}")
            return False
    
    def shutdown(self):
        """Clean up resources and stop monitoring"""
        self.stop_processing()
        self.stop_event.set()  # Stop monitoring thread
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join()
        self.executor.shutdown()
        logger.info("AutoWikiProcessor shutdown complete") 
                # TODO: Implement actual wiki processing logic here
                # For now, just sleep to prevent busy waiting
                time.sleep(5)
                
                if self.on_wiki_update:
                    self.on_wiki_update("Wiki processing active")
                    
            except Exception as e:
                if self.on_wiki_update:
                    self.on_wiki_update(f"Error in wiki processing: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def process_message(self, message: str):
        """Process a message to find relevant wiki content"""
        try:
            # Basic keyword extraction (to be improved)
            keywords = self._extract_keywords(message)
            
            for keyword in keywords:
                if keyword == self.last_search:
                    continue
                    
                try:
                    summary = wikipedia.summary(keyword, sentences=2)
                    self.last_search = keyword
                    if self.on_wiki_update:
                        self.on_wiki_update(f"Related to '{keyword}': {summary}")
                    break
                except wikipedia.exceptions.DisambiguationError as e:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                    
        except Exception as e:
            if self.on_wiki_update:
                self.on_wiki_update(f"Error processing message: {str(e)}")
    
    def _extract_keywords(self, message: str) -> list:
        """Extract potential keywords from a message"""
        # This is a basic implementation that should be improved
        words = message.split()
        # Filter out common words and short words
        keywords = [word for word in words if len(word) > 4 and word.lower() not in {
            'about', 'above', 'after', 'again', 'their', 'would', 'could',
            'should', 'which', 'there', 'where', 'when', 'what', 'have'
        }]
        return keywords[:3]  # Return top 3 potential keywords
    
    def _get_random_wiki_article(self) -> Dict[str, str]:
        """Fetch a random Wikipedia article"""
        try:
            # Get random article
            response = requests.get(self.RANDOM_WIKI_URL, allow_redirects=True)
            response.raise_for_status()
            
            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title and content
            title = soup.find(id='firstHeading').text
            content = soup.find(id='mw-content-text').get_text()[:500]  # First 500 chars
            url = response.url
            
            # Clean up content
            content = ' '.join(content.split())  # Remove extra whitespace
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching random Wikipedia article: {e}")
            return None
    
    def _process_task(self, message: str, context: dict) -> None:
        """Process a single task"""
        try:
            # Get random Wikipedia article
            wiki_data = self._get_random_wiki_article()
            if not wiki_data:
                return
            
            # Save to wiki database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wiki_file = self.wiki_data_dir / f"update_{timestamp}.json"
            
            data = {
                "message": message,
                "context": context,
                "wiki_data": wiki_data
            }
            
            with open(wiki_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self._successful_tasks += 1
            logger.info(f"Task processed successfully: {wiki_data['title']}")
            
            # Notify about new wiki article
            if self.on_wiki_update:
                formatted_wiki = (
                    f"📚 Related Wikipedia Article:\n"
                    f"Title: {wiki_data['title']}\n"
                    f"Summary: {wiki_data['content'][:200]}...\n"
                    f"Read more: {wiki_data['url']}"
                )
                self.on_wiki_update(formatted_wiki)
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
        finally:
            self._active_tasks -= 1
            self._last_update = datetime.now()
    
    def process_article(self, title: str, content: str):
        """Process a Wikipedia article and notify listeners"""
        # ... existing code ...
        
        # Format update message
        update_msg = f"Processed article: {title}\nExtracted {len(self.extracted_info)} key points"
        
        # Notify listeners
        if self.on_wiki_update:
            self.on_wiki_update(update_msg)
            
        return self.extracted_info
        
    def export_info(self, filepath: str):
        """Export processed information and notify listeners"""
        # ... existing code ...
        
        # Format update message
        update_msg = f"Exported {len(self.extracted_info)} articles to {filepath}"
        
        # Notify listeners
        if self.on_wiki_update:
            self.on_wiki_update(update_msg)
            
        return True
    
    def shutdown(self):
        """Clean up resources"""
        self.executor.shutdown()
        logger.info("AutoWikiProcessor shutdown complete") 
                    articles.append(self.article_queue.get())
                
            # Process each article
            start_time = time.time()
            successful_articles = 0
            
            for article in articles:
                try:
                    # Save to wiki database
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wiki_file = self.wiki_data_dir / f"update_{timestamp}.json"
                    
                    with open(wiki_file, 'w', encoding='utf-8') as f:
                        json.dump(article, f, indent=2)
                    
                    self._successful_tasks += 1
                    successful_articles += 1
                    
                    # Emit article processed signal
                    self.articleProcessed.emit(article)
                    
                    logger.info(f"Article processed successfully: {article['title']}")
                    
                except Exception as e:
                    self._error_count += 1
                    self._last_error = str(e)
                    self.errorOccurred.emit(str(e))
                    logger.error(f"Error processing article {article['title']}: {e}")
                finally:
                    self.article_queue.task_done()
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._performance_metrics['response_time'].append(processing_time)
            if len(self._performance_metrics['response_time']) > 100:
                self._performance_metrics['response_time'] = self._performance_metrics['response_time'][-100:]
            
            self._performance_metrics['success_rate'] = successful_articles / len(articles)
            self._performance_metrics['throughput'] = len(articles) / processing_time
            
            # Update health score based on performance
            self._update_health_score()
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            self.errorOccurred.emit(str(e))
            logger.error(f"Error processing article batch: {e}")
    
    def _update_health_score(self):
        """Update health score based on performance metrics"""
        try:
            # Calculate health score components
            success_rate_score = self.success_rate / 100.0
            response_time_score = min(1.0, 1.0 / (sum(self._performance_metrics['response_time']) / len(self._performance_metrics['response_time']) if self._performance_metrics['response_time'] else 1.0))
            error_rate_score = max(0.0, 1.0 - (self._error_count / max(1, self._total_tasks)))
            
            # Weighted average for health score
            self._health_score = (
                0.4 * success_rate_score +
                0.3 * response_time_score +
                0.3 * error_rate_score
            )
            
        except Exception as e:
            logger.error(f"Error updating health score: {e}")
            self._health_score = 0.0
    
    def start_processing(self):
        """Start the wiki processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.processing_thread = Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        logger.info("AutoWiki processing started")
    
    def stop_processing(self):
        """Stop the wiki processing thread"""
        if self.processing_thread:
            self.stop_event.set()
            self.processing_thread.join()
            self.processing_thread = None
            logger.info("AutoWiki processing stopped")
    
    def _process_loop(self):
        """Main processing loop that fetches and processes random articles"""
        while not self.stop_event.is_set():
            try:
                # Fetch new batch of articles
                articles = self._fetch_random_articles()
                
                # Add articles to queue
                with self.queue_lock:
                    for article in articles:
                        self.article_queue.put(article)
                
                # Process the batch
                self._process_article_batch()
                
                # Sleep before next batch
                time.sleep(self._update_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(self.RETRY_DELAY)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'queue_size': self.article_queue.qsize(),
            'active_tasks': self._active_tasks,
            'total_tasks': self._total_tasks,
            'success_rate': self.success_rate,
            'last_update': self._last_update.isoformat()
        }
    
    def request_new_articles(self, count: int = BATCH_SIZE) -> bool:
        """Request a new batch of random articles"""
        try:
            articles = self._fetch_random_articles(count)
            
            with self.queue_lock:
                for article in articles:
                    self.article_queue.put(article)
            
            logger.info(f"Added {len(articles)} new articles to queue")
            return True
            
        except Exception as e:
            logger.error(f"Error requesting new articles: {e}")
            return False
    
    def shutdown(self):
        """Clean up resources and stop monitoring"""
        self.stop_processing()
        self.stop_event.set()  # Stop monitoring thread
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join()
        self.executor.shutdown()
        logger.info("AutoWikiProcessor shutdown complete") 
                # TODO: Implement actual wiki processing logic here
                # For now, just sleep to prevent busy waiting
                time.sleep(5)
                
                if self.on_wiki_update:
                    self.on_wiki_update("Wiki processing active")
                    
            except Exception as e:
                if self.on_wiki_update:
                    self.on_wiki_update(f"Error in wiki processing: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def process_message(self, message: str):
        """Process a message to find relevant wiki content"""
        try:
            # Basic keyword extraction (to be improved)
            keywords = self._extract_keywords(message)
            
            for keyword in keywords:
                if keyword == self.last_search:
                    continue
                    
                try:
                    summary = wikipedia.summary(keyword, sentences=2)
                    self.last_search = keyword
                    if self.on_wiki_update:
                        self.on_wiki_update(f"Related to '{keyword}': {summary}")
                    break
                except wikipedia.exceptions.DisambiguationError as e:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                    
        except Exception as e:
            if self.on_wiki_update:
                self.on_wiki_update(f"Error processing message: {str(e)}")
    
    def _extract_keywords(self, message: str) -> list:
        """Extract potential keywords from a message"""
        # This is a basic implementation that should be improved
        words = message.split()
        # Filter out common words and short words
        keywords = [word for word in words if len(word) > 4 and word.lower() not in {
            'about', 'above', 'after', 'again', 'their', 'would', 'could',
            'should', 'which', 'there', 'where', 'when', 'what', 'have'
        }]
        return keywords[:3]  # Return top 3 potential keywords
    
    def _get_random_wiki_article(self) -> Dict[str, str]:
        """Fetch a random Wikipedia article"""
        try:
            # Get random article
            response = requests.get(self.RANDOM_WIKI_URL, allow_redirects=True)
            response.raise_for_status()
            
            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title and content
            title = soup.find(id='firstHeading').text
            content = soup.find(id='mw-content-text').get_text()[:500]  # First 500 chars
            url = response.url
            
            # Clean up content
            content = ' '.join(content.split())  # Remove extra whitespace
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching random Wikipedia article: {e}")
            return None
    
    def _process_task(self, message: str, context: dict) -> None:
        """Process a single task"""
        try:
            # Get random Wikipedia article
            wiki_data = self._get_random_wiki_article()
            if not wiki_data:
                return
            
            # Save to wiki database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wiki_file = self.wiki_data_dir / f"update_{timestamp}.json"
            
            data = {
                "message": message,
                "context": context,
                "wiki_data": wiki_data
            }
            
            with open(wiki_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self._successful_tasks += 1
            logger.info(f"Task processed successfully: {wiki_data['title']}")
            
            # Notify about new wiki article
            if self.on_wiki_update:
                formatted_wiki = (
                    f"📚 Related Wikipedia Article:\n"
                    f"Title: {wiki_data['title']}\n"
                    f"Summary: {wiki_data['content'][:200]}...\n"
                    f"Read more: {wiki_data['url']}"
                )
                self.on_wiki_update(formatted_wiki)
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
        finally:
            self._active_tasks -= 1
            self._last_update = datetime.now()
    
    def process_article(self, title: str, content: str):
        """Process a Wikipedia article and notify listeners"""
        # ... existing code ...
        
        # Format update message
        update_msg = f"Processed article: {title}\nExtracted {len(self.extracted_info)} key points"
        
        # Notify listeners
        if self.on_wiki_update:
            self.on_wiki_update(update_msg)
            
        return self.extracted_info
        
    def export_info(self, filepath: str):
        """Export processed information and notify listeners"""
        # ... existing code ...
        
        # Format update message
        update_msg = f"Exported {len(self.extracted_info)} articles to {filepath}"
        
        # Notify listeners
        if self.on_wiki_update:
            self.on_wiki_update(update_msg)
            
        return True
    
    def shutdown(self):
        """Clean up resources"""
        self.executor.shutdown()
        logger.info("AutoWikiProcessor shutdown complete") 