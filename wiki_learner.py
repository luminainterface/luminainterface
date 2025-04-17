import wikipedia
import requests
from bs4 import BeautifulSoup
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Set, Any
from tqdm import tqdm
from lumina_training import LuminaTrainer
import random
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wiki_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WikiLearner:
    def __init__(self, save_dir: str = "data/wiki_knowledge", hybrid_node=None, wiki_vocab=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed articles and links
        self.processed_articles: Set[str] = set()
        self.article_queue: Set[str] = set()
        self.knowledge_graph: Dict[str, List[str]] = {}
        
        # Training data storage
        self.training_data: List[str] = []
        self.max_articles = 1000  # Limit total articles to process
        self.max_links_per_article = 5  # Limit links to follow per article
        
        # Load existing progress if any
        self._load_progress()
        
        self.hybrid_node = hybrid_node
        self.wiki_vocab = wiki_vocab
        
        logger.info("WikiLearner initialized with basic configuration")

    def _load_progress(self):
        """Load previously processed articles and knowledge graph"""
        try:
            progress_file = self.save_dir / "learning_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_articles = set(data.get("processed_articles", []))
                    self.knowledge_graph = data.get("knowledge_graph", {})
                    logger.info(f"Loaded {len(self.processed_articles)} processed articles")
        except Exception as e:
            logger.error(f"Error loading progress: {str(e)}")

    def _save_progress(self):
        """Save current progress"""
        try:
            progress_file = self.save_dir / "learning_progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_articles": list(self.processed_articles),
                    "knowledge_graph": self.knowledge_graph
                }, f, indent=4)
            logger.info("Progress saved")
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove citations [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def collect_article(self, title: str) -> Dict[str, Any]:
        """Collect and process a Wikipedia article"""
        try:
            # Get article content
            page = wikipedia.page(title, auto_suggest=False)
            
            # Extract main text and clean it
            content = self._clean_text(page.content)
            
            # Extract links
            links = [link for link in page.links 
                    if link not in self.processed_articles 
                    and not any(c in link for c in [':', '(', ')'])]
            
            # Limit links to prevent exponential growth
            links = random.sample(links, min(len(links), self.max_links_per_article))
            
            # Update knowledge graph
            self.knowledge_graph[title] = links
            
            # Add links to queue
            self.article_queue.update(links)
            
            return {
                "title": title,
                "content": content,
                "links": links,
                "url": page.url
            }
            
        except Exception as e:
            logger.error(f"Error collecting article '{title}': {str(e)}")
            return {}

    def process_articles(self, seed_topics: List[str]):
        """Process articles starting from seed topics"""
        try:
            # Add seed topics to queue
            self.article_queue.update(seed_topics)
            
            # Process articles breadth-first
            while self.article_queue and len(self.processed_articles) < self.max_articles:
                # Get next article
                title = self.article_queue.pop()
                
                # Skip if already processed
                if title in self.processed_articles:
                    continue
                
                logger.info(f"Processing article: {title}")
                
                # Collect article
                article = self.collect_article(title)
                if not article:
                    continue
                
                # Add to training data
                self.training_data.append(article["content"])
                
                # Mark as processed
                self.processed_articles.add(title)
                
                # Save progress periodically
                if len(self.processed_articles) % 10 == 0:
                    self._save_progress()
                    self._save_training_data()
                
                # Add delay to be nice to Wikipedia
                time.sleep(1)
            
            # Final save
            self._save_progress()
            self._save_training_data()
            
            logger.info(f"Processed {len(self.processed_articles)} articles")
            
        except Exception as e:
            logger.error(f"Error processing articles: {str(e)}")

    def _save_training_data(self):
        """Save collected training data"""
        try:
            # Save full text
            training_file = self.save_dir / "english_training.txt"
            with open(training_file, 'w', encoding='utf-8') as f:
                for text in self.training_data:
                    # Split into sentences for better training
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence.split()) >= 3:  # Only keep sentences with at least 3 words
                            f.write(sentence + '\n')
            
            logger.info(f"Saved {len(self.training_data)} articles to training data")
            
        except Exception as e:
            logger.error(f"Error saving training data: {str(e)}")

    def train_on_collected_data(self):
        """Train Lumina on collected Wikipedia data"""
        try:
            # Initialize trainer
            trainer = LuminaTrainer(model_path="models/wiki_trained")
            
            # Train on collected data
            trainer.train(str(self.save_dir / "english_training.txt"))
            
            logger.info("Training completed")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")

    def learn_from_article(self, title, content):
        """Process and learn from a Wikipedia article."""
        if self.hybrid_node and self.wiki_vocab:
            try:
                _, resonance = self.hybrid_node.process(content)
                logger.info(f"Processed article '{title}' with resonance {resonance:.2f}")
                return True
            except Exception as e:
                logger.error(f"Error learning from article '{title}': {e}")
        return False

def run_wiki_learning():
    """Run the Wikipedia learning process"""
    try:
        # Initialize learner
        learner = WikiLearner()
        
        # Define seed topics covering various domains
        seed_topics = [
            "Artificial intelligence",
            "Natural language processing",
            "Machine learning",
            "Neural networks",
            "Deep learning",
            "Cognitive science",
            "Language acquisition",
            "Linguistics",
            "Psychology",
            "Philosophy of mind"
        ]
        
        # Collect and process articles
        logger.info("Starting article collection...")
        learner.process_articles(seed_topics)
        
        # Train on collected data
        logger.info("Starting training on collected data...")
        learner.train_on_collected_data()
        
        logger.info("Wiki learning process completed")
        
    except Exception as e:
        logger.error(f"Error in wiki learning process: {str(e)}")

if __name__ == "__main__":
    run_wiki_learning() 