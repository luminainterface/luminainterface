import logging
from pathlib import Path
import json
from datetime import datetime

class ArticleManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.articles = {}
        self.articles_dir = Path("data/articles")
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self):
        """Initialize article database"""
        try:
            # Load existing articles
            for article_file in self.articles_dir.glob("*.json"):
                with open(article_file, "r") as f:
                    article = json.load(f)
                    self.articles[article["id"]] = article
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize article manager: {str(e)}")
            return False
            
    def get_all_articles(self):
        """Get all articles"""
        return self.articles
        
    def add_article(self, title, content, category):
        """Add a new article"""
        try:
            article_id = str(len(self.articles) + 1)
            article = {
                "id": article_id,
                "title": title,
                "content": content,
                "category": category,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Save to file
            article_path = self.articles_dir / f"article_{article_id}.json"
            with open(article_path, "w") as f:
                json.dump(article, f, indent=4)
                
            # Add to memory
            self.articles[article_id] = article
            return article_id
        except Exception as e:
            self.logger.error(f"Failed to add article: {str(e)}")
            return None
            
    def update_article(self, article_id, content):
        """Update an existing article"""
        try:
            if article_id not in self.articles:
                return False
                
            self.articles[article_id]["content"] = content
            self.articles[article_id]["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            article_path = self.articles_dir / f"article_{article_id}.json"
            with open(article_path, "w") as f:
                json.dump(self.articles[article_id], f, indent=4)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to update article: {str(e)}")
            return False
            
    def delete_article(self, article_id):
        """Delete an article"""
        try:
            if article_id not in self.articles:
                return False
                
            # Remove file
            article_path = self.articles_dir / f"article_{article_id}.json"
            article_path.unlink(missing_ok=True)
            
            # Remove from memory
            del self.articles[article_id]
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete article: {str(e)}")
            return False 