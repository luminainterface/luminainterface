from typing import List, Dict, Optional
from PySide6.QtWidgets import QApplication
import logging
from datetime import datetime
from pathlib import Path

from .article_manager import ArticleManager
from .suggestion_engine import SuggestionEngine, Suggestion
from .wiki_ui import AutoWikiUI
from .content_generator import ContentGenerator
from .learning_engine import AutoLearningEngine

class AutoWiki:
    """Main AutoWiki application class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.article_manager = ArticleManager()
        self.suggestion_engine = SuggestionEngine()
        self.content_generator = ContentGenerator()
        self.learning_engine = AutoLearningEngine()
        self.neural_seed = None
        self.ui = None
        
    def initialize(self):
        """Initialize AutoWiki system"""
        try:
            self.article_manager.initialize()
            self.suggestion_engine.initialize()
            self.content_generator.initialize()
            self.learning_engine.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize AutoWiki: {str(e)}")
            return False
            
    def set_neural_seed(self, neural_seed):
        """Connect to neural seed"""
        self.neural_seed = neural_seed
        
    def handle_content_request(self, request):
        """Handle content generation request"""
        try:
            return self.content_generator.generate(request)
        except Exception as e:
            self.logger.error(f"Error handling content request: {str(e)}")
            return None
            
    def handle_learning_update(self, update):
        """Handle learning system update"""
        try:
            return self.learning_engine.process_update(update)
        except Exception as e:
            self.logger.error(f"Error handling learning update: {str(e)}")
            return None
            
    def get_metrics(self):
        """Get AutoWiki metrics"""
        try:
            return {
                'articles_count': len(self.article_manager.get_all_articles()),
                'suggestions_count': len(self.suggestion_engine.get_pending_suggestions()),
                'learning_progress': self.learning_engine.get_progress(),
                'content_generation_queue': len(self.content_generator.get_queue())
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {}
        
    def start(self):
        """Start the AutoWiki application"""
        try:
            # Create Qt application
            app = QApplication([])
            
            # Create and show UI
            self.ui = AutoWikiUI()
            self.ui.show()
            
            # Connect signals
            self._connect_signals()
            
            # Load initial data
            self._load_articles()
            
            # Start event loop
            return app.exec()
            
        except Exception as e:
            self.logger.error(f"Error starting AutoWiki: {str(e)}")
            raise
            
    def _connect_signals(self):
        """Connect UI signals to handlers"""
        if not self.ui:
            return
            
        # Article list signals
        self.ui.article_list.article_selected.connect(self._handle_article_selected)
        self.ui.article_list.article_search.connect(self._handle_article_search)
        
        # Article editor signals
        self.ui.article_editor.save_clicked.connect(self._handle_save_article)
        self.ui.article_editor.generate_clicked.connect(self._handle_generate_content)
        
        # Suggestion panel signals
        self.ui.suggestion_panel.suggestion_applied.connect(self._handle_apply_suggestion)
        
    def _load_articles(self):
        """Load articles into UI"""
        try:
            articles = self.article_manager.get_all_articles()
            if self.ui:
                self.ui.article_list.load_articles(articles)
                
        except Exception as e:
            self.logger.error(f"Error loading articles: {str(e)}")
            if self.ui:
                self.ui.show_error("Failed to load articles")
                
    def _handle_article_selected(self, article_id: int):
        """Handle article selection"""
        try:
            # Get article
            article = self.article_manager.get_article_by_id(article_id)
            if not article:
                return
                
            # Update editor
            if self.ui:
                self.ui.article_editor.load_article(article)
                
            # Generate suggestions
            suggestions = self.suggestion_engine.analyze_article(
                article['title'],
                article['content'],
                article['category']
            )
            
            # Update suggestion panel
            if self.ui:
                self.ui.suggestion_panel.load_suggestions(suggestions)
                
        except Exception as e:
            self.logger.error(f"Error handling article selection: {str(e)}")
            if self.ui:
                self.ui.show_error("Failed to load article")
                
    def _handle_article_search(self, query: str, category: Optional[str] = None):
        """Handle article search"""
        try:
            articles = self.article_manager.search_articles(query, category)
            if self.ui:
                self.ui.article_list.load_articles(articles)
                
        except Exception as e:
            self.logger.error(f"Error searching articles: {str(e)}")
            if self.ui:
                self.ui.show_error("Failed to search articles")
                
    def _handle_save_article(self, article_data: Dict):
        """Handle article save"""
        try:
            success = self.article_manager.save_article(article_data)
            if success and self.ui:
                self.ui.show_message("Article saved successfully")
                self._load_articles()  # Refresh article list
            elif self.ui:
                self.ui.show_error("Failed to save article")
                
        except Exception as e:
            self.logger.error(f"Error saving article: {str(e)}")
            if self.ui:
                self.ui.show_error("Failed to save article")
                
    def _handle_generate_content(self, article_data: Dict):
        """Handle content generation request"""
        try:
            # TODO: Implement content generation using AI
            # For now, just show a message
            if self.ui:
                self.ui.show_message("Content generation not implemented yet")
                
        except Exception as e:
            self.logger.error(f"Error generating content: {str(e)}")
            if self.ui:
                self.ui.show_error("Failed to generate content")
                
    def _handle_apply_suggestion(self, suggestion: Suggestion):
        """Handle suggestion application"""
        try:
            # Get current article content
            article = self.article_manager.get_article(suggestion.article_title)
            if not article:
                return
                
            # Apply suggestion
            new_content = self.suggestion_engine.apply_suggestion(
                article['content'],
                suggestion
            )
            
            if new_content:
                # Update article
                article['content'] = new_content
                success = self.article_manager.save_article(article)
                
                if success and self.ui:
                    self.ui.show_message("Suggestion applied successfully")
                    self.ui.article_editor.load_article(article)
                elif self.ui:
                    self.ui.show_error("Failed to apply suggestion")
                    
        except Exception as e:
            self.logger.error(f"Error applying suggestion: {str(e)}")
            if self.ui:
                self.ui.show_error("Failed to apply suggestion")
                
def main():
    """Main entry point"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Start application
        app = AutoWiki()
        return app.start()
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        return 1
        
if __name__ == '__main__':
    exit(main()) 