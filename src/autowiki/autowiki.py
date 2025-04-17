import logging
from pathlib import Path
from datetime import datetime

from .article_manager import ArticleManager
from .suggestion_engine import SuggestionEngine
from .content_generator import ContentGenerator
from .learning_engine import AutoLearningEngine

class AutoWiki:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.article_manager = ArticleManager()
        self.suggestion_engine = SuggestionEngine()
        self.content_generator = ContentGenerator()
        self.learning_engine = AutoLearningEngine()
        self.neural_seed = None
        
    def initialize(self):
        """Initialize AutoWiki system"""
        try:
            # Initialize all components
            components = [
                self.article_manager,
                self.suggestion_engine,
                self.content_generator,
                self.learning_engine
            ]
            
            for component in components:
                if not component.initialize():
                    self.logger.error(f"Failed to initialize {component.__class__.__name__}")
                    return False
                    
            self.logger.info("AutoWiki system initialized successfully")
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
            # Add request to queue
            self.content_generator.add_to_queue(request)
            
            # Process queue and get results
            results = self.content_generator.process_queue()
            
            # Add suggestions for generated content
            for content in results:
                self.suggestion_engine.add_suggestion(
                    content=content,
                    source="content_generator",
                    confidence=0.8
                )
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error handling content request: {str(e)}")
            return None
            
    def handle_learning_update(self, update):
        """Handle learning system update"""
        try:
            # Process update in learning engine
            if not self.learning_engine.process_update(update):
                return False
                
            # Generate suggestions based on learned patterns
            if update.get('patterns'):
                for pattern in update['patterns']:
                    self.suggestion_engine.add_suggestion(
                        content=pattern,
                        source="learning_engine",
                        confidence=0.9
                    )
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling learning update: {str(e)}")
            return False
            
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