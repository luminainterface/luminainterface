"""
AutoWiki Integration Module
Integrates all AutoWiki components into a cohesive system
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import json

from .article_manager import ArticleManager
from .suggestion_engine import SuggestionEngine
from .content_generator import ContentGenerator
from .auto_learning import AutoLearningEngine

logger = logging.getLogger(__name__)

class AutoWikiIntegration:
    def __init__(self, data_dir: str = "data/wiki"):
        """Initialize AutoWiki integration"""
        self.logger = logger
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize components
        self.article_manager = ArticleManager(os.path.join(data_dir, "articles.db"))
        self.suggestion_engine = SuggestionEngine()
        self.content_generator = ContentGenerator()
        self.learning_engine = AutoLearningEngine()
        
        # Track active sessions
        self.active_sessions = {}
        
    def create_article(self, title: str, content: str, category: str = None) -> Dict[str, Any]:
        """Create a new article with automatic enhancements"""
        try:
            # Generate article ID and create base article
            article = self.article_manager.create_article(title, content, category)
            article_id = article['id']
            
            # Analyze content and generate suggestions
            suggestions = self.suggestion_engine.analyze_content(content)
            
            # Learn patterns from content
            patterns = self.learning_engine.learn_patterns(
                content,
                metadata={
                    'article_id': article_id,
                    'title': title,
                    'category': category
                }
            )
            
            # Generate outline and summary
            outline = self.content_generator.generate_outline(title)
            summary = self.content_generator.generate_summary(content)
            
            # Update article with enhancements
            enhanced_article = {
                **article,
                'suggestions': suggestions,
                'patterns': patterns,
                'outline': outline,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store enhancements
            self._store_article_metadata(article_id, enhanced_article)
            
            return enhanced_article
            
        except Exception as e:
            logger.error(f"Failed to create enhanced article: {str(e)}")
            raise
            
    def update_article(self, article_id: str, content: str) -> Dict[str, Any]:
        """Update article with automatic enhancements"""
        try:
            # Update base article
            article = self.article_manager.update_article(article_id, content)
            
            # Get existing metadata
            metadata = self._load_article_metadata(article_id)
            
            # Generate new suggestions
            suggestions = self.suggestion_engine.analyze_content(content)
            
            # Learn new patterns
            patterns = self.learning_engine.learn_patterns(
                content,
                metadata={
                    'article_id': article_id,
                    'title': article['title'],
                    'category': article.get('category')
                }
            )
            
            # Generate new summary
            summary = self.content_generator.generate_summary(content)
            
            # Update metadata
            metadata.update({
                'suggestions': suggestions,
                'patterns': patterns,
                'summary': summary,
                'last_updated': datetime.now().isoformat()
            })
            
            # Store updated metadata
            self._store_article_metadata(article_id, metadata)
            
            return {**article, **metadata}
            
        except Exception as e:
            logger.error(f"Failed to update article: {str(e)}")
            raise
            
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article with all enhancements"""
        try:
            # Get base article
            article = self.article_manager.get_article(article_id)
            if not article:
                return None
                
            # Get metadata
            metadata = self._load_article_metadata(article_id)
            
            return {**article, **metadata}
            
        except Exception as e:
            logger.error(f"Failed to get article: {str(e)}")
            raise
            
    def search_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search articles with pattern matching"""
        try:
            # Get base search results
            articles = self.article_manager.search_articles(query)
            
            # Find similar patterns
            patterns = self.learning_engine.find_similar_patterns(query)
            
            # Enhance results with pattern matches
            enhanced_results = []
            for article in articles:
                # Get article metadata
                metadata = self._load_article_metadata(article['id'])
                
                # Calculate relevance score
                relevance = self._calculate_relevance(
                    query,
                    article,
                    patterns,
                    metadata.get('patterns', {})
                )
                
                enhanced_results.append({
                    **article,
                    **metadata,
                    'relevance': relevance
                })
                
            # Sort by relevance
            enhanced_results.sort(key=lambda x: x['relevance'], reverse=True)
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to search articles: {str(e)}")
            raise
            
    def get_article_suggestions(self, article_id: str) -> Dict[str, Any]:
        """Get suggestions for article improvement"""
        try:
            # Get article
            article = self.article_manager.get_article(article_id)
            if not article:
                raise ValueError(f"Article {article_id} not found")
                
            # Generate suggestions
            suggestions = self.suggestion_engine.analyze_content(article['content'])
            
            # Get similar patterns
            patterns = self.learning_engine.find_similar_patterns(article['content'])
            
            # Generate potential improvements
            improvements = self.content_generator.generate_suggestions(
                article['content'],
                patterns[:5]  # Use top 5 similar patterns
            )
            
            return {
                'article_id': article_id,
                'suggestions': suggestions,
                'similar_patterns': patterns,
                'improvements': improvements,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get article suggestions: {str(e)}")
            raise
            
    def start_editing_session(self, article_id: str) -> str:
        """Start an interactive editing session"""
        try:
            # Generate session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize session
            self.active_sessions[session_id] = {
                'article_id': article_id,
                'start_time': datetime.now().isoformat(),
                'suggestions': [],
                'changes': []
            }
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start editing session: {str(e)}")
            raise
            
    def get_session_suggestions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get real-time suggestions for editing session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
                
            # Get article
            article = self.article_manager.get_article(session['article_id'])
            
            # Generate real-time suggestions
            suggestions = self.suggestion_engine.analyze_content(article['content'])
            
            # Update session
            session['suggestions'] = suggestions
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get session suggestions: {str(e)}")
            raise
            
    def end_editing_session(self, session_id: str) -> Dict[str, Any]:
        """End an editing session and save changes"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
                
            # Get final article state
            article = self.article_manager.get_article(session['article_id'])
            
            # Generate final suggestions
            final_suggestions = self.suggestion_engine.analyze_content(article['content'])
            
            # Update session metadata
            session.update({
                'end_time': datetime.now().isoformat(),
                'final_suggestions': final_suggestions
            })
            
            # Store session history
            self._store_session_history(session_id, session)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to end editing session: {str(e)}")
            raise
            
    def _calculate_relevance(self, query: str, article: Dict, 
                           query_patterns: List[Dict], article_patterns: Dict) -> float:
        """Calculate relevance score for search results"""
        try:
            # Base relevance from title/content match
            base_score = 0.5
            
            # Add pattern similarity score
            pattern_score = 0.0
            if query_patterns and article_patterns:
                # Compare patterns
                similarities = []
                for qp in query_patterns:
                    for ap in article_patterns.get('key_terms', []):
                        if qp['key_terms'] == ap:
                            similarities.append(qp['similarity'])
                            
                if similarities:
                    pattern_score = max(similarities)
                    
            # Combine scores
            return base_score + (pattern_score * 0.5)  # Weight pattern matching as 50%
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance: {str(e)}")
            return 0.0
            
    def _store_article_metadata(self, article_id: str, metadata: Dict):
        """Store article metadata"""
        try:
            metadata_path = os.path.join(self.data_dir, f"metadata_{article_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"Failed to store article metadata: {str(e)}")
            raise
            
    def _load_article_metadata(self, article_id: str) -> Dict:
        """Load article metadata"""
        try:
            metadata_path = os.path.join(self.data_dir, f"metadata_{article_id}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load article metadata: {str(e)}")
            return {}
            
    def _store_session_history(self, session_id: str, session_data: Dict):
        """Store editing session history"""
        try:
            history_path = os.path.join(self.data_dir, f"session_{session_id}.json")
            with open(history_path, 'w') as f:
                json.dump(session_data, f)
                
        except Exception as e:
            logger.error(f"Failed to store session history: {str(e)}")
            raise 