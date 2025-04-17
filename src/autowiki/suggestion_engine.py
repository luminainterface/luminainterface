import logging
from datetime import datetime, timedelta

class Suggestion:
    def __init__(self, content, source, confidence):
        self.content = content
        self.source = source
        self.confidence = confidence
        self.created_at = datetime.now()
        self.status = "pending"

class SuggestionEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.suggestions = []
        self.min_confidence = 0.7
        
    def initialize(self):
        """Initialize suggestion engine"""
        try:
            self.suggestions = []
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize suggestion engine: {str(e)}")
            return False
            
    def add_suggestion(self, content, source, confidence):
        """Add a new suggestion"""
        try:
            if confidence < self.min_confidence:
                return False
                
            suggestion = Suggestion(content, source, confidence)
            self.suggestions.append(suggestion)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add suggestion: {str(e)}")
            return False
            
    def get_pending_suggestions(self):
        """Get all pending suggestions"""
        return [s for s in self.suggestions if s.status == "pending"]
        
    def approve_suggestion(self, suggestion):
        """Approve a suggestion"""
        try:
            suggestion.status = "approved"
            return True
        except Exception as e:
            self.logger.error(f"Failed to approve suggestion: {str(e)}")
            return False
            
    def reject_suggestion(self, suggestion):
        """Reject a suggestion"""
        try:
            suggestion.status = "rejected"
            return True
        except Exception as e:
            self.logger.error(f"Failed to reject suggestion: {str(e)}")
            return False
            
    def clear_old_suggestions(self, days=30):
        """Clear suggestions older than specified days"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            self.suggestions = [s for s in self.suggestions 
                              if s.created_at > cutoff or s.status == "pending"]
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear old suggestions: {str(e)}")
            return False 