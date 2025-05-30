#!/usr/bin/env python3
"""
Token Limiter - Hard Word Count Enforcement
===========================================

Enforces hard word count limits by truncating and optimizing text
until it meets the specified constraints.
"""

import re
from typing import List, Optional

class TokenLimiter:
    """Hard enforcement of word count limits"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
    
    def enforce_word_limit(self, text: str, max_words: int, 
                          preserve_structure: bool = True) -> str:
        """
        Enforce hard word limit by truncating and optimizing text
        
        Args:
            text: Input text to limit
            max_words: Maximum number of words allowed
            preserve_structure: Whether to preserve sentence structure
        
        Returns:
            Text limited to max_words
        """
        if not text or max_words <= 0:
            return ""
        
        words = text.split()
        current_count = len(words)
        
        if current_count <= max_words:
            return text
        
        print(f"🔧 TOKEN LIMITER: Reducing {current_count} words to {max_words}")
        
        if preserve_structure:
            return self._smart_truncate(text, max_words)
        else:
            return self._hard_truncate(text, max_words)
    
    def _hard_truncate(self, text: str, max_words: int) -> str:
        """Simple hard truncation at word boundary"""
        words = text.split()
        if len(words) <= max_words:
            return text
        
        truncated = ' '.join(words[:max_words])
        
        # Try to end at sentence boundary if possible
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > len(truncated) * 0.8:  # If sentence end is near the end
            return truncated[:last_sentence_end + 1]
        
        return truncated + "..."
    
    def _smart_truncate(self, text: str, max_words: int) -> str:
        """Smart truncation preserving key information"""
        
        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        
        # Step 2: Prioritize sentences by importance
        prioritized = self._prioritize_sentences(sentences)
        
        # Step 3: Build result within word limit
        result_sentences = []
        current_words = 0
        
        for sentence, priority in prioritized:
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words <= max_words:
                result_sentences.append(sentence)
                current_words += sentence_words
            elif current_words < max_words * 0.9:  # If we have room, try to compress
                compressed = self._compress_sentence(sentence, max_words - current_words)
                if compressed:
                    result_sentences.append(compressed)
                    current_words += len(compressed.split())
                break
            else:
                break
        
        result = ' '.join(result_sentences)
        
        # Final check and hard truncate if needed
        if len(result.split()) > max_words:
            result = self._hard_truncate(result, max_words)
        
        return result
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _prioritize_sentences(self, sentences: List[str]) -> List[tuple]:
        """Prioritize sentences by importance"""
        prioritized = []
        
        for i, sentence in enumerate(sentences):
            priority = 0
            
            # First and last sentences are important
            if i == 0:
                priority += 10
            if i == len(sentences) - 1:
                priority += 5
            
            # Sentences with numbers/data are important
            if re.search(r'\d+', sentence):
                priority += 3
            
            # Sentences with key business terms
            key_terms = ['revenue', 'growth', 'million', 'percent', 'increase', 'decrease', 
                        'target', 'goal', 'strategy', 'performance', 'results']
            for term in key_terms:
                if term.lower() in sentence.lower():
                    priority += 2
                    break
            
            # Longer sentences might be more informative
            if len(sentence.split()) > 10:
                priority += 1
            
            prioritized.append((sentence, priority))
        
        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized
    
    def _compress_sentence(self, sentence: str, max_words: int) -> Optional[str]:
        """Compress a sentence to fit within word limit"""
        words = sentence.split()
        
        if len(words) <= max_words:
            return sentence
        
        if max_words < 3:
            return None
        
        # Remove stop words first
        essential_words = []
        for word in words:
            if word.lower() not in self.stop_words or len(essential_words) < max_words // 2:
                essential_words.append(word)
        
        # If still too long, hard truncate
        if len(essential_words) > max_words:
            essential_words = essential_words[:max_words]
        
        return ' '.join(essential_words)
    
    def optimize_for_length(self, text: str, target_words: int, 
                           tolerance: float = 0.1) -> str:
        """
        Optimize text to be close to target word count
        
        Args:
            text: Input text
            target_words: Target word count
            tolerance: Acceptable deviation (0.1 = 10%)
        
        Returns:
            Optimized text
        """
        current_words = len(text.split())
        min_words = int(target_words * (1 - tolerance))
        max_words = int(target_words * (1 + tolerance))
        
        if min_words <= current_words <= max_words:
            return text
        
        if current_words > max_words:
            # Need to shorten
            return self.enforce_word_limit(text, max_words)
        else:
            # Text is too short - could expand, but for now just return as-is
            return text
    
    def get_word_count(self, text: str) -> int:
        """Get accurate word count"""
        if not text:
            return 0
        return len(text.split())
    
    def check_compliance(self, text: str, max_words: int) -> dict:
        """Check if text complies with word limit"""
        word_count = self.get_word_count(text)
        
        return {
            'compliant': word_count <= max_words,
            'current_words': word_count,
            'max_words': max_words,
            'excess_words': max(0, word_count - max_words),
            'compliance_rate': min(1.0, max_words / word_count) if word_count > 0 else 1.0
        }

def test_token_limiter():
    """Test the token limiter with sample text"""
    limiter = TokenLimiter()
    
    # Test with long business text
    long_text = """
    Our company achieved remarkable financial performance in Q4 2024 with revenue reaching $47.2 million, 
    representing a significant 23% increase year-over-year. This exceptional growth was driven primarily 
    by our innovative new enterprise software platform and our strategic expansion into international markets. 
    The gross margin remained strong at 68% while operating margin improved substantially to 22%. 
    Cash flow from operations was robust at $12.3 million, up from $9.1 million in the previous quarter. 
    We successfully completed our Series C funding round in November, raising $85 million led by 
    Venture Partners Group. Customer acquisition cost decreased to $1,850 from $2,100 last quarter 
    while lifetime value to CAC ratio improved to an impressive 4.2:1. Monthly recurring revenue 
    reached $14.8 million with a net revenue retention rate of 118%. Looking ahead to 2025, 
    our strategic priorities include expanding AI capabilities, entering the Asian market, 
    and achieving SOC 2 Type II compliance.
    """
    
    print(f"Original text: {limiter.get_word_count(long_text)} words")
    
    # Test limiting to 50 words
    limited = limiter.enforce_word_limit(long_text, 50, preserve_structure=True)
    print(f"\nLimited to 50 words: {limiter.get_word_count(limited)} words")
    print(f"Result: {limited}")
    
    # Test compliance check
    compliance = limiter.check_compliance(limited, 50)
    print(f"\nCompliance check: {compliance}")
    
    return limited

if __name__ == "__main__":
    test_token_limiter() 