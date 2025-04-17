#!/usr/bin/env python3
"""
Literal Processing Path

This module implements the literal processing path that handles exact pattern matching
and direct data transformations.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..core import ProcessingPath, PathType, PathConfig
from ...ml.core import MLConfig

@dataclass
class LiteralProcessorConfig:
    """Configuration for literal processing"""
    exact_match_threshold: float = 0.95
    case_sensitive: bool = True
    pattern_weights: Dict[str, float] = None
    max_patterns: int = 1000
    index_type: str = "hash"  # or "tree"

class LiteralProcessor:
    """Processor for literal pattern matching"""
    
    def __init__(self, config: LiteralProcessorConfig):
        self.config = config
        self.patterns: Dict[str, Any] = {}
        self.pattern_index = (
            self._create_hash_index()
            if config.index_type == "hash"
            else self._create_tree_index()
        )
        
    def _create_hash_index(self):
        """Create hash-based pattern index"""
        return {}
        
    def _create_tree_index(self):
        """Create tree-based pattern index"""
        # Implement trie or similar tree structure
        return {}
        
    def add_pattern(self, pattern: str, data: Any) -> None:
        """Add pattern to index"""
        if len(self.patterns) >= self.config.max_patterns:
            # Remove least used pattern
            min_pattern = min(self.patterns.items(), key=lambda x: x[1]['usage'])
            del self.patterns[min_pattern[0]]
            
        pattern_key = pattern if self.config.case_sensitive else pattern.lower()
        self.patterns[pattern_key] = {
            'data': data,
            'usage': 0,
            'weight': self.config.pattern_weights.get(pattern, 1.0)
        }
        
        # Update index
        if self.config.index_type == "hash":
            self.pattern_index[pattern_key] = len(self.patterns) - 1
        else:
            self._add_to_tree(pattern_key, data)
            
    def _add_to_tree(self, pattern: str, data: Any) -> None:
        """Add pattern to tree index"""
        current = self.pattern_index
        for char in pattern:
            if char not in current:
                current[char] = {}
            current = current[char]
        current['$'] = data
        
    def find_exact_matches(self, text: str) -> List[Dict[str, Any]]:
        """Find exact pattern matches"""
        if not self.config.case_sensitive:
            text = text.lower()
            
        matches = []
        if self.config.index_type == "hash":
            # Direct lookup
            if text in self.patterns:
                pattern_data = self.patterns[text]
                matches.append({
                    'pattern': text,
                    'data': pattern_data['data'],
                    'confidence': 1.0 * pattern_data['weight']
                })
        else:
            # Tree search
            matches.extend(self._tree_search(text))
            
        return matches
        
    def _tree_search(self, text: str) -> List[Dict[str, Any]]:
        """Search patterns using tree index"""
        matches = []
        current = self.pattern_index
        
        def search_recursive(node, prefix, start_pos):
            if '$' in node:
                pattern_data = self.patterns[prefix]
                matches.append({
                    'pattern': prefix,
                    'data': pattern_data['data'],
                    'confidence': 1.0 * pattern_data['weight'],
                    'position': start_pos
                })
                
            for char, next_node in node.items():
                if char != '$':
                    search_recursive(next_node, prefix + char, start_pos)
                    
        for i in range(len(text)):
            if text[i] in current:
                search_recursive(current[text[i]], text[i], i)
                
        return matches

class LiteralPath(ProcessingPath):
    """Implementation of literal processing path"""
    
    def __init__(
        self,
        ml_config: MLConfig,
        processor_config: LiteralProcessorConfig
    ):
        super().__init__(PathConfig(
            path_type=PathType.LITERAL,
            ml_config=ml_config,
            bridge_config={
                'exact_match_priority': True,
                'pattern_verification': True
            },
            web_config={
                'connection_type': 'direct',
                'verification_required': True
            },
            processing_weights={
                'input': 1.0,
                'ml_processing': 0.5,
                'bridge_transfer': 1.0,
                'web_integration': 0.8,
                'output': 1.0
            }
        ))
        self.processor = LiteralProcessor(processor_config)
        
    async def process_input(self, data: Any) -> Any:
        """Process input data"""
        # Find exact matches
        matches = self.processor.find_exact_matches(str(data))
        
        # Prepare data for ML processing
        processed_data = {
            'original': data,
            'matches': matches,
            'confidence': max([m['confidence'] for m in matches]) if matches else 0.0
        }
        
        return processed_data
        
    async def verify_output(self, data: Any) -> Any:
        """Verify output matches"""
        if not isinstance(data, dict) or 'matches' not in data:
            return {'verified': False, 'data': data}
            
        # Verify each match
        verified_matches = []
        for match in data['matches']:
            if match['confidence'] >= self.processor.config.exact_match_threshold:
                verified_matches.append(match)
                
        return {
            'verified': len(verified_matches) > 0,
            'matches': verified_matches,
            'confidence': max([m['confidence'] for m in verified_matches]) if verified_matches else 0.0
        }
        
    def add_patterns(self, patterns: Dict[str, Any]) -> None:
        """Add patterns to processor"""
        for pattern, data in patterns.items():
            self.processor.add_pattern(pattern, data)
            
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        return {
            'total_patterns': len(self.processor.patterns),
            'index_type': self.processor.config.index_type,
            'case_sensitive': self.processor.config.case_sensitive,
            'threshold': self.processor.config.exact_match_threshold
        } 