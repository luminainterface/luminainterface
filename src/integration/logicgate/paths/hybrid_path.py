#!/usr/bin/env python3
"""
Hybrid Processing Path

This module implements the hybrid processing path that combines literal and semantic
processing for comprehensive pattern matching and understanding.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..core import ProcessingPath, PathType, PathConfig
from ...ml.core import MLConfig
from .literal_path import LiteralPath, LiteralProcessorConfig
from .semantic_path import SemanticPath, SemanticProcessorConfig

@dataclass
class HybridProcessorConfig:
    """Configuration for hybrid processing"""
    literal_weight: float = 0.4
    semantic_weight: float = 0.6
    combination_strategy: str = "weighted"  # or "adaptive", "selective"
    confidence_threshold: float = 0.7
    adaptive_window: int = 100
    use_context_boost: bool = True
    context_boost_factor: float = 1.2

class HybridProcessor:
    """Processor combining literal and semantic approaches"""
    
    def __init__(self, config: HybridProcessorConfig):
        self.config = config
        self.literal_processor = None
        self.semantic_processor = None
        self.processing_history: List[Dict[str, Any]] = []
        
    def setup_processors(
        self,
        literal_path: LiteralPath,
        semantic_path: SemanticPath
    ) -> None:
        """Setup component processors"""
        self.literal_processor = literal_path.processor
        self.semantic_processor = semantic_path.processor
        
    def process_hybrid(self, text: str) -> Dict[str, Any]:
        """Process text using both approaches"""
        # Get literal matches
        literal_matches = self.literal_processor.find_exact_matches(text)
        
        # Get semantic matches
        semantic_matches = self.semantic_processor.find_semantic_matches(text)
        
        # Combine results based on strategy
        if self.config.combination_strategy == "weighted":
            combined_results = self._weighted_combination(
                literal_matches,
                semantic_matches
            )
        elif self.config.combination_strategy == "adaptive":
            combined_results = self._adaptive_combination(
                literal_matches,
                semantic_matches
            )
        else:  # selective
            combined_results = self._selective_combination(
                literal_matches,
                semantic_matches
            )
            
        # Apply context boost if enabled
        if self.config.use_context_boost:
            combined_results = self._apply_context_boost(combined_results)
            
        # Update processing history
        self._update_history(combined_results)
        
        return combined_results
        
    def _weighted_combination(
        self,
        literal_matches: List[Dict[str, Any]],
        semantic_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine results using fixed weights"""
        combined_matches = []
        
        # Process literal matches
        for match in literal_matches:
            combined_matches.append({
                'text': match['pattern'],
                'data': match['data'],
                'confidence': match['confidence'] * self.config.literal_weight,
                'source': 'literal'
            })
            
        # Process semantic matches
        for match in semantic_matches:
            combined_matches.append({
                'text': match['text'],
                'data': match['context'],
                'confidence': match['similarity'] * self.config.semantic_weight,
                'source': 'semantic'
            })
            
        # Sort by confidence
        combined_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'matches': combined_matches,
            'confidence': max([m['confidence'] for m in combined_matches]) if combined_matches else 0.0,
            'literal_count': len(literal_matches),
            'semantic_count': len(semantic_matches)
        }
        
    def _adaptive_combination(
        self,
        literal_matches: List[Dict[str, Any]],
        semantic_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine results with adaptive weights"""
        # Calculate success rates from history
        literal_success = self._calculate_success_rate('literal')
        semantic_success = self._calculate_success_rate('semantic')
        
        # Adjust weights based on success rates
        total_success = literal_success + semantic_success
        if total_success > 0:
            literal_weight = literal_success / total_success
            semantic_weight = semantic_success / total_success
        else:
            literal_weight = self.config.literal_weight
            semantic_weight = self.config.semantic_weight
            
        # Apply adaptive weights
        combined_matches = []
        
        for match in literal_matches:
            combined_matches.append({
                'text': match['pattern'],
                'data': match['data'],
                'confidence': match['confidence'] * literal_weight,
                'source': 'literal'
            })
            
        for match in semantic_matches:
            combined_matches.append({
                'text': match['text'],
                'data': match['context'],
                'confidence': match['similarity'] * semantic_weight,
                'source': 'semantic'
            })
            
        combined_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'matches': combined_matches,
            'confidence': max([m['confidence'] for m in combined_matches]) if combined_matches else 0.0,
            'literal_weight': literal_weight,
            'semantic_weight': semantic_weight
        }
        
    def _selective_combination(
        self,
        literal_matches: List[Dict[str, Any]],
        semantic_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Selectively combine results based on confidence"""
        combined_matches = []
        
        # Add high-confidence literal matches
        for match in literal_matches:
            if match['confidence'] >= self.config.confidence_threshold:
                combined_matches.append({
                    'text': match['pattern'],
                    'data': match['data'],
                    'confidence': match['confidence'],
                    'source': 'literal'
                })
                
        # Add high-confidence semantic matches
        for match in semantic_matches:
            if match['similarity'] >= self.config.confidence_threshold:
                combined_matches.append({
                    'text': match['text'],
                    'data': match['context'],
                    'confidence': match['similarity'],
                    'source': 'semantic'
                })
                
        combined_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'matches': combined_matches,
            'confidence': max([m['confidence'] for m in combined_matches]) if combined_matches else 0.0,
            'threshold': self.config.confidence_threshold
        }
        
    def _apply_context_boost(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply context-based confidence boost"""
        if not results['matches']:
            return results
            
        # Get context from semantic processor
        for match in results['matches']:
            if match['source'] == 'semantic':
                # Boost confidence based on context relevance
                match['confidence'] *= self.config.context_boost_factor
                
        # Re-sort matches
        results['matches'].sort(key=lambda x: x['confidence'], reverse=True)
        results['confidence'] = max(m['confidence'] for m in results['matches'])
        
        return results
        
    def _update_history(self, results: Dict[str, Any]) -> None:
        """Update processing history"""
        self.processing_history.append(results)
        
        # Maintain history window
        if len(self.processing_history) > self.config.adaptive_window:
            self.processing_history.pop(0)
            
    def _calculate_success_rate(self, source: str) -> float:
        """Calculate success rate for a processing source"""
        if not self.processing_history:
            return 0.5  # Default to equal weighting
            
        successes = sum(
            1 for result in self.processing_history
            for match in result['matches']
            if match['source'] == source and match['confidence'] >= self.config.confidence_threshold
        )
        
        total = sum(
            1 for result in self.processing_history
            for match in result['matches']
            if match['source'] == source
        )
        
        return successes / total if total > 0 else 0.5

class HybridPath(ProcessingPath):
    """Implementation of hybrid processing path"""
    
    def __init__(
        self,
        ml_config: MLConfig,
        literal_config: LiteralProcessorConfig,
        semantic_config: SemanticProcessorConfig,
        hybrid_config: HybridProcessorConfig
    ):
        super().__init__(PathConfig(
            path_type=PathType.HYBRID,
            ml_config=ml_config,
            bridge_config={
                'hybrid_processing': True,
                'adaptive_routing': True
            },
            web_config={
                'connection_type': 'hybrid',
                'dynamic_weighting': True
            },
            processing_weights={
                'input': 1.0,
                'ml_processing': 1.0,
                'bridge_transfer': 1.0,
                'web_integration': 1.0,
                'output': 1.0
            }
        ))
        
        # Create component paths
        self.literal_path = LiteralPath(ml_config, literal_config)
        self.semantic_path = SemanticPath(ml_config, semantic_config)
        
        # Create hybrid processor
        self.processor = HybridProcessor(hybrid_config)
        self.processor.setup_processors(self.literal_path, self.semantic_path)
        
    async def process_input(self, data: Any) -> Any:
        """Process input data"""
        text = str(data)
        
        # Process using hybrid approach
        results = self.processor.process_hybrid(text)
        
        # Add processing metadata
        results['original'] = text
        results['timestamp'] = torch.cuda.Event().record()
        
        return results
        
    async def verify_output(self, data: Any) -> Any:
        """Verify hybrid output"""
        if not isinstance(data, dict) or 'matches' not in data:
            return {'verified': False, 'data': data}
            
        # Verify matches based on confidence
        verified_matches = [
            match for match in data['matches']
            if match['confidence'] >= self.processor.config.confidence_threshold
        ]
        
        return {
            'verified': len(verified_matches) > 0,
            'matches': verified_matches,
            'confidence': max([m['confidence'] for m in verified_matches]) if verified_matches else 0.0,
            'sources': list(set(m['source'] for m in verified_matches))
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid processing statistics"""
        return {
            'literal_stats': self.literal_path.get_pattern_stats(),
            'semantic_stats': self.semantic_path.get_context_stats(),
            'combination_strategy': self.processor.config.combination_strategy,
            'history_size': len(self.processor.processing_history)
        } 
 