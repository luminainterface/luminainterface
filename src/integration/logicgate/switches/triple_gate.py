#!/usr/bin/env python3
"""
Triple Flip Switch Gate

This module implements a dynamic switching mechanism that can route processing
through literal, semantic, and hybrid paths with real-time state changes.
"""

import asyncio
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from ..core import PathType, ProcessingStage
from ..paths.literal_path import LiteralPath
from ..paths.semantic_path import SemanticPath
from ..paths.hybrid_path import HybridPath

class GateState(Enum):
    """States for each gate"""
    OPEN = auto()
    CLOSED = auto()
    PARTIAL = auto()

class FlowDirection(Enum):
    """Direction of information flow"""
    FORWARD = auto()
    BACKWARD = auto()
    BIDIRECTIONAL = auto()

@dataclass
class GateConfig:
    """Configuration for triple gate"""
    switching_threshold: float = 0.75
    min_confidence: float = 0.6
    max_active_paths: int = 2
    auto_switch: bool = True
    feedback_window: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    state_persistence: bool = True

class PathState:
    """State tracking for a processing path"""
    
    def __init__(self, path_type: PathType):
        self.path_type = path_type
        self.state = GateState.CLOSED
        self.confidence = 0.0
        self.flow = FlowDirection.FORWARD
        self.last_active = datetime.now()
        self.success_rate = 0.0
        self.processing_history: List[Dict[str, float]] = []
        self.feedback_scores: List[float] = []
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update path metrics"""
        self.processing_history.append(metrics)
        if len(self.processing_history) > 100:
            self.processing_history.pop(0)
            
        # Update success rate
        recent_successes = sum(
            1 for m in self.processing_history[-50:]
            if m.get('confidence', 0) >= 0.7
        )
        self.success_rate = recent_successes / min(50, len(self.processing_history))

class TripleGate:
    """Dynamic switching gate for three processing paths"""
    
    def __init__(
        self,
        config: GateConfig,
        literal_path: LiteralPath,
        semantic_path: SemanticPath,
        hybrid_path: HybridPath
    ):
        self.config = config
        self.paths = {
            PathType.LITERAL: literal_path,
            PathType.SEMANTIC: semantic_path,
            PathType.HYBRID: hybrid_path
        }
        self.states = {
            path_type: PathState(path_type)
            for path_type in PathType
        }
        self.active_paths: Set[PathType] = set()
        self.switch_history: List[Dict[str, Any]] = []
        self._initialize_states()
        
    def _initialize_states(self) -> None:
        """Initialize gate states"""
        # Start with hybrid path open
        self.states[PathType.HYBRID].state = GateState.OPEN
        self.active_paths.add(PathType.HYBRID)
        
        # Others start closed
        self.states[PathType.LITERAL].state = GateState.CLOSED
        self.states[PathType.SEMANTIC].state = GateState.CLOSED
        
    async def process(self, data: Any) -> Dict[str, Any]:
        """Process data through active paths"""
        results = {}
        confidences = {}
        
        # Process through active paths
        for path_type in self.active_paths:
            path_result = await self.paths[path_type].process(data)
            results[path_type] = path_result
            confidences[path_type] = path_result.get('confidence', 0.0)
            
            # Update path state
            self.states[path_type].update_metrics({
                'confidence': confidences[path_type],
                'timestamp': datetime.now().timestamp()
            })
            
        # Auto-switch if enabled
        if self.config.auto_switch:
            await self._auto_switch(confidences)
            
        # Combine results
        combined_result = self._combine_results(results)
        
        return combined_result
        
    async def _auto_switch(self, confidences: Dict[PathType, float]) -> None:
        """Automatically switch gates based on performance"""
        # Calculate path scores
        scores = self._calculate_path_scores()
        
        # Find best performing paths
        sorted_paths = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Update states
        new_active_paths = set()
        for path_type, score in sorted_paths[:self.config.max_active_paths]:
            if score >= self.config.switching_threshold:
                new_active_paths.add(path_type)
                self.states[path_type].state = GateState.OPEN
            else:
                self.states[path_type].state = GateState.CLOSED
                
        # Record switch
        if new_active_paths != self.active_paths:
            self.switch_history.append({
                'timestamp': datetime.now(),
                'previous': list(self.active_paths),
                'new': list(new_active_paths),
                'scores': scores
            })
            
        self.active_paths = new_active_paths
        
    def _calculate_path_scores(self) -> Dict[PathType, float]:
        """Calculate performance scores for each path"""
        scores = {}
        for path_type, state in self.states.items():
            # Combine multiple factors
            success_weight = 0.4
            confidence_weight = 0.3
            recency_weight = 0.3
            
            # Calculate components
            success_score = state.success_rate
            confidence_score = np.mean([
                m.get('confidence', 0)
                for m in state.processing_history[-10:]
            ]) if state.processing_history else 0
            
            time_since_active = (
                datetime.now() - state.last_active
            ).total_seconds()
            recency_score = 1.0 / (1.0 + time_since_active / 3600)  # Decay over hours
            
            # Combine scores
            scores[path_type] = (
                success_weight * success_score +
                confidence_weight * confidence_score +
                recency_weight * recency_score
            )
            
        return scores
        
    def _combine_results(
        self,
        results: Dict[PathType, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine results from active paths"""
        if not results:
            return {'error': 'No active paths'}
            
        combined_matches = []
        confidences = []
        
        for path_type, result in results.items():
            if 'matches' in result:
                path_matches = result['matches']
                for match in path_matches:
                    match['path_type'] = path_type
                combined_matches.extend(path_matches)
                
            if 'confidence' in result:
                confidences.append(result['confidence'])
                
        # Sort by confidence
        combined_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'matches': combined_matches,
            'confidence': max(confidences) if confidences else 0.0,
            'active_paths': list(self.active_paths),
            'path_states': {
                path_type.value: {
                    'state': state.state.value,
                    'confidence': state.confidence,
                    'success_rate': state.success_rate
                }
                for path_type, state in self.states.items()
            }
        }
        
    def switch_path(
        self,
        path_type: PathType,
        state: GateState,
        flow: Optional[FlowDirection] = None
    ) -> None:
        """Manually switch path state"""
        path_state = self.states[path_type]
        previous_state = path_state.state
        
        path_state.state = state
        if flow:
            path_state.flow = flow
            
        if state == GateState.OPEN:
            self.active_paths.add(path_type)
            path_state.last_active = datetime.now()
        else:
            self.active_paths.discard(path_type)
            
        # Record switch
        self.switch_history.append({
            'timestamp': datetime.now(),
            'path_type': path_type,
            'previous_state': previous_state.value,
            'new_state': state.value,
            'flow': flow.value if flow else None
        })
        
    def get_state(self) -> Dict[str, Any]:
        """Get current gate state"""
        return {
            'active_paths': list(self.active_paths),
            'states': {
                path_type.value: {
                    'state': state.state.value,
                    'flow': state.flow.value,
                    'confidence': state.confidence,
                    'success_rate': state.success_rate,
                    'last_active': state.last_active.isoformat()
                }
                for path_type, state in self.states.items()
            },
            'switch_history': self.switch_history[-10:],  # Last 10 switches
            'auto_switch': self.config.auto_switch
        }
        
    def set_auto_switch(self, enabled: bool) -> None:
        """Enable/disable automatic switching"""
        self.config.auto_switch = enabled
        
    def reset(self) -> None:
        """Reset gate states"""
        self._initialize_states()
        for state in self.states.values():
            state.processing_history.clear()
            state.feedback_scores.clear()
            state.success_rate = 0.0
            state.confidence = 0.0
            
    async def optimize(self) -> None:
        """Optimize gate parameters based on performance history"""
        if len(self.switch_history) < self.config.feedback_window:
            return
            
        # Calculate success rates for different configurations
        configs = {}
        for switch in self.switch_history[-self.config.feedback_window:]:
            active_paths = tuple(switch['new'])
            if active_paths not in configs:
                configs[active_paths] = {
                    'count': 0,
                    'success_sum': 0.0
                }
                
            configs[active_paths]['count'] += 1
            configs[active_paths]['success_sum'] += switch.get('scores', {}).get(
                'confidence', 0.0
            )
            
        # Find best configuration
        best_config = max(
            configs.items(),
            key=lambda x: x[1]['success_sum'] / x[1]['count']
        )
        
        # Update thresholds
        self.config.switching_threshold = max(
            0.5,
            min(0.95, best_config[1]['success_sum'] / best_config[1]['count'])
        )
        
        # Update active paths if significantly better
        best_paths = set(best_config[0])
        if (
            best_config[1]['success_sum'] / best_config[1]['count']
            > self.config.switching_threshold
            and best_paths != self.active_paths
        ):
            for path_type in PathType:
                self.switch_path(
                    path_type,
                    GateState.OPEN if path_type in best_paths else GateState.CLOSED
                ) 
 