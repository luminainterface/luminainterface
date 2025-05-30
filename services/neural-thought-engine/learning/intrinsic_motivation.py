"""
Lightweight Intrinsic Motivation System
Implements novelty detection and curiosity-driven exploration for GTX 1080
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional
import time
import logging
import hashlib

class LightweightIntrinsicMotivation:
    """Simplified curiosity and novelty detection for GTX 1080"""
    
    def __init__(self, embedding_dim: int = 256, history_size: int = 100):
        self.embedding_dim = embedding_dim
        self.thought_history = deque(maxlen=history_size)
        self.novelty_threshold = 0.7
        self.logger = logging.getLogger(__name__)
        
        # Simple embedding network for text
        self.text_embedder = nn.Sequential(
            nn.Linear(1000, 512),  # Assuming bag-of-words input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.Tanh()
        )
        
        # Curiosity metrics
        self.curiosity_factors = {
            "exploration_bonus": 0.1,
            "uncertainty_bonus": 0.2,
            "surprise_bonus": 0.3,
            "complexity_bonus": 0.15
        }
        
        # Concept exploration tracking
        self.explored_concepts = set()
        self.concept_exploration_count = {}
        
        # Novelty statistics
        self.novelty_scores = deque(maxlen=1000)
        self.avg_novelty = 0.5
        
    def simple_text_embedding(self, text: str) -> torch.Tensor:
        """Convert text to simple embedding using bag-of-words"""
        
        # Simple bag-of-words approach
        words = text.lower().split()
        vocab_size = 1000
        
        # Create simple word frequency vector
        word_freq = torch.zeros(vocab_size)
        for word in words:
            # Simple hash-based word indexing
            word_idx = hash(word) % vocab_size
            word_freq[word_idx] += 1
        
        # Normalize
        if word_freq.sum() > 0:
            word_freq = word_freq / word_freq.sum()
        
        # Pass through embedding network
        with torch.no_grad():
            embedding = self.text_embedder(word_freq)
        
        return embedding
    
    def compute_novelty(self, thought: str) -> float:
        """Compute novelty score for a thought"""
        
        if not self.thought_history:
            return 1.0  # First thought is completely novel
        
        try:
            # Get embedding for current thought
            current_embedding = self.simple_text_embedding(thought)
            
            # Compare with recent thoughts (only last 10 for efficiency)
            recent_thoughts = list(self.thought_history)[-10:]
            max_similarity = 0.0
            
            for past_thought in recent_thoughts:
                past_embedding = self.simple_text_embedding(past_thought)
                
                # Cosine similarity
                similarity = torch.cosine_similarity(
                    current_embedding.unsqueeze(0),
                    past_embedding.unsqueeze(0)
                ).item()
                
                max_similarity = max(max_similarity, similarity)
            
            novelty = 1.0 - max_similarity
            novelty = max(0.0, min(1.0, novelty))
            
            # Update statistics
            self.novelty_scores.append(novelty)
            self.avg_novelty = sum(self.novelty_scores) / len(self.novelty_scores)
            
            return novelty
            
        except Exception as e:
            self.logger.error(f"Error computing novelty: {e}")
            return 0.5  # Default novelty
    
    def compute_curiosity_score(self, thought: str, system_state: Dict[str, Any]) -> float:
        """Compute curiosity-driven exploration score"""
        
        try:
            base_curiosity = 0.5
            
            # Exploration bonus for new concepts
            words = set(thought.lower().split())
            historical_words = set()
            for past_thought in self.thought_history:
                historical_words.update(past_thought.lower().split())
            
            new_words = words - historical_words
            exploration_bonus = len(new_words) * self.curiosity_factors["exploration_bonus"]
            
            # Track concept exploration
            for word in new_words:
                if len(word) > 4:  # Only meaningful words
                    self.explored_concepts.add(word)
                    self.concept_exploration_count[word] = self.concept_exploration_count.get(word, 0) + 1
            
            # Uncertainty bonus based on system state
            uncertainty_bonus = 0.0
            services_health = system_state.get("services_health", {})
            services_down = sum(1 for healthy in services_health.values() if not healthy)
            uncertainty_bonus = services_down * self.curiosity_factors["uncertainty_bonus"]
            
            # Complexity bonus for longer, more structured thoughts
            complexity_bonus = 0.0
            words_count = len(thought.split())
            if 10 <= words_count <= 50:  # Sweet spot for complexity
                complexity_bonus = self.curiosity_factors["complexity_bonus"]
            
            # Surprise bonus for unexpected patterns
            surprise_bonus = np.random.uniform(0, self.curiosity_factors["surprise_bonus"])
            
            total_curiosity = (
                base_curiosity + 
                exploration_bonus + 
                uncertainty_bonus + 
                complexity_bonus + 
                surprise_bonus
            )
            
            return min(1.0, total_curiosity)
            
        except Exception as e:
            self.logger.error(f"Error computing curiosity: {e}")
            return 0.5
    
    def compute_intrinsic_reward(self, thought: str, outcome: Dict[str, Any]) -> float:
        """Compute total intrinsic reward for a thought"""
        
        try:
            # Novelty component
            novelty_reward = self.compute_novelty(thought)
            
            # Curiosity component
            curiosity_reward = self.compute_curiosity_score(
                thought, outcome.get("system_state", {})
            )
            
            # Outcome-based reward
            outcome_reward = 1.0 if outcome.get("success", False) else 0.5
            
            # Quality bonus
            quality_bonus = outcome.get("quality_score", 0.5)
            
            # Execution efficiency bonus
            execution_time = outcome.get("execution_time", 1.0)
            efficiency_bonus = max(0.0, 1.0 - (execution_time / 2.0))  # Bonus for fast execution
            
            # Weighted combination
            total_reward = (
                0.25 * novelty_reward +
                0.25 * curiosity_reward +
                0.20 * outcome_reward +
                0.20 * quality_bonus +
                0.10 * efficiency_bonus
            )
            
            # Store thought for future novelty calculations
            self.thought_history.append(thought)
            
            # Log interesting rewards
            if total_reward > 0.8:
                self.logger.info(f"High intrinsic reward ({total_reward:.3f}) for thought: {thought[:50]}...")
            
            return total_reward
            
        except Exception as e:
            self.logger.error(f"Error computing intrinsic reward: {e}")
            return 0.5
    
    def get_exploration_targets(self) -> List[str]:
        """Get suggested exploration targets based on curiosity"""
        
        try:
            # Identify underexplored areas
            word_frequency = {}
            for thought in self.thought_history:
                for word in thought.lower().split():
                    if len(word) > 4:  # Only meaningful words
                        word_frequency[word] = word_frequency.get(word, 0) + 1
            
            # Find rare concepts that might be worth exploring
            rare_concepts = [
                word for word, freq in word_frequency.items()
                if freq == 1 and len(word) > 6
            ]
            
            # Add some domain-specific exploration targets
            domain_targets = [
                "neural_optimization", "memory_efficiency", "gpu_utilization",
                "concept_integration", "learning_dynamics", "attention_mechanisms",
                "transformer_architecture", "gradient_accumulation", "mixed_precision"
            ]
            
            # Combine and prioritize
            exploration_targets = rare_concepts[:3] + domain_targets[:2]
            
            return exploration_targets[:5]  # Return top 5 exploration targets
            
        except Exception as e:
            self.logger.error(f"Error getting exploration targets: {e}")
            return ["neural_networks", "optimization", "learning"]
    
    def get_curiosity_statistics(self) -> Dict[str, Any]:
        """Get curiosity and exploration statistics"""
        
        return {
            "thoughts_processed": len(self.thought_history),
            "unique_concepts_explored": len(self.explored_concepts),
            "average_novelty": self.avg_novelty,
            "exploration_targets": self.get_exploration_targets(),
            "most_explored_concepts": sorted(
                self.concept_exploration_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "novelty_trend": list(self.novelty_scores)[-10:] if self.novelty_scores else []
        }
    
    def should_explore_concept(self, concept: str) -> bool:
        """Determine if a concept should be explored based on curiosity"""
        
        exploration_count = self.concept_exploration_count.get(concept, 0)
        
        # Explore if:
        # 1. Never explored before
        # 2. Explored less than 3 times
        # 3. Random exploration chance
        
        if exploration_count == 0:
            return True
        elif exploration_count < 3:
            return np.random.random() > 0.5
        else:
            return np.random.random() > 0.8  # Low chance for well-explored concepts
    
    def update_exploration_strategy(self, recent_rewards: List[float]):
        """Update exploration strategy based on recent reward patterns"""
        
        if len(recent_rewards) < 5:
            return
        
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
        
        # If rewards are low, increase exploration
        if avg_recent_reward < 0.4:
            self.curiosity_factors["exploration_bonus"] = min(0.3, self.curiosity_factors["exploration_bonus"] * 1.1)
            self.curiosity_factors["surprise_bonus"] = min(0.4, self.curiosity_factors["surprise_bonus"] * 1.1)
        
        # If rewards are high, maintain current strategy
        elif avg_recent_reward > 0.7:
            self.curiosity_factors["exploration_bonus"] = max(0.05, self.curiosity_factors["exploration_bonus"] * 0.95)
            self.curiosity_factors["surprise_bonus"] = max(0.1, self.curiosity_factors["surprise_bonus"] * 0.95)
        
        self.logger.debug(f"Updated exploration strategy: {self.curiosity_factors}") 