"""
Efficient Thought Learning System
Implements memory-efficient learning with gradient accumulation for GTX 1080
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time
import json
import os

@dataclass
class TrainingBatch:
    thoughts: List[str]
    outcomes: List[Dict[str, Any]]
    rewards: List[float]
    system_states: List[Dict[str, Any]]
    quality_scores: List[float]

class EfficientThoughtLearning:
    """Memory-efficient learning with gradient accumulation"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        self.accumulated_loss = 0.0
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        
        # Loss functions
        self.quality_loss_fn = nn.MSELoss()
        self.coherence_loss_fn = nn.CrossEntropyLoss()
        self.novelty_loss_fn = nn.MSELoss()
        
        self.logger = logging.getLogger(__name__)
        
        # Training metrics
        self.training_steps = 0
        self.total_loss = 0.0
        self.loss_history = []
        self.reward_history = []
        
        # Mixed precision training for GTX 1080
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        # Learning statistics
        self.learning_stats = {
            "thoughts_learned": 0,
            "avg_quality_improvement": 0.0,
            "avg_reward": 0.0,
            "convergence_rate": 0.0
        }
        
    async def learn_from_outcome(self, thought: str, outcome: Dict[str, Any]) -> float:
        """Learn from a single thought outcome with gradient accumulation"""
        
        try:
            # Compute loss for this thought
            loss = self._compute_thought_loss(thought, outcome)
            
            # Scale loss for accumulation
            scaled_loss = loss / self.accumulation_steps
            
            # Mixed precision backward pass
            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Accumulate
            self.accumulated_loss += scaled_loss.item()
            self.current_step += 1
            
            # Update weights when accumulation is complete
            if self.current_step >= self.accumulation_steps:
                await self._update_weights()
                self._reset_accumulation()
            
            # Update learning statistics
            reward = outcome.get("reward", 0.5)
            self.reward_history.append(reward)
            self.learning_stats["thoughts_learned"] += 1
            
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in learning step: {e}")
            return 0.0
    
    def _compute_thought_loss(self, thought: str, outcome: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a thought and its outcome"""
        
        # Quality prediction loss
        predicted_quality = outcome.get("predicted_quality", 0.5)
        actual_quality = outcome.get("quality_score", 0.5)
        
        quality_loss = self.quality_loss_fn(
            torch.tensor(predicted_quality, dtype=torch.float32, device=self.device, requires_grad=True),
            torch.tensor(actual_quality, dtype=torch.float32, device=self.device)
        )
        
        # Coherence loss (simplified)
        coherence_score = self._compute_coherence_score(thought)
        coherence_target = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        coherence_loss = self.quality_loss_fn(coherence_score, coherence_target)
        
        # Novelty preservation loss
        novelty_score = outcome.get("novelty_score", 0.5)
        novelty_target = 0.7  # Target moderate novelty
        
        novelty_loss = self.novelty_loss_fn(
            torch.tensor(novelty_score, dtype=torch.float32, device=self.device),
            torch.tensor(novelty_target, dtype=torch.float32, device=self.device)
        )
        
        # Reward-based loss (encourage high-reward thoughts)
        reward = outcome.get("reward", 0.5)
        reward_loss = self.quality_loss_fn(
            torch.tensor(reward, dtype=torch.float32, device=self.device),
            torch.tensor(1.0, dtype=torch.float32, device=self.device)  # Target high reward
        )
        
        # Efficiency loss (encourage fast execution)
        execution_time = outcome.get("execution_time", 1.0)
        efficiency_target = 0.5  # Target 0.5 seconds or less
        efficiency_loss = self.quality_loss_fn(
            torch.tensor(min(execution_time, 2.0), dtype=torch.float32, device=self.device),
            torch.tensor(efficiency_target, dtype=torch.float32, device=self.device)
        )
        
        # Combined loss with adaptive weights
        total_loss = (
            0.35 * quality_loss +
            0.25 * coherence_loss +
            0.20 * novelty_loss +
            0.15 * reward_loss +
            0.05 * efficiency_loss
        )
        
        return total_loss
    
    def _compute_coherence_score(self, thought: str) -> torch.Tensor:
        """Compute coherence score for a thought"""
        
        # Simple coherence metric based on sentence structure
        words = thought.split()
        
        # Basic coherence indicators
        has_subject = any(word.lower() in ["i", "system", "we", "neural", "thought"] for word in words)
        has_verb = any(word.lower() in ["am", "is", "are", "observe", "notice", "analyze", "process"] for word in words)
        has_object = any(word.lower() in ["data", "pattern", "behavior", "system", "information"] for word in words)
        reasonable_length = 5 <= len(words) <= 50
        proper_punctuation = thought.strip().endswith(('.', '!', '?'))
        
        coherence = 0.0
        if has_subject:
            coherence += 0.25
        if has_verb:
            coherence += 0.25
        if has_object:
            coherence += 0.20
        if reasonable_length:
            coherence += 0.20
        if proper_punctuation:
            coherence += 0.10
        
        return torch.tensor(coherence, dtype=torch.float32, device=self.device, requires_grad=True)
    
    async def _update_weights(self):
        """Update model weights with gradient clipping"""
        
        try:
            if self.scaler:
                # Mixed precision update
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard update
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Scheduler step
            self.scheduler.step()
            
            # Update metrics
            self.training_steps += 1
            self.total_loss += self.accumulated_loss
            self.loss_history.append(self.accumulated_loss)
            
            # Update learning statistics
            self._update_learning_statistics()
            
            # Log progress
            if self.training_steps % 10 == 0:
                await self._log_training_progress()
                
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")
    
    def _update_learning_statistics(self):
        """Update learning statistics"""
        
        if len(self.reward_history) >= 10:
            recent_rewards = self.reward_history[-10:]
            self.learning_stats["avg_reward"] = sum(recent_rewards) / len(recent_rewards)
        
        if len(self.loss_history) >= 20:
            recent_losses = self.loss_history[-20:]
            older_losses = self.loss_history[-40:-20] if len(self.loss_history) >= 40 else recent_losses
            
            recent_avg = sum(recent_losses) / len(recent_losses)
            older_avg = sum(older_losses) / len(older_losses)
            
            # Convergence rate (negative means loss is decreasing)
            self.learning_stats["convergence_rate"] = (recent_avg - older_avg) / max(older_avg, 1e-6)
    
    async def _log_training_progress(self):
        """Log training progress"""
        
        avg_loss = self.total_loss / max(self.training_steps, 1)
        current_lr = self.scheduler.get_last_lr()[0] if self.training_steps > 0 else 0.0
        
        self.logger.info(
            f"Training step {self.training_steps}: "
            f"avg_loss={avg_loss:.4f}, "
            f"lr={current_lr:.6f}, "
            f"avg_reward={self.learning_stats['avg_reward']:.3f}, "
            f"convergence_rate={self.learning_stats['convergence_rate']:.4f}"
        )
    
    def _reset_accumulation(self):
        """Reset gradient accumulation counters"""
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    async def batch_learn(self, training_batch: TrainingBatch):
        """Learn from a batch of thoughts efficiently"""
        
        batch_start_time = time.time()
        
        for thought, outcome, reward, quality in zip(
            training_batch.thoughts,
            training_batch.outcomes,
            training_batch.rewards,
            training_batch.quality_scores
        ):
            # Add reward and quality to outcome
            outcome["reward"] = reward
            outcome["quality_score"] = quality
            
            # Learn from this example
            await self.learn_from_outcome(thought, outcome)
        
        batch_time = time.time() - batch_start_time
        self.logger.info(f"Batch learning completed in {batch_time:.2f}s for {len(training_batch.thoughts)} thoughts")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        
        avg_loss = self.total_loss / max(self.training_steps, 1)
        current_lr = self.scheduler.get_last_lr()[0] if self.training_steps > 0 else 0.0
        
        # Memory usage
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2,   # MB
                "gpu_memory_max": torch.cuda.max_memory_allocated() / 1024**2    # MB
            }
        
        return {
            "training_steps": self.training_steps,
            "average_loss": avg_loss,
            "current_learning_rate": current_lr,
            "accumulation_progress": f"{self.current_step}/{self.accumulation_steps}",
            "learning_stats": self.learning_stats,
            "memory_stats": memory_stats,
            "loss_trend": self.loss_history[-10:] if self.loss_history else [],
            "reward_trend": self.reward_history[-10:] if self.reward_history else []
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_steps": self.training_steps,
                "total_loss": self.total_loss,
                "current_step": self.current_step,
                "accumulated_loss": self.accumulated_loss,
                "learning_stats": self.learning_stats,
                "loss_history": self.loss_history,
                "reward_history": self.reward_history
            }
            
            if self.scaler:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load training checkpoint"""
        
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Checkpoint file not found: {filepath}")
                return False
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.training_steps = checkpoint["training_steps"]
            self.total_loss = checkpoint["total_loss"]
            self.current_step = checkpoint["current_step"]
            self.accumulated_loss = checkpoint["accumulated_loss"]
            self.learning_stats = checkpoint.get("learning_stats", self.learning_stats)
            self.loss_history = checkpoint.get("loss_history", [])
            self.reward_history = checkpoint.get("reward_history", [])
            
            if self.scaler and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            self.logger.info(f"Checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def should_save_checkpoint(self) -> bool:
        """Determine if a checkpoint should be saved"""
        
        # Save every 50 training steps
        if self.training_steps % 50 == 0 and self.training_steps > 0:
            return True
        
        # Save if convergence rate is very good
        if self.learning_stats["convergence_rate"] < -0.1:  # Loss decreasing rapidly
            return True
        
        # Save if average reward is very high
        if self.learning_stats["avg_reward"] > 0.8:
            return True
        
        return False
    
    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations for improving learning"""
        
        recommendations = []
        
        # Check convergence
        if self.learning_stats["convergence_rate"] > 0.05:
            recommendations.append("Learning rate may be too high - consider reducing")
        elif self.learning_stats["convergence_rate"] > -0.001:
            recommendations.append("Learning appears to have plateaued - consider curriculum learning")
        
        # Check reward trends
        if self.learning_stats["avg_reward"] < 0.3:
            recommendations.append("Low average reward - increase exploration or adjust reward function")
        elif self.learning_stats["avg_reward"] > 0.9:
            recommendations.append("Very high rewards - may be overfitting, increase task difficulty")
        
        # Check loss trends
        if len(self.loss_history) >= 10:
            recent_loss_trend = sum(self.loss_history[-5:]) / 5 - sum(self.loss_history[-10:-5]) / 5
            if recent_loss_trend > 0.1:
                recommendations.append("Loss is increasing - check for overfitting or reduce learning rate")
        
        # Memory recommendations
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_usage > 0.9:
                recommendations.append("High GPU memory usage - consider reducing batch size or model complexity")
        
        return recommendations if recommendations else ["Learning is progressing well"] 