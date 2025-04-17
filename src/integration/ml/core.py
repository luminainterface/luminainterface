#!/usr/bin/env python3
"""
Core ML Infrastructure

This module provides the core infrastructure for machine learning components,
including base classes and interfaces for models, trainers, and datasets.
"""

import abc
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MLConfig:
    """Configuration for ML components"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "mse"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_type: str = "transformer"
    hidden_size: int = 256
    num_layers: int = 4
    dropout: float = 0.1

class BaseModel(nn.Module, abc.ABC):
    """Base class for all ML models"""
    
    def __init__(self, config: MLConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._setup_model()
        
    @abc.abstractmethod
    def _setup_model(self) -> None:
        """Setup model architecture"""
        pass
        
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
        
    def save(self, path: str) -> None:
        """Save model state"""
        torch.save({
            'model_state': self.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, path)
        
    def load(self, path: str) -> None:
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state'])
        self.config = checkpoint['config']

class BaseTrainer(abc.ABC):
    """Base class for model trainers"""
    
    def __init__(self, model: BaseModel, config: MLConfig):
        self.model = model
        self.config = config
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._setup_loss_function()
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function"""
        if self.config.loss_function.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_function.lower() == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
            
    @abc.abstractmethod
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step"""
        pass
        
    @abc.abstractmethod
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single validation step"""
        pass
        
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics = {
            'train_loss': 0.0,
            'train_accuracy': 0.0,
            'val_loss': 0.0,
            'val_accuracy': 0.0
        }
        
        # Training
        num_batches = 0
        for batch in train_loader:
            batch_metrics = self.train_step(batch)
            for k, v in batch_metrics.items():
                metrics[f'train_{k}'] += v
            num_batches += 1
            
        # Average metrics
        for k in metrics.keys():
            if k.startswith('train_'):
                metrics[k] /= num_batches
                
        # Validation
        if val_loader is not None:
            self.model.eval()
            num_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_metrics = self.validate_step(batch)
                    for k, v in batch_metrics.items():
                        metrics[f'val_{k}'] += v
                    num_batches += 1
                    
            # Average metrics
            for k in metrics.keys():
                if k.startswith('val_'):
                    metrics[k] /= num_batches
                    
        return metrics
        
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        callbacks: Optional[List[Any]] = None
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            
            # Update history
            for k, v in metrics.items():
                history[k].append(v)
                
            # Log progress
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            for k, v in metrics.items():
                logger.info(f"{k}: {v:.4f}")
                
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, metrics, self.model)
                    
        return history

class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    """Base class for datasets"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        
    @abc.abstractmethod
    def __len__(self) -> int:
        """Return dataset size"""
        pass
        
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item"""
        pass
        
    @abc.abstractmethod
    def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess raw data"""
        pass
        
class MLCallback(abc.ABC):
    """Base class for callbacks"""
    
    @abc.abstractmethod
    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: BaseModel
    ) -> None:
        """Execute callback"""
        pass

class EarlyStopping(MLCallback):
    """Early stopping callback"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = 'val_loss'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: BaseModel
    ) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return
            
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True

class ModelCheckpoint(MLCallback):
    """Model checkpoint callback"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float('inf')
        
    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: BaseModel
    ) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return
            
        if self.save_best_only:
            if current < self.best_value:
                self.best_value = current
                model.save(self.filepath)
        else:
            model.save(f"{self.filepath}_epoch_{epoch}")

def create_model(config: MLConfig) -> BaseModel:
    """Create model instance based on configuration"""
    from .models import MODEL_REGISTRY
    
    model_class = MODEL_REGISTRY.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
        
    return model_class(config)

def create_trainer(
    model: BaseModel,
    config: MLConfig
) -> BaseTrainer:
    """Create trainer instance based on configuration"""
    from .trainers import TRAINER_REGISTRY
    
    trainer_class = TRAINER_REGISTRY.get(config.model_type)
    if trainer_class is None:
        raise ValueError(f"Unknown trainer type: {config.model_type}")
        
    return trainer_class(model, config) 