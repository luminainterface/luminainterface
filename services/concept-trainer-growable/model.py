import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LayerMetrics:
    """Metrics for a single layer"""
    layer_id: int
    input_size: int
    output_size: int
    activation: str
    added_at: datetime
    training_metrics: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.training_metrics is None:
            self.training_metrics = {
                'loss': [],
                'accuracy': [],
                'drift': []
            }

class GrowableLayer(nn.Module):
    """A neural network layer that can grow its capacity"""
    def __init__(self, in_features: int, out_features: int, activation: str = 'relu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.current_capacity = out_features
        self.activation = activation
        self.growth_history = []
        
        # Initialize layer
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.linear.bias, 0)
        
        logger.info(f"Created GrowableLayer: in_features={in_features}, out_features={out_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation"""
        x = self.linear(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.batch_norm(x)
        
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:
            return x
    
    def grow(self, new_size: int) -> None:
        """Grow the layer to a new size"""
        if new_size <= self.current_capacity:
            raise ValueError(f"New size {new_size} must be greater than current capacity {self.current_capacity}")
            
        logger.info(f"Growing layer from {self.current_capacity} to {new_size}")
        
        # Create new layer with larger size
        new_linear = nn.Linear(self.in_features, new_size)
        new_batch_norm = nn.BatchNorm1d(new_size)
        
        # Initialize new weights
        nn.init.kaiming_normal_(new_linear.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(new_linear.bias, 0)
        
        # Copy existing weights
        with torch.no_grad():
            new_linear.weight[:self.current_capacity] = self.linear.weight
            new_linear.bias[:self.current_capacity] = self.linear.bias
        
        # Replace old layer with new one
        self.linear = new_linear
        self.batch_norm = new_batch_norm
        self.out_features = new_size
        self.current_capacity = new_size
        
        # Record growth event
        self.growth_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_size': self.current_capacity,
            'new_size': new_size
        })
        
        logger.info(f"Layer grown successfully to {new_size}")

class GrowableConceptNet(nn.Module):
    """A neural network that can grow its capacity over time"""
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = None,
                 output_size: int = 2,  # Binary classification
                 activation: str = 'relu',
                 learning_rate: float = 1e-4):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.last_training = None
        
        # Default to larger hidden sizes if none provided
        if hidden_sizes is None:
            hidden_sizes = [
                input_size,  # First layer same as input
                input_size // 2,  # Half size
                input_size // 4   # Quarter size
            ]
        
        logger.info(f"Creating GrowableConceptNet with input_size={input_size}, "
                   f"hidden_sizes={hidden_sizes}, output_size={output_size}, "
                   f"activation={activation}")
        
        # Create layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.layers.append(GrowableLayer(input_size, hidden_sizes[0], activation))
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(
                GrowableLayer(hidden_sizes[i], hidden_sizes[i+1], activation)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i+1]))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size),
            nn.LogSoftmax(dim=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Training state
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.training_history = []
        self.concept_metrics = {}  # Per-concept metrics
        
        logger.info("GrowableConceptNet initialization complete")
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Ensure input is properly shaped
        if len(x.shape) < 2:
            x = x.view(-1, self.input_size)
        
        # Process through hidden layers
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                x = bn(x)
        
        # Final output layer
        x = self.output_layer(x)
        return x
    
    def should_grow(self, concept_id: str) -> Tuple[bool, int]:
        """Determine if any layer should grow based on concept drift and maturity"""
        if concept_id not in self.concept_metrics:
            logger.info(f"No metrics found for concept {concept_id}")
            return False, -1
            
        metrics = self.concept_metrics[concept_id]
        drift = metrics.get('last_drift', 0.0)
        drift_trend = metrics.get('drift_trend', 0.0)
        maturity = metrics.get('maturity_score', 0.0)
        threshold = metrics.get('growth_threshold', 0.1)
        
        logger.info(f"Checking growth for concept {concept_id}:")
        logger.info(f"Drift: {drift:.4f}, Trend: {drift_trend:.4f}")
        logger.info(f"Maturity: {maturity:.4f}, Threshold: {threshold:.4f}")
        
        # Check for persistent drift (high drift with positive trend)
        persistent_drift = drift > threshold and drift_trend > 0
        
        if persistent_drift:
            # Find layer with highest drift contribution
            max_drift_layer = -1
            max_layer_drift = 0.0
            
            for i, layer in enumerate(self.layers):
                layer_drift = self._calculate_layer_drift(i, concept_id)
                if layer_drift > max_layer_drift:
                    max_layer_drift = layer_drift
                    max_drift_layer = i
            
            if max_drift_layer >= 0:
                logger.info(f"Layer {max_drift_layer} should grow due to persistent drift")
                return True, max_drift_layer
        
        logger.info("No layers need to grow")
        return False, -1
    
    def _calculate_layer_drift(self, layer_idx: int, concept_id: str) -> float:
        """Calculate drift contribution for a specific layer"""
        if concept_id not in self.concept_metrics:
            return 0.0
            
        metrics = self.concept_metrics[concept_id]
        drift_history = metrics.get('drift_history', [])
        
        if not drift_history:
            return 0.0
            
        # Calculate drift based on layer's output statistics
        recent_drifts = [entry.get('layer_drifts', {}).get(str(layer_idx), 0.0) 
                        for entry in drift_history[-5:]]
        
        if not recent_drifts:
            return 0.0
            
        return np.mean(recent_drifts)
    
    def grow_layer(self, layer_idx: int, new_size: int) -> None:
        """Grow a specific layer"""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            error_msg = f"Invalid layer index {layer_idx}. Must be between 0 and {len(self.layers)-1}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Growing layer {layer_idx} to size {new_size}")
        layer = self.layers[layer_idx]
        layer.grow(new_size)
        
        # Update batch norm layer if needed
        if layer_idx < len(self.batch_norms):
            self.batch_norms[layer_idx] = nn.BatchNorm1d(new_size)
            logger.info(f"Updated batch norm for layer {layer_idx}")
        
        # Update optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer.param_groups[0]['lr'])
    
    def record_training(self, 
                       concept_id: str,
                       loss: float,
                       accuracy: float,
                       drift: float) -> None:
        """Record training metrics for a concept"""
        logger.info(f"Recording training metrics for concept {concept_id}: "
                   f"loss={loss:.4f}, accuracy={accuracy:.4f}, drift={drift:.4f}")
        
        if concept_id not in self.concept_metrics:
            self.concept_metrics[concept_id] = {
                'drift_history': [],
                'last_update': datetime.now().isoformat(),
                'last_loss': 0.0,
                'last_accuracy': 0.0,
                'last_drift': 0.0,
                'drift_trend': 0.0,
                'maturity_score': 0.0,
                'growth_threshold': 0.1
            }
        
        # Update metrics
        metrics = self.concept_metrics[concept_id]
        metrics.update({
            'last_update': datetime.now().isoformat(),
            'last_loss': loss,
            'last_accuracy': accuracy,
            'last_drift': drift
        })
        
        # Calculate drift trend
        drift_history = metrics['drift_history']
        drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'drift': drift,
            'loss': loss,
            'accuracy': accuracy
        })
        
        # Keep only last 100 entries
        if len(drift_history) > 100:
            drift_history = drift_history[-100:]
        
        # Calculate drift trend using linear regression
        if len(drift_history) >= 5:
            timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in drift_history[-5:]]
            drifts = [entry['drift'] for entry in drift_history[-5:]]
            x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
            y = np.array(drifts)
            slope = np.polyfit(x, y, 1)[0]
            metrics['drift_trend'] = float(slope)
        
        # Update maturity score (simplified)
        metrics['maturity_score'] = min(1.0, len(drift_history) / 100.0)
        
        # Update growth threshold based on recent performance
        if len(drift_history) >= 5:
            recent_drifts = [entry['drift'] for entry in drift_history[-5:]]
            drift_std = np.std(recent_drifts)
            metrics['growth_threshold'] = max(0.001, min(0.1, drift_std * 2))
        
        # Update last training timestamp
        self.last_training = datetime.now().isoformat()
    
    def get_concept_metrics(self, concept_id: str) -> Dict:
        """Get metrics for a specific concept"""
        logger.info(f"Getting metrics for concept {concept_id}")
        return self.concept_metrics.get(concept_id, {})
    
    def get_network_stats(self) -> Dict:
        """Get overall network statistics"""
        stats = {
            'num_layers': len(self.layers),
            'layer_sizes': [layer.current_capacity for layer in self.layers],
            'total_params': sum(p.numel() for p in self.parameters()),
            'growth_events': sum(len(layer.growth_history) for layer in self.layers),
            'concepts_tracked': len(self.concept_metrics),
            'last_training': self.last_training
        }
        logger.info(f"Network stats: {stats}")
        return stats
        
    def get_concept_embedding(self, concept_id: str) -> torch.Tensor:
        """Get the current embedding for a concept"""
        if concept_id not in self.concept_metrics:
            logger.warning(f"No embedding found for concept {concept_id}")
            return torch.zeros(self.input_size)
            
        # Get the most recent input data for this concept
        drift_history = self.concept_metrics[concept_id]['drift_history']
        if not drift_history:
            logger.warning(f"No drift history for concept {concept_id}")
            return torch.zeros(self.input_size)
            
        # Use the most recent input data as the embedding
        # In a real system, you might want to use a more sophisticated method
        # like averaging recent inputs or using a dedicated embedding layer
        recent_inputs = torch.tensor([
            entry.get('input_data', torch.zeros(self.input_size))
            for entry in drift_history[-5:]  # Last 5 entries
        ])
        
        # Average the recent inputs
        embedding = recent_inputs.mean(dim=0)
        
        # Normalize the embedding
        embedding = embedding / (torch.norm(embedding) + 1e-6)
        
        logger.info(f"Generated embedding for concept {concept_id} with shape {embedding.shape}")
        return embedding 