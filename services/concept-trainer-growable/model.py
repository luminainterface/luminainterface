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
    """A layer that can grow its capacity"""
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.current_capacity = output_size
        self.max_capacity = output_size * 4  # Maximum growth factor
        self.activation = activation
        self.last_growth = datetime.now()
        self.growth_history = []
        self.drift_contributions = []  # Track drift contributions
        
        # Initialize layer with multidimensional support
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        self._init_weights()
        
        logger.info(f"Created GrowableLayer with input_size={input_size}, output_size={output_size}, activation={activation}")
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation and dropout"""
        # Ensure input is properly shaped
        if len(x.shape) < 2:
            x = x.view(-1, self.input_size)
        
        x = self.linear(x)
        x = self.dropout(x)
        
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        return x
    
    def grow(self, new_size: int) -> None:
        """Grow the layer to a new size"""
        if new_size <= self.current_capacity:
            logger.warning(f"New size {new_size} not larger than current size {self.current_capacity}")
            return
            
        if new_size > self.max_capacity:
            logger.warning(f"New size {new_size} exceeds max capacity {self.max_capacity}")
            new_size = self.max_capacity
            
        logger.info(f"Growing layer from {self.current_capacity} to {new_size}")
        
        # Create new layer with larger size
        new_layer = nn.Linear(self.input_size, new_size)
        
        # Initialize new weights
        nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')
        if new_layer.bias is not None:
            nn.init.constant_(new_layer.bias, 0)
            
        # Copy existing weights
        new_layer.weight.data[:self.current_capacity] = self.linear.weight.data
        if new_layer.bias is not None:
            new_layer.bias.data[:self.current_capacity] = self.linear.bias.data
            
        # Replace old layer
        self.linear = new_layer
        self.current_capacity = new_size
        self.last_growth = datetime.now()
        
        # Record growth event
        self.growth_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_size': self.current_capacity,
            'new_size': new_size,
            'growth_factor': new_size / self.current_capacity
        })
        
        logger.info(f"Layer grown successfully to {new_size}")
    
    def get_drift_contribution(self) -> float:
        """Calculate this layer's contribution to overall drift"""
        if not self.drift_contributions:
            return 0.0
            
        # Use exponential moving average of drift contributions
        alpha = 0.3  # Decay factor
        ema = self.drift_contributions[0]
        for contribution in self.drift_contributions[1:]:
            ema = alpha * contribution + (1 - alpha) * ema
        return ema
    
    def update_drift_contribution(self, contribution: float):
        """Update the layer's drift contribution history"""
        self.drift_contributions.append(contribution)
        if len(self.drift_contributions) > 10:  # Keep last 10 contributions
            self.drift_contributions.pop(0)
    
    def should_grow(self, 
                   current_drift: float,
                   drift_threshold: float = 0.001,
                   min_time_between_growth: int = 1) -> bool:
        """Determine if layer should grow based on drift and time"""
        time_since_last_growth = (datetime.now() - self.last_growth).total_seconds()
        
        should_grow = (current_drift > drift_threshold and 
                      time_since_last_growth > min_time_between_growth and
                      self.current_capacity < self.max_capacity)
        
        logger.info(f"Should grow check: drift={current_drift}, threshold={drift_threshold}, "
                   f"time_since_last={time_since_last_growth}, current_capacity={self.current_capacity}, "
                   f"max_capacity={self.max_capacity}, result={should_grow}")
        
        return should_grow

class GrowableConceptNet(nn.Module):
    """A neural network that can grow its capacity over time"""
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = None,
                 output_size: int = 2,  # Binary classification
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
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
            
        # Output layer for binary classification
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 1),  # Single output
            nn.Sigmoid()  # Squash to [0, 1]
        )
        
        # Initialize weights using better initialization
        self.apply(self._init_weights)
        
        # Training history
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
        """Forward pass with binary classification support"""
        logger.info(f"Forward pass input shape: {x.shape}")
        
        # Ensure input is properly shaped
        if len(x.shape) < 2:
            x = x.view(-1, self.input_size)
        
        # Process through hidden layers
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            logger.info(f"Layer {i} input shape: {x.shape}")
            x = layer(x)
            if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                x = bn(x)
            logger.info(f"Layer {i} output shape: {x.shape}")
            
        # Final output layer with binary classification
        x = self.output_layer(x)  # Returns probabilities in [0, 1]
        logger.info(f"Sigmoid output shape: {x.shape}")
        
        # Convert to binary classification format
        x = x.view(-1, 1)  # Ensure shape is (batch_size, 1)
        x = torch.cat([x, 1 - x], dim=1)  # [p, 1-p] for each sample
        x = torch.log(x + 1e-10)  # Add small epsilon to avoid log(0)
        logger.info(f"Final output shape: {x.shape}")
        
        return x
    
    def evaluate_drift(self, 
                      concept_id: str,
                      new_data: torch.Tensor,
                      old_data: torch.Tensor) -> float:
        """Evaluate concept drift between old and new data"""
        logger.info(f"Evaluating drift for concept {concept_id}")
        logger.info(f"New data shape: {new_data.shape}, Old data shape: {old_data.shape}")
        
        with torch.no_grad():
            # Get model outputs
            old_output = self(old_data)
            new_output = self(new_data)
            
            logger.info(f"Old output shape: {old_output.shape}, New output shape: {new_output.shape}")
            
            # Calculate input space drift
            input_mean_old = old_data.mean(0)
            input_mean_new = new_data.mean(0)
            input_mean_drift = torch.norm(input_mean_new - input_mean_old) / (torch.norm(input_mean_old) + 1e-6)
            
            input_std_old = old_data.std(0)
            input_std_new = new_data.std(0)
            input_std_drift = torch.norm(input_std_new - input_std_old) / (torch.norm(input_std_old) + 1e-6)
            
            # Calculate output space drift
            output_mean_old = old_output.mean(0)
            output_mean_new = new_output.mean(0)
            output_mean_drift = torch.norm(output_mean_new - output_mean_old) / (torch.norm(output_mean_old) + 1e-6)
            
            output_std_old = old_output.std(0)
            output_std_new = new_output.std(0)
            output_std_drift = torch.norm(output_std_new - output_std_old) / (torch.norm(output_std_old) + 1e-6)
            
            # Calculate representation change
            old_norm = old_output / (torch.norm(old_output, dim=1, keepdim=True) + 1e-6)
            new_norm = new_output / (torch.norm(new_output, dim=1, keepdim=True) + 1e-6)
            cos_sim = torch.mean(torch.sum(old_norm * new_norm, dim=1))
            cos_drift = 1.0 - cos_sim
            
            # Calculate scale changes
            input_scale_old = torch.norm(old_data)
            input_scale_new = torch.norm(new_data)
            input_scale_drift = abs(input_scale_new - input_scale_old) / (input_scale_old + 1e-6)
            
            # Combine drift metrics
            total_drift = (
                input_mean_drift + 
                input_std_drift + 
                output_mean_drift + 
                output_std_drift + 
                cos_drift + 
                input_scale_drift
            ) / 6.0
            
            logger.info(f"Drift components: input_mean={input_mean_drift:.4f}, "
                       f"input_std={input_std_drift:.4f}, output_mean={output_mean_drift:.4f}, "
                       f"output_std={output_std_drift:.4f}, cos={cos_drift:.4f}, "
                       f"scale={input_scale_drift:.4f}, total={total_drift:.4f}")
            
            # Record drift metrics
            if concept_id not in self.concept_metrics:
                self.concept_metrics[concept_id] = {
                    'drift_history': [],
                    'last_update': datetime.now().isoformat(),
                    'last_loss': 0.0,
                    'last_accuracy': 0.0,
                    'last_drift': 0.0,
                    'drift_trend': 0.0,  # Slope of drift over time
                    'maturity_score': 0.0,  # Concept maturity score
                    'growth_threshold': 0.001  # Initial growth threshold
                }
            
            # Update drift history
            drift_entry = {
                'timestamp': datetime.now().isoformat(),
                'drift': total_drift.item(),
                'input_drift': input_mean_drift.item(),
                'output_drift': output_mean_drift.item(),
                'scale_drift': input_scale_drift.item(),
                'cos_drift': cos_drift.item(),
                'input_data': new_data.mean(0).tolist()  # Store mean input data
            }
            self.concept_metrics[concept_id]['drift_history'].append(drift_entry)
            
            # Calculate drift trend (slope of last 5 points)
            drift_history = self.concept_metrics[concept_id]['drift_history']
            if len(drift_history) >= 5:
                recent_drifts = [entry['drift'] for entry in drift_history[-5:]]
                timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in drift_history[-5:]]
                time_diffs = [(t - timestamps[0]).total_seconds() for t in timestamps]
                
                # Calculate linear regression slope
                x_mean = sum(time_diffs) / len(time_diffs)
                y_mean = sum(recent_drifts) / len(recent_drifts)
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(time_diffs, recent_drifts))
                denominator = sum((x - x_mean) ** 2 for x in time_diffs)
                slope = numerator / (denominator + 1e-6)
                
                self.concept_metrics[concept_id]['drift_trend'] = slope
                
                # Update maturity score based on drift stability
                stability_score = 1.0 / (1.0 + abs(slope))  # Higher score for stable drift
                self.concept_metrics[concept_id]['maturity_score'] = (
                    0.7 * self.concept_metrics[concept_id]['maturity_score'] +
                    0.3 * stability_score
                )
                
                # Adjust growth threshold based on maturity
                base_threshold = 0.001
                maturity_factor = 1.0 - (0.5 * self.concept_metrics[concept_id]['maturity_score'])
                self.concept_metrics[concept_id]['growth_threshold'] = base_threshold * maturity_factor
            
            return total_drift.item()
    
    def should_grow(self, concept_id: str) -> Tuple[bool, int]:
        """Determine if any layer should grow based on concept drift and maturity"""
        if concept_id not in self.concept_metrics:
            logger.info(f"No metrics found for concept {concept_id}")
            return False, -1
            
        metrics = self.concept_metrics[concept_id]
        drift = metrics['last_drift']
        drift_trend = metrics['drift_trend']
        maturity = metrics['maturity_score']
        threshold = metrics['growth_threshold']
        
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
                layer_drift = layer.get_drift_contribution()
                if layer_drift > max_layer_drift:
                    max_layer_drift = layer_drift
                    max_drift_layer = i
            
            if max_drift_layer >= 0:
                logger.info(f"Layer {max_drift_layer} should grow due to persistent drift")
                return True, max_drift_layer
        
        logger.info("No layers need to grow")
        return False, -1
    
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
                'last_drift': 0.0
            }
            
        self.concept_metrics[concept_id].update({
            'last_update': datetime.now().isoformat(),
            'last_loss': loss,
            'last_accuracy': accuracy,
            'last_drift': drift
        })
    
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
            'concepts_tracked': len(self.concept_metrics)
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