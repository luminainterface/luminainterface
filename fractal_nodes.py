"""
Fractal Nodes Module
Implements fractal-based neural network nodes for complex pattern recognition and processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import cmath
import logging

class MandelbrotHybridNode(nn.Module):
    """A hybrid node that combines fractal mathematics with neural processing."""
    
    def __init__(self, dimension=512, max_iter=100):
        """Initialize the MandelbrotHybridNode."""
        super().__init__()
        self.dimension = dimension
        self.max_iter = max_iter
        self.logger = logging.getLogger(__name__)
        self.is_active = True
        self.version = "1.0.0"
        self.state = {
            'complexity': 0.0,
            'stability': 1.0,
            'iterations': 0,
            'last_processed': None
        }
        
        # Fractal field processors
        self.escape_processor = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        self.orbit_processor = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        self.boundary_processor = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        # Fractal parameters
        self.cardioid_center = nn.Parameter(torch.tensor([-0.75, 0.0]))
        self.main_bulb_radius = nn.Parameter(torch.tensor(0.25))
        
    def _compute_escape_time(self, c: complex, max_iter: int) -> Tuple[int, complex]:
        z = 0
        for n in range(max_iter):
            z = z*z + c
            if abs(z) > 2:
                return n, z
        return max_iter, z
        
    def _process_point(self, x: float, y: float) -> torch.Tensor:
        c = complex(x, y)
        escape_time, final_z = self._compute_escape_time(c, self.max_iter)
        
        # Create feature vector
        features = torch.tensor([
            escape_time / self.max_iter,
            abs(final_z),
            final_z.real,
            final_z.imag,
            x, y
        ], dtype=torch.float32)
        
        return features
        
    def forward(self, input_tensor):
        """Process input data using fractal mathematics."""
        try:
            # Update state
            self.state['iterations'] += 1
            self.state['last_processed'] = input_tensor
            
            batch_size = input_tensor.size(0)
            
            # Extract coordinates from input
            x = input_tensor[:, 0]
            y = input_tensor[:, 1]
            
            # Process each point in the batch
            features = []
            for i in range(batch_size):
                point_features = self._process_point(x[i].item(), y[i].item())
                features.append(point_features)
            
            features = torch.stack(features)
            
            # Expand to match dimension
            if features.size(-1) < self.dimension:
                padding = torch.zeros(batch_size, self.dimension - features.size(-1))
                features = torch.cat([features, padding], dim=-1)
            
            # Process through fractal fields
            escape_field = self.escape_processor(features)
            orbit_field = self.orbit_processor(features)
            
            # Combine fields for boundary analysis
            combined = torch.cat([escape_field, orbit_field], dim=-1)
            boundary_field = self.boundary_processor(combined)
            
            # Calculate complexity
            self.state['complexity'] = self._calculate_complexity(boundary_field.detach().cpu().numpy())
            
            # Update stability
            self.state['stability'] = self._update_stability()
            
            return {
                'status': 'success',
                'escape_field': escape_field,
                'orbit_field': orbit_field,
                'boundary_field': boundary_field,
                'is_in_set': (escape_field.mean(dim=-1) > 0.5).float(),
                'fractal_dimension': torch.sigmoid(boundary_field.mean(dim=-1)),
                'complexity': self.state['complexity'],
                'stability': self.state['stability']
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_complexity(self, pattern: np.ndarray) -> float:
        """Calculate the complexity of a fractal pattern."""
        # Calculate entropy as a measure of complexity
        hist = np.histogram(pattern.flatten(), bins=256)[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)
    
    def _update_stability(self) -> float:
        """Update the stability metric based on recent processing."""
        # Simple stability calculation based on iteration count
        stability = 1.0 / (1.0 + np.log(1 + self.state['iterations']))
        return float(stability)
    
    def get_health(self) -> Dict[str, Any]:
        """Get the health status of the node."""
        return {
            'is_active': self.is_active,
            'version': self.version,
            'state': self.state,
            'last_processed': self.state['last_processed']
        }
    
    def compute_cardioid_boundary(self, t: float) -> complex:
        """Compute point on the main cardioid boundary"""
        angle = t * 2 * np.pi
        return 0.5 * cmath.exp(1j * angle) * (1 - cmath.exp(1j * angle))

class JuliaHybridNode(nn.Module):
    def __init__(self, dimension=512, max_iter=100):
        super().__init__()
        self.dimension = dimension
        self.max_iter = max_iter
        
        # Dynamic parameter for Julia set
        self.c = nn.Parameter(torch.tensor([-0.4, 0.6]))  # Default to interesting Julia set
        
        # Field processors
        self.stability_processor = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        self.cycle_processor = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
        self.connection_processor = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.Tanh(),
            nn.Linear(dimension, dimension)
        )
        
    def _compute_julia_orbit(self, z: complex, max_iter: int) -> Tuple[int, complex]:
        c = complex(self.c[0].item(), self.c[1].item())
        for n in range(max_iter):
            z = z*z + c
            if abs(z) > 2:
                return n, z
        return max_iter, z
        
    def _process_point(self, x: float, y: float) -> torch.Tensor:
        z = complex(x, y)
        orbit_time, final_z = self._compute_julia_orbit(z, self.max_iter)
        
        # Create feature vector
        features = torch.tensor([
            orbit_time / self.max_iter,
            abs(final_z),
            final_z.real,
            final_z.imag,
            x, y,
            self.c[0].item(),
            self.c[1].item()
        ], dtype=torch.float32)
        
        return features
        
    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        
        # Extract coordinates from input
        x = input_tensor[:, 0]
        y = input_tensor[:, 1]
        
        # Process each point in the batch
        features = []
        for i in range(batch_size):
            point_features = self._process_point(x[i].item(), y[i].item())
            features.append(point_features)
            
        features = torch.stack(features)
        
        # Expand to match dimension
        if features.size(-1) < self.dimension:
            padding = torch.zeros(batch_size, self.dimension - features.size(-1))
            features = torch.cat([features, padding], dim=-1)
            
        # Process through fractal fields
        stability_field = self.stability_processor(features)
        cycle_field = self.cycle_processor(features)
        
        # Combine fields for connection analysis
        combined = torch.cat([stability_field, cycle_field], dim=-1)
        connection_field = self.connection_processor(combined)
        
        return {
            "stability_field": stability_field,
            "cycle_field": cycle_field,
            "connection_field": connection_field,
            "is_in_set": (stability_field.mean(dim=-1) > 0.5).float(),
            "fractal_dimension": torch.sigmoid(connection_field.mean(dim=-1)),
            "julia_parameter": self.c.detach()
        }
        
    def set_julia_parameter(self, real: float, imag: float):
        """Set the c parameter for the Julia set"""
        with torch.no_grad():
            self.c.copy_(torch.tensor([real, imag]))
            
    def find_periodic_points(self, period: int = 1) -> List[complex]:
        """Find periodic points of given period"""
        c = complex(self.c[0].item(), self.c[1].item())
        points = []
        
        # Grid search for periodic points
        for x in np.linspace(-2, 2, 20):
            for y in np.linspace(-2, 2, 20):
                z = complex(x, y)
                z0 = z
                
                # Iterate to check for periodicity
                for _ in range(period):
                    z = z*z + c
                
                # Check if point is periodic
                if abs(z - z0) < 1e-6:
                    points.append(z0)
                    
        return points 