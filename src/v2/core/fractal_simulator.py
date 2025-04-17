"""
Fractal Simulator for Version 2
This module provides a comprehensive fractal simulation system with support for:
- Julia sets
- Mandelbrot sets
- Custom fractal types
- Real-time rendering
- Color mapping
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class FractalType(Enum):
    MANDELBROT = 'mandelbrot'
    JULIA = 'julia'
    BURNING_SHIP = 'burning_ship'
    TRICORN = 'tricorn'

@dataclass
class FractalConfig:
    """Configuration for fractal generation."""
    width: int = 800
    height: int = 600
    max_iterations: int = 100
    escape_radius: float = 2.0
    zoom: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    julia_c: complex = -0.7 + 0.27j
    color_map: str = 'viridis'
    custom_color_map: Optional[List[Tuple[float, float, float]]] = None

class FractalSimulator:
    def __init__(self, config: Optional[FractalConfig] = None):
        """
        Initialize the fractal simulator.
        
        Args:
            config: Optional fractal configuration
        """
        self.config = config or FractalConfig()
        self._init_color_maps()
        
        logger.info("Initialized FractalSimulator")
    
    def _init_color_maps(self) -> None:
        """Initialize color maps."""
        self.color_maps = {
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
            'cividis': plt.cm.cividis
        }
        
        if self.config.custom_color_map:
            self.color_maps['custom'] = LinearSegmentedColormap.from_list(
                'custom', self.config.custom_color_map
            )
    
    def generate_mandelbrot(self) -> np.ndarray:
        """
        Generate a Mandelbrot set.
        
        Returns:
            Array of iteration counts
        """
        x = np.linspace(
            self.config.center_x - 2.0 / self.config.zoom,
            self.config.center_x + 2.0 / self.config.zoom,
            self.config.width
        )
        y = np.linspace(
            self.config.center_y - 1.5 / self.config.zoom,
            self.config.center_y + 1.5 / self.config.zoom,
            self.config.height
        )
        
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=int)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(z) < self.config.escape_radius
            z[mask] = z[mask] * z[mask] + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def generate_julia(self, c: Optional[complex] = None) -> np.ndarray:
        """
        Generate a Julia set.
        
        Args:
            c: Optional complex parameter for Julia set
            
        Returns:
            Array of iteration counts
        """
        c = c or self.config.julia_c
        
        x = np.linspace(
            self.config.center_x - 2.0 / self.config.zoom,
            self.config.center_x + 2.0 / self.config.zoom,
            self.config.width
        )
        y = np.linspace(
            self.config.center_y - 1.5 / self.config.zoom,
            self.config.center_y + 1.5 / self.config.zoom,
            self.config.height
        )
        
        X, Y = np.meshgrid(x, y)
        z = X + 1j * Y
        iterations = np.zeros(z.shape, dtype=int)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(z) < self.config.escape_radius
            z[mask] = z[mask] * z[mask] + c
            iterations[mask] = i
        
        return iterations
    
    def generate_burning_ship(self) -> np.ndarray:
        """
        Generate a Burning Ship fractal.
        
        Returns:
            Array of iteration counts
        """
        x = np.linspace(
            self.config.center_x - 2.0 / self.config.zoom,
            self.config.center_x + 2.0 / self.config.zoom,
            self.config.width
        )
        y = np.linspace(
            self.config.center_y - 1.5 / self.config.zoom,
            self.config.center_y + 1.5 / self.config.zoom,
            self.config.height
        )
        
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=int)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(z) < self.config.escape_radius
            z[mask] = np.abs(np.real(z[mask])) + 1j * np.abs(np.imag(z[mask]))
            z[mask] = z[mask] * z[mask] + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def generate_tricorn(self) -> np.ndarray:
        """
        Generate a Tricorn fractal.
        
        Returns:
            Array of iteration counts
        """
        x = np.linspace(
            self.config.center_x - 2.0 / self.config.zoom,
            self.config.center_x + 2.0 / self.config.zoom,
            self.config.width
        )
        y = np.linspace(
            self.config.center_y - 1.5 / self.config.zoom,
            self.config.center_y + 1.5 / self.config.zoom,
            self.config.height
        )
        
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=int)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(z) < self.config.escape_radius
            z[mask] = np.conj(z[mask] * z[mask]) + c[mask]
            iterations[mask] = i
        
        return iterations
    
    def render_fractal(self, iterations: np.ndarray) -> np.ndarray:
        """
        Render a fractal with color mapping.
        
        Args:
            iterations: Array of iteration counts
            
        Returns:
            RGB array of the rendered fractal
        """
        cmap = self.color_maps.get(self.config.color_map, plt.cm.viridis)
        normalized = iterations / self.config.max_iterations
        return cmap(normalized)[:, :, :3]
    
    def generate_fractal(self, fractal_type: FractalType) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and render a fractal.
        
        Args:
            fractal_type: Type of fractal to generate
            
        Returns:
            Tuple of (iterations, rendered_image)
        """
        if fractal_type == FractalType.MANDELBROT:
            iterations = self.generate_mandelbrot()
        elif fractal_type == FractalType.JULIA:
            iterations = self.generate_julia()
        elif fractal_type == FractalType.BURNING_SHIP:
            iterations = self.generate_burning_ship()
        elif fractal_type == FractalType.TRICORN:
            iterations = self.generate_tricorn()
        else:
            raise ValueError(f"Unknown fractal type: {fractal_type}")
        
        rendered = self.render_fractal(iterations)
        return iterations, rendered
    
    def update_config(self, **kwargs) -> None:
        """
        Update fractal configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        if 'custom_color_map' in kwargs:
            self._init_color_maps()
        
        logger.info(f"Updated fractal configuration: {kwargs}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current fractal metrics.
        
        Returns:
            Dictionary containing fractal metrics
        """
        return {
            'resolution': (self.config.width, self.config.height),
            'max_iterations': self.config.max_iterations,
            'zoom': self.config.zoom,
            'center': (self.config.center_x, self.config.center_y),
            'color_map': self.config.color_map
        }

# Export functionality for node integration
functionality = {
    'classes': {
        'FractalSimulator': FractalSimulator,
        'FractalConfig': FractalConfig,
        'FractalType': FractalType
    },
    'description': 'Fractal simulation system with support for multiple fractal types'
} 