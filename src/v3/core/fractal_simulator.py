"""
Fractal Simulator for Version 3
This module provides enhanced fractal generation capabilities with support for
multiple fractal types, customizable parameters, and performance optimizations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class FractalType(Enum):
    MANDELBROT = 'mandelbrot'
    JULIA = 'julia'
    BURNING_SHIP = 'burning_ship'
    TRICORN = 'tricorn'
    NEWTON = 'newton'
    PHOENIX = 'phoenix'

@dataclass
class FractalConfig:
    """Configuration for fractal generation."""
    width: int = 800
    height: int = 600
    max_iterations: int = 1000
    escape_radius: float = 2.0
    zoom: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    color_map: str = 'viridis'
    julia_c_real: float = -0.7
    julia_c_imag: float = 0.27015
    newton_polynomial: List[complex] = None
    phoenix_c_real: float = 0.5667
    phoenix_c_imag: float = 0.0
    phoenix_p_real: float = -0.5
    phoenix_p_imag: float = 0.0

class FractalSimulator:
    def __init__(self, config: Optional[FractalConfig] = None):
        """
        Initialize the fractal simulator.
        
        Args:
            config: Optional configuration for fractal generation
        """
        self.config = config or FractalConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = {
            'generation_count': 0,
            'total_iterations': 0,
            'average_generation_time': 0.0,
            'last_generation_time': 0.0
        }
        
        logger.info("Initialized FractalSimulator V3")
    
    def update_config(self, **kwargs) -> None:
        """
        Update fractal configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Invalid configuration parameter: {key}")
    
    async def generate_fractal(self, fractal_type: FractalType) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a fractal asynchronously.
        
        Args:
            fractal_type: Type of fractal to generate
            
        Returns:
            Tuple of (iterations array, rendered image)
        """
        start_time = time.time()
        
        # Create coordinate grids
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
        Z = X + 1j * Y
        
        # Generate fractal based on type
        if fractal_type == FractalType.MANDELBROT:
            iterations = await self._generate_mandelbrot(Z)
        elif fractal_type == FractalType.JULIA:
            iterations = await self._generate_julia(Z)
        elif fractal_type == FractalType.BURNING_SHIP:
            iterations = await self._generate_burning_ship(Z)
        elif fractal_type == FractalType.TRICORN:
            iterations = await self._generate_tricorn(Z)
        elif fractal_type == FractalType.NEWTON:
            iterations = await self._generate_newton(Z)
        elif fractal_type == FractalType.PHOENIX:
            iterations = await self._generate_phoenix(Z)
        else:
            raise ValueError(f"Unsupported fractal type: {fractal_type}")
        
        # Render the fractal
        rendered = await self._render_fractal(iterations)
        
        # Update metrics
        generation_time = time.time() - start_time
        self.metrics['generation_count'] += 1
        self.metrics['total_iterations'] += np.sum(iterations)
        self.metrics['last_generation_time'] = generation_time
        self.metrics['average_generation_time'] = (
            (self.metrics['average_generation_time'] * (self.metrics['generation_count'] - 1) +
             generation_time) / self.metrics['generation_count']
        )
        
        return iterations, rendered
    
    async def _generate_mandelbrot(self, Z: np.ndarray) -> np.ndarray:
        """Generate Mandelbrot set."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._mandelbrot_kernel,
            Z
        )
    
    def _mandelbrot_kernel(self, Z: np.ndarray) -> np.ndarray:
        """Kernel function for Mandelbrot set generation."""
        iterations = np.zeros(Z.shape, dtype=np.int32)
        C = Z.copy()
        
        for i in range(self.config.max_iterations):
            mask = np.abs(Z) < self.config.escape_radius
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            iterations[mask] += 1
        
        return iterations
    
    async def _generate_julia(self, Z: np.ndarray) -> np.ndarray:
        """Generate Julia set."""
        c = self.config.julia_c_real + 1j * self.config.julia_c_imag
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._julia_kernel,
            Z, c
        )
    
    def _julia_kernel(self, Z: np.ndarray, c: complex) -> np.ndarray:
        """Kernel function for Julia set generation."""
        iterations = np.zeros(Z.shape, dtype=np.int32)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(Z) < self.config.escape_radius
            Z[mask] = Z[mask] * Z[mask] + c
            iterations[mask] += 1
        
        return iterations
    
    async def _generate_burning_ship(self, Z: np.ndarray) -> np.ndarray:
        """Generate Burning Ship fractal."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._burning_ship_kernel,
            Z
        )
    
    def _burning_ship_kernel(self, Z: np.ndarray) -> np.ndarray:
        """Kernel function for Burning Ship fractal generation."""
        iterations = np.zeros(Z.shape, dtype=np.int32)
        C = Z.copy()
        
        for i in range(self.config.max_iterations):
            mask = np.abs(Z) < self.config.escape_radius
            Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag)) ** 2 + C[mask]
            iterations[mask] += 1
        
        return iterations
    
    async def _generate_tricorn(self, Z: np.ndarray) -> np.ndarray:
        """Generate Tricorn fractal."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._tricorn_kernel,
            Z
        )
    
    def _tricorn_kernel(self, Z: np.ndarray) -> np.ndarray:
        """Kernel function for Tricorn fractal generation."""
        iterations = np.zeros(Z.shape, dtype=np.int32)
        C = Z.copy()
        
        for i in range(self.config.max_iterations):
            mask = np.abs(Z) < self.config.escape_radius
            Z[mask] = np.conjugate(Z[mask]) ** 2 + C[mask]
            iterations[mask] += 1
        
        return iterations
    
    async def _generate_newton(self, Z: np.ndarray) -> np.ndarray:
        """Generate Newton fractal."""
        if not self.config.newton_polynomial:
            self.config.newton_polynomial = [1, 0, 0, -1]  # Default: z^3 - 1
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._newton_kernel,
            Z
        )
    
    def _newton_kernel(self, Z: np.ndarray) -> np.ndarray:
        """Kernel function for Newton fractal generation."""
        iterations = np.zeros(Z.shape, dtype=np.int32)
        roots = np.roots(self.config.newton_polynomial)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(Z) < self.config.escape_radius
            f = np.polyval(self.config.newton_polynomial, Z[mask])
            df = np.polyval(np.polyder(self.config.newton_polynomial), Z[mask])
            Z[mask] = Z[mask] - f / df
            iterations[mask] += 1
        
        # Find closest root for each point
        distances = np.abs(Z[:, :, np.newaxis] - roots)
        iterations = np.argmin(distances, axis=2)
        
        return iterations
    
    async def _generate_phoenix(self, Z: np.ndarray) -> np.ndarray:
        """Generate Phoenix fractal."""
        c = self.config.phoenix_c_real + 1j * self.config.phoenix_c_imag
        p = self.config.phoenix_p_real + 1j * self.config.phoenix_p_imag
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._phoenix_kernel,
            Z, c, p
        )
    
    def _phoenix_kernel(self, Z: np.ndarray, c: complex, p: complex) -> np.ndarray:
        """Kernel function for Phoenix fractal generation."""
        iterations = np.zeros(Z.shape, dtype=np.int32)
        prev_z = np.zeros_like(Z)
        
        for i in range(self.config.max_iterations):
            mask = np.abs(Z) < self.config.escape_radius
            temp = Z[mask]
            Z[mask] = Z[mask] * Z[mask] + c + p * prev_z[mask]
            prev_z[mask] = temp
            iterations[mask] += 1
        
        return iterations
    
    async def _render_fractal(self, iterations: np.ndarray) -> np.ndarray:
        """Render the fractal with the specified color map."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._render_kernel,
            iterations
        )
    
    def _render_kernel(self, iterations: np.ndarray) -> np.ndarray:
        """Kernel function for fractal rendering."""
        # Normalize iterations
        norm = plt.Normalize(vmin=0, vmax=self.config.max_iterations)
        normalized = norm(iterations)
        
        # Apply color map
        cmap = plt.get_cmap(self.config.color_map)
        rendered = cmap(normalized)
        
        return (rendered * 255).astype(np.uint8)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics of the fractal simulator.
        
        Returns:
            Dictionary containing metrics
        """
        return self.metrics

# Export functionality for node integration
functionality = {
    'classes': {
        'FractalSimulator': FractalSimulator,
        'FractalType': FractalType,
        'FractalConfig': FractalConfig
    },
    'description': 'Fractal simulator with enhanced features'
} 