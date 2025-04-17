"""
Version 3 Core Module
This module contains the core components for Version 3 of the system.
"""

from .spiderweb_bridge import SpiderwebBridge, VersionType, VersionInfo
from .fractal_bridge import FractalBridge
from .fractal_simulator import FractalSimulator, FractalType, FractalConfig

__all__ = [
    'SpiderwebBridge',
    'VersionType',
    'VersionInfo',
    'FractalBridge',
    'FractalSimulator',
    'FractalType',
    'FractalConfig'
] 