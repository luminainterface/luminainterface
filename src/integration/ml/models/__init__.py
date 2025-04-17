"""
Model Registry

This module maintains a registry of all available models.
"""

from typing import Dict, Type
from ..core import BaseModel

# Registry to store model classes
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {} 