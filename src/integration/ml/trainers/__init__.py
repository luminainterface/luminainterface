"""
Trainer Registry

This module maintains a registry of all available trainers.
"""

from typing import Dict, Type
from ..core import BaseTrainer

# Registry to store trainer classes
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {} 