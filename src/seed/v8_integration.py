"""
V8 Integration Module

This module integrates the V8 Spatial Temple and Seed Growth systems with our SeedEngine.
It provides bidirectional communication and concept flow between the systems.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid

from src.seed.seed_engine import SeedEngine
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode
from src.v8.temple_to_seed_bridge import ConceptSeed
from src.v8.auto_seed_growth import SeedGrowthEngine

logger = logging.getLogger(__name__)

class V8Integration:
    """
    Integrates the V8 system with our SeedEngine, allowing bidirectional
    flow of concepts and growth patterns between the systems.
    """
    
    def __init__(self, seed_engine: SeedEngine, temple_mapper: Optional[SpatialTempleMapper] = None):
        self.seed_engine = seed_engine
        self.temple_mapper = temple_mapper
        self.v8_growth_engine = SeedGrowthEngine(temple_mapper)
        self.concept_mapping: Dict[str, str] = {}  # Maps V8 concept IDs to seed phrases
        
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper for V8 integration"""
        self.temple_mapper = temple_mapper
        self.v8_growth_engine.set_temple_mapper(temple_mapper)
        
    def start_integration(self):
        """Start the integration between systems"""
        self.v8_growth_engine.start_growth()
        logger.info("Started V8 integration with SeedEngine")
        
    def stop_integration(self):
        """Stop the integration between systems"""
        self.v8_growth_engine.stop_growth()
        logger.info("Stopped V8 integration with SeedEngine")
        
    def convert_to_seed(self, concept_seed: ConceptSeed) -> Dict:
        """
        Convert a V8 ConceptSeed to our SeedEngine format
        
        Args:
            concept_seed: The V8 ConceptSeed to convert
            
        Returns:
            Dictionary containing the seed information in our format
        """
        # Create context from attributes and connections
        context = {
            "weight": concept_seed.weight,
            "node_type": concept_seed.node_type,
            "connections": list(concept_seed.connections),
            **concept_seed.attributes
        }
        
        # Plant the seed in our engine
        seed = self.seed_engine.plant(concept_seed.concept, context)
        
        # Store mapping
        self.concept_mapping[concept_seed.id] = concept_seed.concept
        
        return seed
        
    def convert_to_concept_seed(self, seed_phrase: str) -> Optional[ConceptSeed]:
        """
        Convert a seed from our engine to a V8 ConceptSeed
        
        Args:
            seed_phrase: The seed phrase to convert
            
        Returns:
            ConceptSeed if successful, None otherwise
        """
        seed = self.seed_engine.get_seed_details(seed_phrase)
        if not seed:
            return None
            
        # Create ConceptSeed
        concept_seed = ConceptSeed(
            concept=seed_phrase,
            weight=seed["emergence_score"],
            node_type=seed["context"].get("node_type", "concept"),
            connections=set(seed["context"].get("connections", [])),
            attributes={
                k: v for k, v in seed["context"].items() 
                if k not in ["weight", "node_type", "connections"]
            }
        )
        
        return concept_seed
        
    def sync_growth(self):
        """
        Synchronize growth between the systems:
        1. Harvest new concepts from V8
        2. Convert and plant them in our engine
        3. Check for growth spurts in our engine
        4. Update V8 with new growth
        """
        if not self.temple_mapper:
            logger.warning("No temple mapper available for sync")
            return
            
        # 1. Harvest new concepts from V8
        self.v8_growth_engine._harvest_seeds_from_temple()
        
        # 2. Convert and plant new seeds
        for seed in self.v8_growth_engine.seed_pool:
            if seed.id not in self.concept_mapping:
                self.convert_to_seed(seed)
                
        # 3. Check for growth spurts in our engine
        for seed_phrase, seed in self.seed_engine.board.items():
            if seed["status"] == "growthspurting":
                # 4. Update V8 with new growth
                concept_seed = self.convert_to_concept_seed(seed_phrase)
                if concept_seed:
                    # Add to V8 growth engine
                    if not any(s.id == concept_seed.id for s in self.v8_growth_engine.seed_pool):
                        self.v8_growth_engine.seed_pool.append(concept_seed)
                        
    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current status of the integration"""
        return {
            "v8_seeds": len(self.v8_growth_engine.seed_pool),
            "our_seeds": len(self.seed_engine.board),
            "mapped_concepts": len(self.concept_mapping),
            "v8_growth_running": self.v8_growth_engine.running
        }
        
    def process_growth_cycle(self):
        """Process one growth cycle for both systems"""
        # Process our engine's growth
        self.seed_engine.process_growth_cycle()
        
        # Sync with V8
        self.sync_growth()
        
        # Process V8 growth
        if self.v8_growth_engine.running:
            self.v8_growth_engine._growth_loop() 