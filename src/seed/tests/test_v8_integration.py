"""
Tests for the V8 Integration module
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.seed.v8_integration import V8Integration
from src.seed.seed_engine import SeedEngine
from src.v8.temple_to_seed_bridge import ConceptSeed
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode

class TestV8Integration(unittest.TestCase):
    """Test cases for V8Integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.seed_engine = MagicMock(spec=SeedEngine)
        self.temple_mapper = MagicMock(spec=SpatialTempleMapper)
        self.integration = V8Integration(self.seed_engine)
        
    def test_initialization(self):
        """Test initialization of V8Integration"""
        self.assertIsNotNone(self.integration)
        self.assertEqual(self.integration.seed_engine, self.seed_engine)
        self.assertIsNone(self.integration.temple_mapper)
        
    def test_set_temple_mapper(self):
        """Test setting the temple mapper"""
        self.integration.set_temple_mapper(self.temple_mapper)
        self.assertEqual(self.integration.temple_mapper, self.temple_mapper)
        
    def test_start_stop_integration(self):
        """Test starting and stopping integration"""
        self.integration.set_temple_mapper(self.temple_mapper)
        
        # Start integration
        self.integration.start_integration()
        self.assertTrue(self.integration.v8_growth_engine.running)
        
        # Stop integration
        self.integration.stop_integration()
        self.assertFalse(self.integration.v8_growth_engine.running)
        
    def test_convert_to_seed(self):
        """Test converting V8 ConceptSeed to our format"""
        concept_seed = ConceptSeed(
            concept="test_concept",
            weight=0.8,
            node_type="concept",
            connections={"connection1", "connection2"},
            attributes={"attr1": "value1"}
        )
        
        # Mock seed engine response
        self.seed_engine.plant.return_value = {
            "seed_phrase": "test_concept",
            "emergence_score": 0.8,
            "context": {
                "weight": 0.8,
                "node_type": "concept",
                "connections": ["connection1", "connection2"],
                "attr1": "value1"
            }
        }
        
        result = self.integration.convert_to_seed(concept_seed)
        
        # Verify conversion
        self.assertEqual(result["seed_phrase"], "test_concept")
        self.assertEqual(result["emergence_score"], 0.8)
        self.assertEqual(result["context"]["weight"], 0.8)
        self.assertEqual(result["context"]["node_type"], "concept")
        self.assertEqual(set(result["context"]["connections"]), {"connection1", "connection2"})
        self.assertEqual(result["context"]["attr1"], "value1")
        
        # Verify mapping
        self.assertEqual(self.integration.concept_mapping[concept_seed.id], "test_concept")
        
    def test_convert_to_concept_seed(self):
        """Test converting our seed to V8 ConceptSeed"""
        seed_phrase = "test_concept"
        seed_details = {
            "seed_phrase": seed_phrase,
            "emergence_score": 0.8,
            "context": {
                "weight": 0.8,
                "node_type": "concept",
                "connections": ["connection1", "connection2"],
                "attr1": "value1"
            }
        }
        
        self.seed_engine.get_seed_details.return_value = seed_details
        
        concept_seed = self.integration.convert_to_concept_seed(seed_phrase)
        
        self.assertIsNotNone(concept_seed)
        self.assertEqual(concept_seed.concept, seed_phrase)
        self.assertEqual(concept_seed.weight, 0.8)
        self.assertEqual(concept_seed.node_type, "concept")
        self.assertEqual(concept_seed.connections, {"connection1", "connection2"})
        self.assertEqual(concept_seed.attributes, {"attr1": "value1"})
        
    def test_sync_growth(self):
        """Test synchronization of growth between systems"""
        self.integration.set_temple_mapper(self.temple_mapper)
        
        # Mock V8 growth engine
        v8_seed = ConceptSeed(
            concept="v8_concept",
            weight=0.8,
            node_type="concept",
            connections=set(),
            attributes={}
        )
        self.integration.v8_growth_engine.seed_pool = [v8_seed]
        
        # Mock our engine's board
        self.seed_engine.board = {
            "our_concept": {
                "status": "growthspurting",
                "emergence_score": 0.9,
                "context": {
                    "weight": 0.9,
                    "node_type": "concept",
                    "connections": []
                }
            }
        }
        
        # Mock seed details
        self.seed_engine.get_seed_details.return_value = {
            "seed_phrase": "our_concept",
            "emergence_score": 0.9,
            "context": {
                "weight": 0.9,
                "node_type": "concept",
                "connections": []
            }
        }
        
        # Perform sync
        self.integration.sync_growth()
        
        # Verify V8 seed was planted
        self.seed_engine.plant.assert_called_once()
        
        # Verify our seed was added to V8 pool
        self.assertEqual(len(self.integration.v8_growth_engine.seed_pool), 2)
        
    def test_get_integration_status(self):
        """Test getting integration status"""
        self.integration.set_temple_mapper(self.temple_mapper)
        
        # Mock V8 growth engine
        self.integration.v8_growth_engine.seed_pool = [MagicMock()] * 3
        self.integration.v8_growth_engine.running = True
        
        # Mock our engine
        self.seed_engine.board = {"seed1": {}, "seed2": {}}
        
        status = self.integration.get_integration_status()
        
        self.assertEqual(status["v8_seeds"], 3)
        self.assertEqual(status["our_seeds"], 2)
        self.assertEqual(status["mapped_concepts"], 0)
        self.assertTrue(status["v8_growth_running"])
        
    def test_process_growth_cycle(self):
        """Test processing a growth cycle"""
        self.integration.set_temple_mapper(self.temple_mapper)
        self.integration.v8_growth_engine.running = True
        
        # Mock sync method
        with patch.object(self.integration, 'sync_growth') as mock_sync:
            self.integration.process_growth_cycle()
            
            # Verify methods were called
            self.seed_engine.process_growth_cycle.assert_called_once()
            mock_sync.assert_called_once()
            self.integration.v8_growth_engine._growth_loop.assert_called_once()

if __name__ == '__main__':
    unittest.main() 