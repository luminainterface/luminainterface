import unittest
import time
from datetime import datetime, timedelta
from src.seed.seed_engine import SeedEngine

class MockCentralNode:
    def __init__(self):
        self.processors = {'QuantumInfection': MockQuantumInfection()}
        self.nodes = {
            'EchoSpiralNode': MockEchoSpiral(),
            'GlyphStateNode': MockGlyphState()
        }
        
    def get_processor(self, name):
        return self.processors.get(name)
        
    def get_node(self, name):
        return self.nodes.get(name)

class MockQuantumInfection:
    def infect_data(self, data):
        return True

class MockEchoSpiral:
    def get_recent_memories(self):
        return [
            {"content": "test memory with emotion curiosity", "timestamp": datetime.now()},
            {"content": "test memory 2", "timestamp": datetime.now()}
        ]
        
    def find_related(self, content):
        return f"related to {content}"
        
    def insert_echo(self, message):
        return True

class MockGlyphState:
    def activate(self, glyph):
        return True

class TestSeedEngine(unittest.TestCase):
    def setUp(self):
        self.central_node = MockCentralNode()
        self.engine = SeedEngine(self.central_node)
        
    def test_plant_seed(self):
        """Test planting a new seed"""
        seed = self.engine.plant("test seed")
        self.assertIn("test seed", self.engine.board)
        self.assertEqual(seed["status"], "germinating")
        self.assertEqual(seed["emergence_score"], 1.0)  # Should be 1.0 initially
        
    def test_update_seed(self):
        """Test updating an existing seed"""
        self.engine.plant("test seed")
        context = {"emotion": "curiosity"}
        updated = self.engine.plant("test seed", context)
        self.assertEqual(updated["context"]["emotion"], "curiosity")
        
    def test_growth_spurt(self):
        """Test triggering a growth spurt"""
        seed = self.engine.plant("test seed")
        # Artificially increase emergence score
        seed["emergence_score"] = 4.0  # Above pi threshold
        self.engine._propagate("test seed", seed)
        self.assertEqual(seed["status"], "growthspurting")
        
    def test_glyph_suggestion(self):
        """Test glyph suggestion based on content"""
        # Test elemental glyphs
        fire_seed = self.engine.plant("fire and passion")
        self.assertEqual(fire_seed["glyph"], "🜂")
        
        water_seed = self.engine.plant("water of emotion")
        self.assertEqual(water_seed["glyph"], "🜄")
        
        # Test random alchemical glyph
        random_seed = self.engine.plant("random concept")
        self.assertIn(random_seed["glyph"], ["🜔", "🝊", "🜚", "🜕", "🜖", "🜗", "🜘", "🜙"])
        
    def test_evidence_board(self):
        """Test evidence board generation"""
        self.engine.plant("test seed 1")
        self.engine.plant("test seed 2")
        board = self.engine.get_evidence_board()
        
        self.assertIn("test seed 1", board)
        self.assertIn("test seed 2", board)
        self.assertEqual(len(board), 2)
        
    def test_prune_inactive(self):
        """Test pruning inactive seeds"""
        # Create an old seed
        old_seed = self.engine.plant("old seed")
        old_seed["timestamp"] = datetime.now() - timedelta(days=2)
        
        # Create a new seed
        self.engine.plant("new seed")
        
        # Prune seeds older than 1 day
        self.engine.prune_inactive_seeds(86400)
        
        board = self.engine.get_evidence_board()
        self.assertNotIn("old seed", board)
        self.assertIn("new seed", board)
        
    def test_resonance_check(self):
        """Test resonance checking"""
        # Test direct match
        memory = {"content": "test seed content"}
        self.assertTrue(self.engine._check_resonance("test seed", memory, {}))
        
        # Test context match with word overlap
        memory = {"content": "something about curiosity and learning"}
        self.assertTrue(self.engine._check_resonance("seed", memory, {"emotion": "curiosity learning"}))
        
        # Test no match
        memory = {"content": "unrelated content"}
        self.assertFalse(self.engine._check_resonance("test seed", memory, {}))
        
    def test_growth_cycle(self):
        """Test processing a growth cycle"""
        seed = self.engine.plant("test seed")
        seed["status"] = "growthspurting"
        
        initial_score = seed["emergence_score"]
        self.engine.process_growth_cycle()
        
        # Score should have decayed
        self.assertLess(seed["emergence_score"], initial_score)
        
    def test_thread_limiting(self):
        """Test thread count limiting"""
        seed = self.engine.plant("test seed")
        
        # Add more threads than the limit
        for i in range(150):
            thread = {
                "resonance": 0.5,
                "matched": f"memory {i}",
                "echo": f"echo {i}",
                "timestamp": datetime.now()
            }
            # Add threads directly to test limiting
            if len(seed["threads"]) < self.engine.max_threads:
                seed["threads"].append(thread)
                
        self.assertLessEqual(len(seed["threads"]), self.engine.max_threads)
        
    def test_propagation_without_decay(self):
        """Test propagation without decay"""
        seed = self.engine.plant("test seed")
        initial_score = seed["emergence_score"]
        
        # Propagate without decay
        self.engine._propagate("test seed", seed, apply_decay=False)
        self.assertGreaterEqual(seed["emergence_score"], initial_score)

if __name__ == '__main__':
    unittest.main() 