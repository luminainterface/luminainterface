#!/usr/bin/env python3
"""
Memory Node Test Suite for V7 Node Consciousness.

This test suite validates the core functionality of the Memory Node:
- Memory creation and storage
- Memory retrieval and search
- Memory update and deletion
- Memory decay and persistence
"""

import sys
import os
import time
import uuid
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

try:
    from src.v7.memory.memory_node import MemoryNode
except ImportError as e:
    print(f"Failed to import MemoryNode: {e}")
    print("Make sure you are running this from the project root directory")
    sys.exit(1)


class TestMemoryNode(unittest.TestCase):
    """Test suite for the Memory Node functionality."""

    def setUp(self):
        """Set up test environment with temporary memory storage."""
        # Create a temporary directory for memory storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize memory node with in-memory SQLite for faster tests
        self.memory_node = MemoryNode(
            storage_type="sqlite",
            memory_path=self.temp_dir,
            enable_persistence=True,
            decay_rate=0.1,  # Higher decay rate for quicker testing
            decay_interval=1,  # Check decay every second
            minimum_strength=0.2  # Higher threshold for testing
        )
        
        # Sample memory for testing
        self.sample_memory_id = str(uuid.uuid4())
        self.sample_memory = {
            "id": self.sample_memory_id,
            "content": "This is a test memory for unit testing",
            "memory_type": "test",
            "strength": 0.8,
            "tags": ["test", "sample"],
            "metadata": {"purpose": "testing", "version": "1.0"}
        }

    def tearDown(self):
        """Clean up after tests."""
        # Close memory node connections
        if hasattr(self.memory_node, "_storage") and hasattr(self.memory_node._storage, "close"):
            self.memory_node._storage.close()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_creation(self):
        """Test creating and storing memories."""
        # Store the sample memory
        self.memory_node.store_memory(
            memory_id=self.sample_memory_id,
            content=self.sample_memory["content"],
            memory_type=self.sample_memory["memory_type"],
            strength=self.sample_memory["strength"],
            tags=self.sample_memory["tags"],
            metadata=self.sample_memory["metadata"]
        )
        
        # Retrieve and verify
        memory = self.memory_node.get_memory(self.sample_memory_id)
        self.assertIsNotNone(memory, "Memory should be retrieved")
        self.assertEqual(memory["id"], self.sample_memory_id, "Memory ID should match")
        self.assertEqual(memory["content"], self.sample_memory["content"], "Memory content should match")
        self.assertEqual(memory["memory_type"], self.sample_memory["memory_type"], "Memory type should match")
        self.assertAlmostEqual(memory["strength"], self.sample_memory["strength"], places=2)
        self.assertEqual(set(memory["tags"]), set(self.sample_memory["tags"]), "Memory tags should match")
        
        # Verify creation and last accessed times
        self.assertIn("created_at", memory, "Memory should have creation timestamp")
        self.assertIn("last_accessed", memory, "Memory should have last accessed timestamp")

    def test_bulk_memory_creation(self):
        """Test creating multiple memories in bulk."""
        # Create a batch of memories
        memory_count = 10
        memories = []
        
        for i in range(memory_count):
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": f"Bulk test memory {i}",
                "memory_type": "bulk_test",
                "strength": 0.7 + (i / 100),  # Slightly different strengths
                "tags": ["bulk", f"test_{i}"],
                "metadata": {"index": i}
            }
            memories.append(memory)
            
            # Store memory
            self.memory_node.store_memory(
                memory_id=memory["id"],
                content=memory["content"],
                memory_type=memory["memory_type"],
                strength=memory["strength"],
                tags=memory["tags"],
                metadata=memory["metadata"]
            )
        
        # Verify all memories were stored
        all_memories = self.memory_node.list_memories()
        self.assertEqual(len(all_memories), memory_count, f"Should have stored {memory_count} memories")
        
        # Verify memory retrieval by type
        bulk_memories = self.memory_node.search_memories(memory_type="bulk_test")
        self.assertEqual(len(bulk_memories), memory_count, f"Should retrieve {memory_count} bulk test memories")

    def test_memory_retrieval(self):
        """Test memory retrieval and search capabilities."""
        # Store multiple memories with different attributes
        memories = [
            {
                "id": str(uuid.uuid4()),
                "content": "Python is a programming language",
                "memory_type": "fact",
                "strength": 0.9,
                "tags": ["fact", "programming", "python"],
                "metadata": {"verified": True}
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Memory testing is important",
                "memory_type": "fact",
                "strength": 0.85,
                "tags": ["fact", "testing", "important"],
                "metadata": {"verified": True}
            },
            {
                "id": str(uuid.uuid4()),
                "content": "I learned about memory retrieval",
                "memory_type": "experience",
                "strength": 0.75,
                "tags": ["experience", "learning", "retrieval"],
                "metadata": {"context": "testing"}
            }
        ]
        
        # Store all memories
        for memory in memories:
            self.memory_node.store_memory(
                memory_id=memory["id"],
                content=memory["content"],
                memory_type=memory["memory_type"],
                strength=memory["strength"],
                tags=memory["tags"],
                metadata=memory["metadata"]
            )
        
        # Test retrieval by ID
        for memory in memories:
            retrieved = self.memory_node.get_memory(memory["id"])
            self.assertEqual(retrieved["content"], memory["content"], "Content should match for retrieved memory")
        
        # Test search by type
        facts = self.memory_node.search_memories(memory_type="fact")
        self.assertEqual(len(facts), 2, "Should find 2 fact memories")
        
        experiences = self.memory_node.search_memories(memory_type="experience")
        self.assertEqual(len(experiences), 1, "Should find 1 experience memory")
        
        # Test search by tag
        python_memories = self.memory_node.search_memories(tags=["python"])
        self.assertEqual(len(python_memories), 1, "Should find 1 memory with python tag")
        
        # Test search by content
        memory_matches = self.memory_node.search_memories(content="memory")
        self.assertEqual(len(memory_matches), 2, "Should find 2 memories containing 'memory'")
        
        # Test search by minimum strength
        strong_memories = self.memory_node.search_memories(min_strength=0.8)
        self.assertEqual(len(strong_memories), 2, "Should find 2 memories with strength >= 0.8")

    def test_memory_update(self):
        """Test updating memory attributes."""
        # Store a memory
        memory_id = str(uuid.uuid4())
        self.memory_node.store_memory(
            memory_id=memory_id,
            content="Original content",
            memory_type="test",
            strength=0.7,
            tags=["original", "test"],
            metadata={"version": 1}
        )
        
        # Verify initial state
        original = self.memory_node.get_memory(memory_id)
        self.assertEqual(original["content"], "Original content", "Content should match initial value")
        self.assertEqual(original["strength"], 0.7, "Strength should match initial value")
        
        # Update the memory
        self.memory_node.update_memory(
            memory_id=memory_id,
            content="Updated content",
            strength=0.8,
            tags=["updated", "test"],
            metadata={"version": 2}
        )
        
        # Verify updates
        updated = self.memory_node.get_memory(memory_id)
        self.assertEqual(updated["content"], "Updated content", "Content should be updated")
        self.assertEqual(updated["strength"], 0.8, "Strength should be updated")
        self.assertEqual(set(updated["tags"]), {"updated", "test"}, "Tags should be updated")
        self.assertEqual(updated["metadata"]["version"], 2, "Metadata should be updated")
        
        # Partial update (only strength)
        self.memory_node.update_memory(
            memory_id=memory_id,
            strength=0.9
        )
        
        # Verify partial update
        partial = self.memory_node.get_memory(memory_id)
        self.assertEqual(partial["content"], "Updated content", "Content should remain unchanged")
        self.assertEqual(partial["strength"], 0.9, "Strength should be updated")
        self.assertEqual(set(partial["tags"]), {"updated", "test"}, "Tags should remain unchanged")

    def test_memory_deletion(self):
        """Test memory deletion."""
        # Store a memory
        memory_id = str(uuid.uuid4())
        self.memory_node.store_memory(
            memory_id=memory_id,
            content="Memory to delete",
            memory_type="test",
            strength=0.7,
            tags=["deletion", "test"],
            metadata={"temporary": True}
        )
        
        # Verify memory exists
        memory = self.memory_node.get_memory(memory_id)
        self.assertIsNotNone(memory, "Memory should exist before deletion")
        
        # Delete the memory
        self.memory_node.delete_memory(memory_id)
        
        # Verify memory no longer exists
        deleted = self.memory_node.get_memory(memory_id)
        self.assertIsNone(deleted, "Memory should not exist after deletion")
        
        # Verify deletion doesn't affect other memories
        other_id = str(uuid.uuid4())
        self.memory_node.store_memory(
            memory_id=other_id,
            content="Another memory",
            memory_type="test",
            strength=0.8,
            tags=["test"],
            metadata={}
        )
        
        # Delete the first memory again (should not raise error)
        self.memory_node.delete_memory(memory_id)
        
        # Verify other memory still exists
        other = self.memory_node.get_memory(other_id)
        self.assertIsNotNone(other, "Other memory should still exist")

    def test_memory_decay(self):
        """Test memory decay mechanism."""
        # Store memories with different strengths
        memories = []
        for i in range(5):
            memory_id = str(uuid.uuid4())
            # Strengths: 0.3, 0.4, 0.5, 0.6, 0.7
            strength = 0.3 + (i * 0.1)
            
            self.memory_node.store_memory(
                memory_id=memory_id,
                content=f"Decay test memory {i}",
                memory_type="decay_test",
                strength=strength,
                tags=["decay", "test"],
                metadata={"index": i}
            )
            memories.append({"id": memory_id, "strength": strength})
        
        # Verify all memories were stored
        initial_count = len(self.memory_node.list_memories())
        self.assertEqual(initial_count, 5, "Should have stored 5 memories")
        
        # Force decay by setting last check time far in the past
        self.memory_node._last_decay_check = time.time() - 100
        
        # Trigger decay check
        self.memory_node._check_memory_decay()
        
        # Verify memories have decayed
        remaining = self.memory_node.list_memories()
        self.assertLess(len(remaining), initial_count, "Some memories should have decayed")
        
        # Verify only memories above minimum_strength remain
        for memory in remaining:
            self.assertGreaterEqual(memory["strength"], self.memory_node._minimum_strength,
                                   "Remaining memories should have strength above minimum")

    def test_memory_access_updates_strength(self):
        """Test that accessing memories updates their strength and last accessed time."""
        # Store a memory
        memory_id = str(uuid.uuid4())
        initial_strength = 0.5
        
        self.memory_node.store_memory(
            memory_id=memory_id,
            content="Access test memory",
            memory_type="test",
            strength=initial_strength,
            tags=["access", "test"],
            metadata={}
        )
        
        # Get initial state
        initial = self.memory_node.get_memory(memory_id)
        initial_last_accessed = initial["last_accessed"]
        
        # Wait a moment to ensure timestamp changes
        time.sleep(0.1)
        
        # Access the memory multiple times
        for _ in range(3):
            self.memory_node.get_memory(memory_id)
            time.sleep(0.01)  # Small delay between accesses
        
        # Get updated state
        updated = self.memory_node.get_memory(memory_id)
        
        # Verify strength increased
        self.assertGreater(updated["strength"], initial_strength,
                          "Memory strength should increase after access")
        
        # Verify last_accessed timestamp updated
        self.assertGreater(updated["last_accessed"], initial_last_accessed,
                          "Last accessed timestamp should be updated")


if __name__ == "__main__":
    unittest.main() 