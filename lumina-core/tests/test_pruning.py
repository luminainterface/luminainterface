import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from lumina_core.memory.pruning import VectorPruner

@pytest.fixture
def mock_qdrant():
    with patch("lumina_core.memory.pruning.QdrantClient") as mock:
        mock_instance = mock.return_value
        mock_instance.get_collection = MagicMock()
        mock_instance.scroll = MagicMock()
        mock_instance.delete = MagicMock()
        yield mock_instance

def test_prune_vectors(mock_qdrant):
    # Setup mock data
    now = datetime.utcnow()
    old_date = now - timedelta(days=31)
    recent_date = now - timedelta(days=15)
    
    # Mock collection info
    mock_qdrant.get_collection.return_value = MagicMock(points_count=1000)
    
    # Mock scroll results with mixed points
    mock_points = [
        # Old point with low similarity
        MagicMock(
            id=1,
            payload={
                "timestamp": old_date.isoformat(),
                "similarity": 0.2
            }
        ),
        # Old point with high similarity
        MagicMock(
            id=2,
            payload={
                "timestamp": old_date.isoformat(),
                "similarity": 0.8
            }
        ),
        # Recent point with low similarity
        MagicMock(
            id=3,
            payload={
                "timestamp": recent_date.isoformat(),
                "similarity": 0.2
            }
        ),
        # Recent point with high similarity
        MagicMock(
            id=4,
            payload={
                "timestamp": recent_date.isoformat(),
                "similarity": 0.8
            }
        )
    ]
    mock_qdrant.scroll.return_value = (mock_points, None)
    
    # Create pruner and run
    pruner = VectorPruner(
        qdrant_url="http://test:6333",
        max_age_days=30,
        min_similarity=0.3
    )
    results = pruner.prune_vectors()
    
    # Verify results
    assert results["pruned"] == 1  # Only the old point with low similarity
    assert results["total_before"] == 1000
    assert results["remaining"] == 999
    
    # Verify Qdrant calls
    mock_qdrant.get_collection.assert_called_once()
    mock_qdrant.scroll.assert_called_once()
    mock_qdrant.delete.assert_called_once()
    
    # Verify delete call arguments
    delete_call = mock_qdrant.delete.call_args
    assert delete_call[1]["collection_name"] == "lumina-chat"
    assert delete_call[1]["points_selector"].points == [1]  # Only the old, low-similarity point

def test_prune_vectors_error_handling(mock_qdrant):
    # Simulate Qdrant error
    mock_qdrant.get_collection.side_effect = Exception("Qdrant error")
    
    pruner = VectorPruner(qdrant_url="http://test:6333")
    
    with pytest.raises(Exception) as exc_info:
        pruner.prune_vectors()
    
    assert "Qdrant error" in str(exc_info.value)

def test_prune_empty_collection(mock_qdrant):
    """Test pruning behavior with an empty collection."""
    # Mock empty collection
    mock_qdrant.get_collection.return_value = MagicMock(points_count=0)
    mock_qdrant.scroll.return_value = ([], None)
    
    pruner = VectorPruner(qdrant_url="http://test:6333")
    results = pruner.prune_vectors()
    
    assert results["pruned"] == 0
    assert results["total_before"] == 0
    assert results["remaining"] == 0
    
    # Verify Qdrant calls
    mock_qdrant.get_collection.assert_called_once()
    mock_qdrant.scroll.assert_called_once()
    mock_qdrant.delete.assert_not_called()

def test_prune_malformed_timestamps(mock_qdrant):
    """Test pruning behavior with malformed timestamps."""
    now = datetime.utcnow()
    
    # Mock collection info
    mock_qdrant.get_collection.return_value = MagicMock(points_count=3)
    
    # Mock points with various timestamp formats
    mock_points = [
        # Valid timestamp
        MagicMock(
            id=1,
            payload={
                "timestamp": now.isoformat(),
                "similarity": 0.2
            }
        ),
        # Invalid timestamp format
        MagicMock(
            id=2,
            payload={
                "timestamp": "invalid-date",
                "similarity": 0.2
            }
        ),
        # Missing timestamp
        MagicMock(
            id=3,
            payload={
                "similarity": 0.2
            }
        )
    ]
    mock_qdrant.scroll.return_value = (mock_points, None)
    
    pruner = VectorPruner(qdrant_url="http://test:6333")
    results = pruner.prune_vectors()
    
    # Should only prune points with valid timestamps
    assert results["pruned"] == 0  # None are old enough
    assert results["total_before"] == 3
    assert results["remaining"] == 3
    
    # Verify Qdrant calls
    mock_qdrant.get_collection.assert_called_once()
    mock_qdrant.scroll.assert_called_once()
    mock_qdrant.delete.assert_not_called()

def test_prune_malformed_similarity(mock_qdrant):
    """Test pruning behavior with malformed similarity scores."""
    now = datetime.utcnow()
    old_date = now - timedelta(days=31)
    
    # Mock collection info
    mock_qdrant.get_collection.return_value = MagicMock(points_count=3)
    
    # Mock points with various similarity formats
    mock_points = [
        # Valid similarity
        MagicMock(
            id=1,
            payload={
                "timestamp": old_date.isoformat(),
                "similarity": 0.2
            }
        ),
        # Invalid similarity type
        MagicMock(
            id=2,
            payload={
                "timestamp": old_date.isoformat(),
                "similarity": "not-a-number"
            }
        ),
        # Missing similarity
        MagicMock(
            id=3,
            payload={
                "timestamp": old_date.isoformat()
            }
        )
    ]
    mock_qdrant.scroll.return_value = (mock_points, None)
    
    pruner = VectorPruner(qdrant_url="http://test:6333")
    results = pruner.prune_vectors()
    
    # Should only prune points with valid similarity scores
    assert results["pruned"] == 1  # Only the point with valid similarity
    assert results["total_before"] == 3
    assert results["remaining"] == 2
    
    # Verify Qdrant calls
    mock_qdrant.get_collection.assert_called_once()
    mock_qdrant.scroll.assert_called_once()
    mock_qdrant.delete.assert_called_once()
    assert mock_qdrant.delete.call_args[1]["points_selector"].points == [1] 