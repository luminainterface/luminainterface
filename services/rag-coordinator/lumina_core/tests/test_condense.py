import pytest
import json
import networkx as nx
from pathlib import Path
from lumina_core.tools.condense import load_graph, louvain_clustering, save_results

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    # Add nodes
    for i in range(10):
        G.add_node(f"node_{i}", value=i)
    # Add edges
    for i in range(9):
        G.add_edge(f"node_{i}", f"node_{i+1}", weight=1)
    return G

@pytest.fixture
def sample_json(tmp_path):
    """Create a sample JSON file for testing."""
    data = {
        "nodes": [
            {"id": f"node_{i}", "attributes": {"value": i}} for i in range(10)
        ],
        "edges": [
            {"source": f"node_{i}", "target": f"node_{i+1}", "attributes": {"weight": 1}}
            for i in range(9)
        ]
    }
    file_path = tmp_path / "test_graph.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

def test_load_graph(sample_json):
    """Test loading graph from JSON."""
    G = load_graph(sample_json)
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 9
    assert G.has_edge("node_0", "node_1")
    assert G["node_0"]["node_1"]["weight"] == 1

def test_louvain_clustering(sample_graph):
    """Test Louvain clustering."""
    hierarchy = louvain_clustering(sample_graph, min_delta=0.01)
    assert len(hierarchy) > 0
    assert all('level' in level for level in hierarchy)
    assert all('modularity' in level for level in hierarchy)
    assert all('clusters' in level for level in hierarchy)
    
    # Check that modularity increases
    mods = [level['modularity'] for level in hierarchy]
    assert all(mods[i] <= mods[i+1] for i in range(len(mods)-1))

def test_save_results(sample_graph, tmp_path):
    """Test saving results."""
    hierarchy = louvain_clustering(sample_graph)
    save_results(hierarchy, sample_graph, tmp_path)
    
    # Check files exist
    assert (tmp_path / "hierarchy.json").exists()
    assert (tmp_path / "meta_edges.parquet").exists()
    assert (tmp_path / "stripped_graph.gpickle").exists()
    
    # Check hierarchy file
    with open(tmp_path / "hierarchy.json") as f:
        saved_hierarchy = json.load(f)
    assert len(saved_hierarchy) == len(hierarchy)
    
    # Check that number of level-1 clusters is reasonable
    level_1_clusters = set(hierarchy[0]['clusters'].values())
    assert len(level_1_clusters) <= 1000  # As per requirements

def test_integration(sample_json, tmp_path):
    """Test the full pipeline."""
    # Load graph
    G = load_graph(sample_json)
    assert G.number_of_nodes() == 10
    
    # Cluster
    hierarchy = louvain_clustering(G)
    assert len(hierarchy) > 0
    
    # Save results
    save_results(hierarchy, G, tmp_path)
    assert (tmp_path / "hierarchy.json").exists()
    assert (tmp_path / "meta_edges.parquet").exists()
    assert (tmp_path / "stripped_graph.gpickle").exists() 