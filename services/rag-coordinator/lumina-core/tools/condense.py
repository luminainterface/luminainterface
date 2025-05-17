import json
import networkx as nx
import community
import pandas as pd
from typing import Dict, Any, List
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_graph(file_path: str) -> nx.Graph:
    """Load graph from JSON or ZIP file."""
    logger.info(f"Loading graph from {file_path}")
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        G = nx.Graph()
        for node in data['nodes']:
            G.add_node(node['id'], **node.get('attributes', {}))
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge.get('attributes', {}))
    else:
        raise ValueError("Unsupported file format")
    return G

def louvain_clustering(G: nx.Graph, min_delta: float = 0.01) -> Dict[str, Any]:
    """Perform multi-level Louvain clustering until modularity delta is small."""
    logger.info("Starting Louvain clustering")
    hierarchy = []
    current_graph = G.copy()
    level = 0
    
    while True:
        # Get partition
        partition = community.best_partition(current_graph)
        
        # Calculate modularity
        mod = community.modularity(partition, current_graph)
        
        # Store level info
        level_info = {
            'level': level,
            'modularity': mod,
            'clusters': partition
        }
        hierarchy.append(level_info)
        
        # Check if we should stop
        if level > 0:
            delta = mod - hierarchy[level-1]['modularity']
            if delta < min_delta:
                break
        
        # Create next level graph
        next_graph = nx.Graph()
        for node, cluster in partition.items():
            if cluster not in next_graph:
                next_graph.add_node(cluster)
        
        # Add edges between clusters
        for u, v, data in current_graph.edges(data=True):
            u_cluster = partition[u]
            v_cluster = partition[v]
            if u_cluster != v_cluster:
                if next_graph.has_edge(u_cluster, v_cluster):
                    next_graph[u_cluster][v_cluster]['weight'] += data.get('weight', 1)
                else:
                    next_graph.add_edge(u_cluster, v_cluster, weight=data.get('weight', 1))
        
        current_graph = next_graph
        level += 1
    
    return hierarchy

def save_results(hierarchy: Dict[str, Any], G: nx.Graph, output_dir: str):
    """Save hierarchy, meta edges, and stripped graph."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save hierarchy
    logger.info("Saving hierarchy")
    with open(output_path / "hierarchy.json", 'w') as f:
        json.dump(hierarchy, f)
    
    # Save meta edges
    logger.info("Saving meta edges")
    meta_edges = []
    for level in hierarchy:
        for node, cluster in level['clusters'].items():
            meta_edges.append({
                'node': node,
                'cluster': cluster,
                'level': level['level']
            })
    pd.DataFrame(meta_edges).to_parquet(output_path / "meta_edges.parquet")
    
    # Save stripped graph
    logger.info("Saving stripped graph")
    nx.write_gpickle(G, output_path / "stripped_graph.gpickle")

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Condense graph data")
    parser.add_argument("input", help="Input graph file (JSON)")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum modularity delta")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load graph
    G = load_graph(args.input)
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Perform clustering
    hierarchy = louvain_clustering(G, args.min_delta)
    logger.info(f"Generated {len(hierarchy)} levels of hierarchy")
    
    # Save results
    save_results(hierarchy, G, args.output)
    
    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 