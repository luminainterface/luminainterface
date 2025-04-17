#!/usr/bin/env python3
"""
Demo Data Generator for Spatial Temple (v8)

This module provides functions to generate sample data for the Spatial Temple
visualization system. It creates realistic-looking demo nodes and connections
to demonstrate the Spatial Temple interface without requiring external data.
"""

import random
import math
import uuid
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

# Try to import from the correct path
try:
    from src.v8.spatial_temple_mapper import SpatialNode
except ImportError:
    # If in the same directory, try direct import
    try:
        from spatial_temple_mapper import SpatialNode
    except ImportError:
        # Define a minimal SpatialNode for standalone usage
        class SpatialNode:
            """Represents a concept in 3D spatial coordinates"""
            
            def __init__(self, 
                        concept: str, 
                        position: Optional[Tuple[float, float, float]] = None,
                        weight: float = 1.0,
                        node_type: str = "concept"):
                self.id = str(uuid.uuid4())
                self.concept = concept
                self.position = position or (0.0, 0.0, 0.0)
                self.weight = weight
                self.node_type = node_type
                self.connections = set()  # IDs of connected nodes
                self.attributes = {}
                self.created_at = datetime.now().isoformat()

# Sample concepts for demo nodes
SAMPLE_CONCEPTS = [
    # AI and ML concepts
    "Neural Network", "Deep Learning", "Machine Learning", "Artificial Intelligence",
    "Transformer", "Attention Mechanism", "Natural Language Processing", "Computer Vision",
    "Reinforcement Learning", "Supervised Learning", "Unsupervised Learning", "Transfer Learning",
    
    # Knowledge concepts
    "Knowledge Graph", "Semantic Network", "Ontology", "Taxonomy",
    "Information Retrieval", "Data Mining", "Knowledge Representation", "Cognitive System",
    
    # Temple and spatial concepts
    "Spatial Memory", "Memory Palace", "Method of Loci", "Cognitive Map",
    "Mental Model", "Temple Architecture", "Sacred Geometry", "Ritual Space",
    
    # Philosophical concepts
    "Consciousness", "Epistemology", "Metaphysics", "Ontology",
    "Mind", "Perception", "Reality", "Truth",
    
    # Technical concepts
    "Graph Database", "Vector Space", "Embedding", "Dimensionality Reduction",
    "Clustering", "Classification", "Visualization", "Interface",
    
    # Additional domain concepts
    "Language", "Memory", "Learning", "Reasoning",
    "Creativity", "Intelligence", "Adaptation", "Evolution"
]

# Node types
NODE_TYPES = [
    "concept", "entity", "process", "attribute", 
    "action", "event", "relation", "property"
]

def generate_position(radius_min: float = 30.0, 
                     radius_max: float = 100.0, 
                     center: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
    """
    Generate a random 3D position within a spherical shell around a center point
    
    Args:
        radius_min: Minimum radius from center
        radius_max: Maximum radius from center
        center: Center point of the spherical shell
        
    Returns:
        Tuple of (x, y, z) coordinates
    """
    # Random radius within the shell
    radius = random.uniform(radius_min, radius_max)
    
    # Random angles
    theta = random.uniform(0, 2 * math.pi)  # Longitude angle (0 to 2π)
    phi = random.uniform(0, math.pi)        # Latitude angle (0 to π)
    
    # Convert spherical to cartesian coordinates
    x = center[0] + radius * math.sin(phi) * math.cos(theta)
    y = center[1] + radius * math.sin(phi) * math.sin(theta)
    z = center[2] + radius * math.cos(phi)
    
    return (x, y, z)

def generate_demo_nodes(count: int = 50) -> List[SpatialNode]:
    """
    Generate a list of demo nodes for the Spatial Temple visualization
    
    Args:
        count: Number of nodes to generate
        
    Returns:
        List of SpatialNode objects
    """
    nodes = []
    
    # Ensure we have enough concepts
    concepts = SAMPLE_CONCEPTS.copy()
    # Add numbered concepts if we need more
    while len(concepts) < count:
        concepts.append(f"Concept {len(concepts) + 1}")
    
    # Shuffle concepts to randomize
    random.shuffle(concepts)
    
    # Create nodes
    for i in range(count):
        # Generate random position
        position = generate_position()
        
        # Random weight between 0.5 and 2.0
        weight = random.uniform(0.5, 2.0)
        
        # Random node type
        node_type = random.choice(NODE_TYPES)
        
        # Create node
        node = SpatialNode(
            concept=concepts[i],
            position=position,
            weight=weight,
            node_type=node_type
        )
        
        # Add some random attributes
        if random.random() < 0.7:  # 70% chance to have attributes
            num_attributes = random.randint(1, 3)
            for _ in range(num_attributes):
                attribute_type = random.choice(["importance", "confidence", "relevance", "frequency"])
                attribute_value = random.uniform(0.1, 1.0)
                node.attributes[attribute_type] = attribute_value
        
        nodes.append(node)
    
    # Create random connections between nodes (each node connects to 1-5 others)
    for i, node in enumerate(nodes):
        # Determine number of connections for this node
        num_connections = random.randint(1, min(5, count-1))
        
        # Create connections
        connected_indices = random.sample([j for j in range(count) if j != i], num_connections)
        for j in connected_indices:
            node.connections.add(nodes[j].id)
            nodes[j].connections.add(node.id)
    
    return nodes

def generate_themed_demo_nodes(count: int = 50, theme: str = "ai") -> List[SpatialNode]:
    """
    Generate themed demo nodes with positions based on the specified theme
    
    Args:
        count: Number of nodes to generate
        theme: Theme to use for positioning ("ai", "temple", "network", "brain")
        
    Returns:
        List of SpatialNode objects
    """
    nodes = []
    
    # Copy and shuffle concepts
    concepts = SAMPLE_CONCEPTS.copy()
    while len(concepts) < count:
        concepts.append(f"Concept {len(concepts) + 1}")
    random.shuffle(concepts)
    
    # Theme-specific positioning
    if theme == "temple":
        # Temple layout with nodes arranged in chambers and corridors
        for i in range(count):
            # Determine if this is a chamber node or corridor node
            is_chamber = (i % 10 == 0)
            
            if is_chamber:
                # Chamber nodes are larger and at key points
                chamber_idx = i // 10
                angle = chamber_idx * (2 * math.pi / (count // 10))
                r = 80.0
                position = (r * math.cos(angle), 20 * math.sin(angle // 2), r * math.sin(angle))
                weight = random.uniform(1.5, 2.5)
            else:
                # Corridor nodes connect chambers
                section = i // 10
                pos_in_section = i % 10
                angle1 = section * (2 * math.pi / (count // 10))
                angle2 = ((section + 1) % (count // 10)) * (2 * math.pi / (count // 10))
                t = pos_in_section / 10
                r1, r2 = 80.0, 80.0
                x = r1 * math.cos(angle1) * (1-t) + r2 * math.cos(angle2) * t
                z = r1 * math.sin(angle1) * (1-t) + r2 * math.sin(angle2) * t
                y = 20 * math.sin(angle1 // 2) * (1-t) + 20 * math.sin(angle2 // 2) * t
                position = (x, y, z)
                weight = random.uniform(0.5, 1.0)
            
            node = SpatialNode(
                concept=concepts[i],
                position=position,
                weight=weight,
                node_type=random.choice(NODE_TYPES)
            )
            nodes.append(node)
            
    elif theme == "network":
        # Network layout with clusters of related nodes
        num_clusters = 5
        for i in range(count):
            # Determine which cluster this node belongs to
            cluster = i % num_clusters
            
            # Generate position based on cluster
            cluster_center = (
                (cluster - num_clusters // 2) * 50,
                0,
                (cluster % 2) * 50 - 25
            )
            
            # Nodes are positioned around their cluster center
            if i % 10 == 0:  # Main cluster nodes
                jitter = 10
                weight = random.uniform(1.5, 2.0)
            else:  # Satellite nodes
                jitter = 30
                weight = random.uniform(0.5, 1.2)
            
            position = (
                cluster_center[0] + random.uniform(-jitter, jitter),
                cluster_center[1] + random.uniform(-jitter, jitter),
                cluster_center[2] + random.uniform(-jitter, jitter)
            )
            
            node = SpatialNode(
                concept=concepts[i],
                position=position,
                weight=weight,
                node_type=random.choice(NODE_TYPES)
            )
            nodes.append(node)
            
    elif theme == "brain":
        # Brain-like structure with hemisphere organization
        for i in range(count):
            # Left or right hemisphere
            hemisphere = -1 if i % 2 == 0 else 1
            
            # Different brain regions have different y positions
            region = i % 4  # 4 regions: frontal, parietal, temporal, occipital
            y_pos = 40 - 30 * (region / 3)
            
            # Position within the hemisphere
            region_angle = random.uniform(0, math.pi)
            region_height = random.uniform(-20, 20)
            
            # Calculate position
            x = hemisphere * (30 + 20 * math.cos(region_angle))
            y = y_pos + region_height
            z = 30 * math.sin(region_angle)
            
            position = (x, y, z)
            weight = 0.8 + 0.8 * math.sin(i * 0.1)  # Varying weights
            
            node = SpatialNode(
                concept=concepts[i],
                position=position,
                weight=weight,
                node_type=random.choice(NODE_TYPES)
            )
            nodes.append(node)
            
    else:  # Default "ai" theme or any other theme
        # AI-themed layout with layers like a neural network
        num_layers = 4
        for i in range(count):
            layer = i % num_layers
            position_in_layer = i // num_layers
            nodes_per_layer = count // num_layers + (1 if layer < count % num_layers else 0)
            
            # Position nodes in a grid-like pattern
            x = layer * 50 - 75  # Layers along x-axis
            
            # Arrange in a circular pattern within each layer
            angle = 2 * math.pi * position_in_layer / nodes_per_layer
            radius = 30 + 5 * math.sin(layer * 0.5)
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            position = (x, y, z)
            weight = 0.5 + 1.0 * ((num_layers - layer) / num_layers)  # Higher weights in earlier layers
            
            node = SpatialNode(
                concept=concepts[i],
                position=position,
                weight=weight,
                node_type=random.choice(NODE_TYPES)
            )
            nodes.append(node)
    
    # Create connections between nodes
    for i, node in enumerate(nodes):
        # Connect to 1-5 other nodes, with higher probability for nearby nodes
        num_connections = random.randint(1, min(5, count-1))
        
        # Calculate distances to all other nodes
        distances = []
        for j, other_node in enumerate(nodes):
            if i != j:
                x1, y1, z1 = node.position
                x2, y2, z2 = other_node.position
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                distances.append((j, distance))
        
        # Sort by distance and prefer closer nodes with some randomness
        distances.sort(key=lambda x: x[1])
        
        # Select connection candidates with bias towards closer nodes
        candidates = []
        for idx, (j, _) in enumerate(distances):
            # Higher probability for closer nodes
            probability = 1.0 - (idx / len(distances))
            if random.random() < probability and len(candidates) < num_connections:
                candidates.append(j)
                
        # Ensure we have enough connections
        while len(candidates) < num_connections:
            j = random.choice([j for j in range(count) if j != i and j not in candidates])
            candidates.append(j)
        
        # Create the connections
        for j in candidates:
            node.connections.add(nodes[j].id)
            nodes[j].connections.add(node.id)
    
    return nodes

if __name__ == "__main__":
    # Test the generator
    import json
    
    nodes = generate_demo_nodes(20)
    print(f"Generated {len(nodes)} demo nodes")
    
    # Display a sample node
    sample = nodes[0]
    print(f"Sample node: {sample.concept}")
    print(f"  Position: {sample.position}")
    print(f"  Weight: {sample.weight}")
    print(f"  Connections: {len(sample.connections)}")
    
    # Try themed generator
    themed_nodes = generate_themed_demo_nodes(20, "temple")
    print(f"\nGenerated {len(themed_nodes)} themed demo nodes")
    
    # Display a sample themed node
    sample = themed_nodes[0]
    print(f"Sample themed node: {sample.concept}")
    print(f"  Position: {sample.position}")
    print(f"  Weight: {sample.weight}")
    print(f"  Connections: {len(sample.connections)}") 