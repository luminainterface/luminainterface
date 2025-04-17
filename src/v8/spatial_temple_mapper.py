#!/usr/bin/env python3
"""
Spatial Temple Mapper Module (v8)

Part of the Spatial Temple Interface for Lumina Neural Network v8.
This module provides spatial organization capabilities for language memory,
enabling 3D conceptual navigation and temple-based knowledge mapping.
"""

import logging
import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.spatial_temple_mapper")

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
        self.connections: Set[str] = set()  # IDs of connected nodes
        self.attributes: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "id": self.id,
            "concept": self.concept,
            "position": self.position,
            "weight": self.weight,
            "node_type": self.node_type,
            "connections": list(self.connections),
            "attributes": self.attributes,
            "created_at": self.created_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpatialNode':
        """Create node from dictionary"""
        node = cls(
            concept=data["concept"],
            position=data["position"],
            weight=data["weight"],
            node_type=data["node_type"]
        )
        node.id = data["id"]
        node.connections = set(data["connections"])
        node.attributes = data["attributes"]
        node.created_at = data["created_at"]
        return node

class SpatialConnection:
    """Represents a connection between nodes in the spatial temple"""
    
    def __init__(self, 
                source_id: str, 
                target_id: str,
                strength: float = 1.0,
                connection_type: str = "association"):
        self.id = str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.strength = strength
        self.connection_type = connection_type
        self.attributes: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert connection to dictionary for serialization"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "connection_type": self.connection_type,
            "attributes": self.attributes,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpatialConnection':
        """Create connection from dictionary"""
        connection = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            strength=data["strength"],
            connection_type=data["connection_type"]
        )
        connection.id = data["id"]
        connection.attributes = data["attributes"]
        connection.created_at = data["created_at"]
        return connection

class TempleZone:
    """Represents a zone within the spatial temple"""
    
    ZONE_TYPES = [
        "knowledge", "reflection", "integration", 
        "contradiction", "synthesis", "memory", 
        "consciousness", "query", "ritual"
    ]
    
    def __init__(self, 
                name: str,
                zone_type: str,
                center: Tuple[float, float, float],
                radius: float):
        self.id = str(uuid.uuid4())
        self.name = name
        self.zone_type = zone_type if zone_type in self.ZONE_TYPES else "knowledge"
        self.center = center
        self.radius = radius
        self.nodes: Set[str] = set()  # IDs of nodes in this zone
        self.attributes: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is within this zone"""
        x1, y1, z1 = self.center
        x2, y2, z2 = point
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        return distance <= self.radius
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert zone to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "zone_type": self.zone_type,
            "center": self.center,
            "radius": self.radius,
            "nodes": list(self.nodes),
            "attributes": self.attributes,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TempleZone':
        """Create zone from dictionary"""
        zone = cls(
            name=data["name"],
            zone_type=data["zone_type"],
            center=data["center"],
            radius=data["radius"]
        )
        zone.id = data["id"]
        zone.nodes = set(data["nodes"])
        zone.attributes = data["attributes"]
        zone.created_at = data["created_at"]
        return zone

class SpatialTempleMapper:
    """
    Spatial Temple Mapper for organizing concepts in 3D space
    
    This class maps language concepts into a 3D spatial temple for navigation
    and knowledge organization. It uses temple metaphors like chambers, paths,
    and ritual spaces to organize information.
    """
    
    def __init__(self, language_memory=None):
        """
        Initialize the Spatial Temple Mapper
        
        Args:
            language_memory: Optional LanguageMemory instance to extract concepts
        """
        logger.info("Initializing Spatial Temple Mapper")
        self.language_memory = language_memory
        
        # Core data structures
        self.nodes: Dict[str, SpatialNode] = {}
        self.connections: Dict[str, SpatialConnection] = {}
        self.zones: Dict[str, TempleZone] = {}
        
        # Create default temple structure
        self._initialize_temple_structure()
        
        # Configuration
        self.config = {
            "default_spacing": 5.0,
            "min_connection_strength": 0.3,
            "max_concepts_per_zone": 50,
            "temple_radius": 100.0
        }
        
    def _initialize_temple_structure(self):
        """Initialize the basic temple structure with core zones"""
        # Create central chamber (origin)
        central = TempleZone(
            name="Central Chamber",
            zone_type="integration",
            center=(0, 0, 0),
            radius=20.0
        )
        self.zones[central.id] = central
        
        # Create main chambers in cardinal directions
        zones = [
            # Knowledge zones (north/south axis)
            ("Knowledge Chamber", "knowledge", (0, 50, 0), 15.0),
            ("Memory Chamber", "memory", (0, -50, 0), 15.0),
            
            # Processing zones (east/west axis)
            ("Reflection Chamber", "reflection", (50, 0, 0), 15.0),
            ("Contradiction Chamber", "contradiction", (-50, 0, 0), 15.0),
            
            # Consciousness zones (up/down axis)
            ("Consciousness Temple", "consciousness", (0, 0, 50), 15.0),
            ("Ritual Foundation", "ritual", (0, 0, -50), 15.0),
        ]
        
        for name, zone_type, center, radius in zones:
            zone = TempleZone(name=name, zone_type=zone_type, center=center, radius=radius)
            self.zones[zone.id] = zone
    
    def map_concepts(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Map concepts from text into spatial temple organization
        
        Args:
            text: Text to analyze and map
            context: Optional context information
            
        Returns:
            Mapping results with spatial positions and relationships
        """
        logger.info(f"Mapping concepts from text: {text[:50]}...")
        
        # Extract key concepts from text
        concepts = self._extract_concepts(text)
        
        # Map concepts to spatial positions
        mapped_concepts = self._position_concepts(concepts, context)
        
        # Create connections between related concepts
        connections = self._create_concept_connections(mapped_concepts)
        
        # Organize concepts into temple zones
        zone_mapping = self._organize_into_zones(mapped_concepts)
        
        # Return the mapping results
        return {
            "text": text,
            "concepts": [node.to_dict() for node in mapped_concepts],
            "connections": [conn.to_dict() for conn in connections],
            "zones": {zone_id: self.zones[zone_id].to_dict() for zone_id in zone_mapping.keys()},
            "zone_mapping": {zone_id: list(concepts) for zone_id, concepts in zone_mapping.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_concepts(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract key concepts from text with their importance weights
        
        Returns:
            List of (concept, weight) tuples
        """
        # First try to use language_memory if available
        if self.language_memory and hasattr(self.language_memory, "extract_topics"):
            try:
                topics = self.language_memory.extract_topics(text)
                if isinstance(topics, dict) and "topics" in topics:
                    return [(topic["text"], topic["relevance"]) for topic in topics["topics"]]
            except Exception as e:
                logger.warning(f"Error using language_memory to extract topics: {e}")
        
        # Fallback: simple word frequency-based extraction
        words = text.lower().split()
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "in", "on", "at", "of", "to", "for", "with", "is", "are"}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Normalize weights
        max_count = max(word_counts.values()) if word_counts else 1
        concepts = [(word, count / max_count) for word, count in word_counts.items()]
        
        # Sort by weight and limit
        concepts.sort(key=lambda x: x[1], reverse=True)
        return concepts[:10]  # Limit to top 10 concepts
    
    def _position_concepts(self, 
                          concepts: List[Tuple[str, float]], 
                          context: Optional[Dict[str, Any]] = None) -> List[SpatialNode]:
        """
        Position concepts in 3D space based on relationships
        
        Args:
            concepts: List of (concept, weight) tuples
            context: Optional context information
            
        Returns:
            List of SpatialNode objects with 3D positions
        """
        result_nodes = []
        concept_texts = [c[0] for c in concepts]
        
        # Look for existing nodes first
        existing_concepts = {}
        for node_id, node in self.nodes.items():
            if node.concept in concept_texts:
                existing_concepts[node.concept] = node
        
        # Create embedding-based positions for new concepts
        for concept_text, weight in concepts:
            # Reuse existing node if available
            if concept_text in existing_concepts:
                node = existing_concepts[concept_text]
                # Update weight if the concept is more significant now
                if weight > node.weight:
                    node.weight = weight
                result_nodes.append(node)
                continue
            
            # Create a new node
            # For now, position in pseudo-random but deterministic locations based on concept text
            # In a real implementation, this would use NLP embeddings mapped to 3D space
            
            # Simple hash-based position generator (deterministic for same concept)
            h = sum(ord(c) for c in concept_text)
            r = 30.0 + (h % 40)  # Radius from center: 30-70
            theta = (h % 360) * (math.pi / 180)  # Angle in radians
            phi = ((h * 7) % 180) * (math.pi / 180)  # Elevation angle
            
            # Convert spherical to cartesian coordinates
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            # Create node with position
            node = SpatialNode(
                concept=concept_text,
                position=(x, y, z),
                weight=weight
            )
            
            # Store in overall nodes dictionary
            self.nodes[node.id] = node
            result_nodes.append(node)
        
        return result_nodes
    
    def _create_concept_connections(self, nodes: List[SpatialNode]) -> List[SpatialConnection]:
        """
        Create connections between related concepts
        
        Args:
            nodes: List of positioned concept nodes
            
        Returns:
            List of connections between the nodes
        """
        connections = []
        
        # For each pair of nodes, check if they should be connected
        for i, node1 in enumerate(nodes):
            for j in range(i+1, len(nodes)):
                node2 = nodes[j]
                
                # In a real implementation, we would calculate semantic similarity
                # For now, we'll simulate it with a distance-based approach
                x1, y1, z1 = node1.position
                x2, y2, z2 = node2.position
                
                # Calculate Euclidean distance
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                # Normalize to a 0-1 range (closer = stronger connection)
                # Assuming max distance of 200 units in the temple
                max_distance = 200.0
                similarity = max(0.0, 1.0 - (distance / max_distance))
                
                # Only create connections above threshold strength
                if similarity >= self.config["min_connection_strength"]:
                    connection = SpatialConnection(
                        source_id=node1.id,
                        target_id=node2.id,
                        strength=similarity
                    )
                    
                    # Update node connection sets
                    node1.connections.add(node2.id)
                    node2.connections.add(node1.id)
                    
                    # Store connection
                    self.connections[connection.id] = connection
                    connections.append(connection)
        
        return connections
    
    def _organize_into_zones(self, nodes: List[SpatialNode]) -> Dict[str, Set[str]]:
        """
        Organize nodes into temple zones based on their positions
        
        Args:
            nodes: List of positioned concept nodes
            
        Returns:
            Dictionary mapping zone IDs to sets of node IDs
        """
        zone_mapping = {}
        
        # Initialize empty sets for each zone
        for zone_id in self.zones:
            zone_mapping[zone_id] = set()
        
        # For each node, find zones that contain it
        for node in nodes:
            assigned = False
            for zone_id, zone in self.zones.items():
                if zone.contains_point(node.position):
                    zone_mapping[zone_id].add(node.id)
                    zone.nodes.add(node.id)
                    assigned = True
            
            # If not in any zone, assign to nearest zone
            if not assigned:
                nearest_zone_id = None
                min_distance = float('inf')
                
                for zone_id, zone in self.zones.items():
                    x1, y1, z1 = zone.center
                    x2, y2, z2 = node.position
                    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_zone_id = zone_id
                
                if nearest_zone_id:
                    zone_mapping[nearest_zone_id].add(node.id)
                    self.zones[nearest_zone_id].nodes.add(node.id)
        
        return zone_mapping
    
    def get_node(self, node_id: str) -> Optional[SpatialNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_connection(self, connection_id: str) -> Optional[SpatialConnection]:
        """Get a connection by ID"""
        return self.connections.get(connection_id)
    
    def get_zone(self, zone_id: str) -> Optional[TempleZone]:
        """Get a zone by ID"""
        return self.zones.get(zone_id)
    
    def get_all_nodes(self) -> List[SpatialNode]:
        """Get all nodes in the temple"""
        return list(self.nodes.values())
    
    def get_all_connections(self) -> List[SpatialConnection]:
        """Get all connections in the temple"""
        return list(self.connections.values())
    
    def get_all_zones(self) -> List[TempleZone]:
        """Get all zones in the temple"""
        return list(self.zones.values())
    
    def get_nodes_in_zone(self, zone_id: str) -> List[SpatialNode]:
        """Get all nodes in a specific zone"""
        zone = self.get_zone(zone_id)
        if not zone:
            return []
        return [self.nodes[node_id] for node_id in zone.nodes if node_id in self.nodes]
    
    def navigate_temple(self, 
                       start_position: Tuple[float, float, float], 
                       direction: Tuple[float, float, float],
                       distance: float) -> Dict[str, Any]:
        """
        Navigate through the temple from a starting position in a given direction
        
        Args:
            start_position: Starting position (x, y, z)
            direction: Direction vector (x, y, z)
            distance: Distance to travel
            
        Returns:
            Dictionary with navigation results
        """
        # Normalize direction vector
        dx, dy, dz = direction
        magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
        if magnitude > 0:
            dx, dy, dz = dx/magnitude, dy/magnitude, dz/magnitude
        else:
            dx, dy, dz = 0, 0, 0
        
        # Calculate end position
        x, y, z = start_position
        end_position = (x + dx * distance, y + dy * distance, z + dz * distance)
        
        # Find nodes along the path
        encountered_nodes = []
        encountered_zones = []
        
        # Check nodes near the path
        for node_id, node in self.nodes.items():
            node_x, node_y, node_z = node.position
            
            # Calculate distance from the line segment
            # Using vector projection to find closest point on line
            t = max(0, min(1, ((node_x - x) * dx + (node_y - y) * dy + (node_z - z) * dz) / distance))
            closest_x = x + t * dx * distance
            closest_y = y + t * dy * distance
            closest_z = z + t * dz * distance
            
            # Distance from node to closest point on line
            node_distance = math.sqrt(
                (node_x - closest_x)**2 + 
                (node_y - closest_y)**2 + 
                (node_z - closest_z)**2
            )
            
            # If node is close enough to path, include it
            if node_distance < 10.0:  # Proximity threshold
                encountered_nodes.append({
                    "node": node.to_dict(),
                    "distance": node_distance,
                    "position_on_path": t  # Normalized position along path (0-1)
                })
        
        # Check which zones are crossed
        for zone_id, zone in self.zones.items():
            # Simplified check - in a full implementation, we would do proper
            # ray-sphere intersection tests
            zone_x, zone_y, zone_z = zone.center
            
            # Similar calculation to find closest point on line
            t = max(0, min(1, ((zone_x - x) * dx + (zone_y - y) * dy + (zone_z - z) * dz) / distance))
            closest_x = x + t * dx * distance
            closest_y = y + t * dy * distance
            closest_z = z + t * dz * distance
            
            # Distance from zone center to closest point on line
            zone_distance = math.sqrt(
                (zone_x - closest_x)**2 + 
                (zone_y - closest_y)**2 + 
                (zone_z - closest_z)**2
            )
            
            # If path passes through or near zone, include it
            if zone_distance < zone.radius + 5.0:  # Zone radius plus tolerance
                encountered_zones.append({
                    "zone": zone.to_dict(),
                    "distance": zone_distance,
                    "position_on_path": t  # Normalized position along path (0-1)
                })
        
        # Sort by position on path
        encountered_nodes.sort(key=lambda x: x["position_on_path"])
        encountered_zones.sort(key=lambda x: x["position_on_path"])
        
        return {
            "start_position": start_position,
            "end_position": end_position,
            "direction": (dx, dy, dz),
            "distance": distance,
            "encountered_nodes": encountered_nodes,
            "encountered_zones": encountered_zones,
            "timestamp": datetime.now().isoformat()
        }
    
    def find_path(self, 
                 source_concept: str, 
                 target_concept: str) -> Dict[str, Any]:
        """
        Find a path between two concepts in the temple
        
        Args:
            source_concept: Starting concept text
            target_concept: Target concept text
            
        Returns:
            Dictionary with path results
        """
        # Find matching nodes
        source_nodes = [n for n in self.nodes.values() if n.concept.lower() == source_concept.lower()]
        target_nodes = [n for n in self.nodes.values() if n.concept.lower() == target_concept.lower()]
        
        if not source_nodes or not target_nodes:
            return {
                "path_found": False,
                "error": "Source or target concept not found in temple",
                "source_concept": source_concept,
                "target_concept": target_concept,
                "timestamp": datetime.now().isoformat()
            }
        
        # Use first matching node for each concept
        source_node = source_nodes[0]
        target_node = target_nodes[0]
        
        # Simple breadth-first search for path
        queue = [(source_node.id, [source_node.id])]
        visited = {source_node.id}
        
        while queue:
            node_id, path = queue.pop(0)
            
            # If we've reached the target
            if node_id == target_node.id:
                # Build path details
                path_details = []
                for i in range(len(path) - 1):
                    curr_id, next_id = path[i], path[i+1]
                    # Find the connection between these nodes
                    connection = next((c for c in self.connections.values() 
                                     if (c.source_id == curr_id and c.target_id == next_id) or
                                        (c.source_id == next_id and c.target_id == curr_id)),
                                     None)
                    path_details.append({
                        "from_node": self.nodes[curr_id].to_dict(),
                        "to_node": self.nodes[next_id].to_dict(),
                        "connection": connection.to_dict() if connection else None
                    })
                
                return {
                    "path_found": True,
                    "source_concept": source_concept,
                    "target_concept": target_concept,
                    "path_ids": path,
                    "path_details": path_details,
                    "path_length": len(path) - 1,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Explore connections
            current_node = self.nodes[node_id]
            for connected_id in current_node.connections:
                if connected_id not in visited:
                    visited.add(connected_id)
                    queue.append((connected_id, path + [connected_id]))
        
        # No path found
        return {
            "path_found": False,
            "error": "No path found between concepts",
            "source_concept": source_concept,
            "target_concept": target_concept,
            "timestamp": datetime.now().isoformat()
        }

# Get a configured mapper instance
def get_spatial_mapper(language_memory=None):
    """Get a configured Spatial Temple Mapper instance"""
    return SpatialTempleMapper(language_memory) 