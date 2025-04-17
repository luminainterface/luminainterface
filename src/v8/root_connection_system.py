#!/usr/bin/env python3
"""
Root Connection System (v8)

This module implements the connection from fruits back to the root system,
ensuring a complete cycle of knowledge flow. Like a tree's vascular system
that transports nutrients from leaves back to roots, this system ensures 
concepts and insights discovered at the peripheral "fruits" flow back to 
strengthen the core root system.
"""

import os
import sys
import logging
import json
import time
import threading
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import uuid

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode, TempleZone
from src.v8.temple_to_seed_bridge import ConceptSeed
from src.v8.auto_seed_growth import AutoSeedGrowthSystem, KnowledgeSource

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.root_connection")

class NutrientPacket:
    """
    Represents synthesized information flowing back from fruits to roots.
    Like how a plant converts sunlight to sugars that feed the roots,
    this represents processed knowledge returning to the core system.
    """
    
    def __init__(self, 
                 source_concept: str,
                 source_id: str,
                 enriched_data: Dict[str, Any],
                 strength: float = 1.0):
        self.id = str(uuid.uuid4())
        self.source_concept = source_concept
        self.source_id = source_id  # ID of the originating concept/node
        self.enriched_data = enriched_data  # Knowledge extracted and processed from source
        self.strength = strength  # Importance of this nutrient
        self.created_at = datetime.now().isoformat()
        self.status = "created"  # created -> in_transit -> delivered -> absorbed
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "source_concept": self.source_concept,
            "source_id": self.source_id,
            "enriched_data": self.enriched_data,
            "strength": self.strength,
            "created_at": self.created_at,
            "status": self.status
        }

class RootNexus:
    """Represents the central root system that receives nutrients from fruits"""
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.core_concepts = {}  # Dictionary of core concepts and their strength
        self.connections = {}  # Mapping of concept interconnections
        self.last_update = datetime.now().isoformat()
        self.nutrient_history = []  # History of received nutrients
        self.core_zones = []  # Core conceptual zones
        
    def add_nutrient(self, nutrient: NutrientPacket) -> bool:
        """Process a nutrient packet and incorporate into the root system"""
        try:
            # Check if concept already exists in core
            concept_key = nutrient.source_concept.lower()
            
            if concept_key in self.core_concepts:
                # Update existing concept
                current_strength = self.core_concepts[concept_key]["strength"]
                # Blend the strengths with bias toward new information
                new_strength = current_strength * 0.7 + nutrient.strength * 0.3
                self.core_concepts[concept_key]["strength"] = new_strength
                self.core_concepts[concept_key]["updates"] += 1
                self.core_concepts[concept_key]["last_update"] = datetime.now().isoformat()
                
                # Merge enriched data
                for key, value in nutrient.enriched_data.items():
                    if key in self.core_concepts[concept_key]["data"]:
                        # If numeric, take average
                        if isinstance(value, (int, float)) and isinstance(self.core_concepts[concept_key]["data"][key], (int, float)):
                            self.core_concepts[concept_key]["data"][key] = (self.core_concepts[concept_key]["data"][key] + value) / 2
                        else:
                            # For non-numeric, prefer new value
                            self.core_concepts[concept_key]["data"][key] = value
                    else:
                        # Add new attribute
                        self.core_concepts[concept_key]["data"][key] = value
            else:
                # Add new concept to core
                self.core_concepts[concept_key] = {
                    "concept": nutrient.source_concept,
                    "strength": nutrient.strength,
                    "created_at": datetime.now().isoformat(),
                    "last_update": datetime.now().isoformat(),
                    "updates": 1,
                    "data": nutrient.enriched_data.copy()
                }
            
            # Update connections based on nutrient data
            if "connections" in nutrient.enriched_data:
                for connection in nutrient.enriched_data["connections"]:
                    # Each connection is a tuple of (concept, strength)
                    target_concept = connection[0].lower()
                    conn_strength = connection[1]
                    
                    # Create connection key (alphabetical order to avoid duplicates)
                    conn_key = tuple(sorted([concept_key, target_concept]))
                    
                    if conn_key in self.connections:
                        # Update existing connection strength
                        current_strength = self.connections[conn_key]["strength"]
                        new_strength = max(current_strength, conn_strength)
                        self.connections[conn_key]["strength"] = new_strength
                        self.connections[conn_key]["updates"] += 1
                        self.connections[conn_key]["last_update"] = datetime.now().isoformat()
                    else:
                        # Add new connection
                        self.connections[conn_key] = {
                            "concepts": [concept_key, target_concept],
                            "strength": conn_strength,
                            "created_at": datetime.now().isoformat(),
                            "last_update": datetime.now().isoformat(),
                            "updates": 1
                        }
            
            # Record in history
            nutrient.status = "absorbed"
            self.nutrient_history.append(nutrient.to_dict())
            self.last_update = datetime.now().isoformat()
            
            # Reorganize zones periodically
            if len(self.nutrient_history) % 10 == 0:
                self._reorganize_zones()
                
            return True
        
        except Exception as e:
            logger.error(f"Error processing nutrient: {e}")
            return False
    
    def _reorganize_zones(self):
        """Reorganize conceptual zones based on current data"""
        # Group concepts into zones based on relationships
        # This is a simplified implementation - more advanced clustering would be used in practice
        
        # Reset zones
        self.core_zones = []
        
        # Get the top concepts by strength
        top_concepts = sorted(self.core_concepts.items(), key=lambda x: x[1]["strength"], reverse=True)
        
        # Create zones around the top concepts
        for i, (concept_key, data) in enumerate(top_concepts[:5]):  # Create up to 5 zones
            # Find all directly connected concepts
            zone_concepts = [concept_key]
            
            for conn_key, conn_data in self.connections.items():
                if concept_key in conn_data["concepts"] and conn_data["strength"] > 0.5:
                    # Add the other concept in the connection
                    other_concept = conn_data["concepts"][0] if conn_data["concepts"][1] == concept_key else conn_data["concepts"][1]
                    if other_concept not in zone_concepts:
                        zone_concepts.append(other_concept)
            
            # Create zone
            zone = {
                "id": str(uuid.uuid4()),
                "name": f"{data['concept']} Zone",
                "central_concept": concept_key,
                "concepts": zone_concepts,
                "strength": data["strength"],
                "created_at": datetime.now().isoformat()
            }
            
            self.core_zones.append(zone)
    
    def get_top_concepts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the top concepts by strength"""
        concepts = sorted(self.core_concepts.values(), key=lambda x: x["strength"], reverse=True)
        return concepts[:limit]
    
    def apply_to_temple(self, temple_mapper: SpatialTempleMapper) -> int:
        """
        Apply root system knowledge to strengthen the temple mapper
        Returns the number of updates made
        """
        if not temple_mapper:
            return 0
            
        updates = 0
        
        # Enhance existing temple nodes with root knowledge
        for node_id, node in temple_mapper.nodes.items():
            concept_key = node.concept.lower()
            
            if concept_key in self.core_concepts:
                # Update node with root knowledge
                root_data = self.core_concepts[concept_key]
                
                # Update weight based on root strength
                old_weight = node.weight
                new_weight = (node.weight + root_data["strength"]) / 2
                node.weight = new_weight
                
                # Update attributes from root data
                for key, value in root_data["data"].items():
                    if key not in ["connections", "concept", "strength"]:
                        node.attributes[f"root_{key}"] = value
                
                # Add root strengthening marker
                node.attributes["root_strengthened"] = True
                node.attributes["root_strengthened_at"] = datetime.now().isoformat()
                
                updates += 1
        
        # Create missing concepts that are important in the root system
        top_concepts = self.get_top_concepts(20)
        existing_concepts = [node.concept.lower() for node in temple_mapper.nodes.values()]
        
        for concept_data in top_concepts:
            if concept_data["concept"].lower() not in existing_concepts:
                # Create new node for this important root concept
                node = SpatialNode(
                    concept=concept_data["concept"],
                    weight=concept_data["strength"],
                    node_type="root_concept"  # Special type for root-sourced concepts
                )
                
                # Set position in central area
                node.position = (
                    random.uniform(-30, 30),
                    random.uniform(-30, 30),
                    random.uniform(-30, 30)
                )
                
                # Add attributes
                for key, value in concept_data["data"].items():
                    if key not in ["connections", "concept", "strength"]:
                        node.attributes[key] = value
                
                node.attributes["root_origin"] = True
                node.attributes["root_strength"] = concept_data["strength"]
                
                # Add to temple
                temple_mapper.nodes[node.id] = node
                updates += 1
                
                # Create connections based on root knowledge
                self._create_node_connections(node, temple_mapper)
        
        # Update zones based on root core zones
        for zone_data in self.core_zones:
            # Check if similar zone exists
            similar_zone = None
            for zone_id, zone in temple_mapper.zones.items():
                if zone.name.lower() == zone_data["name"].lower():
                    similar_zone = zone
                    break
            
            if similar_zone:
                # Update existing zone
                similar_zone.attributes["root_strengthened"] = True
                similar_zone.attributes["root_strength"] = zone_data["strength"]
            else:
                # Create new zone
                center = (
                    random.uniform(-50, 50),
                    random.uniform(-50, 50),
                    random.uniform(-50, 50)
                )
                
                zone = TempleZone(
                    name=zone_data["name"],
                    zone_type="knowledge",
                    center=center,
                    radius=30.0 * zone_data["strength"]  # Size based on importance
                )
                
                zone.attributes["root_origin"] = True
                zone.attributes["root_strength"] = zone_data["strength"]
                
                # Add to temple
                temple_mapper.zones[zone.id] = zone
                updates += 1
        
        return updates
    
    def _create_node_connections(self, node: SpatialNode, temple_mapper: SpatialTempleMapper):
        """Create connections for a node based on root knowledge"""
        concept_key = node.concept.lower()
        
        # Find connections in root system
        for conn_key, conn_data in self.connections.items():
            if concept_key in conn_data["concepts"] and conn_data["strength"] > 0.3:
                # Get the other concept in the connection
                other_concept_key = conn_data["concepts"][0] if conn_data["concepts"][1] == concept_key else conn_data["concepts"][1]
                
                # Find corresponding node in temple
                for other_id, other_node in temple_mapper.nodes.items():
                    if other_node.concept.lower() == other_concept_key and other_id != node.id:
                        # Create connection
                        node.connections.add(other_id)
                        other_node.connections.add(node.id)
                        break

class VascularSystem:
    """
    System that transports nutrients (information) from fruits back to roots.
    This represents the phloem in a tree that carries sugars from leaves to roots.
    """
    
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None, auto_growth: Optional[AutoSeedGrowthSystem] = None):
        self.temple_mapper = temple_mapper
        self.auto_growth = auto_growth
        self.root_nexus = RootNexus()
        self.nutrient_queue = []  # Queue of nutrients in transit
        self.running = False
        self.transport_thread = None
        self.flow_rate = 5  # Seconds between transport cycles
    
    def start_transport(self):
        """Start the nutrient transport process"""
        if self.running:
            logger.info("Vascular system already running")
            return
            
        self.running = True
        self.transport_thread = threading.Thread(target=self._transport_loop)
        self.transport_thread.daemon = True
        self.transport_thread.start()
        logger.info("Started nutrient transport from fruits to roots")
    
    def stop_transport(self):
        """Stop the nutrient transport process"""
        self.running = False
        if self.transport_thread:
            self.transport_thread.join(timeout=2.0)
        logger.info("Stopped nutrient transport from fruits to roots")
    
    def _transport_loop(self):
        """Main loop for the transport process"""
        while self.running:
            try:
                # 1. Harvest nutrients from fruits/nodes
                self._harvest_nutrients()
                
                # 2. Process transport queue
                self._process_nutrient_queue()
                
                # 3. Apply root knowledge back to temple
                if self.temple_mapper:
                    updates = self.root_nexus.apply_to_temple(self.temple_mapper)
                    if updates > 0:
                        logger.info(f"Applied {updates} updates from root system to temple")
                
                # Wait before next cycle
                time.sleep(self.flow_rate)
                
            except Exception as e:
                logger.error(f"Error in transport loop: {e}")
                time.sleep(self.flow_rate * 2)  # Wait longer after error
    
    def _harvest_nutrients(self):
        """Harvest nutrients from temple nodes and growth system"""
        self._harvest_from_temple()
        self._harvest_from_growth_system()
    
    def _harvest_from_temple(self):
        """Harvest nutrients from temple nodes"""
        if not self.temple_mapper:
            return
            
        # Select nodes to harvest nutrients from
        nodes = list(self.temple_mapper.nodes.values())
        
        # Prefer nodes with higher weight and those not recently harvested
        harvestable_nodes = []
        for node in nodes:
            last_harvest = node.attributes.get("last_nutrient_harvest")
            
            if not last_harvest or (datetime.now() - datetime.fromisoformat(last_harvest)).total_seconds() > 60:
                harvestable_nodes.append(node)
        
        # Sort by weight and take a subset
        harvestable_nodes.sort(key=lambda n: n.weight, reverse=True)
        harvest_count = min(5, len(harvestable_nodes))
        nodes_to_harvest = harvestable_nodes[:harvest_count]
        
        # Generate nutrient packets
        for node in nodes_to_harvest:
            # Create enriched data based on node
            enriched_data = {
                "weight": node.weight,
                "node_type": node.node_type,
                "connections": []
            }
            
            # Add attributes
            for key, value in node.attributes.items():
                if not key.startswith("_") and key != "last_nutrient_harvest":
                    enriched_data[key] = value
            
            # Add connection information
            for conn_id in node.connections:
                if conn_id in self.temple_mapper.nodes:
                    target_node = self.temple_mapper.nodes[conn_id]
                    enriched_data["connections"].append((
                        target_node.concept,
                        min(node.weight, target_node.weight)  # Connection strength based on node weights
                    ))
            
            # Create nutrient packet
            nutrient = NutrientPacket(
                source_concept=node.concept,
                source_id=node.id,
                enriched_data=enriched_data,
                strength=node.weight
            )
            
            # Add to transport queue
            self.nutrient_queue.append(nutrient)
            
            # Mark node as harvested
            node.attributes["last_nutrient_harvest"] = datetime.now().isoformat()
            
            logger.info(f"Harvested nutrient from '{node.concept}' node")
    
    def _harvest_from_growth_system(self):
        """Harvest nutrients from auto growth system"""
        if not self.auto_growth or not hasattr(self.auto_growth, 'growth_engine'):
            return
            
        growth_engine = self.auto_growth.growth_engine
        
        # Check if growth system has knowledge sources and history
        if not hasattr(growth_engine, 'knowledge_sources') or not hasattr(growth_engine, 'growth_history'):
            return
            
        # Get recent growth events that have not been harvested
        recent_events = []
        for event in growth_engine.growth_history:
            if event.get("harvested_for_roots", False) == False:
                recent_events.append(event)
                event["harvested_for_roots"] = True
        
        # For each event, create a nutrient packet from the source
        for event in recent_events[:5]:  # Limit to avoid overwhelming
            source_name = event.get("source_name", "Unknown Source")
            seed_concept = event.get("seed_concept", "Unknown Concept")
            
            # Find matching source in knowledge sources
            source = None
            for source_id, src in growth_engine.knowledge_sources.items():
                if src.name == source_name:
                    source = src
                    break
            
            if not source:
                continue
                
            # Create enriched data from source
            enriched_data = {
                "source_type": source.source_type,
                "content_summary": source.content_summary,
                "extracted_concepts": source.extracted_concepts,
                "connections": []
            }
            
            # Add source metadata
            for key, value in source.metadata.items():
                enriched_data[key] = value
            
            # Create connections between extracted concepts
            for i, concept1 in enumerate(source.extracted_concepts):
                for j in range(i+1, len(source.extracted_concepts)):
                    concept2 = source.extracted_concepts[j]
                    enriched_data["connections"].append((
                        concept1,
                        0.7  # Default connection strength for concepts from same source
                    ))
            
            # Create nutrient packet
            nutrient = NutrientPacket(
                source_concept=seed_concept,
                source_id="growth_" + source.id,
                enriched_data=enriched_data,
                strength=0.8  # High strength for externally verified concepts
            )
            
            # Add to transport queue
            self.nutrient_queue.append(nutrient)
            
            logger.info(f"Harvested nutrient from growth system source: {source_name}")
    
    def _process_nutrient_queue(self):
        """Process nutrients in the queue and deliver to roots"""
        # Check if queue is empty
        if not self.nutrient_queue:
            return
            
        # Process a batch of nutrients
        batch_size = min(10, len(self.nutrient_queue))
        batch = self.nutrient_queue[:batch_size]
        self.nutrient_queue = self.nutrient_queue[batch_size:]
        
        # Deliver to root nexus
        for nutrient in batch:
            # Mark as in transit
            nutrient.status = "in_transit"
            
            # Simulate transport time
            time.sleep(0.1)
            
            # Deliver to roots
            nutrient.status = "delivered"
            success = self.root_nexus.add_nutrient(nutrient)
            
            if success:
                logger.info(f"Successfully delivered nutrient from '{nutrient.source_concept}' to root system")
            else:
                logger.warning(f"Failed to deliver nutrient from '{nutrient.source_concept}' to root system")
    
    def get_transport_statistics(self) -> Dict[str, Any]:
        """Get statistics about the transport process"""
        stats = {
            "nutrients_in_queue": len(self.nutrient_queue),
            "root_concepts": len(self.root_nexus.core_concepts),
            "root_connections": len(self.root_nexus.connections),
            "root_zones": len(self.root_nexus.core_zones),
            "nutrients_processed": len(self.root_nexus.nutrient_history),
            "running": self.running,
            "last_root_update": self.root_nexus.last_update
        }
        return stats

class RootConnectionSystem:
    """
    Master system that manages the bidirectional flow of knowledge in the tree,
    connecting fruits back to roots to complete the nutrient cycle.
    """
    
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None, auto_growth: Optional[AutoSeedGrowthSystem] = None):
        """Initialize with optional temple mapper and auto growth system"""
        self.temple_mapper = temple_mapper
        self.auto_growth = auto_growth
        self.vascular_system = VascularSystem(temple_mapper, auto_growth)
        
    def start(self):
        """Start the bidirectional knowledge flow"""
        self.vascular_system.start_transport()
        
    def stop(self):
        """Stop the bidirectional knowledge flow"""
        self.vascular_system.stop_transport()
        
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper"""
        self.temple_mapper = temple_mapper
        self.vascular_system.temple_mapper = temple_mapper
        
    def set_auto_growth(self, auto_growth: AutoSeedGrowthSystem):
        """Set the auto growth system"""
        self.auto_growth = auto_growth
        self.vascular_system.auto_growth = auto_growth
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the system"""
        stats = self.vascular_system.get_transport_statistics()
        
        # Add top concepts from root
        top_concepts = self.vascular_system.root_nexus.get_top_concepts(5)
        stats["top_root_concepts"] = [
            {"concept": c["concept"], "strength": c["strength"]}
            for c in top_concepts
        ]
        
        return stats

def create_demo_temple_mapper() -> SpatialTempleMapper:
    """Create a demo temple mapper for testing"""
    from src.v8.demo_data_generator import generate_demo_nodes
    
    mapper = SpatialTempleMapper()
    
    # Generate demo nodes
    demo_nodes = generate_demo_nodes(20)
    
    # Add to mapper
    for node in demo_nodes:
        mapper.nodes[node.id] = node
        
    return mapper

if __name__ == "__main__":
    # Create a demo temple mapper
    temple_mapper = create_demo_temple_mapper()
    
    # Create a demo auto growth system
    from src.v8.auto_seed_growth import AutoSeedGrowthSystem
    auto_growth = AutoSeedGrowthSystem(temple_mapper)
    
    # Initialize root connection system
    root_system = RootConnectionSystem(temple_mapper, auto_growth)
    
    # Start the auto growth system
    auto_growth.start()
    
    # Start the root connection system
    root_system.start()
    
    try:
        # Run for a while to simulate the complete cycle
        print("Complete knowledge cycle system running...")
        print("This system connects fruits back to roots for complete circulation")
        print("Press Ctrl+C to stop")
        
        # Check stats periodically
        for _ in range(12):
            time.sleep(5)
            stats = root_system.get_statistics()
            print(f"\nTransport stats: {stats['nutrients_in_queue']} in queue, {stats['nutrients_processed']} processed")
            print(f"Root system: {stats['root_concepts']} core concepts, {stats['root_connections']} connections")
            
            if stats['top_root_concepts']:
                print("Top root concepts:")
                for concept in stats['top_root_concepts']:
                    print(f"  - {concept['concept']} ({concept['strength']:.2f})")
            
    except KeyboardInterrupt:
        print("\nStopping systems...")
    finally:
        # Stop systems
        root_system.stop()
        auto_growth.stop()
        print("Knowledge cycle systems stopped") 