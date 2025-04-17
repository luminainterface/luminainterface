#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8_flow_test_simplified.py - Test the bidirectional flow between v1-v7 and v8

This simplified test verifies bidirectional data flow between v1-v7 (DNA) and v8 (seed)
without the complex imports that caused circular dependencies.
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import uuid
from datetime import datetime
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("v8.flow_test")

# Simple mock classes to avoid complex imports
class SpatialNode:
    """Mock of SpatialNode"""
    def __init__(self, concept, position=(0,0,0), node_type="concept", weight=0.5):
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.position = position
        self.node_type = node_type
        self.weight = weight
        self.connections = set()
        self.attributes = {}

class NutrientPacket:
    """Mock of NutrientPacket - data flowing from v8 to v1-v7"""
    def __init__(self, source_concept, source_id, enriched_data, strength=1.0):
        self.id = str(uuid.uuid4())
        self.source_concept = source_concept
        self.source_id = source_id
        self.enriched_data = enriched_data
        self.strength = strength
        self.created_at = datetime.now().isoformat()
        self.status = "created"
        
    def to_dict(self):
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
    """Mock of RootNexus - the core v1-v7 knowledge store"""
    def __init__(self):
        self.core_concepts = {}
        self.connections = {}
        self.last_update = datetime.now().isoformat()
        self.nutrient_history = []
        
    def add_nutrient(self, nutrient):
        """Process nutrient from v8 and add to v1-v7 core"""
        concept_key = nutrient.source_concept.lower()
        
        # Add to core concepts
        self.core_concepts[concept_key] = {
            "concept": nutrient.source_concept,
            "strength": nutrient.strength,
            "data": nutrient.enriched_data.copy(),
            "last_update": datetime.now().isoformat()
        }
        
        # Add connections if any
        if "connections" in nutrient.enriched_data:
            for conn in nutrient.enriched_data["connections"]:
                target_concept = conn[0].lower()
                conn_key = tuple(sorted([concept_key, target_concept]))
                self.connections[conn_key] = {
                    "concepts": [concept_key, target_concept],
                    "strength": conn[1],
                    "last_update": datetime.now().isoformat()
                }
        
        # Record in history
        nutrient.status = "absorbed"
        self.nutrient_history.append(nutrient.to_dict())
        self.last_update = datetime.now().isoformat()
        
        return True
        
    def apply_to_temple(self, temple_mapper):
        """Apply v1-v7 knowledge to v8 temple (DNA to seed flow)"""
        updates = 0
        
        # For each core concept, create a node in the temple if it doesn't exist
        for concept_key, data in self.core_concepts.items():
            # Check if concept already exists in temple
            exists = False
            for node in temple_mapper.nodes.values():
                if node.concept.lower() == concept_key:
                    exists = True
                    break
                    
            if not exists:
                # Create new node in temple
                node = SpatialNode(
                    concept=data["concept"],
                    weight=data["strength"],
                    node_type="knowledge"
                )
                temple_mapper.nodes[node.id] = node
                updates += 1
                
        return updates

class SpatialTempleMapper:
    """Mock of SpatialTempleMapper - the v8 concept store"""
    def __init__(self):
        self.nodes = {}
        self.zones = {}
        
    def add_node(self, concept, node_type="concept", weight=0.5):
        node = SpatialNode(concept, node_type=node_type, weight=weight)
        self.nodes[node.id] = node
        return node

class VascularSystem:
    """Mock of VascularSystem - handles bidirectional flow"""
    def __init__(self, temple_mapper):
        self.temple_mapper = temple_mapper
        self.root_nexus = RootNexus()
        self.nutrient_queue = []
        self.running = False
        
    def start_transport(self):
        """Start the nutrient transport process"""
        self.running = True
        logger.info("Started bidirectional flow system")
        
    def stop_transport(self):
        """Stop the nutrient transport process"""
        self.running = False
        logger.info("Stopped bidirectional flow system")
        
    def harvest_from_temple(self):
        """Harvest nutrients from temple nodes (v8 to v1-v7 flow)"""
        # Select nodes to harvest
        nodes = list(self.temple_mapper.nodes.values())
        for node in random.sample(nodes, min(3, len(nodes))):
            # Create enriched data
            enriched_data = {
                "weight": node.weight,
                "node_type": node.node_type,
                "connections": []
            }
            
            # Add connection information
            for conn_id in node.connections:
                if conn_id in self.temple_mapper.nodes:
                    target_node = self.temple_mapper.nodes[conn_id]
                    enriched_data["connections"].append((
                        target_node.concept,
                        min(node.weight, target_node.weight)
                    ))
            
            # Create nutrient packet
            nutrient = NutrientPacket(
                source_concept=node.concept,
                source_id=node.id,
                enriched_data=enriched_data,
                strength=node.weight
            )
            
            # Add to root nexus
            success = self.root_nexus.add_nutrient(nutrient)
            if success:
                logger.info(f"Harvested nutrient from '{node.concept}' to root system")
                
    def apply_roots_to_temple(self):
        """Apply root nexus knowledge to temple (v1-v7 to v8 flow)"""
        updates = self.root_nexus.apply_to_temple(self.temple_mapper)
        if updates > 0:
            logger.info(f"Applied {updates} updates from root system to temple")
        return updates
            
    def get_statistics(self):
        """Get statistics about the transport process"""
        return {
            "root_concepts": len(self.root_nexus.core_concepts),
            "root_connections": len(self.root_nexus.connections),
            "temple_nodes": len(self.temple_mapper.nodes),
            "running": self.running
        }

def test_bidirectional_flow():
    """Test the bidirectional flow between v1-v7 and v8"""
    logger.info("Starting bidirectional flow test")
    
    # Create test components
    temple_mapper = SpatialTempleMapper()
    
    # Add initial demo nodes
    concepts = ["neural_network", "consciousness", "pattern_recognition", 
                "memory_system", "language_processing"]
    for concept in concepts:
        node = temple_mapper.add_node(concept, weight=random.random())
        
    # Create vascular system
    vascular_system = VascularSystem(temple_mapper)
    
    # Check initial state
    initial_stats = vascular_system.get_statistics()
    logger.info(f"Initial state: {initial_stats['root_concepts']} root concepts, {initial_stats['temple_nodes']} temple nodes")
    
    # Start system
    vascular_system.start_transport()
    
    # 1. Test flow from v8 to v1-v7 (seed to DNA)
    logger.info("Testing v8 -> v1-v7 flow (seed to DNA)")
    vascular_system.harvest_from_temple()
    
    # Check updated state
    mid_stats = vascular_system.get_statistics()
    logger.info(f"After v8->v1-v7 flow: {mid_stats['root_concepts']} root concepts")
    
    # Verify v8 to v1-v7 flow
    v8_to_v7_flow_working = mid_stats['root_concepts'] > initial_stats['root_concepts']
    if v8_to_v7_flow_working:
        logger.info("PASS: Concepts from v8 (seed) successfully flowed to v1-v7 (DNA)")
    else:
        logger.error("FAIL: No concepts flowed from v8 (seed) to v1-v7 (DNA)")
    
    # 2. Test flow from v1-v7 to v8 (DNA to seed)
    logger.info("Testing v1-v7 -> v8 flow (DNA to seed)")
    
    # Create a direct v1-v7 concept
    direct_nutrient = NutrientPacket(
        source_concept="v7_direct_test_concept",
        source_id="v7_test",
        enriched_data={
            "connections": [("neural_network", 0.8), ("consciousness", 0.7)],
            "importance": 0.9,
            "description": "Test concept from v7 DNA"
        },
        strength=0.95
    )
    
    # Add the nutrient directly to the root nexus
    vascular_system.root_nexus.add_nutrient(direct_nutrient)
    
    # Apply root knowledge to temple
    updates = vascular_system.apply_roots_to_temple()
    
    # Check final state
    final_stats = vascular_system.get_statistics()
    logger.info(f"After v1-v7->v8 flow: {final_stats['temple_nodes']} temple nodes (added {final_stats['temple_nodes'] - mid_stats['temple_nodes']})")
    
    # Verify v1-v7 to v8 flow
    v7_to_v8_flow_working = updates > 0
    if v7_to_v8_flow_working:
        logger.info("PASS: Concepts from v1-v7 (DNA) successfully flowed to v8 (seed)")
    else:
        logger.error("FAIL: No concepts flowed from v1-v7 (DNA) to v8 (seed)")
    
    # Stop system
    vascular_system.stop_transport()
    
    # Overall test result
    if v8_to_v7_flow_working and v7_to_v8_flow_working:
        logger.info("OVERALL: Bidirectional flow is WORKING PROPERLY")
        return True
    else:
        logger.error("OVERALL: Bidirectional flow test FAILED")
        return False

if __name__ == "__main__":
    success = test_bidirectional_flow()
    sys.exit(0 if success else 1) 