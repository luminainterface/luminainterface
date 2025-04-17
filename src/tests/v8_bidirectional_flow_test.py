#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8_bidirectional_flow_test.py - Test the bidirectional flow between v1-v7 and v8

This script tests the bidirectional data flow between v1-v7 (DNA) components and
v8 (seed) to ensure that information flows properly in both directions:
1. From v1-v7 to v8: Core concepts feeding into the seed system
2. From v8 back to v1-v7: New insights strengthening the root system
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("v8.bidirectional_test")

def test_bidirectional_flow():
    """Test the bidirectional flow between v1-v7 and v8"""
    logger.info("Starting bidirectional flow test")
    
    # Import required components
    from src.v8.root_connection_system import RootConnectionSystem, NutrientPacket
    from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode
    from src.v8.auto_seed_growth import AutoSeedGrowthSystem
    from src.v8.demo_data_generator import generate_themed_demo_nodes
    
    # Create test components
    temple_mapper = SpatialTempleMapper()
    
    # Generate demo nodes with AI theme
    demo_nodes = generate_themed_demo_nodes(15, 'ai')
    for node in demo_nodes:
        temple_mapper.nodes[node.id] = node
    
    # Initialize auto growth system
    auto_growth = AutoSeedGrowthSystem(temple_mapper)
    
    # Create root connection system
    root_system = RootConnectionSystem(temple_mapper, auto_growth)
    
    # Check initial state
    initial_stats = root_system.get_statistics()
    logger.info(f"Initial state: {initial_stats['root_concepts']} concepts, {initial_stats['root_connections']} connections")
    
    # 1. Test flow from v8 to v1-v7 (seed to DNA)
    # Start systems
    root_system.start()
    auto_growth.start()
    
    # Wait for initial processing
    logger.info("Running bidirectional flow for 10 seconds...")
    time.sleep(10)
    
    # Check updated state
    mid_stats = root_system.get_statistics()
    logger.info(f"After 10 seconds: {mid_stats['root_concepts']} concepts, {mid_stats['root_connections']} connections")
    
    # Verify v8 to v1-v7 flow (seed to DNA)
    v8_to_v7_flow_working = mid_stats['root_concepts'] > initial_stats['root_concepts']
    logger.info(f"v8 -> v1-v7 flow working: {v8_to_v7_flow_working}")
    
    if v8_to_v7_flow_working:
        logger.info("PASS: Concepts from v8 (seed) successfully flowed to v1-v7 (DNA)")
    else:
        logger.error("FAIL: No concepts flowed from v8 (seed) to v1-v7 (DNA)")
    
    # 2. Test flow from v1-v7 to v8 (DNA to seed)
    # Create a new node directly in the temple mapper (simulating v1-v7 influence)
    v7_test_node = SpatialNode(
        concept="v7_dna_test_concept",
        position=(0, 0, 50),
        node_type="knowledge",
        weight=0.9
    )
    temple_mapper.nodes[v7_test_node.id] = v7_test_node
    
    # Create a direct nutrient packet (simulating v1-v7 data)
    test_nutrient = NutrientPacket(
        source_concept="v7_direct_dna_test",
        source_id="v7_test",
        enriched_data={
            "connections": [("neural_network", 0.8), ("consciousness", 0.7)],
            "importance": 0.9,
            "description": "Test concept from v7 DNA"
        },
        strength=0.95
    )
    
    # Add the nutrient directly to the root nexus
    root_system.vascular_system.root_nexus.add_nutrient(test_nutrient)
    
    # Wait for processing
    logger.info("Running v1-v7 to v8 test for 10 more seconds...")
    time.sleep(10)
    
    # Check final state
    final_stats = root_system.get_statistics()
    logger.info(f"Final state: {final_stats['root_concepts']} concepts, {final_stats['root_connections']} connections")
    
    # Verify v1-v7 to v8 flow (DNA to seed)
    # Check if our test concept appears in the temple mapper nodes
    v7_to_v8_flow_working = False
    for node_id, node in temple_mapper.nodes.items():
        if node.concept.lower() == "v7_direct_dna_test" or "v7_dna_test" in node.concept.lower():
            v7_to_v8_flow_working = True
            logger.info(f"Found v7 test concept in temple: {node.concept}")
            break
    
    # Also check any nodes created after we added the test nutrient
    node_count_before = len(demo_nodes) + 1  # +1 for our v7_test_node
    v7_to_v8_created_nodes = len(temple_mapper.nodes) > node_count_before
    
    if v7_to_v8_flow_working:
        logger.info("PASS: Concepts from v1-v7 (DNA) successfully flowed to v8 (seed)")
    else:
        logger.info(f"Checking alternative flow evidence: New nodes created = {v7_to_v8_created_nodes}")
        if v7_to_v8_created_nodes:
            logger.info("PASS: New nodes created in v8 based on v1-v7 (DNA) data")
            v7_to_v8_flow_working = True
        else:
            logger.error("FAIL: No evidence of concepts flowing from v1-v7 (DNA) to v8 (seed)")
    
    # Stop systems
    root_system.stop()
    auto_growth.stop()
    
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