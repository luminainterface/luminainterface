#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_tests.py - Test runner for the Lumina v8 system

This script runs all tests for the Lumina v8 components to ensure
that the system is functioning correctly.
"""

import os
import sys
import logging
import unittest
from pathlib import Path

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("v8.tests")

class SpatialTempleBasicTest(unittest.TestCase):
    """Basic tests for the Spatial Temple system"""
    
    def test_import_spatial_temple_mapper(self):
        """Test that the SpatialTempleMapper can be imported"""
        from src.v8.spatial_temple_mapper import SpatialTempleMapper
        self.assertIsNotNone(SpatialTempleMapper)
        
    def test_create_spatial_temple_mapper(self):
        """Test that the SpatialTempleMapper can be created"""
        from src.v8.spatial_temple_mapper import SpatialTempleMapper
        mapper = SpatialTempleMapper()
        self.assertIsNotNone(mapper)
        
    def test_demo_data_generator(self):
        """Test that the demo data generator functions work"""
        from src.v8.demo_data_generator import generate_demo_nodes, generate_themed_demo_nodes
        
        # Test generate_demo_nodes
        nodes = generate_demo_nodes(10)
        self.assertEqual(len(nodes), 10)
        
        # Test generate_themed_demo_nodes
        themed_nodes = generate_themed_demo_nodes(10, 'ai')
        self.assertEqual(len(themed_nodes), 10)

class TempleVisualizationTest(unittest.TestCase):
    """Tests for the Spatial Temple visualization system"""
    
    def test_import_visualization(self):
        """Test that the visualization module can be imported"""
        from src.v8.spatial_temple_visualization import SpatialTempleVisualizationWindow
        self.assertIsNotNone(SpatialTempleVisualizationWindow)

def run_tests():
    """Run all tests for the Lumina v8 system"""
    logger.info("Starting Lumina v8 test suite")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases to suite
    test_suite.addTest(unittest.makeSuite(SpatialTempleBasicTest))
    test_suite.addTest(unittest.makeSuite(TempleVisualizationTest))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success/failure
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 