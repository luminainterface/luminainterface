#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_spatial_temple.py - Launcher for Lumina v8 Spatial Temple

This script serves as the main entry point for the Spatial Temple system,
providing 3D conceptual navigation and knowledge mapping functionality.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import traceback
from PySide6.QtWidgets import QApplication

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from src.v8.spatial_temple_mapper import SpatialTempleMapper
from src.v8.spatial_temple_visualization import SpatialTempleVisualizationWindow
from src.v8.demo_data_generator import generate_demo_nodes, generate_themed_demo_nodes

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "v8_spatial_temple.log")),
        logging.StreamHandler()
    ]
)

def load_language_memory():
    """Load the language memory system if available"""
    # Skip using CentralLanguageNode due to compatibility issues
    try:
        from src.language.language_memory import LanguageMemory
        logger.info("Loading standalone Language Memory")
        return LanguageMemory()
    except ImportError:
        logger.warning("Language Memory not available, running with basic functionality")
    
    return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Lumina v8 Spatial Temple - 3D Conceptual Navigation")
    parser.add_argument("--text", "-t", type=str, help="Text to analyze for spatial temple")
    parser.add_argument("--demo", "-d", action="store_true", help="Use demo data instead of analyzing text")
    parser.add_argument("--theme", type=str, choices=['ai', 'temple', 'network', 'brain'], default='ai', 
                        help="Theme for demo data node arrangement (default: ai)")
    parser.add_argument("--nodes", type=int, default=50, help="Number of demo nodes to generate (default: 50)")
    parser.add_argument("--mode", "-m", type=str, choices=['2d', '3d'], default='3d', 
                        help="Visualization mode: 2D or 3D (default: 3D)")
    return parser.parse_args()

def main():
    """
    Main function to run the spatial temple
    
    This will parse arguments, set up the mapper, and launch the visualization
    """
    # Create QApplication at the beginning of main
    app = QApplication(sys.argv)
    
    args = parse_arguments()
    
    # Set up the temple mapper
    mapper = SpatialTempleMapper()
    
    # Handle input
    if not args.text and not args.demo:
        logger.info("No text specified, using demo data")
        args.demo = True
    
    if args.demo:
        logger.info(f"Loading demo nodes with '{args.theme}' theme for spatial temple")
        demo_nodes = generate_themed_demo_nodes(args.nodes, args.theme)
        
        for node in demo_nodes:
            mapper.nodes[node.id] = node
    else:
        logger.info(f"Analyzing text: {args.text[:50]}...")
        mapper.process_text(args.text)
    
    # Print mode information
    if args.mode.lower() == '3d':
        print(f"Starting Spatial Temple in 3D visualization mode.")
        print("Note: 3D mode requires OpenGL support. If you experience issues with Intel integrated GPUs:")
        print("- Update your graphics drivers to the latest version")
        print("- Consider using the 2D mode with --mode 2d if problems persist")
    else:
        print(f"Starting Spatial Temple in 2D visualization mode.")
    
    # Create and show visualization
    try:
        vis_window = SpatialTempleVisualizationWindow(mapper, mode=args.mode.lower())
        vis_window.show()
        return app.exec()
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 