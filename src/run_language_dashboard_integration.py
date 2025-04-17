#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUMINA V7 Language Dashboard Integration
========================================

This script provides a single entry point to launch the integrated
language system with the dashboard panels.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/language_dashboard_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LanguageDashboardIntegration")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LUMINA V7 Language Dashboard Integration"
    )
    
    parser.add_argument(
        "--db-path",
        default="data/neural_metrics.db",
        help="Path to the metrics database"
    )
    
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the data directory"
    )
    
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.5,
        help="LLM weight (0.0-1.0)"
    )
    
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.5,
        help="Neural network weight (0.0-1.0)"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data mode"
    )
    
    parser.add_argument(
        "--gui-framework",
        choices=["PyQt5", "PySide6"],
        default="PySide6",
        help="GUI framework to use"
    )
    
    return parser.parse_args()

def initialize_directories(args):
    """Create necessary directories"""
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(args.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Directory structure initialized")

def initialize_language_system(args):
    """Initialize the language system"""
    try:
        # Import the Central Language Node
        try:
            from src.language.central_language_node import CentralLanguageNode
            
            # Create a new CentralLanguageNode instance
            central_node = CentralLanguageNode(
                data_dir=args.data_dir,
                llm_weight=args.llm_weight,
                nn_weight=args.nn_weight
            )
            
            logger.info("Central Language Node initialized")
            
            # Get references to components
            language_memory = central_node.language_memory
            neural_processor = central_node.neural_processor
            conscious_mirror = central_node.conscious_mirror
            pattern_analyzer = central_node.pattern_analyzer
            
            logger.info("Language components initialized")
            
            # Return the central node reference
            return central_node
            
        except ImportError as e:
            logger.error(f"Error importing Central Language Node: {e}")
            logger.warning("Will run in mock mode instead")
            return None
            
    except Exception as e:
        logger.error(f"Error initializing language system: {e}")
        return None

def initialize_language_dashboard_bridge(args, central_node=None):
    """Initialize the Language Dashboard Bridge"""
    try:
        # Import the Language Dashboard Bridge
        try:
            from src.language.language_dashboard_bridge import get_language_dashboard_bridge
            
            # Configure the bridge
            config = {
                "db_path": args.db_path,
                "data_dir": args.data_dir,
                "mock_mode": args.mock or (central_node is None),
                "llm_weight": args.llm_weight,
                "nn_weight": args.nn_weight
            }
            
            # Get the bridge instance
            bridge = get_language_dashboard_bridge(config)
            
            # Connect to language components if provided
            if central_node:
                components = {
                    "central_language_node": central_node,
                    "language_memory": central_node.language_memory,
                    "neural_linguistic_processor": central_node.neural_processor,
                    "conscious_mirror": central_node.conscious_mirror,
                    "pattern_analyzer": central_node.pattern_analyzer
                }
                
                if hasattr(central_node, "neural_flex_bridge"):
                    components["neural_flex_bridge"] = central_node.neural_flex_bridge
                
                bridge.connect_language_components(components)
            else:
                # Try to connect automatically
                bridge.connect_language_components()
            
            # Start the bridge
            bridge.start()
            
            logger.info("Language Dashboard Bridge initialized and started")
            
            # Return the bridge reference
            return bridge
            
        except ImportError as e:
            logger.error(f"Error importing Language Dashboard Bridge: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error initializing Language Dashboard Bridge: {e}")
        return None

def run_dashboard(args, language_bridge=None):
    """Run the dashboard with language integration"""
    try:
        # Save bridge reference to global variable if needed by Qt modules
        if language_bridge:
            global _language_bridge
            _language_bridge = language_bridge
        
        # Import necessary Qt modules
        if args.gui_framework == "PySide6":
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QTimer
            logger.info("Using PySide6 framework")
        else:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer
            logger.info("Using PyQt5 framework")
            
        # Import dashboard module
        from src.visualization.run_qt_dashboard import LuminaQtDashboard
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create dashboard
        dashboard = LuminaQtDashboard(
            db_path=args.db_path,
            mock_mode=args.mock,
            nn_weight=args.nn_weight,
            llm_weight=args.llm_weight,
            gui_framework=args.gui_framework
        )
        
        # Show dashboard
        dashboard.show()
        
        # Run the application
        return app.exec() if hasattr(app, 'exec') else app.exec_()
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return 1

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize directories
    initialize_directories(args)
    
    # Initialize language system
    central_node = None
    if not args.mock:
        central_node = initialize_language_system(args)
    
    # Initialize Language Dashboard Bridge
    language_bridge = initialize_language_dashboard_bridge(args, central_node)
    
    # Run the dashboard
    return run_dashboard(args, language_bridge)

if __name__ == "__main__":
    sys.exit(main()) 