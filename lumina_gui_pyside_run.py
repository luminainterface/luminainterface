#!/usr/bin/env python
"""
Lumina GUI PySide6 Launcher - A launcher script for the PySide6 version of Lumina GUI

This launcher script provides a way to start the upgraded Lumina GUI system with PySide6.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

def main():
    """Main entry point for Lumina GUI PySide6"""
    parser = argparse.ArgumentParser(description="Lumina GUI PySide6 System")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Set debug mode by default for now to help diagnose issues
    debug_mode = True
    
    # Configure basic logging to console
    import logging
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=log_level, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                      handlers=[
                          logging.FileHandler("lumina_pyside.log"),
                          logging.StreamHandler()
                      ])
    logger = logging.getLogger("LuminaPySideLauncher")
    
    logger.info("Starting Lumina GUI PySide6...")
    
    # Try to import and run the Lumina GUI PySide6
    try:
        # Check if PySide6 is installed
        try:
            logger.info("Checking for PySide6...")
            from PySide6.QtWidgets import QApplication
            logger.info("PySide6 is installed.")
        except ImportError as e:
            logger.error(f"Error importing PySide6: {e}")
            print("Error: PySide6 is not installed.")
            print("Please install it with: pip install PySide6")
            sys.exit(1)
            
        logger.info("Importing LuminaGUIPySide...")
        try:
            from lumina_gui_pyside import LuminaGUIPySide
            logger.info("Successfully imported LuminaGUIPySide")
        except Exception as import_error:
            logger.error(f"Failed to import LuminaGUIPySide: {import_error}")
            if debug_mode:
                traceback.print_exc()
            raise
        
        logger.info("Creating QApplication...")
        app = QApplication(sys.argv)
        
        # Set application-wide stylesheet (PySide6 style)
        logger.info("Setting application stylesheet...")
        app.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        logger.info("Creating LuminaGUIPySide window...")
        try:
            window = LuminaGUIPySide()
            logger.info("LuminaGUIPySide window created successfully")
        except Exception as window_error:
            logger.error(f"Error creating window: {window_error}")
            if debug_mode:
                traceback.print_exc()
            raise
            
        logger.info("Showing window...")
        window.show()
        
        logger.info("Entering event loop...")
        sys.exit(app.exec())  # In PySide6, exec() is used instead of exec_()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"Error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install PySide6")
        if debug_mode:
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting Lumina GUI PySide6: {e}")
        print(f"Error starting Lumina GUI PySide6: {e}")
        if debug_mode or args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 