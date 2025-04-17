#!/usr/bin/env python3
"""
V5 PySide6 Client

A comprehensive PySide6-based client for the V5 Fractal Echo Visualization System,
integrating with the Language Memory System.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/v5_pyside6_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v5_pyside6_client")

# Set environment variable to force PySide6 usage
os.environ["V5_QT_FRAMEWORK"] = "PySide6"

# Add parent directory to path if needed
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Global application instance
qapp = None

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="V5 PySide6 Client")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for testing without backend services")
    parser.add_argument("--no-plugins", action="store_true", help="Disable plugin discovery")
    parser.add_argument("--theme", choices=["light", "dark", "system"], default="system", help="UI theme")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def ensure_application():
    """Ensure Qt application exists"""
    global qapp
    try:
        from PySide6.QtWidgets import QApplication
        if qapp is None:
            logger.info("Creating QApplication instance")
            qapp = QApplication.instance()
            if qapp is None:
                qapp = QApplication(sys.argv)
                qapp.setApplicationName("V5 Fractal Echo Visualization")
                qapp.setOrganizationName("Lumina Neural Network System")
                logger.info("QApplication created")
            else:
                logger.info("Using existing QApplication instance")
        return qapp
    except ImportError as e:
        logger.error(f"Failed to import PySide6: {e}")
        print(f"Error: PySide6 is required but not installed.")
        print("Please install PySide6 with: pip install PySide6>=6.2.0")
        return None

def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_args()
    
    # Set log level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Import PySide6
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication
        logger.info("Successfully imported PySide6")
    except ImportError as e:
        logger.error(f"Failed to import PySide6: {e}")
        print(f"Error: PySide6 is required but not installed.")
        print("Please install PySide6 with: pip install PySide6>=6.2.0")
        return 1
    
    # Create Qt Application - MUST be done before creating any widgets
    app = ensure_application()
    if app is None:
        return 1
    
    # Set theme if specified
    if args.theme != "system":
        try:
            from v5_client.ui.theme_manager import ThemeManager
            theme_manager = ThemeManager()
            theme_manager.apply_theme(args.theme)
            logger.info(f"Applied {args.theme} theme")
        except ImportError as e:
            logger.warning(f"Could not load theme manager: {e}")
    
    # Import and create main window
    try:
        from v5_client.ui.main_window import MainWindow
        logger.info("Creating main window")
        window = MainWindow(mock_mode=args.mock, enable_plugins=not args.no_plugins)
        window.show()
    except ImportError as e:
        logger.error(f"Failed to import MainWindow: {e}")
        print(f"Error: Could not load application components: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error creating main window: {e}")
        print(f"Error: {e}")
        return 1
    
    # Start a timer to initialize components after the event loop starts
    # (this prevents UI freezing during initialization)
    QTimer.singleShot(100, lambda: window.initialize_components())
    
    # Run the application
    logger.info("Starting application event loop")
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 