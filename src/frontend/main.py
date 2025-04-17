import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from core.main_controller import MainController
from core.version_manager import VersionManager
from core.system_monitor import SystemMonitor
from ui.windows.main_window import MainWindow

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lumina_frontend.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main application entry point."""
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Lumina Frontend...")
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        # Initialize core components
        main_controller = MainController()
        version_manager = VersionManager()
        system_monitor = SystemMonitor()
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        # Start system monitoring
        system_monitor.start_monitoring()
        
        # Run application
        exit_code = app.exec()
        
        # Cleanup
        system_monitor.stop_monitoring()
        main_controller.shutdown()
        
        logger.info("Lumina Frontend shutdown complete")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Fatal error in main application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 