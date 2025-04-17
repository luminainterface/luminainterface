#!/usr/bin/env python3
"""
Memory System Launcher

This script launches all components of the memory system, including:
- GUI interface
- Background memory services
- Synthesis modules
- Data storage management

Usage:
  python src/launch_memory_system.py [options]

Options:
  --no-gui        Run without the GUI interface (console mode)
  --no-background Disable background memory services
  --debug         Enable debug logging
"""

import os
import sys
import logging
import argparse
import threading
import time
import tkinter as tk
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data/logs/memory_system.log'
)
# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("memory_system_launcher")


class MemorySystemLauncher:
    """
    Manages the launch and coordination of all memory system components
    """
    
    def __init__(self, options=None):
        """
        Initialize the memory system launcher
        
        Args:
            options: Dictionary of launch options
        """
        self.options = options or {}
        self.gui_enabled = not self.options.get('no_gui', False)
        self.background_enabled = not self.options.get('no_background', False)
        
        if self.options.get('debug', False):
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Initialize paths
        self.root_path = Path(__file__).resolve().parent.parent
        self.data_path = self.root_path / 'data'
        
        # Create required directories
        self._ensure_directories()
        
        # Track components
        self.components = {}
        self.threads = {}
        self.gui_app = None
        
        logger.info("Memory System Launcher initialized")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        # List of required directories
        required_dirs = [
            self.data_path / 'memory',
            self.data_path / 'synthesis',
            self.data_path / 'logs',
            self.data_path / 'backups'
        ]
        
        # Create each directory
        for directory in required_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
    
    def start(self):
        """Start the memory system"""
        logger.info("Starting Memory System...")
        
        # Start components in sequence
        self._start_components()
        
        # Start background services if enabled
        if self.background_enabled:
            self._start_background_services()
        
        # Start GUI or run in console mode
        if self.gui_enabled:
            self._start_gui()
        else:
            self._run_console_mode()
        
        logger.info("Memory System started successfully")
    
    def _start_components(self):
        """Start and initialize all required components"""
        try:
            # Import components here to avoid circular imports
            from src.conversation_memory import ConversationMemory
            from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
            
            # Initialize conversation memory
            logger.info("Initializing conversation memory...")
            self.components['conversation_memory'] = ConversationMemory()
            
            # Initialize language memory synthesis
            logger.info("Initializing language memory synthesis...")
            self.components['synthesis'] = LanguageMemorySynthesisIntegration()
            
            logger.info("All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            return False
    
    def _start_background_services(self):
        """Start background services for memory management"""
        logger.info("Starting background services...")
        
        # Start memory cleanup service
        cleanup_thread = threading.Thread(
            target=self._run_memory_cleanup_service,
            daemon=True
        )
        cleanup_thread.start()
        self.threads['cleanup'] = cleanup_thread
        
        # Start backup service
        backup_thread = threading.Thread(
            target=self._run_backup_service,
            daemon=True
        )
        backup_thread.start()
        self.threads['backup'] = backup_thread
        
        logger.info("Background services started")
    
    def _run_memory_cleanup_service(self):
        """Background service to periodically clean up outdated memories"""
        logger.info("Memory cleanup service started")
        
        while True:
            try:
                # Sleep for a while (run cleanup every 24 hours)
                time.sleep(24 * 60 * 60)  # 24 hours
                
                logger.info("Running scheduled memory cleanup")
                # This is a placeholder for actual cleanup logic
                # For a real implementation, you might want to archive old memories
                # or compress them based on timestamp, etc.
                
            except Exception as e:
                logger.error(f"Error in memory cleanup service: {str(e)}")
                # Sleep for a bit before retrying
                time.sleep(60)
    
    def _run_backup_service(self):
        """Background service to periodically backup memory data"""
        logger.info("Backup service started")
        
        while True:
            try:
                # Sleep for a while (run backup every 12 hours)
                time.sleep(12 * 60 * 60)  # 12 hours
                
                logger.info("Running scheduled memory backup")
                # This is a placeholder for actual backup logic
                backup_path = self.data_path / 'backups' / f"memory_backup_{time.strftime('%Y%m%d_%H%M%S')}"
                
                # For a real implementation, you would copy files to backup_path
                # or compress them into an archive
                
            except Exception as e:
                logger.error(f"Error in backup service: {str(e)}")
                # Sleep for a bit before retrying
                time.sleep(60)
    
    def _start_gui(self):
        """Start the graphical user interface"""
        try:
            # Import GUI
            from src.language_memory_gui import LanguageMemoryGUI
            
            logger.info("Starting GUI...")
            root = tk.Tk()
            self.gui_app = LanguageMemoryGUI(root)
            
            # Run the GUI main loop
            root.mainloop()
            
            logger.info("GUI closed, shutting down...")
            self._shutdown()
        except Exception as e:
            logger.error(f"Error starting GUI: {str(e)}")
            # Fall back to console mode if GUI fails
            self._run_console_mode()
    
    def _run_console_mode(self):
        """Run the system in console mode (no GUI)"""
        logger.info("Running in console mode")
        try:
            print("\nMemory System running in console mode.")
            print("Press Ctrl+C to exit.")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            print("\nShutting down Memory System...")
            self._shutdown()
    
    def _shutdown(self):
        """Shutdown all components and threads"""
        logger.info("Shutting down Memory System...")
        
        # Perform component-specific shutdown if needed
        for name, component in self.components.items():
            try:
                # Check if component has a shutdown method
                if hasattr(component, 'shutdown') and callable(component.shutdown):
                    component.shutdown()
                logger.info(f"Component '{name}' shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down component '{name}': {str(e)}")
        
        logger.info("Memory System shut down successfully")


def main():
    """Main entry point for the Memory System launcher"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch the Memory System")
    parser.add_argument('--no-gui', action='store_true', help="Run without GUI (console mode)")
    parser.add_argument('--no-background', action='store_true', help="Disable background services")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    # Convert args to dictionary
    options = vars(args)
    
    # Create and start the launcher
    launcher = MemorySystemLauncher(options)
    launcher.start()


if __name__ == "__main__":
    main() 