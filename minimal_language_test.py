#!/usr/bin/env python
"""
Minimal Language Module Test

The simplest possible test for the language module - just a text box and results area.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("minimal_language_test.log")
    ]
)
logger = logging.getLogger("MinimalTest")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/memory/language_memory",
        "data/neural_linguistic",
        "data/v10",
        "data/central_language"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    return True

def main():
    """Main entry point for the minimal test"""
    print("Starting Minimal Language Module Test...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Check for PySide6
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        logger.info("PySide6 is available")
    except ImportError:
        logger.error("PySide6 is required but not installed!")
        print("Error: PySide6 is required but not installed.")
        print("Please install with: pip install PySide6")
        return 1
    
    # Create a minimal language module
    try:
        try:
            from src.language.pyside6_adapter import (
                LanguageMemoryAdapter,
                NeuralLinguisticProcessorAdapter,
                ConsciousMirrorLanguageAdapter,
                CentralLanguageNodeAdapter
            )
            has_language_system = True
            logger.info("Successfully imported language adapter modules")
        except ImportError:
            has_language_system = False
            logger.error("Could not import language adapter modules")
            return 1
        
        # Define a minimalist UI
        class MinimalLanguageUI(QtWidgets.QWidget):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("Minimal Language Test")
                self.resize(800, 600)
                
                # Initialize language components - very minimal
                try:
                    # Initialize central node directly
                    self.central_node = CentralLanguageNodeAdapter()
                    if not self.central_node.available:
                        raise RuntimeError("Central language node not available")
                    
                    # Connect signals 
                    self.central_node.processing_complete.connect(self.on_processing_complete)
                    self.central_node.error_occurred.connect(self.on_error)
                    
                    logger.info("Language system initialized successfully")
                    self.available = True
                except Exception as e:
                    logger.error(f"Error initializing language system: {e}")
                    self.available = False
                
                # Create layout
                layout = QtWidgets.QVBoxLayout(self)
                
                # Instructions
                instr_label = QtWidgets.QLabel(
                    "Enter text below and click Process to analyze it using the language system.")
                instr_label.setWordWrap(True)
                layout.addWidget(instr_label)
                
                # Input text area
                self.text_input = QtWidgets.QTextEdit()
                self.text_input.setPlaceholderText("Enter text to process...")
                layout.addWidget(self.text_input)
                
                # Process button
                self.process_button = QtWidgets.QPushButton("Process Text")
                self.process_button.clicked.connect(self.process_text)
                layout.addWidget(self.process_button)
                
                # Status label
                self.status_label = QtWidgets.QLabel("Ready")
                layout.addWidget(self.status_label)
                
                # Results area
                self.results_text = QtWidgets.QTextEdit()
                self.results_text.setReadOnly(True)
                self.results_text.setPlaceholderText("Results will appear here")
                layout.addWidget(self.results_text)
                
                # Set availability
                if not self.available:
                    self.process_button.setEnabled(False)
                    self.status_label.setText("Error: Language system not available")
            
            def process_text(self):
                """Process the entered text"""
                if not self.available:
                    QtWidgets.QMessageBox.critical(self, "Component Unavailable", 
                                    "Language system is not available")
                    return
                    
                text = self.text_input.toPlainText().strip()
                
                if not text:
                    QtWidgets.QMessageBox.warning(self, "Empty Input", "Please enter text to process")
                    return
                
                # Update status
                self.status_label.setText("Processing...")
                self.results_text.clear()
                
                # Process the text with safety measures
                try:
                    # Use a local reference to avoid thread issues
                    self.central_node.process_text(
                        text=text,
                        use_consciousness=True,
                        use_neural_linguistics=True
                    )
                    logger.info(f"Submitted text for processing: {text[:50]}...")
                except Exception as e:
                    logger.error(f"Error processing text: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.on_error(f"Error processing text: {str(e)}")
            
            @Slot(dict)
            def on_processing_complete(self, results):
                """Handle processing completion"""
                try:
                    # Format and display detailed results
                    from datetime import datetime
                    results_text = "Processing Results:\n\n"
                    
                    # Add timestamp
                    results_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    
                    # Add scores
                    results_text += f"Unified Score: {results.get('unified_language_score', 0):.3f}\n"
                    if 'neural_linguistic_score' in results:
                        results_text += f"Neural Score: {results.get('neural_linguistic_score', 0):.3f}\n"
                    if 'consciousness_level' in results:
                        results_text += f"Consciousness: {results.get('consciousness_level', 0):.3f}\n"
                    results_text += f"Final Score: {results.get('final_score', 0):.3f}\n\n"
                    
                    # Add analysis if available
                    if 'analysis' in results:
                        results_text += f"Analysis:\n{results['analysis']}\n\n"
                    
                    # Set results
                    self.results_text.setText(results_text)
                    self.status_label.setText("Processing complete")
                    
                except Exception as e:
                    logger.error(f"Error displaying results: {str(e)}")
                    self.on_error(f"Error displaying results: {str(e)}")
            
            @Slot(str)
            def on_error(self, error_message):
                """Handle error"""
                self.status_label.setText(f"Error: {error_message}")
                self.results_text.setText(f"Error occurred: {error_message}")
                logger.error(f"Error in language processing: {error_message}")
        
        # Create application
        app = QtWidgets.QApplication(sys.argv)
        window = MinimalLanguageUI()
        window.show()
        return app.exec()
        
    except Exception as e:
        logger.error(f"Global error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 