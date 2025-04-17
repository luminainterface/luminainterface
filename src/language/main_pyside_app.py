#!/usr/bin/env python3
"""
Enhanced Language System - PySide6 Application

This is the main application for the Enhanced Language System using PySide6.
It provides a GUI interface for interacting with Language Memory,
Neural Linguistic Processor, Conscious Mirror Language, and the 
Central Language Node.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_language_system_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main_pyside_app")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# PySide6 imports
try:
    from PySide6.QtCore import Qt, Slot, QTimer, QSettings
    from PySide6.QtGui import QAction, QIcon, QFont
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget, 
        QSplitter, QStatusBar, QToolBar, QDockWidget, QSlider,
        QComboBox, QMessageBox, QProgressBar, QGroupBox, QFormLayout
    )
    logger.info("Successfully imported PySide6")
except ImportError as e:
    logger.error(f"Failed to import PySide6: {e}")
    print(f"Error: PySide6 is required but not installed.")
    print("Please install PySide6 with: pip install PySide6")
    sys.exit(1)

# Import our adapter classes
try:
    from .pyside6_adapter import (
        LanguageMemoryAdapter,
        NeuralLinguisticProcessorAdapter,
        ConsciousMirrorLanguageAdapter,
        CentralLanguageNodeAdapter
    )
    logger.info("Successfully imported language component adapters")
except ImportError as e:
    logger.error(f"Failed to import language component adapters: {e}")
    print(f"Error: Language component adapters not found.")
    sys.exit(1)


class TextProcessingTab(QWidget):
    """Tab for processing text through the language system"""
    
    def __init__(self, central_node_adapter=None, parent=None):
        super().__init__(parent)
        self.central_node = central_node_adapter
        self.init_ui()
        
        # Connect signals from central node adapter
        if self.central_node and self.central_node.available:
            self.central_node.processing_complete.connect(self.on_processing_complete)
            self.central_node.error_occurred.connect(self.on_error)
    
    def init_ui(self):
        """Initialize the UI elements"""
        main_layout = QVBoxLayout(self)
        
        # Input section
        input_group = QGroupBox("Input Text")
        input_layout = QVBoxLayout(input_group)
        
        # Text input area
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to process through the language system...")
        input_layout.addWidget(self.text_input)
        
        # Processing options
        options_layout = QHBoxLayout()
        
        # Use consciousness checkbox
        self.use_consciousness_cb = QComboBox()
        self.use_consciousness_cb.addItems(["Use Consciousness: Yes", "Use Consciousness: No"])
        options_layout.addWidget(self.use_consciousness_cb)
        
        # Use neural linguistics checkbox
        self.use_neural_cb = QComboBox()
        self.use_neural_cb.addItems(["Use Neural Linguistics: Yes", "Use Neural Linguistics: No"])
        options_layout.addWidget(self.use_neural_cb)
        
        # Process button
        self.process_button = QPushButton("Process Text")
        self.process_button.clicked.connect(self.process_text)
        options_layout.addWidget(self.process_button)
        
        input_layout.addLayout(options_layout)
        
        # Output section
        output_group = QGroupBox("Processing Results")
        output_layout = QVBoxLayout(output_group)
        
        # Splitter for results area
        results_splitter = QSplitter(Qt.Vertical)
        
        # Unified language score
        score_widget = QWidget()
        score_layout = QFormLayout(score_widget)
        self.unified_score_label = QLabel("N/A")
        score_layout.addRow("Unified Language Score:", self.unified_score_label)
        self.neural_score_label = QLabel("N/A")
        score_layout.addRow("Neural Linguistic Score:", self.neural_score_label)
        self.consciousness_level_label = QLabel("N/A")
        score_layout.addRow("Consciousness Level:", self.consciousness_level_label)
        self.final_score_label = QLabel("N/A")
        score_layout.addRow("Final Score (with LLM weight):", self.final_score_label)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_splitter.addWidget(score_widget)
        results_splitter.addWidget(self.results_text)
        results_splitter.setSizes([100, 300])
        
        output_layout.addWidget(results_splitter)
        
        # Add to main layout
        main_layout.addWidget(input_group, 1)
        main_layout.addWidget(output_group, 2)
    
    def process_text(self):
        """Process the entered text"""
        text = self.text_input.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "Empty Input", "Please enter text to process")
            return
        
        if not self.central_node or not self.central_node.available:
            QMessageBox.critical(self, "Component Unavailable", 
                              "Central Language Node is not available")
            return
        
        # Get processing options
        use_consciousness = self.use_consciousness_cb.currentIndex() == 0  # Yes is index 0
        use_neural = self.use_neural_cb.currentIndex() == 0  # Yes is index 0
        
        # Clear previous results
        self.unified_score_label.setText("Processing...")
        self.neural_score_label.setText("Processing...")
        self.consciousness_level_label.setText("Processing...")
        self.final_score_label.setText("Processing...")
        self.results_text.clear()
        
        # Process the text
        try:
            self.central_node.process_text(
                text=text,
                use_consciousness=use_consciousness,
                use_neural_linguistics=use_neural
            )
            logger.info(f"Submitted text for processing: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            self.on_error(str(e))
    
    @Slot(dict)
    def on_processing_complete(self, results):
        """Handle processing completion"""
        try:
            # Update score labels
            self.unified_score_label.setText(f"{results.get('unified_language_score', 0):.3f}")
            
            # Update neural score if available
            if 'neural_linguistic_score' in results:
                self.neural_score_label.setText(f"{results.get('neural_linguistic_score', 0):.3f}")
            
            # Update consciousness level if available
            if 'consciousness_level' in results:
                self.consciousness_level_label.setText(f"{results.get('consciousness_level', 0):.3f}")
            
            # Update final score
            self.final_score_label.setText(f"{results.get('final_score', 0):.3f}")
            
            # Format and display detailed results
            results_text = "Processing Results:\n\n"
            
            # Add timestamp
            results_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add LLM weight information
            results_text += f"LLM Weight Applied: {results.get('llm_weight_applied', 'N/A')}\n\n"
            
            # Add cross-domain mappings if available
            if 'cross_domain_mappings' in results and results['cross_domain_mappings']:
                results_text += "Cross-Domain Mappings:\n"
                for mapping_name, mapping_data in results['cross_domain_mappings'].items():
                    results_text += f"  - {mapping_name}:\n"
                    for key, value in mapping_data.items():
                        if key != 'description':
                            results_text += f"    {key}: {value}\n"
                    results_text += f"    Description: {mapping_data.get('description', 'N/A')}\n"
                results_text += "\n"
            
            # Add component-specific results
            if 'language_memory_results' in results:
                memory_results = results['language_memory_results']
                results_text += "Language Memory Results:\n"
                
                if 'word_associations' in memory_results:
                    results_text += "  Word Associations:\n"
                    for assoc in memory_results['word_associations'][:5]:  # Show top 5
                        results_text += f"    - {assoc['word']}: {assoc['metadata'].get('strength', 0):.2f}\n"
                    
                    if len(memory_results['word_associations']) > 5:
                        results_text += f"    (and {len(memory_results['word_associations']) - 5} more...)\n"
                
                results_text += "\n"
            
            if 'neural_linguistic_results' in results:
                neural_results = results['neural_linguistic_results']
                results_text += "Neural Linguistic Results:\n"
                results_text += f"  Score: {neural_results.get('score', 0):.3f}\n"
                results_text += f"  Word Count: {neural_results.get('word_count', 0)}\n"
                results_text += f"  Unique Words: {neural_results.get('unique_words', 0)}\n"
                
                if 'patterns' in neural_results and neural_results['patterns']:
                    results_text += "  Detected Patterns:\n"
                    for pattern in neural_results['patterns']:
                        results_text += f"    - {pattern.get('pattern', 'unknown')} (confidence: {pattern.get('confidence', 0):.2f})\n"
                
                results_text += "\n"
            
            if 'consciousness_results' in results:
                consciousness_results = results['consciousness_results']
                results_text += "Consciousness Results:\n"
                results_text += f"  Level: {consciousness_results.get('consciousness_level', 0):.3f}\n"
                results_text += f"  Memory Continuity: {consciousness_results.get('memory_continuity', 0):.2f}\n"
                results_text += f"  Recursive Depth: {consciousness_results.get('recursive_depth', 0)}\n"
                results_text += f"  Self-Reference Detected: {consciousness_results.get('self_reference_detected', False)}\n"
                results_text += "\n"
            
            # Display the formatted results
            self.results_text.setText(results_text)
            logger.info("Text processing results displayed successfully")
            
        except Exception as e:
            logger.error(f"Error displaying processing results: {str(e)}")
            self.results_text.setText(f"Error displaying results: {str(e)}")
    
    @Slot(str)
    def on_error(self, error_message):
        """Handle processing errors"""
        self.unified_score_label.setText("Error")
        self.neural_score_label.setText("Error")
        self.consciousness_level_label.setText("Error")
        self.final_score_label.setText("Error")
        self.results_text.setText(f"Error processing text: {error_message}")
        
        QMessageBox.critical(self, "Processing Error", f"Error: {error_message}")
        logger.error(f"Processing error: {error_message}")


class LLMWeightControlTab(QWidget):
    """Tab for controlling LLM weights across components"""
    
    def __init__(self, central_node_adapter=None, 
                 memory_adapter=None, 
                 neural_adapter=None, 
                 consciousness_adapter=None,
                 parent=None):
        super().__init__(parent)
        self.central_node = central_node_adapter
        self.memory_adapter = memory_adapter
        self.neural_adapter = neural_adapter
        self.consciousness_adapter = consciousness_adapter
        self.init_ui()
        
        # Connect signals
        if self.central_node and self.central_node.available:
            self.central_node.llm_weight_changed.connect(self.on_central_weight_changed)
            self.central_node.component_status_changed.connect(self.on_component_status_changed)
    
    def init_ui(self):
        """Initialize the UI elements"""
        main_layout = QVBoxLayout(self)
        
        # Central node weight control
        central_group = QGroupBox("Central Language Node LLM Weight")
        central_layout = QVBoxLayout(central_group)
        
        # Current weight label
        self.central_weight_label = QLabel("Current Weight: 0.50")
        central_layout.addWidget(self.central_weight_label)
        
        # Central node slider
        self.central_slider = QSlider(Qt.Horizontal)
        self.central_slider.setMinimum(0)
        self.central_slider.setMaximum(100)
        self.central_slider.setValue(50)  # Default to 0.5
        self.central_slider.setTickPosition(QSlider.TicksBelow)
        self.central_slider.setTickInterval(10)
        self.central_slider.valueChanged.connect(self.on_central_slider_changed)
        central_layout.addWidget(self.central_slider)
        
        # Preset buttons
        presets_layout = QHBoxLayout()
        preset_values = [(0, "0.0"), (20, "0.2"), (50, "0.5"), (80, "0.8"), (100, "1.0")]
        
        for value, label in preset_values:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, v=value: self.central_slider.setValue(v))
            presets_layout.addWidget(button)
        
        central_layout.addLayout(presets_layout)
        
        # Apply button
        self.apply_button = QPushButton("Apply Weight to All Components")
        self.apply_button.clicked.connect(self.apply_weight)
        central_layout.addWidget(self.apply_button)
        
        # Component weights status
        components_group = QGroupBox("Component LLM Weights")
        components_layout = QFormLayout(components_group)
        
        self.memory_weight_label = QLabel("N/A")
        components_layout.addRow("Language Memory Weight:", self.memory_weight_label)
        
        self.neural_weight_label = QLabel("N/A")
        components_layout.addRow("Neural Linguistic Processor Weight:", self.neural_weight_label)
        
        self.consciousness_weight_label = QLabel("N/A")
        components_layout.addRow("Conscious Mirror Language Weight:", self.consciousness_weight_label)
        
        # Effects description
        effects_group = QGroupBox("LLM Weight Effects")
        effects_layout = QVBoxLayout(effects_group)
        
        effects_text = """
<b>LLM Weight Effects:</b>

The LLM weight controls the influence of Large Language Model suggestions 
on the system's operation:

- <b>0.0:</b> No LLM influence, pure algorithmic processing
- <b>0.2:</b> Minimal LLM influence, primarily algorithmic
- <b>0.5:</b> Balanced between algorithmic and LLM processing
- <b>0.8:</b> Strong LLM influence with algorithmic grounding
- <b>1.0:</b> Maximum LLM influence, minimal algorithmic constraints

Experiment with different weights to find the optimal balance for your use case.
        """
        
        effects_label = QLabel(effects_text)
        effects_label.setWordWrap(True)
        effects_layout.addWidget(effects_label)
        
        # Add to main layout
        main_layout.addWidget(central_group)
        main_layout.addWidget(components_group)
        main_layout.addWidget(effects_group)
        main_layout.addStretch(1)
        
        # Initialize with current values if available
        self.update_component_weights()
    
    def on_central_slider_changed(self, value):
        """Update the central weight label when the slider changes"""
        weight = value / 100.0
        self.central_weight_label.setText(f"Current Weight: {weight:.2f}")
    
    def apply_weight(self):
        """Apply the selected weight to all components"""
        if not self.central_node or not self.central_node.available:
            QMessageBox.critical(self, "Component Unavailable", 
                              "Central Language Node is not available")
            return
        
        weight = self.central_slider.value() / 100.0
        
        try:
            self.central_node.adjust_llm_weight(weight)
            logger.info(f"Applied LLM weight {weight} to all components")
            QMessageBox.information(self, "Weight Applied", 
                                   f"LLM weight {weight:.2f} applied to all components")
        except Exception as e:
            logger.error(f"Error applying LLM weight: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to apply weight: {str(e)}")
    
    @Slot(float)
    def on_central_weight_changed(self, weight):
        """Update UI when central weight changes"""
        self.central_slider.setValue(int(weight * 100))
        self.central_weight_label.setText(f"Current Weight: {weight:.2f}")
        
        # Also update the component weights
        QTimer.singleShot(500, self.update_component_weights)
    
    @Slot(dict)
    def on_component_status_changed(self, status):
        """Update UI when component status changes"""
        # Update component availability indicators
        self.update_component_weights()
    
    def update_component_weights(self):
        """Update the component weight labels"""
        # Language Memory
        if self.memory_adapter and self.memory_adapter.available:
            try:
                stats = self.memory_adapter.get_llm_integration_stats()
                weight = stats.get("llm_weight", "N/A")
                self.memory_weight_label.setText(f"{weight:.2f}")
            except:
                self.memory_weight_label.setText("Error")
        else:
            self.memory_weight_label.setText("Not available")
        
        # Neural Linguistic Processor
        if self.neural_adapter and self.neural_adapter.available:
            if hasattr(self.neural_adapter.processor, "llm_weight"):
                weight = self.neural_adapter.processor.llm_weight
                self.neural_weight_label.setText(f"{weight:.2f}")
            else:
                self.neural_weight_label.setText("N/A")
        else:
            self.neural_weight_label.setText("Not available")
        
        # Conscious Mirror Language
        if self.consciousness_adapter and self.consciousness_adapter.available:
            if hasattr(self.consciousness_adapter.consciousness, "llm_weight"):
                weight = self.consciousness_adapter.consciousness.llm_weight
                self.consciousness_weight_label.setText(f"{weight:.2f}")
            else:
                self.consciousness_weight_label.setText("N/A")
        else:
            self.consciousness_weight_label.setText("Not available")


class MainWindow(QMainWindow):
    """Main window for the Enhanced Language System GUI"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize adapters
        self.init_adapters()
        
        # Set up the UI
        self.init_ui()
        
        # Set up the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect status updates
        if self.central_node and self.central_node.available:
            self.central_node.system_status_updated.connect(self.update_status)
    
    def init_adapters(self):
        """Initialize the component adapters"""
        try:
            # Create the component adapters
            self.memory_adapter = LanguageMemoryAdapter()
            self.neural_adapter = NeuralLinguisticProcessorAdapter()
            self.consciousness_adapter = ConsciousMirrorLanguageAdapter()
            
            # Create the central node adapter using the components
            self.central_node = CentralLanguageNodeAdapter(
                language_memory_adapter=self.memory_adapter,
                neural_processor_adapter=self.neural_adapter,
                consciousness_adapter=self.consciousness_adapter
            )
            
            logger.info("Successfully initialized language component adapters")
        except Exception as e:
            logger.error(f"Error initializing adapters: {str(e)}")
            self.memory_adapter = None
            self.neural_adapter = None
            self.consciousness_adapter = None
            self.central_node = None
    
    def init_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("Enhanced Language System")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create and add tabs
        self.text_processing_tab = TextProcessingTab(self.central_node)
        self.llm_weight_tab = LLMWeightControlTab(
            self.central_node,
            self.memory_adapter,
            self.neural_adapter,
            self.consciousness_adapter
        )
        
        self.tabs.addTab(self.text_processing_tab, "Text Processing")
        self.tabs.addTab(self.llm_weight_tab, "LLM Weight Control")
        
        main_layout.addWidget(self.tabs)
        
        # Create actions
        self.create_actions()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
    
    def create_actions(self):
        """Create actions for menus and toolbars"""
        # Exit action
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        
        # About action
        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self.show_about)
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(self.about_action)
    
    def create_toolbar(self):
        """Create the application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add LLM weight label and value
        toolbar.addWidget(QLabel("LLM Weight:"))
        self.llm_weight_toolbar_label = QLabel("0.50")
        toolbar.addWidget(self.llm_weight_toolbar_label)
        
        # Add system status label
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("System Status:"))
        self.system_status_label = QLabel("Initializing...")
        toolbar.addWidget(self.system_status_label)
    
    @Slot(dict)
    def update_status(self, status):
        """Update status bar and toolbar with system status"""
        if not status:
            return
        
        # Update toolbar LLM weight
        if "llm_weight" in status:
            weight = status["llm_weight"]
            self.llm_weight_toolbar_label.setText(f"{weight:.2f}")
        
        # Update system status label
        components_count = status.get("components_initialized", 0)
        processed_count = status.get("processed_text_count", 0)
        
        status_text = f"Components: {components_count}/3 | Processed texts: {processed_count}"
        self.system_status_label.setText(status_text)
        
        # Update status bar
        self.status_bar.showMessage(f"System ready - {components_count}/3 components initialized")
    
    def show_about(self):
        """Show the about dialog"""
        about_text = """
<h2>Enhanced Language System</h2>
<p>Version 1.0</p>
<p>A system that integrates Language Memory, Neural Linguistic Processing, 
and Conscious Mirror Language with LLM weighing capabilities.</p>
<p>Components:</p>
<ul>
  <li>Language Memory - Stores and recalls language patterns</li>
  <li>Neural Linguistic Processor - Analyzes text for patterns and connections</li>
  <li>Conscious Mirror Language - Models consciousness in language</li>
  <li>Central Language Node - Integrates all components</li>
</ul>
"""
        QMessageBox.about(self, "About Enhanced Language System", about_text)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop component adapters
        if self.central_node and self.central_node.available:
            if hasattr(self.central_node.node, "stop"):
                self.central_node.node.stop()
        
        event.accept()


def main():
    """Run the main application"""
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Language System")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    main_window = MainWindow()
    main_window.show()
    
    # Run the application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 