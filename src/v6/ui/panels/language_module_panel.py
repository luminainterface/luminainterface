#!/usr/bin/env python
"""
V6 Language Module Panel

Integrates the Enhanced Language System with the V6 Portal of Contradiction,
providing a holographic interface for language processing capabilities.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
    from src.v5.ui.qt_compat import get_widgets, get_gui, get_core
except ImportError:
    logging.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt, Signal, Slot
        
        # Simple compatibility functions
        def get_widgets():
            return QtWidgets
            
        def get_gui():
            return QtGui
            
        def get_core():
            return QtCore
    except ImportError:
        logging.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Import V6 panel base
from src.v6.ui.panel_base import V6PanelBase

# Import language system adapters
try:
    from src.language.pyside6_adapter import (
        LanguageMemoryAdapter,
        NeuralLinguisticProcessorAdapter,
        ConsciousMirrorLanguageAdapter,
        CentralLanguageNodeAdapter
    )
    HAS_LANGUAGE_SYSTEM = True
except ImportError:
    HAS_LANGUAGE_SYSTEM = False
    logging.error("Language system adapters not found. Some functionality will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)

class LanguageModulePanel(V6PanelBase):
    """
    Language Module Panel for the V6 Portal of Contradiction
    
    This panel integrates the Enhanced Language System into V6, providing:
    - Text processing with neural linguistic analysis
    - Consciousness level evaluation of language
    - Memory storage and retrieval
    - LLM weight control across all language components
    """
    
    # Define signals
    textProcessed = Signal(dict)  # Emitted when text is processed
    memoryStored = Signal(dict)   # Emitted when a memory is stored
    llmWeightChanged = Signal(float)  # Emitted when LLM weight changes
    
    def __init__(self, socket_manager=None, parent=None):
        """Initialize the Language Module Panel"""
        super().__init__(parent)
        self.socket_manager = socket_manager
        self.init_language_adapters()
        self.initUI()
        
        if self.socket_manager:
            self.socket_manager.register_handler("process_language", self.handle_language_request)
            self.socket_manager.register_handler("adjust_llm_weight", self.handle_llm_adjustment)
        
    def init_language_adapters(self):
        """Initialize language component adapters"""
        if not HAS_LANGUAGE_SYSTEM:
            self.central_node = None
            self.language_memory = None
            self.neural_processor = None
            self.consciousness = None
            self.available = False
            return
            
        try:
            # Initialize adapters
            self.language_memory = LanguageMemoryAdapter(
                data_dir="data/memory/language_memory", 
                llm_weight=0.5
            )
            
            self.neural_processor = NeuralLinguisticProcessorAdapter(
                data_dir="data/neural_linguistic", 
                llm_weight=0.5
            )
            
            self.consciousness = ConsciousMirrorLanguageAdapter(
                data_dir="data/v10", 
                llm_weight=0.5
            )
            
            # Initialize central node with references to other adapters
            self.central_node = CentralLanguageNodeAdapter(
                data_dir="data/central_language",
                llm_weight=0.5,
                language_memory_adapter=self.language_memory,
                neural_processor_adapter=self.neural_processor,
                consciousness_adapter=self.consciousness
            )
            
            # Connect signals
            if self.central_node.available:
                self.central_node.processing_complete.connect(self.on_processing_complete)
                self.central_node.llm_weight_changed.connect(self.on_llm_weight_changed)
                self.central_node.error_occurred.connect(self.on_error)
                self.available = True
                logger.info("Language adapters initialized successfully")
            else:
                self.available = False
                logger.warning("Central node not available")
                
        except Exception as e:
            logger.error(f"Error initializing language adapters: {str(e)}")
            self.central_node = None
            self.available = False
    
    def initUI(self):
        """Initialize the UI components"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
                background: rgba(16, 30, 43, 120);
            }
            QTabBar::tab {
                background: rgba(32, 60, 86, 180);
                border: 1px solid rgba(52, 152, 219, 100);
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                color: #ECF0F1;
            }
            QTabBar::tab:selected {
                background: rgba(52, 152, 219, 150);
                border: 1px solid rgba(52, 152, 219, 200);
                border-bottom: none;
            }
        """)
        
        # Create processing tab
        self.processing_tab = self.create_processing_tab()
        self.tabs.addTab(self.processing_tab, "Text Processing")
        
        # Create LLM weight control tab
        self.weight_control_tab = self.create_weight_control_tab()
        self.tabs.addTab(self.weight_control_tab, "LLM Weight Control")
        
        # Add status bar
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: rgba(16, 30, 43, 150);
                color: #ECF0F1;
                border-top: 1px solid rgba(52, 152, 219, 100);
            }
        """)
        
        # Update status
        status_msg = "Language system ready" if self.available else "Language system not available"
        self.status_bar.showMessage(status_msg)
        
        # Add components to layout
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status_bar)
    
    def create_processing_tab(self):
        """Create the text processing tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Input section
        input_group = QtWidgets.QGroupBox("Input Text")
        input_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
                margin-top: 12px;
                font-weight: bold;
                color: #3498DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        input_layout = QtWidgets.QVBoxLayout(input_group)
        
        # Text input area
        self.text_input = QtWidgets.QTextEdit()
        self.text_input.setStyleSheet("""
            QTextEdit {
                background: rgba(25, 35, 45, 150);
                color: #ECF0F1;
                border: 1px solid rgba(52, 152, 219, 70);
                border-radius: 3px;
                padding: 5px;
            }
        """)
        self.text_input.setPlaceholderText("Enter text to process through the language system...")
        input_layout.addWidget(self.text_input)
        
        # Processing options
        options_layout = QtWidgets.QHBoxLayout()
        
        # Use consciousness checkbox
        self.use_consciousness_cb = QtWidgets.QComboBox()
        self.use_consciousness_cb.addItems(["Use Consciousness: Yes", "Use Consciousness: No"])
        self.use_consciousness_cb.setStyleSheet("""
            QComboBox {
                background: rgba(32, 60, 86, 180);
                color: #ECF0F1;
                border: 1px solid rgba(52, 152, 219, 70);
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        options_layout.addWidget(self.use_consciousness_cb)
        
        # Use neural linguistics checkbox
        self.use_neural_cb = QtWidgets.QComboBox()
        self.use_neural_cb.addItems(["Use Neural Linguistics: Yes", "Use Neural Linguistics: No"])
        self.use_neural_cb.setStyleSheet("""
            QComboBox {
                background: rgba(32, 60, 86, 180);
                color: #ECF0F1;
                border: 1px solid rgba(52, 152, 219, 70);
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        options_layout.addWidget(self.use_neural_cb)
        
        # Process button
        self.process_button = QtWidgets.QPushButton("Process Text")
        self.process_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #2980B9, stop:1 #3498DB);
                color: white;
                border-radius: 3px;
                border: 1px solid #2980B9;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #3498DB, stop:1 #2980B9);
            }
            QPushButton:pressed {
                background: #2980B9;
            }
            QPushButton:disabled {
                background: #34495E;
                color: #95A5A6;
                border: 1px solid #34495E;
            }
        """)
        self.process_button.clicked.connect(self.process_text)
        options_layout.addWidget(self.process_button)
        
        input_layout.addLayout(options_layout)
        
        # Output section
        output_group = QtWidgets.QGroupBox("Processing Results")
        output_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
                margin-top: 12px;
                font-weight: bold;
                color: #3498DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        output_layout = QtWidgets.QVBoxLayout(output_group)
        
        # Splitter for results area
        results_splitter = QtWidgets.QSplitter(Qt.Vertical)
        
        # Unified language score
        score_widget = QtWidgets.QWidget()
        score_layout = QtWidgets.QFormLayout(score_widget)
        score_layout.setLabelAlignment(Qt.AlignRight)
        
        # Create styled labels
        score_label_style = """
            QLabel {
                color: #ECF0F1;
                font-weight: bold;
            }
        """
        score_value_style = """
            QLabel {
                color: #2ECC71;
                font-weight: bold;
                font-size: 14px;
            }
        """
        
        self.unified_score_label = QtWidgets.QLabel("N/A")
        self.unified_score_label.setStyleSheet(score_value_style)
        label = QtWidgets.QLabel("Unified Language Score:")
        label.setStyleSheet(score_label_style)
        score_layout.addRow(label, self.unified_score_label)
        
        self.neural_score_label = QtWidgets.QLabel("N/A")
        self.neural_score_label.setStyleSheet(score_value_style)
        label = QtWidgets.QLabel("Neural Linguistic Score:")
        label.setStyleSheet(score_label_style)
        score_layout.addRow(label, self.neural_score_label)
        
        self.consciousness_level_label = QtWidgets.QLabel("N/A")
        self.consciousness_level_label.setStyleSheet(score_value_style)
        label = QtWidgets.QLabel("Consciousness Level:")
        label.setStyleSheet(score_label_style)
        score_layout.addRow(label, self.consciousness_level_label)
        
        self.final_score_label = QtWidgets.QLabel("N/A")
        self.final_score_label.setStyleSheet(score_value_style)
        label = QtWidgets.QLabel("Final Score (with LLM weight):")
        label.setStyleSheet(score_label_style)
        score_layout.addRow(label, self.final_score_label)
        
        # Results text area
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background: rgba(25, 35, 45, 150);
                color: #ECF0F1;
                border: 1px solid rgba(52, 152, 219, 70);
                border-radius: 3px;
                padding: 5px;
            }
        """)
        
        results_splitter.addWidget(score_widget)
        results_splitter.addWidget(self.results_text)
        results_splitter.setSizes([100, 300])
        
        output_layout.addWidget(results_splitter)
        
        # Add to main layout
        layout.addWidget(input_group, 1)
        layout.addWidget(output_group, 2)
        
        return tab
    
    def create_weight_control_tab(self):
        """Create the LLM weight control tab"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Create global weight control
        global_group = QtWidgets.QGroupBox("Global LLM Weight")
        global_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
                margin-top: 12px;
                font-weight: bold;
                color: #3498DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        global_layout = QtWidgets.QVBoxLayout(global_group)
        
        # Description
        description = QtWidgets.QLabel(
            "Adjust the LLM weight to control the influence of Large Language Models "
            "on the language processing system. A higher weight means more LLM "
            "influence, while a lower weight relies more on algorithmic processing."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #ECF0F1;")
        global_layout.addWidget(description)
        
        # Slider layout
        slider_layout = QtWidgets.QHBoxLayout()
        
        # Weight labels
        slider_layout.addWidget(QtWidgets.QLabel("0.0"))
        
        # Global weight slider
        self.global_weight_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.global_weight_slider.setMinimum(0)
        self.global_weight_slider.setMaximum(100)
        self.global_weight_slider.setValue(50)  # Default 0.5
        self.global_weight_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: rgba(25, 35, 45, 150);
                border-radius: 4px;
                border: 1px solid rgba(52, 152, 219, 70);
            }
            
            QSlider::handle:horizontal {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                          stop:0 #3498DB, stop:1 #2980B9);
                width: 18px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 9px;
                border: 1px solid rgba(52, 152, 219, 200);
            }
            
            QSlider::handle:horizontal:hover {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                          stop:0 #5DADE2, stop:1 #3498DB);
            }
        """)
        self.global_weight_slider.valueChanged.connect(self.on_central_slider_changed)
        slider_layout.addWidget(self.global_weight_slider, 1)
        
        slider_layout.addWidget(QtWidgets.QLabel("1.0"))
        
        global_layout.addLayout(slider_layout)
        
        # Current weight value and apply button layout
        weight_button_layout = QtWidgets.QHBoxLayout()
        
        # Current value
        self.current_weight_label = QtWidgets.QLabel("Current: 0.50")
        self.current_weight_label.setStyleSheet("""
            QLabel {
                color: #ECF0F1;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        weight_button_layout.addWidget(self.current_weight_label)
        
        weight_button_layout.addStretch(1)
        
        # Apply button
        self.apply_weight_button = QtWidgets.QPushButton("Apply Weight to All Components")
        self.apply_weight_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #2980B9, stop:1 #3498DB);
                color: white;
                border-radius: 3px;
                border: 1px solid #2980B9;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #3498DB, stop:1 #2980B9);
            }
            QPushButton:pressed {
                background: #2980B9;
            }
            QPushButton:disabled {
                background: #34495E;
                color: #95A5A6;
                border: 1px solid #34495E;
            }
        """)
        self.apply_weight_button.clicked.connect(self.apply_weight)
        
        if not self.available:
            self.apply_weight_button.setEnabled(False)
            
        weight_button_layout.addWidget(self.apply_weight_button)
        
        global_layout.addLayout(weight_button_layout)
        
        # Component weights display
        components_group = QtWidgets.QGroupBox("Component Weights")
        components_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid rgba(52, 152, 219, 100);
                border-radius: 5px;
                margin-top: 12px;
                font-weight: bold;
                color: #3498DB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        components_layout = QtWidgets.QFormLayout(components_group)
        components_layout.setLabelAlignment(Qt.AlignRight)
        
        # Create component weight labels
        component_label_style = """
            QLabel {
                color: #ECF0F1;
                font-weight: bold;
            }
        """
        component_value_style = """
            QLabel {
                color: #2ECC71;
                font-weight: bold;
            }
        """
        
        label = QtWidgets.QLabel("Central Language Node:")
        label.setStyleSheet(component_label_style)
        self.central_weight_label = QtWidgets.QLabel("0.50")
        self.central_weight_label.setStyleSheet(component_value_style)
        components_layout.addRow(label, self.central_weight_label)
        
        label = QtWidgets.QLabel("Language Memory:")
        label.setStyleSheet(component_label_style)
        self.memory_weight_label = QtWidgets.QLabel("0.50")
        self.memory_weight_label.setStyleSheet(component_value_style)
        components_layout.addRow(label, self.memory_weight_label)
        
        label = QtWidgets.QLabel("Neural Linguistic Processor:")
        label.setStyleSheet(component_label_style)
        self.neural_weight_label = QtWidgets.QLabel("0.50")
        self.neural_weight_label.setStyleSheet(component_value_style)
        components_layout.addRow(label, self.neural_weight_label)
        
        label = QtWidgets.QLabel("Conscious Mirror Language:")
        label.setStyleSheet(component_label_style)
        self.consciousness_weight_label = QtWidgets.QLabel("0.50")
        self.consciousness_weight_label.setStyleSheet(component_value_style)
        components_layout.addRow(label, self.consciousness_weight_label)
        
        # Add to main layout
        layout.addWidget(global_group)
        layout.addWidget(components_group)
        layout.addStretch(1)
        
        return tab
    
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
        
        # Get processing options
        use_consciousness = self.use_consciousness_cb.currentIndex() == 0  # Yes is index 0
        use_neural = self.use_neural_cb.currentIndex() == 0  # Yes is index 0
        
        # Clear previous results
        self.unified_score_label.setText("Processing...")
        self.neural_score_label.setText("Processing...")
        self.consciousness_level_label.setText("Processing...")
        self.final_score_label.setText("Processing...")
        self.results_text.clear()
        
        # Update status
        self.status_bar.showMessage("Processing text...")
        
        # Process the text with safety measures
        try:
            # Make sure the central node exists and is available
            if not self.central_node or not hasattr(self.central_node, 'process_text'):
                raise AttributeError("Central language node is not properly initialized")
                
            # Use a local reference to avoid thread issues
            central_node = self.central_node
            
            # Process the text with the central node
            central_node.process_text(
                text=text,
                use_consciousness=use_consciousness,
                use_neural_linguistics=use_neural
            )
            logger.info(f"Submitted text for processing: {text[:50]}...")
            
            # If socket manager is available, emit event
            if self.socket_manager:
                self.socket_manager.emit("language_processing_started", {
                    "text_length": len(text),
                    "use_consciousness": use_consciousness,
                    "use_neural": use_neural
                })
                
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.on_error(f"Error processing text: {str(e)}")
            
            # Reset display
            self.unified_score_label.setText("N/A")
            self.neural_score_label.setText("N/A")
            self.consciousness_level_label.setText("N/A")
            self.final_score_label.setText("N/A")
            self.status_bar.showMessage(f"Error: {str(e)}")
            
            # Show error message
            QtWidgets.QMessageBox.critical(
                self, "Processing Error", 
                f"Failed to process text: {str(e)}\n\nCheck log for details."
            )
    
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
            from datetime import datetime
            results_text = "Processing Results:\n\n"
            
            # Add timestamp
            results_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add main scores section
            results_text += "=== Scores ===\n"
            results_text += f"Unified Language Score: {results.get('unified_language_score', 0):.3f}\n"
            
            if 'neural_linguistic_score' in results:
                results_text += f"Neural Linguistic Score: {results.get('neural_linguistic_score', 0):.3f}\n"
                
            if 'consciousness_level' in results:
                results_text += f"Consciousness Level: {results.get('consciousness_level', 0):.3f}\n"
                
            results_text += f"Final Score (with LLM weight): {results.get('final_score', 0):.3f}\n\n"
            
            # Add analysis section if available
            if 'analysis' in results:
                results_text += "=== Analysis ===\n"
                results_text += results['analysis'] + "\n\n"
            
            # Add patterns if available
            if 'patterns' in results:
                results_text += "=== Detected Patterns ===\n"
                patterns = results['patterns']
                if isinstance(patterns, list):
                    for pattern in patterns:
                        results_text += f"- {pattern}\n"
                elif isinstance(patterns, dict):
                    for pattern_type, patterns_list in patterns.items():
                        results_text += f"{pattern_type}:\n"
                        for pattern in patterns_list:
                            results_text += f"  - {pattern}\n"
                else:
                    results_text += str(patterns) + "\n"
                results_text += "\n"
            
            # Add consciousness insights if available
            if 'consciousness_insights' in results:
                results_text += "=== Consciousness Insights ===\n"
                results_text += results['consciousness_insights'] + "\n\n"
            
            # Set the results text
            self.results_text.setText(results_text)
            
            # Update status
            self.status_bar.showMessage("Processing complete")
            
            # Emit signal with results
            self.textProcessed.emit(results)
            
            # If socket manager is available, emit event
            if self.socket_manager:
                self.socket_manager.emit("language_processing_complete", {
                    "unified_score": results.get('unified_language_score', 0),
                    "neural_score": results.get('neural_linguistic_score', 0),
                    "consciousness_level": results.get('consciousness_level', 0),
                    "final_score": results.get('final_score', 0)
                })
                
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")
            self.on_error(str(e))
    
    @Slot(str)
    def on_error(self, error_message):
        """Handle errors"""
        # Show error in results
        self.results_text.setText(f"Error: {error_message}")
        
        # Reset score labels
        self.unified_score_label.setText("N/A")
        self.neural_score_label.setText("N/A")
        self.consciousness_level_label.setText("N/A")
        self.final_score_label.setText("N/A")
        
        # Update status bar
        self.status_bar.showMessage(f"Error: {error_message}")
        
        # Show error dialog
        QtWidgets.QMessageBox.critical(self, "Processing Error", error_message)
        
        # If socket manager is available, emit event
        if self.socket_manager:
            self.socket_manager.emit("language_processing_error", {
                "error": error_message
            })
    
    def on_central_slider_changed(self, value):
        """Handle central slider value changes"""
        weight = value / 100.0
        self.current_weight_label.setText(f"Current: {weight:.2f}")
    
    def apply_weight(self):
        """Apply the weight to all components"""
        if not self.available:
            QtWidgets.QMessageBox.critical(self, "Component Unavailable", 
                              "Language system is not available")
            return
            
        weight = self.global_weight_slider.value() / 100.0
        
        try:
            # Apply to central node (which will propagate to other components)
            self.central_node.adjust_llm_weight(weight)
            
            # Update UI
            self.central_weight_label.setText(f"{weight:.2f}")
            self.memory_weight_label.setText(f"{weight:.2f}")
            self.neural_weight_label.setText(f"{weight:.2f}")
            self.consciousness_weight_label.setText(f"{weight:.2f}")
            
            # Update status
            self.status_bar.showMessage(f"Applied LLM weight: {weight:.2f}")
            
            # Emit signal
            self.llmWeightChanged.emit(weight)
            
            # If socket manager is available, emit event
            if self.socket_manager:
                self.socket_manager.emit("language_llm_weight_changed", {
                    "weight": weight
                })
                
        except Exception as e:
            logger.error(f"Error applying weight: {str(e)}")
            self.on_error(str(e))
    
    @Slot(float)
    def on_llm_weight_changed(self, weight):
        """Handle LLM weight changes from the system"""
        # Update UI
        self.global_weight_slider.setValue(int(weight * 100))
        self.current_weight_label.setText(f"Current: {weight:.2f}")
        
        self.update_component_weights()
    
    def update_component_weights(self):
        """Update the component weight displays"""
        if not self.available:
            return
            
        try:
            # Get weights from components
            central_weight = 0.5
            memory_weight = 0.5
            neural_weight = 0.5
            consciousness_weight = 0.5
            
            if self.central_node and self.central_node.available:
                central_weight = self.central_node.node.llm_weight
                
            if self.language_memory and self.language_memory.available:
                memory_weight = self.language_memory.memory.llm_weight
                
            if self.neural_processor and self.neural_processor.available:
                neural_weight = self.neural_processor.processor.llm_weight
                
            if self.consciousness and self.consciousness.available:
                consciousness_weight = self.consciousness.mirror.llm_weight
            
            # Update labels
            self.central_weight_label.setText(f"{central_weight:.2f}")
            self.memory_weight_label.setText(f"{memory_weight:.2f}")
            self.neural_weight_label.setText(f"{neural_weight:.2f}")
            self.consciousness_weight_label.setText(f"{consciousness_weight:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating component weights: {str(e)}")
    
    def handle_language_request(self, data):
        """Handle language processing requests from socket"""
        if not self.available:
            if self.socket_manager:
                self.socket_manager.emit("language_processing_error", {
                    "error": "Language system not available"
                })
            return
            
        text = data.get("text", "").strip()
        if not text:
            if self.socket_manager:
                self.socket_manager.emit("language_processing_error", {
                    "error": "No text provided"
                })
            return
        
        # Get processing options
        use_consciousness = data.get("use_consciousness", True)
        use_neural = data.get("use_neural_linguistics", True)
        
        # Set the text in the UI
        self.text_input.setText(text)
        self.use_consciousness_cb.setCurrentIndex(0 if use_consciousness else 1)
        self.use_neural_cb.setCurrentIndex(0 if use_neural else 1)
        
        # Process the text
        try:
            self.central_node.process_text(
                text=text,
                use_consciousness=use_consciousness,
                use_neural_linguistics=use_neural
            )
            logger.info(f"Processing text from socket: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing text from socket: {str(e)}")
            if self.socket_manager:
                self.socket_manager.emit("language_processing_error", {
                    "error": str(e)
                })
    
    def handle_llm_adjustment(self, data):
        """Handle LLM weight adjustment requests from socket"""
        if not self.available:
            if self.socket_manager:
                self.socket_manager.emit("language_llm_weight_error", {
                    "error": "Language system not available"
                })
            return
            
        weight = data.get("weight", 0.5)
        if not 0 <= weight <= 1:
            if self.socket_manager:
                self.socket_manager.emit("language_llm_weight_error", {
                    "error": "Weight must be between 0 and 1"
                })
            return
        
        # Set the weight in the UI
        self.global_weight_slider.setValue(int(weight * 100))
        
        # Apply the weight
        self.apply_weight() 