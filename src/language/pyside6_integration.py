#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySide6 Integration for Enhanced Language System

This module provides integration between the Enhanced Language System
and PySide6-based user interfaces, offering signal-based communication,
visualization components, and UI helpers for language processing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Add parent directory to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import PySide6 - handle cases where it might not be installed
try:
    from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt, QThread
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
        QProgressBar, QPushButton, QComboBox, QSlider, QSplitter,
        QTabWidget, QFrame, QGroupBox, QRadioButton, QCheckBox
    )
    from PySide6.QtGui import QFont, QColor, QPalette
    PYSIDE6_AVAILABLE = True
except ImportError:
    # Create dummy base classes to allow the file to be imported even without PySide6
    class QObject:
        pass
    
    class Signal:
        def __init__(self, *args):
            pass
        
        def emit(self, *args, **kwargs):
            pass
    
    class Slot:
        def __init__(self, *args):
            pass
        
        def __call__(self, func):
            return func
    
    PYSIDE6_AVAILABLE = False
    logging.warning("PySide6 not available. UI integration disabled.")

# Import Enhanced Language System components
try:
    from src.language.central_language_node import CentralLanguageNode
    from src.language.language_memory import LanguageMemory
    from src.language.conscious_mirror_language import ConsciousMirrorLanguage
    from src.language.neural_linguistic_processor import NeuralLinguisticProcessor
    from src.language.recursive_pattern_analyzer import RecursivePatternAnalyzer
    LANGUAGE_SYSTEM_AVAILABLE = True
except ImportError:
    LANGUAGE_SYSTEM_AVAILABLE = False
    logging.warning("Enhanced Language System components not available.")


class LanguageSystemSignals(QObject):
    """
    Signal class for Enhanced Language System integration with PySide6
    
    This class provides signals for asynchronous communication between
    language system components and UI elements.
    """
    
    # Processing signals
    processing_started = Signal(str)  # Component name
    processing_progress = Signal(str, float)  # Component name, progress (0-1)
    processing_completed = Signal(str, dict)  # Component name, results
    processing_error = Signal(str, str)  # Component name, error message
    
    # Metrics signals
    consciousness_level_changed = Signal(float)  # New consciousness level
    llm_weight_changed = Signal(float)  # New LLM weight
    nn_weight_changed = Signal(float)  # New NN weight
    
    # Memory signals
    memory_added = Signal(str, dict)  # Memory ID, memory data
    memory_removed = Signal(str)  # Memory ID
    memory_updated = Signal(str, dict)  # Memory ID, updated memory data
    
    # Status signals
    component_status_changed = Signal(str, str)  # Component name, status
    system_status_updated = Signal(dict)  # Full system status


class LanguageProcessingThread(QThread):
    """
    Thread for processing language without blocking the UI
    
    This thread handles processing text through language components
    and emits signals for UI updates.
    """
    
    def __init__(self, parent=None, signals=None, language_node=None):
        """Initialize the language processing thread"""
        super().__init__(parent)
        self.signals = signals or LanguageSystemSignals()
        self.language_node = language_node
        self.text = ""
        self.component = ""
        self.options = {}
        
    def set_task(self, text: str, component: str = "central", options: Dict = None):
        """Set the processing task"""
        self.text = text
        self.component = component
        self.options = options or {}
        
    def run(self):
        """Run the processing task"""
        if not self.text or not self.language_node:
            self.signals.processing_error.emit(
                self.component, "No text or language node provided"
            )
            return
            
        try:
            # Emit processing started signal
            self.signals.processing_started.emit(self.component)
            
            # Process text through the appropriate component
            if self.component == "central":
                result = self.language_node.process_text(self.text, **self.options)
            elif self.component == "memory":
                result = self.language_node.language_memory.process_text(self.text, **self.options)
            elif self.component == "conscious":
                result = self.language_node.conscious_mirror_language.process_text(self.text, **self.options)
            elif self.component == "neural":
                result = self.language_node.neural_linguistic_processor.process_text(self.text, **self.options)
            elif self.component == "recursive":
                result = self.language_node.recursive_pattern_analyzer.analyze_text(self.text, **self.options)
            else:
                self.signals.processing_error.emit(
                    self.component, f"Unknown component: {self.component}"
                )
                return
                
            # Emit completed signal with results
            self.signals.processing_completed.emit(self.component, result)
            
        except Exception as e:
            # Emit error signal
            self.signals.processing_error.emit(
                self.component, f"Error processing text: {str(e)}"
            )


class EnhancedLanguageIntegration:
    """
    Main integration class for Enhanced Language System with PySide6
    
    This class provides methods for integrating the Enhanced Language System
    with PySide6-based UIs, including configuration, component management,
    and signal-based communication.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        llm_weight: float = 0.5,
        nn_weight: float = 0.5,
        config: Dict[str, Any] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the Enhanced Language Integration
        
        Args:
            data_dir: Directory for data storage
            llm_weight: Initial LLM weight (0.0-1.0)
            nn_weight: Initial NN weight (0.0-1.0)
            config: Additional configuration options
            mock_mode: Whether to use mock mode (True) or real processing (False)
        """
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        self.config = config or {}
        self.mock_mode = mock_mode
        
        # Initialize signals
        self.signals = LanguageSystemSignals()
        
        # Setup components
        self._setup_components()
        
        # Create processing thread
        self.processing_thread = LanguageProcessingThread(
            signals=self.signals,
            language_node=self.central_node
        )
        
        # Status timer
        self.status_timer = None
        
        logging.info("Enhanced Language Integration initialized")
    
    def _setup_components(self):
        """Setup language system components"""
        if not LANGUAGE_SYSTEM_AVAILABLE:
            logging.warning("Language system components not available. Using mock mode.")
            self.mock_mode = True
            self.central_node = None
            return
            
        if self.mock_mode:
            logging.info("Using mock mode for language components")
            self.central_node = self._create_mock_node()
            return
            
        try:
            # Create real components
            self.central_node = CentralLanguageNode(
                data_dir=self.data_dir,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            
            # Configure with additional options if provided
            if self.config:
                if "db_enabled" in self.config:
                    # Configure database settings
                    pass
                    
                if "conversation_memory" in self.config:
                    # Configure conversation memory settings
                    pass
            
            logging.info("Enhanced Language System components initialized")
            
        except Exception as e:
            logging.error(f"Error initializing language components: {str(e)}")
            logging.info("Falling back to mock mode")
            self.mock_mode = True
            self.central_node = self._create_mock_node()
    
    def _create_mock_node(self):
        """Create a mock central node for testing without dependencies"""
        # This is a simple mock that returns predefined outputs
        class MockCentralNode:
            def __init__(self, llm_weight=0.5, nn_weight=0.5):
                self.llm_weight = llm_weight
                self.nn_weight = nn_weight
                self.language_memory = MockComponent("Language Memory")
                self.conscious_mirror_language = MockComponent("Conscious Mirror Language")
                self.neural_linguistic_processor = MockComponent("Neural Linguistic Processor")
                self.recursive_pattern_analyzer = MockComponent("Recursive Pattern Analyzer")
                
            def process_text(self, text, **kwargs):
                return {
                    "text": text,
                    "consciousness_level": 0.7,
                    "neural_linguistic_score": 0.6,
                    "recursive_pattern_depth": 2,
                    "memory_associations": ["neural", "network", "language"],
                    "processed_by": "mock_central_node",
                    "mock_mode": True
                }
                
            def set_llm_weight(self, weight):
                self.llm_weight = weight
                return True
                
            def set_nn_weight(self, weight):
                self.nn_weight = weight
                return True
                
        class MockComponent:
            def __init__(self, name):
                self.name = name
                
            def process_text(self, text, **kwargs):
                return {
                    "text": text,
                    "component": self.name,
                    "mock_mode": True,
                    "timestamp": "2023-01-01T00:00:00Z"
                }
                
            def analyze_text(self, text, **kwargs):
                return self.process_text(text, **kwargs)
                
        return MockCentralNode(llm_weight=self.llm_weight, nn_weight=self.nn_weight)
    
    def process_text(self, text, component="central", options=None, async_mode=True):
        """
        Process text through a language system component
        
        Args:
            text: Text to process
            component: Component to use ("central", "memory", "conscious", "neural", "recursive")
            options: Additional processing options
            async_mode: Whether to process asynchronously (True) or synchronously (False)
            
        Returns:
            Dict with results if async_mode=False, otherwise None (results via signals)
        """
        if not text:
            return None
            
        if async_mode:
            # Set task and start processing thread
            self.processing_thread.set_task(text, component, options)
            self.processing_thread.start()
            return None
        else:
            # Process synchronously
            try:
                self.signals.processing_started.emit(component)
                
                if component == "central":
                    result = self.central_node.process_text(text, **(options or {}))
                elif component == "memory":
                    result = self.central_node.language_memory.process_text(text, **(options or {}))
                elif component == "conscious":
                    result = self.central_node.conscious_mirror_language.process_text(text, **(options or {}))
                elif component == "neural":
                    result = self.central_node.neural_linguistic_processor.process_text(text, **(options or {}))
                elif component == "recursive":
                    result = self.central_node.recursive_pattern_analyzer.analyze_text(text, **(options or {}))
                else:
                    raise ValueError(f"Unknown component: {component}")
                
                self.signals.processing_completed.emit(component, result)
                return result
                
            except Exception as e:
                error_msg = f"Error processing text: {str(e)}"
                self.signals.processing_error.emit(component, error_msg)
                logging.error(error_msg)
                return {"error": error_msg}
    
    def set_llm_weight(self, weight):
        """Set the LLM weight"""
        if not 0.0 <= weight <= 1.0:
            logging.warning(f"Invalid LLM weight {weight}. Must be between 0.0 and 1.0.")
            return False
            
        if self.central_node:
            self.central_node.set_llm_weight(weight)
            self.llm_weight = weight
            self.signals.llm_weight_changed.emit(weight)
            return True
        return False
    
    def set_nn_weight(self, weight):
        """Set the neural network weight"""
        if not 0.0 <= weight <= 1.0:
            logging.warning(f"Invalid NN weight {weight}. Must be between 0.0 and 1.0.")
            return False
            
        if self.central_node:
            self.central_node.set_nn_weight(weight)
            self.nn_weight = weight
            self.signals.nn_weight_changed.emit(weight)
            return True
        return False
    
    def start_status_timer(self, interval=5000):
        """Start the status update timer"""
        if not PYSIDE6_AVAILABLE:
            logging.warning("PySide6 not available. Status timer not started.")
            return False
            
        if self.status_timer is None:
            self.status_timer = QTimer()
            self.status_timer.timeout.connect(self._update_status)
            self.status_timer.start(interval)
            return True
        return False
    
    def stop_status_timer(self):
        """Stop the status update timer"""
        if self.status_timer:
            self.status_timer.stop()
            self.status_timer = None
            return True
        return False
    
    def _update_status(self):
        """Update system status"""
        if not self.central_node:
            return
            
        # Collect status information from all components
        status = {
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "mock_mode": self.mock_mode,
            "central_node_active": bool(self.central_node),
            "components": {}
        }
        
        # Add consciousness level if available
        try:
            if hasattr(self.central_node, "get_consciousness_level"):
                status["consciousness_level"] = self.central_node.get_consciousness_level()
            elif hasattr(self.central_node, "conscious_mirror_language") and \
                 hasattr(self.central_node.conscious_mirror_language, "get_consciousness_level"):
                status["consciousness_level"] = self.central_node.conscious_mirror_language.get_consciousness_level()
            else:
                status["consciousness_level"] = 0.0
        except:
            status["consciousness_level"] = 0.0
            
        # Emit status signal
        self.signals.system_status_updated.emit(status)
        
        # Emit consciousness level if it changed
        if "consciousness_level" in status and status["consciousness_level"] != getattr(self, "_last_consciousness_level", None):
            self._last_consciousness_level = status["consciousness_level"]
            self.signals.consciousness_level_changed.emit(status["consciousness_level"])


class LanguageComponentPanel(QWidget):
    """
    UI panel for displaying and interacting with a language component
    
    This widget provides a UI for interacting with a specific language component,
    including text input, processing, and result display.
    """
    
    def __init__(self, parent=None, integration=None, component_type="central"):
        """
        Initialize the language component panel
        
        Args:
            parent: Parent widget
            integration: EnhancedLanguageIntegration instance
            component_type: Component type ("central", "memory", "conscious", "neural", "recursive")
        """
        super().__init__(parent)
        
        # Store parameters
        self.integration = integration
        self.component_type = component_type
        self.component_name = self._get_component_name()
        
        # Connect signals if available
        if integration and integration.signals:
            self.signals = integration.signals
            self.signals.processing_started.connect(self._on_processing_started)
            self.signals.processing_progress.connect(self._on_processing_progress)
            self.signals.processing_completed.connect(self._on_processing_completed)
            self.signals.processing_error.connect(self._on_processing_error)
        
        # Create UI
        self._create_ui()
    
    def _get_component_name(self):
        """Get display name for the component type"""
        component_names = {
            "central": "Central Language Node",
            "memory": "Language Memory",
            "conscious": "Conscious Mirror Language",
            "neural": "Neural Linguistic Processor",
            "recursive": "Recursive Pattern Analyzer"
        }
        return component_names.get(self.component_type, "Unknown Component")
    
    def _create_ui(self):
        """Create the UI elements"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Component title
        title_label = QLabel(self.component_name)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        
        # Text input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(f"Enter text to process with {self.component_name}...")
        input_layout.addWidget(self.text_input)
        
        # Button row
        button_layout = QHBoxLayout()
        
        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self._on_process_clicked)
        button_layout.addWidget(self.process_button)
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)
        
        input_layout.addLayout(button_layout)
        layout.addWidget(input_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Results will appear here...")
        results_layout.addWidget(self.results_display)
        
        layout.addWidget(results_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def _on_process_clicked(self):
        """Handle process button click"""
        text = self.text_input.toPlainText()
        if not text:
            self.status_label.setText("Error: No text provided")
            return
            
        if self.integration:
            self.integration.process_text(text, self.component_type)
        else:
            self.status_label.setText("Error: No language integration available")
    
    def _on_clear_clicked(self):
        """Handle clear button click"""
        self.text_input.clear()
        self.results_display.clear()
        self.status_label.setText("Ready")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
    
    def _on_processing_started(self, component):
        """Handle processing started signal"""
        if component != self.component_type:
            return
            
        self.status_label.setText(f"Processing with {self.component_name}...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.process_button.setEnabled(False)
    
    def _on_processing_progress(self, component, progress):
        """Handle processing progress signal"""
        if component != self.component_type:
            return
            
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)
    
    def _on_processing_completed(self, component, results):
        """Handle processing completed signal"""
        if component != self.component_type:
            return
            
        # Format results
        result_text = "Processing Results:\n\n"
        
        # Add different types of results depending on the component
        if self.component_type == "central":
            if "consciousness_level" in results:
                result_text += f"Consciousness Level: {results['consciousness_level']:.2f}\n"
                
            if "neural_linguistic_score" in results:
                result_text += f"Neural Linguistic Score: {results['neural_linguistic_score']:.2f}\n"
                
            if "recursive_pattern_depth" in results:
                result_text += f"Recursive Pattern Depth: {results['recursive_pattern_depth']}\n"
                
            if "memory_associations" in results:
                result_text += f"Memory Associations: {', '.join(results['memory_associations'])}\n"
                
        elif self.component_type == "memory":
            if "associations" in results:
                result_text += "Word Associations:\n"
                for word, associations in results.get("associations", {}).items():
                    result_text += f"  {word}: {associations}\n"
                    
        elif self.component_type == "conscious":
            if "consciousness_level" in results:
                result_text += f"Consciousness Level: {results['consciousness_level']:.2f}\n"
                
            if "self_awareness" in results:
                result_text += f"Self-Awareness: {results['self_awareness']:.2f}\n"
                
            if "mirror_depth" in results:
                result_text += f"Mirror Depth: {results['mirror_depth']}\n"
                
        elif self.component_type == "neural":
            if "patterns" in results:
                result_text += "Neural Patterns:\n"
                for pattern in results.get("patterns", []):
                    result_text += f"  {pattern}\n"
                    
            if "neural_score" in results:
                result_text += f"Neural Score: {results['neural_score']:.2f}\n"
                
        elif self.component_type == "recursive":
            if "recursive_depth" in results:
                result_text += f"Recursive Depth: {results['recursive_depth']}\n"
                
            if "self_references" in results:
                result_text += f"Self References: {len(results.get('self_references', []))}\n"
                
            if "loops" in results:
                result_text += f"Linguistic Loops: {len(results.get('loops', []))}\n"
        
        # Add raw results
        result_text += "\nRaw Results:\n"
        for key, value in results.items():
            result_text += f"{key}: {value}\n"
        
        # Update UI
        self.results_display.setPlainText(result_text)
        self.status_label.setText("Processing complete")
        self.progress_bar.setValue(100)
        self.process_button.setEnabled(True)
    
    def _on_processing_error(self, component, error_message):
        """Handle processing error signal"""
        if component != self.component_type:
            return
            
        self.status_label.setText(f"Error: {error_message}")
        self.results_display.setPlainText(f"Error during processing:\n{error_message}")
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)


class CentralLanguagePanel(QWidget):
    """
    Main panel for Central Language Node with all components
    
    This widget provides access to all language components through a
    tabbed interface with controls for LLM and NN weights.
    """
    
    def __init__(self, parent=None, integration=None):
        """
        Initialize the central language panel
        
        Args:
            parent: Parent widget
            integration: EnhancedLanguageIntegration instance
        """
        super().__init__(parent)
        self.integration = integration
        
        # Create UI
        self._create_ui()
        
        # Connect signals if available
        if integration and integration.signals:
            self.signals = integration.signals
            self.signals.llm_weight_changed.connect(self._on_llm_weight_changed)
            self.signals.nn_weight_changed.connect(self._on_nn_weight_changed)
            self.signals.system_status_updated.connect(self._on_status_updated)
    
    def _create_ui(self):
        """Create the UI elements"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("Enhanced Language System")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        desc_label = QLabel(
            "Access and interact with all components of the Enhanced Language System"
        )
        layout.addWidget(desc_label)
        
        # Weight sliders
        weight_group = QGroupBox("LLM and Neural Network Weights")
        weight_layout = QVBoxLayout(weight_group)
        
        # LLM weight slider
        llm_layout = QVBoxLayout()
        self.llm_label = QLabel(f"LLM Weight: {self.integration.llm_weight:.1f}")
        llm_layout.addWidget(self.llm_label)
        
        self.llm_slider = QSlider(Qt.Horizontal)
        self.llm_slider.setMinimum(0)
        self.llm_slider.setMaximum(100)
        self.llm_slider.setValue(int(self.integration.llm_weight * 100))
        self.llm_slider.valueChanged.connect(self._on_llm_slider_changed)
        llm_layout.addWidget(self.llm_slider)
        
        weight_layout.addLayout(llm_layout)
        
        # NN weight slider
        nn_layout = QVBoxLayout()
        self.nn_label = QLabel(f"Neural Network Weight: {self.integration.nn_weight:.1f}")
        nn_layout.addWidget(self.nn_label)
        
        self.nn_slider = QSlider(Qt.Horizontal)
        self.nn_slider.setMinimum(0)
        self.nn_slider.setMaximum(100)
        self.nn_slider.setValue(int(self.integration.nn_weight * 100))
        self.nn_slider.valueChanged.connect(self._on_nn_slider_changed)
        nn_layout.addWidget(self.nn_slider)
        
        weight_layout.addLayout(nn_layout)
        
        # Update button
        self.update_button = QPushButton("Update Weights")
        self.update_button.clicked.connect(self._on_update_weights)
        weight_layout.addWidget(self.update_button)
        
        layout.addWidget(weight_group)
        
        # Component tabs
        self.tab_widget = QTabWidget()
        
        # Create tabs for each component
        self.central_tab = LanguageComponentPanel(
            integration=self.integration,
            component_type="central"
        )
        self.tab_widget.addTab(self.central_tab, "Central Node")
        
        self.memory_tab = LanguageComponentPanel(
            integration=self.integration,
            component_type="memory"
        )
        self.tab_widget.addTab(self.memory_tab, "Language Memory")
        
        self.conscious_tab = LanguageComponentPanel(
            integration=self.integration,
            component_type="conscious"
        )
        self.tab_widget.addTab(self.conscious_tab, "Conscious Mirror")
        
        self.neural_tab = LanguageComponentPanel(
            integration=self.integration,
            component_type="neural"
        )
        self.tab_widget.addTab(self.neural_tab, "Neural Linguistic")
        
        self.recursive_tab = LanguageComponentPanel(
            integration=self.integration,
            component_type="recursive"
        )
        self.tab_widget.addTab(self.recursive_tab, "Recursive Patterns")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
    
    def _on_llm_slider_changed(self, value):
        """Handle LLM slider value change"""
        llm_weight = value / 100.0
        self.llm_label.setText(f"LLM Weight: {llm_weight:.1f}")
    
    def _on_nn_slider_changed(self, value):
        """Handle NN slider value change"""
        nn_weight = value / 100.0
        self.nn_label.setText(f"Neural Network Weight: {nn_weight:.1f}")
    
    def _on_update_weights(self):
        """Handle update weights button click"""
        if not self.integration:
            self.status_label.setText("Error: No language integration available")
            return
            
        llm_weight = self.llm_slider.value() / 100.0
        nn_weight = self.nn_slider.value() / 100.0
        
        # Update weights
        self.integration.set_llm_weight(llm_weight)
        self.integration.set_nn_weight(nn_weight)
        
        self.status_label.setText(f"Weights updated: LLM={llm_weight:.1f}, NN={nn_weight:.1f}")
    
    def _on_llm_weight_changed(self, weight):
        """Handle LLM weight changed signal"""
        self.llm_slider.setValue(int(weight * 100))
        self.llm_label.setText(f"LLM Weight: {weight:.1f}")
    
    def _on_nn_weight_changed(self, weight):
        """Handle NN weight changed signal"""
        self.nn_slider.setValue(int(weight * 100))
        self.nn_label.setText(f"Neural Network Weight: {weight:.1f}")
    
    def _on_status_updated(self, status):
        """Handle system status updated signal"""
        status_text = "Status: "
        
        if status.get("mock_mode", False):
            status_text += "MOCK MODE | "
        
        if status.get("central_node_active", False):
            status_text += "Central Node Active | "
        else:
            status_text += "Central Node Inactive | "
        
        status_text += f"LLM: {status.get('llm_weight', 0.0):.1f} | "
        status_text += f"NN: {status.get('nn_weight', 0.0):.1f}"
        
        if "consciousness_level" in status:
            status_text += f" | Consciousness: {status['consciousness_level']:.2f}"
        
        self.status_label.setText(status_text)


def get_language_pyside6_integration(
    data_dir="data",
    llm_weight=0.5,
    nn_weight=0.5,
    config=None,
    mock_mode=False
):
    """
    Get an initialized Enhanced Language Integration instance
    
    This is a convenience function for getting a pre-configured
    instance of the EnhancedLanguageIntegration class.
    
    Args:
        data_dir: Directory for data storage
        llm_weight: Initial LLM weight (0.0-1.0)
        nn_weight: Initial NN weight (0.0-1.0)
        config: Additional configuration options
        mock_mode: Whether to use mock mode
        
    Returns:
        EnhancedLanguageIntegration instance
    """
    return EnhancedLanguageIntegration(
        data_dir=data_dir,
        llm_weight=llm_weight,
        nn_weight=nn_weight,
        config=config,
        mock_mode=mock_mode
    )


def create_language_ui_panel(parent=None, integration=None):
    """
    Create a language UI panel with the specified integration
    
    This is a convenience function for creating a pre-configured
    CentralLanguagePanel instance.
    
    Args:
        parent: Parent widget
        integration: EnhancedLanguageIntegration instance (will create if None)
        
    Returns:
        CentralLanguagePanel instance
    """
    if integration is None:
        integration = get_language_pyside6_integration()
    
    return CentralLanguagePanel(parent=parent, integration=integration) 