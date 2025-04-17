#!/usr/bin/env python3
"""
PySide6 Adapter for Enhanced Language System

This module provides adapter classes and signals to make the Enhanced Language System
components work well in a PySide6 GUI application. It handles:
1. Non-blocking processing using worker threads
2. Signal-based communication for UI updates
3. Data visualization adapters
4. Common interfaces for all language components
"""

import os
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyside6_adapter")

try:
    from PySide6.QtCore import QObject, Signal, Slot, QThread, QTimer
    HAS_PYSIDE6 = True
    logger.info("Successfully imported PySide6")
except ImportError:
    HAS_PYSIDE6 = False
    logger.error("PySide6 not found. The adapter will be limited to non-GUI functionality.")
    # Create dummy classes and signals for non-GUI environments
    class QObject:
        pass
    
    class Signal:
        def __init__(self, *args):
            self.callbacks = []
        
        def connect(self, callback):
            self.callbacks.append(callback)
        
        def emit(self, *args):
            for callback in self.callbacks:
                callback(*args)
    
    class Slot:
        def __init__(self, *args):
            pass
        
        def __call__(self, func):
            return func
    
    class QThread:
        def start(self):
            pass
        
        def quit(self):
            pass
    
    class QTimer:
        def __init__(self):
            pass
        
        def start(self, interval):
            pass
        
        def stop(self):
            pass


# Import components conditionally
try:
    from .language_memory import LanguageMemory
    HAS_LANGUAGE_MEMORY = True
except ImportError:
    HAS_LANGUAGE_MEMORY = False
    logger.warning("Language Memory component not found")

try:
    from .neural_linguistic_processor import NeuralLinguisticProcessor
    HAS_NEURAL_PROCESSOR = True
except ImportError:
    HAS_NEURAL_PROCESSOR = False
    logger.warning("Neural Linguistic Processor component not found")

try:
    from .conscious_mirror_language import ConsciousMirrorLanguage
    HAS_CONSCIOUS_MIRROR = True
except ImportError:
    HAS_CONSCIOUS_MIRROR = False
    logger.warning("Conscious Mirror Language component not found")

try:
    from .central_language_node import CentralLanguageNode
    HAS_CENTRAL_NODE = True
except ImportError:
    HAS_CENTRAL_NODE = False
    logger.warning("Central Language Node component not found")


class LanguageWorker(QThread):
    """Worker thread for language processing tasks"""
    task_complete = Signal(dict)
    task_error = Signal(str)
    
    def __init__(self, target_function, args=None, kwargs=None):
        """
        Initialize the worker
        
        Args:
            target_function: Function to execute in the thread
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
        """
        super().__init__()
        self.target_function = target_function
        self.args = args or []
        self.kwargs = kwargs or {}
    
    def run(self):
        """Execute the target function"""
        try:
            result = self.target_function(*self.args, **self.kwargs)
            self.task_complete.emit({"result": result, "timestamp": datetime.now().isoformat()})
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            self.task_error.emit(str(e))


class LanguageMemoryAdapter(QObject):
    """PySide6 adapter for the Language Memory component"""
    # Signals
    memory_stored = Signal(dict)  # Emitted when a memory is stored
    associations_retrieved = Signal(dict)  # Emitted when associations are retrieved
    memory_stats_updated = Signal(dict)  # Emitted when statistics are updated
    llm_weight_changed = Signal(float)  # Emitted when LLM weight is changed
    error_occurred = Signal(str)  # Emitted on errors
    
    def __init__(self, data_dir="data/memory/language_memory", llm_weight=0.5):
        """
        Initialize the Language Memory adapter
        
        Args:
            data_dir: Directory for memory storage
            llm_weight: Initial LLM weight (0.0-1.0)
        """
        super().__init__()
        self.data_dir = data_dir
        
        # Create component instance if available
        if HAS_LANGUAGE_MEMORY:
            try:
                # Try to initialize with data_dir first
                try:
                    self.memory = LanguageMemory(data_dir=data_dir, llm_weight=llm_weight)
                except TypeError:
                    # If that fails, try without data_dir
                    self.memory = LanguageMemory(llm_weight=llm_weight)
                
                self.available = True
                logger.info(f"Language Memory adapter initialized with LLM weight {llm_weight}")
            except Exception as e:
                self.memory = None
                self.available = False
                logger.error(f"Failed to initialize Language Memory: {str(e)}")
        else:
            self.memory = None
            self.available = False
        
        # For background stats updates
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(10000)  # Update every 10 seconds
    
    def store_word_association(self, word, associated_word, strength=0.8, metadata=None):
        """Store word association in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Language Memory component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.memory.store_word_association,
            kwargs={
                "word": word,
                "associated_word": associated_word,
                "strength": strength,
                "metadata": metadata or {}
            }
        )
        worker.task_complete.connect(self._on_word_association_stored)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def remember_word_associations(self, word, min_strength=0.1, max_results=10):
        """Retrieve word associations in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Language Memory component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.memory.remember_word_associations,
            kwargs={
                "word": word,
                "min_strength": min_strength,
                "max_results": max_results
            }
        )
        worker.task_complete.connect(self._on_associations_retrieved)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def store_sentence(self, sentence, metadata=None):
        """Store a sentence in memory in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Language Memory component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.memory.store_sentence,
            kwargs={
                "sentence": sentence,
                "metadata": metadata or {}
            }
        )
        worker.task_complete.connect(self._on_sentence_stored)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def adjust_llm_weight(self, weight):
        """Adjust the LLM weight"""
        if not self.available:
            self.error_occurred.emit("Language Memory component not available")
            return
        
        try:
            self.memory.adjust_llm_weight(weight)
            self.llm_weight_changed.emit(weight)
            logger.info(f"Language Memory LLM weight adjusted to {weight}")
        except Exception as e:
            self.error_occurred.emit(f"Failed to adjust LLM weight: {str(e)}")
    
    @Slot()
    def update_stats(self):
        """Update the memory statistics in a non-blocking way"""
        if not self.available:
            return
        
        try:
            # Check for different method names that might exist
            if hasattr(self.memory, 'get_memory_stats'):
                stats = self.memory.get_memory_stats()
                self.memory_stats_updated.emit(stats)
            elif hasattr(self.memory, 'get_stats'):
                stats = self.memory.get_stats()
                self.memory_stats_updated.emit(stats)
            else:
                # Missing method - create dummy stats to avoid crashes
                stats = {
                    "word_associations": len(getattr(self.memory, 'word_associations', [])),
                    "sentences": len(getattr(self.memory, 'sentences', [])),
                    "llm_weight": self.memory.llm_weight
                }
                self.memory_stats_updated.emit(stats)
        except Exception as e:
            logger.error(f"Error updating memory stats: {str(e)}")
    
    @Slot(dict)
    def _on_word_association_stored(self, result):
        """Handle word association storage completion"""
        self.memory_stored.emit({
            "type": "word_association",
            "result": result.get("result", {}),
            "timestamp": result.get("timestamp")
        })
    
    @Slot(dict)
    def _on_sentence_stored(self, result):
        """Handle sentence storage completion"""
        self.memory_stored.emit({
            "type": "sentence",
            "result": result.get("result", {}),
            "timestamp": result.get("timestamp")
        })
    
    @Slot(dict)
    def _on_associations_retrieved(self, result):
        """Handle association retrieval completion"""
        self.associations_retrieved.emit({
            "associations": result.get("result", []),
            "timestamp": result.get("timestamp")
        })
    
    def get_llm_integration_stats(self):
        """Get LLM integration statistics"""
        if not self.available:
            return {"available": False}
        
        try:
            return self.memory.get_llm_integration_stats()
        except Exception as e:
            logger.error(f"Error getting LLM integration stats: {str(e)}")
            return {"error": str(e)}


class NeuralLinguisticProcessorAdapter(QObject):
    """PySide6 adapter for the Neural Linguistic Processor component"""
    # Signals
    processing_complete = Signal(dict)  # Emitted when text processing is complete
    patterns_detected = Signal(dict)  # Emitted when patterns are detected
    semantic_network_updated = Signal(dict)  # Emitted when the semantic network is updated
    stats_updated = Signal(dict)  # Emitted when statistics are updated
    llm_weight_changed = Signal(float)  # Emitted when LLM weight is changed
    error_occurred = Signal(str)  # Emitted on errors
    
    def __init__(self, data_dir="data/neural_linguistic", llm_weight=0.5):
        """
        Initialize the Neural Linguistic Processor adapter
        
        Args:
            data_dir: Directory for processor data
            llm_weight: Initial LLM weight (0.0-1.0)
        """
        super().__init__()
        self.data_dir = data_dir
        
        # Create component instance if available
        if HAS_NEURAL_PROCESSOR:
            try:
                self.processor = NeuralLinguisticProcessor(llm_weight=llm_weight)
                self.available = True
                logger.info(f"Neural Linguistic Processor adapter initialized with LLM weight {llm_weight}")
            except Exception as e:
                self.processor = None
                self.available = False
                logger.error(f"Failed to initialize Neural Linguistic Processor: {str(e)}")
        else:
            self.processor = None
            self.available = False
        
        # For background stats updates
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(10000)  # Update every 10 seconds
    
    def process_text(self, text):
        """Process text in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Neural Linguistic Processor component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.processor.process_text,
            kwargs={"text": text}
        )
        worker.task_complete.connect(self._on_processing_complete)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def get_patterns(self, text=None, pattern_type=None):
        """Get detected patterns in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Neural Linguistic Processor component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.processor.get_patterns,
            kwargs={
                "text": text,
                "pattern_type": pattern_type
            }
        )
        worker.task_complete.connect(self._on_patterns_retrieved)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def get_semantic_network(self, word, depth=2):
        """Get semantic network for a word in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Neural Linguistic Processor component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.processor.get_semantic_network,
            kwargs={
                "word": word,
                "depth": depth
            }
        )
        worker.task_complete.connect(self._on_semantic_network_retrieved)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def adjust_llm_weight(self, weight):
        """Adjust the LLM weight"""
        if not self.available:
            self.error_occurred.emit("Neural Linguistic Processor component not available")
            return
        
        try:
            self.processor.adjust_llm_weight(weight)
            self.llm_weight_changed.emit(weight)
            logger.info(f"Neural Linguistic Processor LLM weight adjusted to {weight}")
        except Exception as e:
            self.error_occurred.emit(f"Failed to adjust LLM weight: {str(e)}")
    
    @Slot()
    def update_stats(self):
        """Update the processor statistics in a non-blocking way"""
        if not self.available:
            return
        
        try:
            # Check if get_metrics exists
            if hasattr(self.processor, 'get_metrics'):
                stats = self.processor.get_metrics()
                self.stats_updated.emit(stats)
            else:
                # Missing method - create dummy stats to avoid crashes
                stats = {
                    "patterns_detected": 0,
                    "semantic_nodes": 0,
                    "semantic_connections": 0,
                    "processing_time_ms": 0,
                    "llm_weight": self.processor.llm_weight
                }
                self.stats_updated.emit(stats)
        except Exception as e:
            logger.error(f"Error updating processor stats: {str(e)}")
    
    @Slot(dict)
    def _on_processing_complete(self, result):
        """Handle text processing completion"""
        self.processing_complete.emit({
            "score": result.get("result", {}).get("score", 0),
            "word_count": result.get("result", {}).get("word_count", 0),
            "unique_words": result.get("result", {}).get("unique_words", 0),
            "patterns": result.get("result", {}).get("patterns", []),
            "timestamp": result.get("timestamp")
        })
    
    @Slot(dict)
    def _on_patterns_retrieved(self, result):
        """Handle pattern retrieval completion"""
        self.patterns_detected.emit({
            "patterns": result.get("result", []),
            "timestamp": result.get("timestamp")
        })
    
    @Slot(dict)
    def _on_semantic_network_retrieved(self, result):
        """Handle semantic network retrieval completion"""
        self.semantic_network_updated.emit({
            "network": result.get("result", {}),
            "timestamp": result.get("timestamp")
        })


class ConsciousMirrorLanguageAdapter(QObject):
    """PySide6 adapter for the Conscious Mirror Language component"""
    # Signals
    processing_complete = Signal(dict)  # Emitted when text processing is complete
    consciousness_level_updated = Signal(float)  # Emitted when consciousness level changes
    metrics_updated = Signal(dict)  # Emitted when consciousness metrics are updated
    llm_weight_changed = Signal(float)  # Emitted when LLM weight is changed
    error_occurred = Signal(str)  # Emitted on errors
    
    def __init__(self, data_dir="data/v10", llm_weight=0.5):
        """
        Initialize the Conscious Mirror Language adapter
        
        Args:
            data_dir: Directory for consciousness data
            llm_weight: Initial LLM weight (0.0-1.0)
        """
        super().__init__()
        self.data_dir = data_dir
        
        # Create component instance if available
        if HAS_CONSCIOUS_MIRROR:
            try:
                self.mirror = ConsciousMirrorLanguage(data_dir=data_dir, llm_weight=llm_weight)
                self.available = True
                logger.info(f"Conscious Mirror Language adapter initialized with LLM weight {llm_weight}")
            except Exception as e:
                self.mirror = None
                self.available = False
                logger.error(f"Failed to initialize Conscious Mirror Language: {str(e)}")
        else:
            self.mirror = None
            self.available = False
        
        # For background metrics updates
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(5000)  # Update every 5 seconds
    
    def process_text(self, text):
        """Process text in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Conscious Mirror Language component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.mirror.process_text,
            kwargs={"text": text}
        )
        worker.task_complete.connect(self._on_processing_complete)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def adjust_llm_weight(self, weight):
        """Adjust the LLM weight"""
        if not self.available:
            self.error_occurred.emit("Conscious Mirror Language component not available")
            return
        
        try:
            self.mirror.adjust_llm_weight(weight)
            self.llm_weight_changed.emit(weight)
            logger.info(f"Conscious Mirror Language LLM weight adjusted to {weight}")
        except Exception as e:
            self.error_occurred.emit(f"Failed to adjust LLM weight: {str(e)}")
    
    @Slot()
    def update_metrics(self):
        """Update the consciousness metrics in a non-blocking way"""
        if not self.available:
            return
        
        try:
            # Check for different method names that might exist
            if hasattr(self.mirror, 'get_metrics'):
                metrics = self.mirror.get_metrics()
                self.metrics_updated.emit(metrics)
            elif hasattr(self.mirror, 'get_consciousness_metrics'):
                metrics = self.mirror.get_consciousness_metrics()
                self.metrics_updated.emit(metrics)
            else:
                # Missing method - create dummy metrics to avoid crashes
                metrics = {
                    "consciousness_level": getattr(self.mirror, 'consciousness_level', 0.5),
                    "continuity": getattr(self.mirror, 'continuity', 0.7),
                    "llm_weight": self.mirror.llm_weight
                }
                self.metrics_updated.emit(metrics)
            
            # Also emit consciousness level for simple tracking
            consciousness_level = metrics.get("consciousness_level", 0)
            self.consciousness_level_updated.emit(consciousness_level)
        except Exception as e:
            logger.error(f"Error updating consciousness metrics: {str(e)}")
    
    @Slot(dict)
    def _on_processing_complete(self, result):
        """Handle text processing completion"""
        if "result" in result:
            self.processing_complete.emit(result["result"])
            
            # Also update consciousness level if available
            consciousness_level = result["result"].get("consciousness_level", None)
            if consciousness_level is not None:
                self.consciousness_level_updated.emit(consciousness_level)


class CentralLanguageNodeAdapter(QObject):
    """PySide6 adapter for the Central Language Node"""
    # Signals
    processing_complete = Signal(dict)  # Emitted when text processing is complete
    system_status_updated = Signal(dict)  # Emitted when system status is updated
    cross_mappings_updated = Signal(dict)  # Emitted when cross mappings are updated
    llm_weight_changed = Signal(float)  # Emitted when LLM weight is changed
    component_status_changed = Signal(dict)  # Emitted when component status changes
    error_occurred = Signal(str)  # Emitted on errors
    
    def __init__(self, data_dir="data/central_language", llm_weight=0.5,
                 language_memory_adapter=None,
                 neural_processor_adapter=None,
                 consciousness_adapter=None):
        """
        Initialize the Central Language Node adapter
        
        Args:
            data_dir: Directory for node data
            llm_weight: Initial LLM weight (0.0-1.0)
            language_memory_adapter: Optional Language Memory adapter
            neural_processor_adapter: Optional Neural Linguistic Processor adapter
            consciousness_adapter: Optional Conscious Mirror Language adapter
        """
        super().__init__()
        self.data_dir = data_dir
        
        # Store adapter references
        self.language_memory_adapter = language_memory_adapter
        self.neural_processor_adapter = neural_processor_adapter
        self.consciousness_adapter = consciousness_adapter
        
        # Create or extract component instances
        language_memory = None
        neural_processor = None
        consciousness = None
        
        if language_memory_adapter and language_memory_adapter.available:
            language_memory = language_memory_adapter.memory
        
        if neural_processor_adapter and neural_processor_adapter.available:
            neural_processor = neural_processor_adapter.processor
        
        if consciousness_adapter and consciousness_adapter.available:
            consciousness = consciousness_adapter.mirror
        
        # Create component instance if available
        if HAS_CENTRAL_NODE:
            try:
                self.node = CentralLanguageNode(
                    data_dir=data_dir,
                    llm_weight=llm_weight,
                    language_memory=language_memory,
                    neural_linguistic_processor=neural_processor,
                    conscious_mirror_language=consciousness
                )
                self.available = True
                logger.info(f"Central Language Node adapter initialized with LLM weight {llm_weight}")
            except Exception as e:
                self.node = None
                self.available = False
                logger.error(f"Failed to initialize Central Language Node: {str(e)}")
        else:
            self.node = None
            self.available = False
        
        # For background status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # Update every 5 seconds
        
        # Connect adapter signals if provided
        if language_memory_adapter:
            language_memory_adapter.llm_weight_changed.connect(self._on_component_llm_weight_changed)
        
        if neural_processor_adapter:
            neural_processor_adapter.llm_weight_changed.connect(self._on_component_llm_weight_changed)
        
        if consciousness_adapter:
            consciousness_adapter.llm_weight_changed.connect(self._on_component_llm_weight_changed)
    
    def process_text(self, text, use_consciousness=True, use_neural_linguistics=True):
        """Process text in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Central Language Node component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.node.process_text,
            kwargs={
                "text": text,
                "use_consciousness": use_consciousness,
                "use_neural_linguistics": use_neural_linguistics
            }
        )
        worker.task_complete.connect(self._on_processing_complete)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def process_with_consciousness(self, text):
        """Process text with focus on consciousness in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Central Language Node component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.node.process_with_consciousness,
            kwargs={"text": text}
        )
        worker.task_complete.connect(self._on_consciousness_processing_complete)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def process_with_neural_linguistics(self, text):
        """Process text with focus on neural linguistics in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Central Language Node component not available")
            return
        
        worker = LanguageWorker(
            target_function=self.node.process_with_neural_linguistics,
            kwargs={"text": text}
        )
        worker.task_complete.connect(self._on_neural_processing_complete)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def adjust_llm_weight(self, weight):
        """Adjust the LLM weight across all components"""
        if not self.available:
            self.error_occurred.emit("Central Language Node component not available")
            return
        
        try:
            self.node.adjust_llm_weight(weight)
            self.llm_weight_changed.emit(weight)
            logger.info(f"Central Language Node LLM weight adjusted to {weight}")
        except Exception as e:
            self.error_occurred.emit(f"Failed to adjust LLM weight: {str(e)}")
    
    @Slot()
    def update_status(self):
        """Update system status"""
        if not self.available:
            return
        
        try:
            status = self.node.get_system_status()
            self.system_status_updated.emit(status)
            
            # Also emit component status
            if "components_status" in status:
                self.component_status_changed.emit(status["components_status"])
            
            # Update cross mappings if available
            if hasattr(self.node, "cross_mappings"):
                self.cross_mappings_updated.emit(self.node.cross_mappings)
        except Exception as e:
            logger.error(f"Error updating system status: {str(e)}")
    
    @Slot(dict)
    def _on_processing_complete(self, result):
        """Handle text processing completion"""
        if "result" in result:
            self.processing_complete.emit(result["result"])
    
    @Slot(dict)
    def _on_consciousness_processing_complete(self, result):
        """Handle consciousness-focused processing completion"""
        if "result" in result:
            self.processing_complete.emit({
                "type": "consciousness",
                "result": result["result"],
                "timestamp": result.get("timestamp")
            })
    
    @Slot(dict)
    def _on_neural_processing_complete(self, result):
        """Handle neural-focused processing completion"""
        if "result" in result:
            self.processing_complete.emit({
                "type": "neural",
                "result": result["result"],
                "timestamp": result.get("timestamp")
            })
    
    @Slot(float)
    def _on_component_llm_weight_changed(self, weight):
        """Handle component LLM weight changes"""
        # Only update if there's a significant difference to avoid loops
        if self.available and hasattr(self.node, "llm_weight"):
            if abs(self.node.llm_weight - weight) > 0.01:
                logger.debug(f"Component LLM weight changed to {weight}, syncing central node")
                self.adjust_llm_weight(weight) 