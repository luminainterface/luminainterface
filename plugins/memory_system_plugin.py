"""
Memory System Plugin for V7 Template

Connects the V7 PySide6 template with the V7 Onsite Memory System.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QLineEdit, QPushButton, QLabel, 
    QComboBox, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QTabWidget, QCheckBox,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QIcon, QColor

# Import the plugin interface
try:
    from v7_pyside6_template import PluginInterface
except ImportError:
    # For development/testing
    class PluginInterface:
        def __init__(self, app_context):
            self.app_context = app_context

# Set up logging
logger = logging.getLogger("MemorySystemPlugin")

# Try to import the V7 memory system components
try:
    # Add src path to system path if needed
    if os.path.exists("src"):
        sys.path.insert(0, os.path.abspath("src"))
    
    # Import V7 memory system components - Fix import path
    from src.v7.memory import OnsiteMemory, MemoryAnalyzer
    MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory system import error: {e}")
    MEMORY_AVAILABLE = False
    
    # Define mock classes for development
    class OnsiteMemory:
        def __init__(self, storage_path="data/onsite_memory", mock_mode=True):
            self.storage_path = storage_path
            self.mock_mode = mock_mode
            self.memories = {}
            self.categories = ["conversation", "knowledge", "system", "user_preference"]
            
            # Create storage directory if it doesn't exist
            os.makedirs(storage_path, exist_ok=True)
            
            # Load existing memories if available
            self.load_memories()
        
        def load_memories(self):
            """Load memories from storage"""
            try:
                memory_file = os.path.join(self.storage_path, "memory_store.json")
                if os.path.exists(memory_file):
                    with open(memory_file, "r") as f:
                        self.memories = json.load(f)
                        logger.info(f"Loaded {len(self.memories)} memories from storage")
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
                # Initialize with empty dict if loading fails
                self.memories = {}
        
        def save_memories(self):
            """Save memories to storage"""
            try:
                memory_file = os.path.join(self.storage_path, "memory_store.json")
                with open(memory_file, "w") as f:
                    json.dump(self.memories, f, indent=2)
                logger.info(f"Saved {len(self.memories)} memories to storage")
                return True
            except Exception as e:
                logger.error(f"Error saving memories: {e}")
                return False
        
        def add_memory(self, content, category="conversation", metadata=None):
            """Add a new memory"""
            if not metadata:
                metadata = {}
            
            memory_id = f"mem_{int(time.time())}_{len(self.memories)}"
            timestamp = datetime.now().isoformat()
            
            memory = {
                "id": memory_id,
                "content": content,
                "category": category,
                "timestamp": timestamp,
                "last_accessed": timestamp,
                "access_count": 0,
                "importance": 0.5,
                "metadata": metadata
            }
            
            self.memories[memory_id] = memory
            return memory_id
        
        def get_memory(self, memory_id):
            """Retrieve a specific memory by ID"""
            if memory_id in self.memories:
                # Update access stats
                self.memories[memory_id]["access_count"] += 1
                self.memories[memory_id]["last_accessed"] = datetime.now().isoformat()
                return self.memories[memory_id]
            return None
        
        def search_memories(self, query, category=None, limit=10):
            """Search memories by simple text matching"""
            results = []
            
            for mem_id, memory in self.memories.items():
                if category and memory["category"] != category:
                    continue
                
                # Simple text search (replace with vector search in real implementation)
                if query.lower() in memory["content"].lower():
                    results.append(memory)
                    
                    # Update access stats
                    self.memories[mem_id]["access_count"] += 1
                    self.memories[mem_id]["last_accessed"] = datetime.now().isoformat()
                
                if len(results) >= limit:
                    break
            
            return results
        
        def get_memories_by_category(self, category, limit=50):
            """Get memories by category"""
            results = []
            
            for memory in self.memories.values():
                if memory["category"] == category:
                    results.append(memory)
                
                if len(results) >= limit:
                    break
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results
        
        def update_memory(self, memory_id, updates):
            """Update an existing memory"""
            if memory_id in self.memories:
                for key, value in updates.items():
                    if key != "id" and key != "timestamp":  # Don't allow changing ID or creation timestamp
                        self.memories[memory_id][key] = value
                return True
            return False
        
        def delete_memory(self, memory_id):
            """Delete a memory"""
            if memory_id in self.memories:
                del self.memories[memory_id]
                return True
            return False
        
        def get_stats(self):
            """Get memory system stats"""
            category_counts = {}
            for cat in self.categories:
                category_counts[cat] = 0
            
            for memory in self.memories.values():
                cat = memory["category"]
                if cat in category_counts:
                    category_counts[cat] += 1
            
            return {
                "total_memories": len(self.memories),
                "category_counts": category_counts,
                "storage_path": self.storage_path
            }
    
    class MemoryAnalyzer:
        def __init__(self, memory_system=None):
            self.memory_system = memory_system
        
        def analyze_conversation(self, conversation_text):
            """Analyze conversation and extract key points"""
            # Mock implementation
            return {
                "key_points": ["Sample key point 1", "Sample key point 2"],
                "sentiment": "positive",
                "topics": ["topic1", "topic2"]
            }
        
        def suggest_memories(self, context, limit=3):
            """Suggest relevant memories based on context"""
            # Mock implementation
            if not self.memory_system:
                return []
            
            # Get a sample of memories for demonstration
            all_memories = list(self.memory_system.memories.values())
            if not all_memories:
                return []
            
            # Return up to limit memories
            return all_memories[:min(limit, len(all_memories))]
        
        def compute_importance(self, memory):
            """Compute importance score for a memory"""
            # Mock implementation - in a real system, this would use ML/heuristics
            importance = 0.5  # Base importance
            
            # Adjust based on access count (more accessed = more important)
            access_boost = min(0.3, memory.get("access_count", 0) * 0.02)
            importance += access_boost
            
            # Adjust based on recency
            try:
                timestamp = datetime.fromisoformat(memory.get("timestamp", datetime.now().isoformat()))
                days_old = (datetime.now() - timestamp).days
                recency_boost = max(0, 0.2 - (days_old * 0.01))
                importance += recency_boost
            except:
                pass
            
            return min(0.95, importance)

class MemoryBrowserWidget(QWidget):
    """Widget for browsing and searching memories"""
    
    memory_selected = Signal(dict)
    refresh_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_category = "all"
        self.memories = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Memory Browser")
        title.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 12, QFont.Bold)
        title.setFont(font)
        layout.addWidget(title)
        
        # Search area
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search memories...")
        self.search_input.returnPressed.connect(self.search_memories)
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_memories)
        search_layout.addWidget(self.search_button)
        
        layout.addLayout(search_layout)
        
        # Category filter
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))
        
        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories", "all")
        self.category_combo.addItem("Conversations", "conversation")
        self.category_combo.addItem("Knowledge", "knowledge")
        self.category_combo.addItem("System", "system")
        self.category_combo.addItem("User Preferences", "user_preference")
        self.category_combo.currentIndexChanged.connect(self.filter_by_category)
        category_layout.addWidget(self.category_combo)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_memories)
        category_layout.addWidget(self.refresh_button)
        
        layout.addLayout(category_layout)
        
        # Memory list (table)
        self.memory_table = QTableWidget(0, 4)  # rows, columns
        self.memory_table.setHorizontalHeaderLabels(["Time", "Category", "Importance", "Content"])
        self.memory_table.setColumnWidth(0, 150)  # Time column
        self.memory_table.setColumnWidth(1, 100)  # Category column
        self.memory_table.setColumnWidth(2, 80)   # Importance column
        self.memory_table.setColumnWidth(3, 400)  # Content column
        self.memory_table.cellClicked.connect(self.memory_selected_in_table)
        layout.addWidget(self.memory_table)
        
        # Memory detail area
        self.memory_detail = QTextEdit()
        self.memory_detail.setReadOnly(True)
        self.memory_detail.setMinimumHeight(100)
        layout.addWidget(self.memory_detail)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.update_button = QPushButton("Update Selected")
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self.update_memory)
        button_layout.addWidget(self.update_button)
        
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_memory)
        button_layout.addWidget(self.delete_button)
        
        layout.addLayout(button_layout)
    
    def refresh_memories(self):
        """Request memory refresh"""
        self.refresh_requested.emit()
    
    def search_memories(self):
        """Search memories based on query"""
        query = self.search_input.text()
        self.current_category = self.category_combo.currentData()
        
        # Request search through the refresh signal
        # The plugin will handle the actual search and call update_memories
        self.refresh_requested.emit()
    
    def filter_by_category(self):
        """Filter memories by selected category"""
        self.current_category = self.category_combo.currentData()
        self.refresh_requested.emit()
    
    def update_memories(self, memories):
        """Update the memory table with new data"""
        self.memories = memories
        
        # Clear table
        self.memory_table.setRowCount(0)
        self.memory_detail.clear()
        self.update_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        
        # Add memories to table
        for i, memory in enumerate(memories):
            self.memory_table.insertRow(i)
            
            # Format timestamp
            try:
                timestamp = datetime.fromisoformat(memory.get("timestamp", ""))
                time_str = timestamp.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = memory.get("timestamp", "Unknown")
            
            # Add data to table
            self.memory_table.setItem(i, 0, QTableWidgetItem(time_str))
            self.memory_table.setItem(i, 1, QTableWidgetItem(memory.get("category", "")))
            
            importance_item = QTableWidgetItem(f"{memory.get('importance', 0.0):.2f}")
            importance_item.setTextAlignment(Qt.AlignCenter)
            self.memory_table.setItem(i, 2, importance_item)
            
            # Truncate content if too long
            content = memory.get("content", "")
            if len(content) > 100:
                content = content[:97] + "..."
            self.memory_table.setItem(i, 3, QTableWidgetItem(content))
        
        # Resize table
        self.memory_table.resizeRowsToContents()
    
    def memory_selected_in_table(self, row, column):
        """Handle memory selection in table"""
        if row < 0 or row >= len(self.memories):
            return
        
        # Get selected memory
        memory = self.memories[row]
        
        # Display memory details
        detail_text = f"<h3>Memory Details</h3>"
        detail_text += f"<p><b>ID:</b> {memory.get('id', '')}</p>"
        detail_text += f"<p><b>Category:</b> {memory.get('category', '')}</p>"
        detail_text += f"<p><b>Created:</b> {memory.get('timestamp', '')}</p>"
        detail_text += f"<p><b>Last Accessed:</b> {memory.get('last_accessed', '')}</p>"
        detail_text += f"<p><b>Access Count:</b> {memory.get('access_count', 0)}</p>"
        detail_text += f"<p><b>Importance:</b> {memory.get('importance', 0.0):.2f}</p>"
        detail_text += f"<p><b>Content:</b></p>"
        detail_text += f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{memory.get('content', '')}</div>"
        
        # Add metadata if available
        if memory.get("metadata"):
            detail_text += f"<p><b>Metadata:</b></p>"
            detail_text += "<ul>"
            for key, value in memory.get("metadata", {}).items():
                detail_text += f"<li><b>{key}:</b> {value}</li>"
            detail_text += "</ul>"
        
        self.memory_detail.setHtml(detail_text)
        
        # Enable control buttons
        self.update_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        
        # Emit signal
        self.memory_selected.emit(memory)
    
    def update_memory(self):
        """Open dialog to update selected memory"""
        row = self.memory_table.currentRow()
        if row < 0 or row >= len(self.memories):
            return
        
        # TODO: Implement memory update dialog
        QMessageBox.information(self, "Update Memory", "Memory update functionality not yet implemented")
    
    def delete_memory(self):
        """Delete selected memory"""
        row = self.memory_table.currentRow()
        if row < 0 or row >= len(self.memories):
            return
        
        memory_id = self.memories[row].get("id")
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self, 
            "Confirm Deletion", 
            "Are you sure you want to delete this memory? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Emit signal to delete memory
            # The plugin will handle the actual deletion
            self.memory_selected.emit({"id": memory_id, "action": "delete"})

class MemoryInputWidget(QWidget):
    """Widget for adding new memories"""
    
    memory_added = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Add New Memory")
        title.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 12, QFont.Bold)
        title.setFont(font)
        layout.addWidget(title)
        
        # Memory input form
        form_layout = QVBoxLayout()
        
        # Category selector
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))
        
        self.category_combo = QComboBox()
        self.category_combo.addItem("Conversation", "conversation")
        self.category_combo.addItem("Knowledge", "knowledge")
        self.category_combo.addItem("System", "system")
        self.category_combo.addItem("User Preference", "user_preference")
        category_layout.addWidget(self.category_combo)
        
        form_layout.addLayout(category_layout)
        
        # Memory content
        form_layout.addWidget(QLabel("Memory Content:"))
        
        self.content_input = QTextEdit()
        self.content_input.setMinimumHeight(100)
        self.content_input.setPlaceholderText("Enter memory content here...")
        form_layout.addWidget(self.content_input)
        
        # Importance slider
        importance_layout = QHBoxLayout()
        importance_layout.addWidget(QLabel("Importance:"))
        
        self.importance_slider = QSlider(Qt.Horizontal)
        self.importance_slider.setRange(0, 100)
        self.importance_slider.setValue(50)
        self.importance_slider.valueChanged.connect(self.update_importance_label)
        importance_layout.addWidget(self.importance_slider)
        
        self.importance_label = QLabel("0.50")
        importance_layout.addWidget(self.importance_label)
        
        form_layout.addLayout(importance_layout)
        
        # Add metadata checkbox
        self.metadata_checkbox = QCheckBox("Add Metadata")
        self.metadata_checkbox.stateChanged.connect(self.toggle_metadata_input)
        form_layout.addWidget(self.metadata_checkbox)
        
        # Metadata input
        self.metadata_input = QTextEdit()
        self.metadata_input.setPlaceholderText("Enter metadata as key:value pairs, one per line")
        self.metadata_input.setMaximumHeight(80)
        self.metadata_input.setVisible(False)
        form_layout.addWidget(self.metadata_input)
        
        layout.addLayout(form_layout)
        
        # Add button
        self.add_button = QPushButton("Add Memory")
        self.add_button.clicked.connect(self.add_memory)
        layout.addWidget(self.add_button)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Add spacer
        layout.addStretch(1)
    
    def update_importance_label(self):
        """Update importance label based on slider value"""
        importance = self.importance_slider.value() / 100.0
        self.importance_label.setText(f"{importance:.2f}")
    
    def toggle_metadata_input(self, state):
        """Toggle metadata input visibility"""
        self.metadata_input.setVisible(bool(state))
    
    def add_memory(self):
        """Add a new memory"""
        category = self.category_combo.currentData()
        content = self.content_input.text() if hasattr(self.content_input, "text") else self.content_input.toPlainText()
        importance = self.importance_slider.value() / 100.0
        
        if not content:
            self.status_label.setText("Error: Memory content cannot be empty")
            return
        
        # Parse metadata
        metadata = {}
        if self.metadata_checkbox.isChecked():
            metadata_text = self.metadata_input.toPlainText()
            for line in metadata_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
        
        # Create memory object
        memory = {
            "content": content,
            "category": category,
            "importance": importance,
            "metadata": metadata
        }
        
        # Emit signal
        self.memory_added.emit(memory)
        
        # Clear form
        self.content_input.clear()
        self.importance_slider.setValue(50)
        self.metadata_input.clear()
        self.metadata_checkbox.setChecked(False)
        
        # Update status
        self.status_label.setText("Memory added successfully")
        QTimer.singleShot(3000, lambda: self.status_label.setText(""))

class MemoryAnalyticsWidget(QWidget):
    """Widget for memory system analytics and statistics"""
    
    refresh_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stats = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Memory System Analytics")
        title.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 12, QFont.Bold)
        title.setFont(font)
        layout.addWidget(title)
        
        # Statistics
        self.stats_label = QLabel("Loading statistics...")
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)
        
        # Category breakdown
        layout.addWidget(QLabel("Memory Categories:"))
        
        self.category_table = QTableWidget(0, 2)  # rows, columns
        self.category_table.setHorizontalHeaderLabels(["Category", "Count"])
        self.category_table.setColumnWidth(0, 150)
        self.category_table.setColumnWidth(1, 100)
        layout.addWidget(self.category_table)
        
        # System status
        self.system_status = QLabel("System Status: Unknown")
        layout.addWidget(self.system_status)
        
        # Memory path
        self.memory_path = QLabel("Storage Path: Unknown")
        layout.addWidget(self.memory_path)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh Stats")
        self.refresh_button.clicked.connect(self.refresh_stats)
        control_layout.addWidget(self.refresh_button)
        
        self.backup_button = QPushButton("Backup Memories")
        self.backup_button.clicked.connect(self.backup_memories)
        control_layout.addWidget(self.backup_button)
        
        layout.addLayout(control_layout)
        
        # Set up auto-refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_stats)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
    
    def refresh_stats(self):
        """Request statistics refresh"""
        self.refresh_requested.emit()
    
    def update_stats(self, stats):
        """Update the statistics display"""
        self.stats = stats
        
        # Update total count
        total = stats.get("total_memories", 0)
        self.stats_label.setText(f"Total Memories: {total}")
        
        # Update category table
        category_counts = stats.get("category_counts", {})
        self.category_table.setRowCount(len(category_counts))
        
        for i, (category, count) in enumerate(category_counts.items()):
            # Category name
            self.category_table.setItem(i, 0, QTableWidgetItem(category))
            
            # Count
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.category_table.setItem(i, 1, count_item)
        
        # Update system status
        self.system_status.setText(f"System Status: Active")
        
        # Update memory path
        path = stats.get("storage_path", "Unknown")
        self.memory_path.setText(f"Storage Path: {path}")
    
    def backup_memories(self):
        """Backup memory store to a user-selected location"""
        # Open file dialog
        backup_path, _ = QFileDialog.getSaveFileName(
            self,
            "Backup Memories",
            "memory_backup.json",
            "JSON Files (*.json)"
        )
        
        if not backup_path:
            return
        
        # Emit signal to request backup
        self.refresh_requested.emit()
        
        # The actual backup would be handled by the plugin
        # For now, show a message
        QMessageBox.information(
            self,
            "Backup",
            f"Memory backup functionality not fully implemented. Would save to: {backup_path}"
        )

class Plugin(PluginInterface):
    """
    Memory System Plugin
    
    Connects the V7 PySide6 template with the V7 Onsite Memory System.
    """
    
    def __init__(self, app_context):
        super().__init__(app_context)
        self.name = "Memory System"
        self.version = "1.0.0"
        self.author = "LUMINA"
        self.dependencies = []
        
        # Integration instances
        self.memory_system = None
        self.memory_analyzer = None
        
        # Component status
        self.status = {
            "initialized": False,
            "memory_count": 0
        }
        
        # Current search parameters
        self.search_params = {
            "query": "",
            "category": "all"
        }
        
        # Setup UI components
        self.setup_ui()
    
    def setup_ui(self):
        """Set up UI components for this plugin"""
        # Memory browser widget
        self.memory_browser = MemoryBrowserWidget()
        self.memory_browser.memory_selected.connect(self.handle_memory_selection)
        self.memory_browser.refresh_requested.connect(self.refresh_memories)
        
        # Create browser dock widget
        self.browser_dock = QDockWidget("Memory Browser")
        self.browser_dock.setWidget(self.memory_browser)
        
        # Memory input widget
        self.memory_input = MemoryInputWidget()
        self.memory_input.memory_added.connect(self.add_memory)
        
        # Create input dock widget
        self.input_dock = QDockWidget("Add Memory")
        self.input_dock.setWidget(self.memory_input)
        
        # Memory analytics widget
        self.memory_analytics = MemoryAnalyticsWidget()
        self.memory_analytics.refresh_requested.connect(self.refresh_stats)
        
        # Create analytics dock widget
        self.analytics_dock = QDockWidget("Memory Analytics")
        self.analytics_dock.setWidget(self.memory_analytics)
        
        # Combined memory tab widget
        self.memory_tab_widget = QTabWidget()
        self.memory_tab_widget.addTab(self.memory_browser, "Browse")
        self.memory_tab_widget.addTab(self.memory_input, "Add")
        self.memory_tab_widget.addTab(self.memory_analytics, "Analytics")
    
    def initialize(self) -> bool:
        """Initialize the memory system integration"""
        try:
            # Initialize memory system with default path
            storage_path = os.path.join("data", "onsite_memory")
            os.makedirs(storage_path, exist_ok=True)
            
            self.memory_system = OnsiteMemory(
                storage_path=storage_path,
                mock_mode=not MEMORY_AVAILABLE
            )
            
            # Initialize memory analyzer
            self.memory_analyzer = MemoryAnalyzer(memory_system=self.memory_system)
            
            # Update status
            self.status["initialized"] = True
            self.status["memory_count"] = len(getattr(self.memory_system, "memories", {}))
            
            # Update UI with initial data
            self.refresh_memories()
            self.refresh_stats()
            
            # Register for events
            self.app_context["register_event_handler"]("mistral_response", self.handle_mistral_response)
            
            # Success
            logger.info("Memory system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
            return False
    
    def refresh_memories(self):
        """Refresh memories in browser based on current search/filter"""
        if not self.memory_system:
            return
        
        try:
            # Get current search parameters from browser
            query = self.memory_browser.search_input.text()
            category = self.memory_browser.current_category
            
            # Update search params
            self.search_params["query"] = query
            self.search_params["category"] = category
            
            # Get memories based on search parameters
            memories = []
            
            if query:
                # Search by query
                memories = self.memory_system.search_memories(
                    query=query,
                    category=None if category == "all" else category,
                    limit=50
                )
            else:
                # Just filter by category
                if category == "all":
                    # Get all memories (up to limit)
                    memories = list(self.memory_system.memories.values())[:50]
                else:
                    # Get by category
                    memories = self.memory_system.get_memories_by_category(
                        category=category,
                        limit=50
                    )
            
            # Update browser
            self.memory_browser.update_memories(memories)
            
        except Exception as e:
            logger.error(f"Error refreshing memories: {e}")
    
    def refresh_stats(self):
        """Refresh memory system statistics"""
        if not self.memory_system:
            return
        
        try:
            # Get system stats
            stats = self.memory_system.get_stats()
            
            # Update analytics widget
            self.memory_analytics.update_stats(stats)
            
            # Update status
            self.status["memory_count"] = stats.get("total_memories", 0)
            
        except Exception as e:
            logger.error(f"Error refreshing memory stats: {e}")
    
    def add_memory(self, memory_data):
        """Add a new memory to the system"""
        if not self.memory_system:
            return
        
        try:
            # Extract data
            content = memory_data.get("content", "")
            category = memory_data.get("category", "conversation")
            importance = memory_data.get("importance", 0.5)
            metadata = memory_data.get("metadata", {})
            
            # Add memory
            memory_id = self.memory_system.add_memory(
                content=content,
                category=category,
                metadata=metadata
            )
            
            # Update importance if needed
            if importance != 0.5:
                self.memory_system.update_memory(
                    memory_id=memory_id,
                    updates={"importance": importance}
                )
            
            # Save memories
            self.memory_system.save_memories()
            
            # Refresh UI
            self.refresh_memories()
            self.refresh_stats()
            
            # Notify user of success
            logger.info(f"Added new memory with ID: {memory_id}")
            
            # Trigger event
            self.app_context["trigger_event"](
                "memory_added", 
                {
                    "memory_id": memory_id,
                    "category": category
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
    
    def handle_memory_selection(self, memory_data):
        """Handle memory selection in browser"""
        if not self.memory_system:
            return
        
        # Check if this is a delete action
        if memory_data.get("action") == "delete":
            memory_id = memory_data.get("id")
            if memory_id:
                try:
                    # Delete memory
                    self.memory_system.delete_memory(memory_id)
                    
                    # Save changes
                    self.memory_system.save_memories()
                    
                    # Refresh UI
                    self.refresh_memories()
                    self.refresh_stats()
                    
                    logger.info(f"Deleted memory with ID: {memory_id}")
                except Exception as e:
                    logger.error(f"Error deleting memory: {e}")
            return
        
        # Regular memory selection - trigger event
        self.app_context["trigger_event"](
            "memory_selected", 
            memory_data
        )
    
    def handle_mistral_response(self, data):
        """Handle response event from Mistral plugin"""
        if not isinstance(data, dict) or not self.memory_system or not self.memory_analyzer:
            return
        
        # Process the response if needed
        query = data.get("query", "")
        response = data.get("response", "")
        
        if not query or not response:
            return
        
        try:
            # Analyze conversation
            analysis = self.memory_analyzer.analyze_conversation(
                f"User: {query}\nAssistant: {response}"
            )
            
            # Check if we should store this as a memory
            if analysis.get("key_points"):
                # Create a memory from this conversation
                memory_id = self.memory_system.add_memory(
                    content=f"User: {query}\nAssistant: {response}",
                    category="conversation",
                    metadata={
                        "sentiment": analysis.get("sentiment", "neutral"),
                        "topics": ", ".join(analysis.get("topics", [])),
                        "source": "conversation"
                    }
                )
                
                # Save memories
                self.memory_system.save_memories()
                
                # Refresh stats
                self.refresh_stats()
                
                logger.info(f"Added conversation memory with ID: {memory_id}")
        except Exception as e:
            logger.error(f"Error processing Mistral response for memory: {e}")
    
    def get_dock_widgets(self) -> List[QDockWidget]:
        """Return list of dock widgets provided by this plugin"""
        return [self.browser_dock, self.input_dock, self.analytics_dock]
    
    def get_tab_widgets(self) -> List[tuple]:
        """Return list of (name, widget) tuples for tab widgets"""
        return [
            ("Memory System", self.memory_tab_widget)
        ]
    
    def shutdown(self) -> None:
        """Clean shutdown of the plugin"""
        # Save memories
        if self.memory_system:
            try:
                self.memory_system.save_memories()
                logger.info("Memory system saved successfully during shutdown")
            except Exception as e:
                logger.error(f"Error saving memory system during shutdown: {e}")
        
        # Clean up
        self.memory_system = None
        self.memory_analyzer = None
        
        # Update status
        self.status["initialized"] = False 