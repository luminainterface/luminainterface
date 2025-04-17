#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Onsite Memory PySide6 Integration

This module integrates the onsite memory system with the PySide6 UI for Mistral,
providing components to display, search, and manage persistent memory.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# Add parent directory to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import PySide6
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem, 
    QTabWidget, QSplitter, QComboBox, QScrollArea, QDialog,
    QDialogButtonBox, QFormLayout, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QFont

# Import onsite memory system
from src.v7.onsite_memory import OnsiteMemory

# Configure logging
logger = logging.getLogger(__name__)

class MemoryEntryWidget(QWidget):
    """Widget for displaying a memory entry"""
    
    delete_requested = Signal(str)  # Signal to request deletion of an entry
    
    def __init__(self, topic: str, content: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.topic = topic
        self.content = content
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Create header row with topic and delete button
        header_layout = QHBoxLayout()
        
        # Topic label with bold font
        topic_label = QLabel(topic)
        font = QFont()
        font.setBold(True)
        topic_label.setFont(font)
        header_layout.addWidget(topic_label, 1)
        
        # Delete button
        delete_button = QPushButton("Delete")
        delete_button.setMaximumWidth(60)
        delete_button.clicked.connect(self._on_delete_clicked)
        header_layout.addWidget(delete_button)
        
        layout.addLayout(header_layout)
        
        # Content text
        content_text = QTextEdit()
        content_text.setReadOnly(True)
        content_text.setPlainText(content.get("content", ""))
        content_text.setMaximumHeight(100)
        layout.addWidget(content_text)
        
        # Metadata row (created date, sources)
        meta_layout = QHBoxLayout()
        
        # Created date
        created_label = QLabel(f"Created: {content.get('created', '').split('T')[0]}")
        created_label.setStyleSheet("color: #666;")
        meta_layout.addWidget(created_label)
        
        # Sources (if any)
        sources = content.get("sources", [])
        if sources:
            sources_label = QLabel(f"Sources: {', '.join(sources)}")
            sources_label.setStyleSheet("color: #666;")
            meta_layout.addWidget(sources_label)
        
        layout.addLayout(meta_layout)
        
        # Set styling
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
    
    def _on_delete_clicked(self):
        """Handle delete button click"""
        self.delete_requested.emit(self.topic)

class ConversationWidget(QWidget):
    """Widget for displaying a conversation entry"""
    
    def __init__(self, conversation: Dict[str, Any], parent=None):
        super().__init__(parent)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Timestamp
        timestamp = conversation.get("timestamp", "").split("T")[0]
        time_label = QLabel(f"Date: {timestamp}")
        time_label.setStyleSheet("color: #666;")
        layout.addWidget(time_label)
        
        # User message
        user_layout = QHBoxLayout()
        user_label = QLabel("User:")
        user_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        user_label.setFixedWidth(50)
        user_layout.addWidget(user_label)
        
        user_text = QLabel(conversation.get("user_message", ""))
        user_text.setWordWrap(True)
        user_layout.addWidget(user_text)
        layout.addLayout(user_layout)
        
        # Assistant message
        assistant_layout = QHBoxLayout()
        assistant_label = QLabel("Mistral:")
        assistant_label.setStyleSheet("font-weight: bold; color: #006633;")
        assistant_label.setFixedWidth(50)
        assistant_layout.addWidget(assistant_label)
        
        assistant_text = QLabel(conversation.get("assistant_response", ""))
        assistant_text.setWordWrap(True)
        assistant_layout.addWidget(assistant_text)
        layout.addLayout(assistant_layout)
        
        # Set styling
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 5px;
                border: 1px solid #ddd;
                margin-bottom: 5px;
            }
        """)

class AddKnowledgeDialog(QDialog):
    """Dialog for adding new knowledge to the memory"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Knowledge")
        self.resize(500, 300)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Topic input
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("Enter topic/keyword")
        form_layout.addRow("Topic:", self.topic_input)
        
        # Content input
        self.content_input = QTextEdit()
        self.content_input.setPlaceholderText("Enter knowledge content/definition")
        form_layout.addRow("Content:", self.content_input)
        
        # Source input
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Optional: Enter source of information")
        form_layout.addRow("Source:", self.source_input)
        
        layout.addLayout(form_layout)
        
        # Create buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_knowledge_data(self) -> Dict[str, Any]:
        """Get the entered knowledge data"""
        return {
            "topic": self.topic_input.text(),
            "content": self.content_input.toPlainText(),
            "source": self.source_input.text() if self.source_input.text() else None
        }

class OnsiteMemoryPanel(QWidget):
    """
    Onsite Memory Panel for the Mistral PySide6 application
    
    This widget provides a UI for viewing and interacting with the onsite memory system.
    """
    
    def __init__(self, memory: OnsiteMemory, parent=None):
        super().__init__(parent)
        self.memory = memory
        
        # Set up main layout
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Knowledge tab
        knowledge_tab = self._create_knowledge_tab()
        tabs.addTab(knowledge_tab, "Knowledge")
        
        # Conversations tab
        conversations_tab = self._create_conversations_tab()
        tabs.addTab(conversations_tab, "Conversations")
        
        # Preferences tab
        preferences_tab = self._create_preferences_tab()
        tabs.addTab(preferences_tab, "Preferences")
        
        layout.addWidget(tabs)
        
        # Create status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Memory Status: Ready")
        status_layout.addWidget(self.status_label)
        
        # Stats button
        stats_button = QPushButton("View Stats")
        stats_button.clicked.connect(self._show_stats)
        status_layout.addWidget(stats_button)
        
        layout.addLayout(status_layout)
        
        # Update display
        self.refresh_knowledge()
        self.refresh_conversations()
        
        logger.info("Onsite Memory Panel initialized")
    
    def _create_knowledge_tab(self) -> QWidget:
        """Create the knowledge tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Search bar
        search_layout = QHBoxLayout()
        
        self.knowledge_search = QLineEdit()
        self.knowledge_search.setPlaceholderText("Search knowledge...")
        self.knowledge_search.returnPressed.connect(self._search_knowledge)
        search_layout.addWidget(self.knowledge_search)
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._search_knowledge)
        search_layout.addWidget(search_button)
        
        layout.addLayout(search_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        add_button = QPushButton("Add Knowledge")
        add_button.clicked.connect(self._add_knowledge)
        action_layout.addWidget(add_button)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_knowledge)
        action_layout.addWidget(refresh_button)
        
        layout.addLayout(action_layout)
        
        # Knowledge list in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.knowledge_container = QWidget()
        self.knowledge_layout = QVBoxLayout(self.knowledge_container)
        self.knowledge_layout.setAlignment(Qt.AlignTop)
        self.knowledge_layout.setSpacing(10)
        
        scroll.setWidget(self.knowledge_container)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_conversations_tab(self) -> QWidget:
        """Create the conversations tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Search bar
        search_layout = QHBoxLayout()
        
        self.conversation_search = QLineEdit()
        self.conversation_search.setPlaceholderText("Search conversations...")
        self.conversation_search.returnPressed.connect(self._search_conversations)
        search_layout.addWidget(self.conversation_search)
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._search_conversations)
        search_layout.addWidget(search_button)
        
        layout.addLayout(search_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_conversations)
        action_layout.addWidget(refresh_button)
        
        # Limit selector
        self.limit_combo = QComboBox()
        for limit in [10, 20, 50, 100]:
            self.limit_combo.addItem(str(limit))
        self.limit_combo.currentIndexChanged.connect(self.refresh_conversations)
        action_layout.addWidget(QLabel("Limit:"))
        action_layout.addWidget(self.limit_combo)
        
        layout.addLayout(action_layout)
        
        # Conversations list in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.conversations_container = QWidget()
        self.conversations_layout = QVBoxLayout(self.conversations_container)
        self.conversations_layout.setAlignment(Qt.AlignTop)
        self.conversations_layout.setSpacing(10)
        
        scroll.setWidget(self.conversations_container)
        layout.addWidget(scroll)
        
        return tab
    
    def _create_preferences_tab(self) -> QWidget:
        """Create the preferences tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Auto-save settings
        autosave_layout = QHBoxLayout()
        
        self.autosave_checkbox = QCheckBox("Auto-save")
        self.autosave_checkbox.setChecked(self.memory.auto_save)
        self.autosave_checkbox.stateChanged.connect(self._toggle_autosave)
        autosave_layout.addWidget(self.autosave_checkbox)
        
        # Save interval
        self.save_interval_input = QLineEdit(str(self.memory.save_interval))
        self.save_interval_input.setMaximumWidth(80)
        autosave_layout.addWidget(QLabel("Save interval (s):"))
        autosave_layout.addWidget(self.save_interval_input)
        
        # Apply button
        apply_interval_button = QPushButton("Apply")
        apply_interval_button.clicked.connect(self._update_save_interval)
        autosave_layout.addWidget(apply_interval_button)
        
        autosave_layout.addStretch(1)
        layout.addLayout(autosave_layout)
        
        # Max entries settings
        max_entries_layout = QHBoxLayout()
        
        # Max conversations
        self.max_conversations_input = QLineEdit(str(self.memory.max_conversations))
        self.max_conversations_input.setMaximumWidth(80)
        max_entries_layout.addWidget(QLabel("Max conversations:"))
        max_entries_layout.addWidget(self.max_conversations_input)
        
        # Max knowledge entries
        self.max_entries_input = QLineEdit(str(self.memory.max_entries))
        self.max_entries_input.setMaximumWidth(80)
        max_entries_layout.addWidget(QLabel("Max knowledge entries:"))
        max_entries_layout.addWidget(self.max_entries_input)
        
        # Apply button
        apply_max_button = QPushButton("Apply")
        apply_max_button.clicked.connect(self._update_max_entries)
        max_entries_layout.addWidget(apply_max_button)
        
        max_entries_layout.addStretch(1)
        layout.addLayout(max_entries_layout)
        
        # Memory file location
        file_layout = QHBoxLayout()
        
        memory_path = str(self.memory.memory_file)
        file_layout.addWidget(QLabel("Memory file:"))
        file_layout.addWidget(QLabel(memory_path))
        
        save_now_button = QPushButton("Save Now")
        save_now_button.clicked.connect(self._save_now)
        file_layout.addWidget(save_now_button)
        
        layout.addLayout(file_layout)
        
        # Add spacer
        layout.addStretch(1)
        
        return tab
    
    def refresh_knowledge(self):
        """Refresh the knowledge display"""
        # Clear existing widgets
        while self.knowledge_layout.count():
            item = self.knowledge_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get all topics
        topics = self.memory.get_all_topics()
        
        if not topics:
            self.knowledge_layout.addWidget(QLabel("No knowledge entries found."))
            return
        
        # Add widgets for each topic
        for topic in topics:
            entry = self.memory.get_knowledge(topic)
            if entry:
                widget = MemoryEntryWidget(topic, entry)
                widget.delete_requested.connect(self._delete_knowledge)
                self.knowledge_layout.addWidget(widget)
    
    def refresh_conversations(self):
        """Refresh the conversations display"""
        # Clear existing widgets
        while self.conversations_layout.count():
            item = self.conversations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get limit from combo box
        limit = int(self.limit_combo.currentText())
        
        # Get conversation history
        conversations = self.memory.get_conversation_history(limit=limit)
        
        if not conversations:
            self.conversations_layout.addWidget(QLabel("No conversation history found."))
            return
        
        # Add widgets for each conversation
        for conversation in conversations:
            widget = ConversationWidget(conversation)
            self.conversations_layout.addWidget(widget)
    
    def _search_knowledge(self):
        """Search knowledge entries"""
        query = self.knowledge_search.text().strip()
        if not query:
            self.refresh_knowledge()
            return
        
        # Clear existing widgets
        while self.knowledge_layout.count():
            item = self.knowledge_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Search for matching entries
        results = self.memory.search_knowledge(query)
        
        if not results:
            self.knowledge_layout.addWidget(QLabel(f"No results found for '{query}'."))
            return
        
        # Add widgets for each result
        for result in results:
            topic = result.get("topic", "")
            entry = result.get("entry", {})
            widget = MemoryEntryWidget(topic, entry)
            widget.delete_requested.connect(self._delete_knowledge)
            self.knowledge_layout.addWidget(widget)
    
    def _search_conversations(self):
        """Search conversations"""
        query = self.conversation_search.text().strip()
        if not query:
            self.refresh_conversations()
            return
        
        # Clear existing widgets
        while self.conversations_layout.count():
            item = self.conversations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Search for matching conversations
        results = self.memory.search_conversations(query)
        
        if not results:
            self.conversations_layout.addWidget(QLabel(f"No results found for '{query}'."))
            return
        
        # Add widgets for each result
        for conversation in results:
            widget = ConversationWidget(conversation)
            self.conversations_layout.addWidget(widget)
    
    def _add_knowledge(self):
        """Add new knowledge entry"""
        dialog = AddKnowledgeDialog(self)
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_knowledge_data()
            
            if not data["topic"] or not data["content"]:
                QMessageBox.warning(self, "Invalid Input", 
                                   "Topic and content are required.")
                return
            
            # Add to memory
            success = self.memory.add_knowledge(
                data["topic"], 
                data["content"],
                data["source"]
            )
            
            if success:
                self.refresh_knowledge()
                self.status_label.setText(f"Memory Status: Added '{data['topic']}'")
            else:
                QMessageBox.warning(self, "Error", 
                                   "Failed to add knowledge entry.")
    
    def _delete_knowledge(self, topic: str):
        """Delete a knowledge entry"""
        reply = QMessageBox.question(
            self, 
            "Confirm Delete",
            f"Are you sure you want to delete '{topic}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.memory.delete_knowledge(topic)
            if success:
                self.refresh_knowledge()
                self.status_label.setText(f"Memory Status: Deleted '{topic}'")
            else:
                QMessageBox.warning(self, "Error", 
                                   f"Failed to delete '{topic}'.")
    
    def _toggle_autosave(self, state):
        """Toggle auto-save feature"""
        is_checked = state == Qt.Checked
        
        if is_checked != self.memory.auto_save:
            if is_checked:
                # Enable auto-save
                self.memory.auto_save = True
                self.memory._start_auto_save()
                self.status_label.setText("Memory Status: Auto-save enabled")
            else:
                # Disable auto-save
                self.memory.auto_save = False
                self.memory.stop_auto_save.set()
                if self.memory.save_thread:
                    self.memory.save_thread.join(timeout=1.0)
                    self.memory.save_thread = None
                self.status_label.setText("Memory Status: Auto-save disabled")
    
    def _update_save_interval(self):
        """Update save interval"""
        try:
            new_interval = int(self.save_interval_input.text())
            if new_interval < 1:
                raise ValueError("Interval must be at least 1 second")
            
            self.memory.save_interval = new_interval
            
            # Restart auto-save thread if active
            if self.memory.auto_save and self.memory.save_thread:
                self.memory.stop_auto_save.set()
                self.memory.save_thread.join(timeout=1.0)
                self.memory.stop_auto_save.clear()
                self.memory._start_auto_save()
            
            self.status_label.setText(f"Memory Status: Save interval updated to {new_interval}s")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
    
    def _update_max_entries(self):
        """Update maximum entries settings"""
        try:
            max_conversations = int(self.max_conversations_input.text())
            max_entries = int(self.max_entries_input.text())
            
            if max_conversations < 1 or max_entries < 1:
                raise ValueError("Values must be at least 1")
            
            self.memory.max_conversations = max_conversations
            self.memory.max_entries = max_entries
            
            self.status_label.setText("Memory Status: Max entries updated")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
    
    def _save_now(self):
        """Save memory immediately"""
        success = self.memory.save_memory()
        if success:
            self.status_label.setText("Memory Status: Saved successfully")
        else:
            self.status_label.setText("Memory Status: Save failed")
            QMessageBox.warning(self, "Save Error", 
                               "Failed to save memory. Check logs for details.")
    
    def _show_stats(self):
        """Show memory statistics"""
        stats = self.memory.get_stats()
        
        # Format stats for display
        stats_text = "Memory Statistics:\n\n"
        for key, value in stats.items():
            if key == "last_conversation" and value:
                # Format timestamp
                date_part = value.split("T")[0]
                time_part = value.split("T")[1].split(".")[0]
                stats_text += f"Last conversation: {date_part} {time_part}\n"
            else:
                # Format other stats
                stats_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        # Show in message box
        QMessageBox.information(self, "Memory Statistics", stats_text)

class OnsiteMemoryIntegration:
    """
    Integration between the Onsite Memory system and PySide6 UI
    
    This class provides methods to integrate the onsite memory system
    with the Mistral PySide6 application.
    """
    
    def __init__(
        self, 
        data_dir: str = "data/onsite_memory",
        memory_file: str = "memory.json"
    ):
        """
        Initialize the onsite memory integration
        
        Args:
            data_dir: Directory for storing memory files
            memory_file: Filename for the main memory file
        """
        # Initialize memory system
        self.memory = OnsiteMemory(
            data_dir=data_dir,
            memory_file=memory_file,
            auto_save=True,
            save_interval=60
        )
        
        logger.info("Onsite Memory Integration initialized")
    
    def get_memory_panel(self, parent=None) -> OnsiteMemoryPanel:
        """
        Get the memory panel widget for the UI
        
        Args:
            parent: Parent widget
            
        Returns:
            OnsiteMemoryPanel: Panel for displaying and managing memory
        """
        return OnsiteMemoryPanel(self.memory, parent)
    
    def add_conversation(
        self, 
        user_message: str, 
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a conversation to memory
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            metadata: Optional metadata for the conversation
            
        Returns:
            str: Conversation ID
        """
        return self.memory.add_conversation(
            user_message, 
            assistant_response,
            metadata
        )
    
    def add_knowledge_from_conversation(
        self, 
        topic: str,
        user_message: str,
        assistant_response: str,
        source: str = "conversation"
    ) -> bool:
        """
        Extract knowledge from a conversation and add it to memory
        
        Args:
            topic: Topic/key for the knowledge
            user_message: User message from conversation
            assistant_response: Assistant response from conversation
            source: Source identifier (default: "conversation")
            
        Returns:
            bool: True if successful
        """
        # Create content from the conversation
        content = f"From user query: {user_message}\n\nAnswer: {assistant_response}"
        
        # Add to knowledge
        return self.memory.add_knowledge(
            topic=topic,
            content=content,
            source=source,
            metadata={
                "extracted_from": "conversation",
                "user_query": user_message
            }
        )
    
    def search_context_for_query(self, query: str) -> Optional[str]:
        """
        Search memory for relevant context to enhance a query
        
        Args:
            query: The user's query
            
        Returns:
            Optional[str]: Context to add to the query, or None if no relevant context found
        """
        # Search knowledge first
        knowledge_results = self.memory.search_knowledge(query, limit=3)
        
        if knowledge_results:
            context = "Based on my memory:\n\n"
            
            for idx, result in enumerate(knowledge_results):
                topic = result.get("topic", "")
                content = result.get("entry", {}).get("content", "")
                context += f"{idx+1}. {topic}: {content}\n\n"
            
            return context
        
        # If no knowledge results, try conversations
        conversation_results = self.memory.search_conversations(query, limit=2)
        
        if conversation_results:
            context = "From previous conversations:\n\n"
            
            for idx, conv in enumerate(conversation_results):
                user_msg = conv.get("user_message", "")
                assistant_msg = conv.get("assistant_response", "")
                
                # Truncate long messages
                if len(assistant_msg) > 200:
                    assistant_msg = assistant_msg[:200] + "..."
                
                context += f"{idx+1}. You asked: {user_msg}\n"
                context += f"   I answered: {assistant_msg}\n\n"
            
            return context
        
        # No relevant context found
        return None
    
    def shutdown(self):
        """Shut down the memory system"""
        if self.memory:
            self.memory.stop()
        logger.info("Onsite Memory Integration shut down") 