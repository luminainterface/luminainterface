#!/usr/bin/env python3
"""
Simple Standalone Mistral GUI with Onsite Memory
A minimal GUI application for Mistral with memory storage capabilities
"""

import sys
import os
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path

# Configure basic parameters
API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MOCK_MODE = True  # Set to True to run without API key

# Import language database components if available
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from language.database_manager import DatabaseManager
    from language.language_database_bridge import LanguageDatabaseBridge, get_language_database_bridge
    from language.database_connection_manager import DatabaseConnectionManager, get_database_connection_manager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Language database modules not found. Running with file-based storage only.")

class OnsiteMemory:
    """Simple implementation of onsite memory storage"""
    
    def __init__(self, data_dir="data/onsite_memory", memory_file="mistral_memory.json"):
        self.data_dir = Path(data_dir)
        self.memory_file = self.data_dir / memory_file
        self.conversation_history = []
        self.knowledge_base = {}
        
        # Create directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory if available
        self.load_memory()
    
    def load_memory(self):
        """Load memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = data.get('conversations', [])
                    self.knowledge_base = data.get('knowledge', {})
                print(f"Loaded {len(self.conversation_history)} conversations and {len(self.knowledge_base)} knowledge entries")
            except Exception as e:
                print(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save memory to file"""
        try:
            data = {
                'conversations': self.conversation_history,
                'knowledge': self.knowledge_base,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False
    
    def add_conversation(self, user_message, assistant_response, metadata=None):
        """Add a conversation to memory"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'assistant_response': assistant_response,
            'metadata': metadata or {}
        }
        self.conversation_history.append(conversation)
        self.save_memory()
    
    def add_knowledge(self, topic, content, source=None):
        """Add knowledge to the knowledge base"""
        knowledge_entry = {
            'content': content,
            'source': source,
            'added': datetime.now().isoformat()
        }
        self.knowledge_base[topic] = knowledge_entry
        self.save_memory()
    
    def search_context(self, query, limit=3):
        """Search conversations and knowledge base for relevant context"""
        # Simple keyword-based search
        query_words = set(query.lower().split())
        results = []
        
        # Search knowledge base first
        for topic, entry in self.knowledge_base.items():
            topic_words = set(topic.lower().split())
            content_words = set(entry['content'].lower().split())
            matching_words = query_words.intersection(topic_words.union(content_words))
            
            if matching_words:
                score = len(matching_words) / len(query_words)
                results.append({
                    'type': 'knowledge',
                    'topic': topic,
                    'content': entry['content'],
                    'score': score
                })
        
        # Search conversations
        for conv in self.conversation_history:
            user_words = set(conv['user_message'].lower().split())
            response_words = set(conv['assistant_response'].lower().split())
            matching_words = query_words.intersection(user_words.union(response_words))
            
            if matching_words:
                score = len(matching_words) / len(query_words)
                results.append({
                    'type': 'conversation',
                    'user_message': conv['user_message'],
                    'assistant_response': conv['assistant_response'],
                    'score': score
                })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def format_context(self, search_results):
        """Format search results into a context prompt"""
        if not search_results:
            return None
        
        context = "Based on my memory, I have some relevant information:\n\n"
        
        for result in search_results:
            if result['type'] == 'knowledge':
                context += f"• Topic: {result['topic']}\n{result['content']}\n\n"
            else:
                context += f"• Previous conversation:\nQuestion: {result['user_message']}\nAnswer: {result['assistant_response']}\n\n"
        
        return context
    
    def get_stats(self):
        """Get memory statistics"""
        return {
            'conversations': len(self.conversation_history),
            'knowledge_entries': len(self.knowledge_base),
            'last_updated': datetime.now().isoformat(),
            'memory_file': str(self.memory_file)
        }


class DatabaseMemory(OnsiteMemory):
    """Enhanced memory implementation with database synchronization"""
    
    def __init__(self, data_dir="data/onsite_memory", memory_file="mistral_memory.json"):
        """Initialize with both file storage and database connection"""
        super().__init__(data_dir, memory_file)
        
        self.db_available = DATABASE_AVAILABLE
        self.db_connected = False
        self.db_manager = None
        self.db_bridge = None
        self.central_language_node = None
        
        # Try to connect to the database system
        if self.db_available:
            try:
                self._connect_database()
                # Initial sync from database to memory
                self._sync_from_database()
            except Exception as e:
                print(f"Error connecting to database: {e}")
    
    def _connect_database(self):
        """Connect to the language database system and central language node"""
        try:
            # Get the database manager
            self.db_manager = DatabaseManager.get_instance()
            
            # Get the database bridge
            self.db_bridge = get_language_database_bridge()
            
            # Register with connection manager
            try:
                db_conn_manager = get_database_connection_manager()
                # Make sure it's initialized first
                if hasattr(db_conn_manager, 'initialize'):
                    db_conn_manager.initialize()
                # Then register our component
                if hasattr(db_conn_manager, 'register_component'):
                    db_conn_manager.register_component("mistral_memory", self)
            except Exception as e:
                print(f"Error registering with connection manager: {e}")
            
            # Try to connect to central language node if available
            try:
                from src.language.central_language_node import CentralLanguageNode
                if hasattr(CentralLanguageNode, 'get_instance'):
                    self.central_language_node = CentralLanguageNode.get_instance(data_dir="data")
                else:
                    self.central_language_node = CentralLanguageNode(data_dir="data")
                print("Connected to Central Language Node")
            except Exception as e:
                print(f"Central Language Node not available: {e}")
                self.central_language_node = None
            
            # Connect to database verification tools
            try:
                # Register component hooks for periodic verification if available
                if db_conn_manager and hasattr(db_conn_manager, 'register_component_hook'):
                    db_conn_manager.register_component_hook(
                        "mistral_memory", 
                        "verify_sync", 
                        self._verify_database_sync,
                        interval=900  # 15 minutes
                    )
            except Exception as e:
                print(f"Could not register verification hooks: {e}")
            
            self.db_connected = True
            print("Connected to language database system")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            self.db_connected = False
            return False
    
    def _verify_database_sync(self):
        """Verify database synchronization and connection health"""
        if not self.db_connected or not self.db_bridge:
            return
            
        try:
            # Check bridge status
            bridge_status = self.db_bridge.get_status()
            
            # If last sync was more than 30 minutes ago, trigger sync
            if bridge_status.get('last_sync_time'):
                from datetime import datetime
                last_sync = datetime.fromisoformat(bridge_status['last_sync_time'])
                now = datetime.now()
                if (now - last_sync).total_seconds() > 1800:  # 30 minutes
                    print("Triggering database sync due to elapsed time")
                    self.db_bridge.sync_now()
            
            # Optimize connection if needed
            if self.db_manager and hasattr(self.db_manager, 'optimize_database'):
                self.db_manager.optimize_database()
                
        except Exception as e:
            print(f"Error verifying database sync: {e}")
    
    def _sync_from_database(self):
        """Synchronize memory from database to local storage"""
        if not self.db_connected or not self.db_manager:
            return False
            
        try:
            # Get recent conversations from database
            cutoff_time = datetime.now().replace(year=datetime.now().year - 1)  # Last year
            db_conversations = self.db_manager.get_recent_conversations(cutoff_time)
            
            if db_conversations:
                # Convert format and merge with existing conversations
                for db_conv in db_conversations:
                    # Check if conversation already exists
                    exists = False
                    for conv in self.conversation_history:
                        if (db_conv.get('user_message') == conv.get('user_message') and 
                            db_conv.get('assistant_response') == conv.get('assistant_response')):
                            exists = True
                            break
                    
                    if not exists:
                        self.conversation_history.append({
                            'timestamp': db_conv.get('timestamp', datetime.now().isoformat()),
                            'user_message': db_conv.get('user_message', ''),
                            'assistant_response': db_conv.get('assistant_response', ''),
                            'metadata': db_conv.get('metadata', {})
                        })
            
            # Get knowledge entries from database
            knowledge_entries = self.db_manager.get_knowledge_entries()
            
            if knowledge_entries:
                # Merge with existing knowledge base
                for entry in knowledge_entries:
                    topic = entry.get('topic')
                    if topic and topic not in self.knowledge_base:
                        self.knowledge_base[topic] = {
                            'content': entry.get('content', ''),
                            'source': entry.get('source', 'database'),
                            'added': entry.get('timestamp', datetime.now().isoformat())
                        }
            
            # If central language node is available, get additional context
            if self.central_language_node:
                try:
                    # Get language memory associations
                    language_memory = self.central_language_node.language_memory
                    if language_memory:
                        # Get top associations and concepts
                        top_concepts = language_memory.get_top_concepts(10)
                        for concept in top_concepts:
                            if concept and concept not in self.knowledge_base:
                                associations = language_memory.recall_associations(concept)
                                if associations:
                                    # Convert associations to readable text
                                    content = f"Associated with: {', '.join([a[0] for a in associations[:5] if a])}"
                                    self.knowledge_base[concept] = {
                                        'content': content,
                                        'source': 'language_memory',
                                        'added': datetime.now().isoformat()
                                    }
                except Exception as e:
                    print(f"Error getting language memory associations: {e}")
            
            # Save merged data to file
            self.save_memory()
            print(f"Synchronized data from database: {len(db_conversations)} conversations, {len(knowledge_entries)} knowledge entries")
            return True
        except Exception as e:
            print(f"Error synchronizing from database: {e}")
            return False
    
    def _sync_to_database(self):
        """Synchronize memory from local storage to database"""
        if not self.db_connected or not self.db_manager:
            return False
            
        try:
            # Check available methods for storing conversations
            has_create_conversation = hasattr(self.db_manager, 'create_conversation')
            has_store_conversation = hasattr(self.db_manager, 'store_conversation')
            has_add_conversation = hasattr(self.db_manager, 'add_conversation')
            
            # Sync conversations to database
            for conv in self.conversation_history:
                try:
                    # Check if recently added
                    timestamp = conv.get('timestamp')
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp)
                        # Only sync conversations from the last day
                        if (datetime.now() - dt).days <= 1:
                            user_message = conv.get('user_message', '')
                            assistant_response = conv.get('assistant_response', '')
                            metadata = {
                                'source': 'mistral_memory',
                                'timestamp': timestamp,
                                'app': 'simple_mistral_gui'
                            }
                            
                            # Try different methods based on what's available
                            if has_create_conversation:
                                # Try to adapt to the available method signature
                                import inspect
                                sig = inspect.signature(self.db_manager.create_conversation)
                                params = list(sig.parameters.keys())
                                
                                if 'user_message' in params and 'assistant_response' in params:
                                    # Direct parameters
                                    self.db_manager.create_conversation(
                                        user_message=user_message,
                                        assistant_response=assistant_response,
                                        metadata=metadata
                                    )
                                elif 'text' in params:
                                    # Simple text parameter
                                    combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
                                    self.db_manager.create_conversation(
                                        text=combined_text,
                                        metadata=metadata
                                    )
                                elif 'content' in params:
                                    # Simple content parameter
                                    combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
                                    self.db_manager.create_conversation(
                                        content=combined_text,
                                        metadata=metadata
                                    )
                                else:
                                    # Just try with metadata
                                    metadata['user_message'] = user_message
                                    metadata['assistant_response'] = assistant_response
                                    self.db_manager.create_conversation(
                                        metadata=metadata
                                    )
                            elif has_store_conversation:
                                # Try alternative method
                                self.db_manager.store_conversation(
                                    user_message=user_message,
                                    assistant_response=assistant_response,
                                    metadata=metadata
                                )
                            elif has_add_conversation:
                                # Try another alternative method
                                self.db_manager.add_conversation(
                                    user_message=user_message,
                                    assistant_response=assistant_response,
                                    metadata=metadata
                                )
                except Exception as e:
                    print(f"Error syncing conversation to database: {e}")
            
            # Sync knowledge entries to database
            for topic, entry in self.knowledge_base.items():
                try:
                    # Check if recently added
                    added = entry.get('added')
                    if added:
                        dt = datetime.fromisoformat(added)
                        # Only sync entries from the last day
                        if (datetime.now() - dt).days <= 1:
                            # Try different methods
                            if hasattr(self.db_manager, 'create_knowledge_entry'):
                                self.db_manager.create_knowledge_entry(
                                    topic=topic,
                                    content=entry.get('content', ''),
                                    source=entry.get('source', 'mistral_memory'),
                                    metadata={
                                        'app': 'simple_mistral_gui',
                                        'timestamp': added
                                    }
                                )
                            elif hasattr(self.db_manager, 'store_knowledge'):
                                # Alternative method name
                                self.db_manager.store_knowledge(
                                    topic=topic,
                                    content=entry.get('content', ''),
                                    source=entry.get('source', 'mistral_memory'),
                                    metadata={
                                        'app': 'simple_mistral_gui',
                                        'timestamp': added
                                    }
                                )
                            elif hasattr(self.db_manager, 'add_knowledge'):
                                # Another alternative
                                self.db_manager.add_knowledge(
                                    topic=topic,
                                    content=entry.get('content', ''),
                                    source=entry.get('source', 'mistral_memory')
                                )
                                
                            # If central language node is available, add to language memory
                            if self.central_language_node and hasattr(self.central_language_node, 'language_memory'):
                                try:
                                    content = entry.get('content', '')
                                    words = content.split()
                                    if words and len(words) >= 2:
                                        # Store first few word associations
                                        for i in range(min(len(words)-1, 5)):
                                            self.central_language_node.language_memory.store_word_association(
                                                words[i], words[i+1], 0.7
                                            )
                                except Exception as e:
                                    print(f"Error adding to language memory: {e}")
                except Exception as e:
                    print(f"Error syncing knowledge entry to database: {e}")
            
            # Trigger database bridge synchronization
            if self.db_bridge and hasattr(self.db_bridge, 'sync_now'):
                self.db_bridge.sync_now()
                
            print("Synchronized memory to database")
            return True
        except Exception as e:
            print(f"Error synchronizing to database: {e}")
            return False
    
    def connect_database(self, db_manager):
        """
        Connect to a database manager (used by DatabaseConnectionManager)
        
        Args:
            db_manager: Database manager instance
            
        Returns:
            bool: True if connection successful
        """
        self.db_manager = db_manager
        self.db_connected = True
        return True
    
    def process_input_with_language_system(self, text):
        """
        Process input text with the language system if available
        
        Args:
            text: Input text to process
            
        Returns:
            dict: Results from language processing or None if not available
        """
        if not self.central_language_node:
            return None
            
        try:
            # Process text through central language node
            results = self.central_language_node.process_text(text)
            
            # If we got valid results, add associations to memory
            if results and 'memory_associations' in results:
                associations = results.get('memory_associations', [])
                if associations:
                    # Get the first few associations
                    for association in associations[:3]:
                        # Add to knowledge base if not already present
                        if association and association not in self.knowledge_base:
                            # Get details about the association
                            content = f"Association found in language memory. Related to: {text}"
                            self.add_knowledge(
                                topic=f"Association: {association}",
                                content=content,
                                source="language_system"
                            )
            
            return results
        except Exception as e:
            print(f"Error processing with language system: {e}")
            return None
            
    def add_conversation(self, user_message, assistant_response, metadata=None):
        """Add a conversation to memory and sync to database"""
        # Process with language system first if available
        language_results = self.process_input_with_language_system(user_message)
        
        # If we got language results, add to metadata
        if language_results and metadata is None:
            metadata = {}
        
        if language_results and metadata is not None:
            metadata['consciousness_level'] = language_results.get('consciousness_level', 0)
            metadata['neural_linguistic_score'] = language_results.get('neural_linguistic_score', 0)
            if 'memory_associations' in language_results:
                metadata['memory_associations'] = language_results.get('memory_associations', [])
        
        # Call parent method to add to local storage
        super().add_conversation(user_message, assistant_response, metadata)
        
        # Sync to database if connected
        if self.db_connected:
            try:
                self._sync_to_database()
            except Exception as e:
                print(f"Error syncing to database after adding conversation: {e}")
    
    def add_knowledge(self, topic, content, source=None):
        """Add knowledge to the knowledge base and sync to database"""
        # Call parent method to add to local storage
        super().add_knowledge(topic, content, source)
        
        # Add to language memory if central node is available
        if self.central_language_node and hasattr(self.central_language_node, 'language_memory'):
            try:
                # Process content to extract key words
                words = content.lower().split()
                if len(words) >= 2:
                    # Store first few word associations in language memory
                    for i in range(min(len(words)-1, 5)):
                        self.central_language_node.language_memory.store_word_association(
                            words[i], words[i+1], 0.7
                        )
            except Exception as e:
                print(f"Error adding to language memory: {e}")
        
        # Sync to database if connected
        if self.db_connected:
            try:
                self._sync_to_database()
            except Exception as e:
                print(f"Error syncing to database after adding knowledge: {e}")
    
    def search_context(self, query, limit=3):
        """Search conversations and knowledge base for relevant context"""
        # Get baseline results from parent method
        results = super().search_context(query, limit)
        
        # Try to enhance with language system if available
        if self.central_language_node and hasattr(self.central_language_node, 'language_memory'):
            try:
                # Get associations from language memory
                words = query.lower().split()
                associations = []
                for word in words:
                    if word.isalnum() and len(word) > 3:  # Skip short words and non-alphanumeric
                        word_associations = self.central_language_node.language_memory.recall_associations(word)
                        if word_associations:
                            for assoc, strength in word_associations:
                                if assoc and strength > 0.5:  # Only use strong associations
                                    associations.append((assoc, strength, word))
                
                # Add top associations as context if they exist in knowledge base
                if associations:
                    # Sort by strength
                    associations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Check if any associations match knowledge base entries
                    for assoc, strength, source_word in associations[:5]:
                        # Look for exact match
                        if assoc in self.knowledge_base:
                            entry = self.knowledge_base[assoc]
                            results.append({
                                'type': 'knowledge',
                                'topic': assoc,
                                'content': entry['content'],
                                'score': strength
                            })
                        
                        # Also look for partial matches in topics
                        for topic in self.knowledge_base:
                            if assoc.lower() in topic.lower() and topic not in [r.get('topic') for r in results if r['type'] == 'knowledge']:
                                entry = self.knowledge_base[topic]
                                results.append({
                                    'type': 'knowledge',
                                    'topic': topic,
                                    'content': entry['content'],
                                    'score': strength * 0.8  # Slightly lower score for partial matches
                                })
                                break  # Only add one partial match per association
            except Exception as e:
                print(f"Error enhancing search with language system: {e}")
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def get_stats(self):
        """Get memory statistics including database connection status"""
        stats = super().get_stats()
        
        # Add database status
        stats['database_available'] = self.db_available
        stats['database_connected'] = self.db_connected
        
        # Add database bridge stats if available
        if self.db_bridge and hasattr(self.db_bridge, 'get_status'):
            try:
                bridge_status = self.db_bridge.get_status()
                stats['database_bridge_status'] = bridge_status
            except:
                stats['database_bridge_status'] = 'Error getting bridge status'
        
        # Add language system stats if available
        if self.central_language_node:
            stats['language_system_available'] = True
            try:
                if hasattr(self.central_language_node, 'get_status'):
                    lang_status = self.central_language_node.get_status()
                    stats['language_system_status'] = lang_status
                else:
                    # Get basic component statuses
                    components = {
                        'language_memory': hasattr(self.central_language_node, 'language_memory'),
                        'consciousness': hasattr(self.central_language_node, 'conscious_mirror_language'),
                        'neural': hasattr(self.central_language_node, 'neural_linguistic_processor'),
                        'recursive': hasattr(self.central_language_node, 'recursive_pattern_analyzer')
                    }
                    stats['language_system_components'] = components
            except Exception as e:
                stats['language_system_error'] = str(e)
        else:
            stats['language_system_available'] = False
        
        return stats


def main():
    """Main entry point for the simple Mistral app with memory"""
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QTabWidget,
        QCheckBox, QMessageBox, QInputDialog, QListWidget, QListWidgetItem, QSlider,
        QStatusBar
    )
    from PySide6.QtCore import Qt, QSize, QTimer
    from PySide6.QtGui import QFont
    
    class SimpleMistralWindow(QMainWindow):
        """Simple Mistral Chat Window with Memory"""
        
        def __init__(self):
            super().__init__()
            
            # Set window properties
            self.setWindowTitle("Mistral Chat with Onsite Memory")
            self.resize(1000, 700)
            
            # Initialize memory with database connection if available
            if DATABASE_AVAILABLE:
                self.memory = DatabaseMemory()
                db_status = "with database connection" if self.memory.db_connected else "with database available but not connected"
            else:
                self.memory = OnsiteMemory()
                db_status = "without database connection"
            
            print(f"Initialized memory system {db_status}")
            
            self.use_memory = True
            
            # Create tab widget
            self.tab_widget = QTabWidget()
            self.setCentralWidget(self.tab_widget)
            
            # Create chat tab
            self.chat_tab = QWidget()
            self.tab_widget.addTab(self.chat_tab, "Chat")
            
            # Create memory tab
            self.memory_tab = QWidget()
            self.tab_widget.addTab(self.memory_tab, "Memory")
            
            # Set up chat UI
            self.setup_chat_ui()
            
            # Set up memory UI
            self.setup_memory_ui()
            
            # Create status bar
            self.setup_status_bar()
            
            # Start timer for periodic database sync if connected
            if isinstance(self.memory, DatabaseMemory) and self.memory.db_connected:
                self.sync_timer = QTimer(self)
                self.sync_timer.timeout.connect(self.check_database_sync)
                self.sync_timer.start(300000)  # 5 minutes
        
        def setup_status_bar(self):
            """Set up the status bar with system information"""
            self.statusBar().showMessage("Memory system initialized")
            
            # Add permanent widgets to status bar
            if isinstance(self.memory, DatabaseMemory):
                # Database status indicator
                db_status = QLabel()
                if self.memory.db_connected:
                    db_status.setText("DB: Connected")
                    db_status.setStyleSheet("color: green")
                else:
                    db_status.setText("DB: Disconnected")
                    db_status.setStyleSheet("color: red")
                self.statusBar().addPermanentWidget(db_status)
                
                # Language system status indicator
                lang_status = QLabel()
                if self.memory.central_language_node:
                    lang_status.setText("Language System: Active")
                    lang_status.setStyleSheet("color: green")
                else:
                    lang_status.setText("Language System: Inactive")
                    lang_status.setStyleSheet("color: red")
                self.statusBar().addPermanentWidget(lang_status)
        
        def check_database_sync(self):
            """Periodically check and potentially sync with database"""
            if not isinstance(self.memory, DatabaseMemory) or not self.memory.db_connected:
                return
                
            try:
                # Check bridge status
                if self.memory.db_bridge and hasattr(self.memory.db_bridge, 'get_status'):
                    bridge_status = self.memory.db_bridge.get_status()
                    
                    # Update status bar
                    if bridge_status.get('last_sync_time'):
                        self.statusBar().showMessage(f"Last DB sync: {bridge_status.get('last_sync_time')}")
                    
                    # Check if sync is needed based on time
                    if bridge_status.get('last_sync_time'):
                        from datetime import datetime
                        last_sync = datetime.fromisoformat(bridge_status['last_sync_time'])
                        now = datetime.now()
                        if (now - last_sync).total_seconds() > 1800:  # 30 minutes
                            # Trigger sync in background without notification
                            self.memory._sync_to_database()
            except Exception as e:
                print(f"Error in periodic sync check: {e}")
        
        def setup_chat_ui(self):
            """Set up the chat interface"""
            # Create main layout
            main_layout = QVBoxLayout(self.chat_tab)
            
            # Create splitter for chat and settings
            splitter = QSplitter(Qt.Horizontal)
            main_layout.addWidget(splitter)
            
            # Chat area
            chat_widget = QWidget()
            chat_layout = QVBoxLayout(chat_widget)
            
            # Chat display
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            font = QFont("Arial", 11)
            self.chat_display.setFont(font)
            chat_layout.addWidget(self.chat_display, 1)
            
            # Input area
            input_layout = QHBoxLayout()
            
            self.message_input = QLineEdit()
            self.message_input.setPlaceholderText("Type your message here...")
            self.message_input.returnPressed.connect(self.send_message)
            font = QFont("Arial", 11)
            self.message_input.setFont(font)
            input_layout.addWidget(self.message_input)
            
            send_button = QPushButton("Send")
            send_button.clicked.connect(self.send_message)
            input_layout.addWidget(send_button)
            
            chat_layout.addLayout(input_layout)
            
            # Settings area
            settings_widget = QWidget()
            settings_layout = QVBoxLayout(settings_widget)
            
            # Memory settings
            settings_layout.addWidget(QLabel("<b>Memory Settings</b>"))
            
            # Memory checkbox
            self.memory_checkbox = QCheckBox("Use memory for context")
            self.memory_checkbox.setChecked(self.use_memory)
            self.memory_checkbox.stateChanged.connect(self.toggle_memory)
            settings_layout.addWidget(self.memory_checkbox)
            
            # Extract knowledge button
            add_knowledge_button = QPushButton("Add Knowledge Entry")
            add_knowledge_button.clicked.connect(self.add_knowledge)
            settings_layout.addWidget(add_knowledge_button)
            
            # View memory stats button
            stats_button = QPushButton("Memory Stats")
            stats_button.clicked.connect(self.show_memory_stats)
            settings_layout.addWidget(stats_button)
            
            # Language system settings
            if isinstance(self.memory, DatabaseMemory) and self.memory.central_language_node:
                settings_layout.addWidget(QLabel("<b>Language System</b>"))
                
                # Language settings section
                language_settings_group = QWidget()
                language_settings_layout = QVBoxLayout(language_settings_group)
                
                # LLM weight slider
                llm_layout = QHBoxLayout()
                llm_layout.addWidget(QLabel("LLM Weight:"))
                self.llm_slider = QSlider(Qt.Horizontal)
                self.llm_slider.setMinimum(0)
                self.llm_slider.setMaximum(100)
                self.llm_slider.setValue(70)  # Default 0.7
                self.llm_slider.setTickPosition(QSlider.TicksBelow)
                self.llm_slider.setTickInterval(10)
                llm_layout.addWidget(self.llm_slider)
                self.llm_value_label = QLabel("0.70")
                llm_layout.addWidget(self.llm_value_label)
                language_settings_layout.addLayout(llm_layout)
                
                # Neural network weight slider
                nn_layout = QHBoxLayout()
                nn_layout.addWidget(QLabel("NN Weight:"))
                self.nn_slider = QSlider(Qt.Horizontal)
                self.nn_slider.setMinimum(0)
                self.nn_slider.setMaximum(100)
                self.nn_slider.setValue(50)  # Default 0.5
                self.nn_slider.setTickPosition(QSlider.TicksBelow)
                self.nn_slider.setTickInterval(10)
                nn_layout.addWidget(self.nn_slider)
                self.nn_value_label = QLabel("0.50")
                nn_layout.addWidget(self.nn_value_label)
                language_settings_layout.addLayout(nn_layout)
                
                # Update weights button
                update_weights_button = QPushButton("Update Weights")
                update_weights_button.clicked.connect(self.update_language_weights)
                language_settings_layout.addWidget(update_weights_button)
                
                # Add language settings to main settings
                settings_layout.addWidget(language_settings_group)
                
                # Connect slider signals
                self.llm_slider.valueChanged.connect(self.on_llm_slider_changed)
                self.nn_slider.valueChanged.connect(self.on_nn_slider_changed)
            
            # Database sync button (if available)
            if isinstance(self.memory, DatabaseMemory) and self.memory.db_connected:
                settings_layout.addWidget(QLabel("<b>Database</b>"))
                sync_button = QPushButton("Sync with Database")
                sync_button.clicked.connect(self.sync_with_database)
                settings_layout.addWidget(sync_button)
                
                # Verify database button
                verify_button = QPushButton("Verify Database")
                verify_button.clicked.connect(self.verify_database)
                settings_layout.addWidget(verify_button)
            
            settings_layout.addStretch(1)
            
            # Add widgets to splitter
            splitter.addWidget(chat_widget)
            splitter.addWidget(settings_widget)
            
            # Set splitter sizes
            splitter.setSizes([700, 300])
            
            # Add welcome message
            welcome_msg = "Welcome to Mistral Chat with Onsite Memory!"
            if isinstance(self.memory, DatabaseMemory):
                if self.memory.db_connected:
                    welcome_msg += " Connected to the language database system."
                if self.memory.central_language_node:
                    welcome_msg += " Language system integration is active."
            self.chat_display.append(f"<b>System:</b> {welcome_msg} Type a message to begin.")
        
        def setup_memory_ui(self):
            """Set up the memory management interface"""
            layout = QVBoxLayout(self.memory_tab)
            
            # Knowledge base section
            layout.addWidget(QLabel("<b>Knowledge Base</b>"))
            
            # Knowledge list
            self.knowledge_list = QListWidget()
            self.knowledge_list.itemDoubleClicked.connect(self.view_knowledge)
            layout.addWidget(self.knowledge_list)
            
            # Knowledge buttons
            knowledge_buttons = QHBoxLayout()
            
            add_button = QPushButton("Add Entry")
            add_button.clicked.connect(self.add_knowledge)
            knowledge_buttons.addWidget(add_button)
            
            view_button = QPushButton("View Selected")
            view_button.clicked.connect(lambda: self.view_knowledge(self.knowledge_list.currentItem()))
            knowledge_buttons.addWidget(view_button)
            
            delete_button = QPushButton("Delete Selected")
            delete_button.clicked.connect(self.delete_knowledge)
            knowledge_buttons.addWidget(delete_button)
            
            layout.addLayout(knowledge_buttons)
            
            # Conversation history section
            layout.addWidget(QLabel("<b>Conversation History</b>"))
            
            # Conversation list
            self.conversation_list = QListWidget()
            self.conversation_list.itemDoubleClicked.connect(self.view_conversation)
            layout.addWidget(self.conversation_list)
            
            # Conversation buttons
            conversation_buttons = QHBoxLayout()
            
            view_conv_button = QPushButton("View Selected")
            view_conv_button.clicked.connect(lambda: self.view_conversation(self.conversation_list.currentItem()))
            conversation_buttons.addWidget(view_conv_button)
            
            clear_button = QPushButton("Clear All History")
            clear_button.clicked.connect(self.clear_history)
            conversation_buttons.addWidget(clear_button)
            
            layout.addLayout(conversation_buttons)
            
            # Refresh memory lists
            self.refresh_memory_lists()
        
        def refresh_memory_lists(self):
            """Refresh the memory lists"""
            # Clear lists
            self.knowledge_list.clear()
            self.conversation_list.clear()
            
            # Add knowledge entries
            for topic in self.memory.knowledge_base.keys():
                item = QListWidgetItem(topic)
                self.knowledge_list.addItem(item)
            
            # Add conversations
            for i, conv in enumerate(self.memory.conversation_history):
                # Truncate message for display
                msg = conv['user_message']
                if len(msg) > 50:
                    msg = msg[:47] + "..."
                
                item = QListWidgetItem(f"{i+1}. {msg}")
                self.conversation_list.addItem(item)
        
        def send_message(self):
            """Handle sending a message"""
            message = self.message_input.text().strip()
            if not message:
                return
            
            # Add user message to chat
            self.chat_display.append(f"<b>You:</b> {message}")
            
            # Clear input
            self.message_input.clear()
            
            # Process with language system if available (for DatabaseMemory)
            language_results = None
            if isinstance(self.memory, DatabaseMemory) and hasattr(self.memory, 'process_input_with_language_system'):
                language_results = self.memory.process_input_with_language_system(message)
                if language_results:
                    consciousness = language_results.get('consciousness_level', 0)
                    neural_score = language_results.get('neural_linguistic_score', 0)
                    self.chat_display.append(
                        f"<i>Language metrics: Consciousness level {consciousness:.2f}, "
                        f"Neural-linguistic score {neural_score:.2f}</i>"
                    )
            
            # Find relevant context if memory is enabled
            context = None
            if self.use_memory:
                search_results = self.memory.search_context(message)
                if search_results:
                    context = self.memory.format_context(search_results)
                    self.chat_display.append("<i>Using memory to enhance response...</i>")
            
            # Generate response based on context and language processing
            response = None
            
            # If we have language results, use them to influence the response
            if language_results:
                consciousness = language_results.get('consciousness_level', 0)
                if context:
                    response = self.generate_enhanced_response(message, context, consciousness)
                else:
                    response = self.generate_enhanced_response(message, None, consciousness)
            else:
                # Use standard response generation
                if context:
                    response = self.generate_response(message, context)
                else:
                    response = self.generate_response(message)
            
            # Add response
            self.chat_display.append(f"<b>Assistant:</b> {response}")
            
            # Store in memory with metadata from language processing
            if self.use_memory:
                metadata = {}
                if language_results:
                    metadata['consciousness_level'] = language_results.get('consciousness_level', 0)
                    metadata['neural_linguistic_score'] = language_results.get('neural_linguistic_score', 0)
                    if 'memory_associations' in language_results:
                        metadata['memory_associations'] = language_results.get('memory_associations', [])
                
                self.memory.add_conversation(message, response, metadata)
            
            # Refresh memory lists
            self.refresh_memory_lists()
        
        def generate_enhanced_response(self, message, context=None, consciousness_level=0.5):
            """Generate enhanced response using both memory and external processing"""
            api_key = os.environ.get("MISTRAL_API_KEY", "")
            
            # Process with neural processor if available
            neural_results = None
            if hasattr(self, 'neural_processor') and self.neural_processor:
                try:
                    # Get neural processing state
                    processing_state = self.neural_processor.process_text(message)
                    
                    # Extract neural results
                    neural_results = {
                        'activations': processing_state.activations.detach().cpu().numpy().flatten()[:10].tolist() if processing_state.activations is not None else [],
                        'resonance': float(processing_state.resonance) if hasattr(processing_state, 'resonance') else 0.0,
                        'concepts': []  # Would be populated if we had concept mapping
                    }
                    
                    # Add neural insights to context
                    highest_activations = sorted(
                        [(i, v) for i, v in enumerate(neural_results['activations'][:20])], 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )[:5]
                    
                    if context:
                        context += "\n\nNeural analysis insights:\n"
                        context += f"- Activation patterns detected: {[f'Concept {idx}: {val:.2f}' for idx, val in highest_activations]}\n"
                        context += f"- Resonance level: {neural_results['resonance']:.2f}\n"
                except Exception as e:
                    print(f"Error in neural processing: {e}")
            
            # Process with RSEN if available
            rsen_results = None
            try:
                import sys
                if 'RSEN_node' in sys.modules:
                    rsen_module = sys.modules['RSEN_node']
                    if hasattr(rsen_module, 'RSEN') and hasattr(rsen_module.RSEN, 'get_instance'):
                        rsen_instance = rsen_module.RSEN.get_instance()
                        if rsen_instance and hasattr(rsen_instance, 'train_epoch'):
                            # Process through RSEN
                            rsen_output = rsen_instance.train_epoch(message)
                            
                            # Extract key metrics
                            if rsen_output and isinstance(rsen_output, dict):
                                rsen_results = {
                                    'quantum_metrics': rsen_output.get('quantum_metrics', {}),
                                    'resonance_metrics': rsen_output.get('resonance_metrics', {}),
                                    'topology_metrics': rsen_output.get('topology_metrics', {})
                                }
                                
                                # Add insights to context
                                if context and rsen_results:
                                    context += "\nRSEN quantum analysis:\n"
                                    if 'quantum_metrics' in rsen_results and rsen_results['quantum_metrics']:
                                        context += f"- Quantum field state: {rsen_results['quantum_metrics'].get('field_state', 'Unknown')}\n"
                                    if 'resonance_metrics' in rsen_results and rsen_results['resonance_metrics']:
                                        context += f"- Resonance score: {rsen_results['resonance_metrics'].get('resonance_score', 0)}\n"
            except Exception as e:
                print(f"Error in RSEN processing: {e}")
            
            # Use external Mistral API if available, otherwise simulate response
            if api_key:
                # Import Mistral client
                try:
                    from mistralai.client import MistralClient
                    from mistralai.models.chat_completion import ChatMessage
                    
                    # Initialize client
                    client = MistralClient(api_key=api_key)
                    
                    # Add context to system message if available
                    messages = []
                    if context:
                        messages.append(ChatMessage(role="system", content=f"You are Mistral AI assistant. Use this context to help with your response:\n\n{context}"))
                    
                    # Add user message
                    messages.append(ChatMessage(role="user", content=message))
                    
                    # Get chat completion
                    chat_response = client.chat(
                        model="mistral-small-latest",  # or other models
                        messages=messages,
                    )
                    
                    # Extract response
                    return chat_response.choices[0].message.content
                except Exception as e:
                    print(f"Error calling Mistral API: {e}")
                    return f"Error: Failed to get response from Mistral API - {e}"
            else:
                # Use our neural processor for simple responses if available
                if neural_results:
                    # Get LLM/NN weights
                    llm_weight = float(os.environ.get("LLM_WEIGHT", "0.5"))
                    nn_weight = float(os.environ.get("NN_WEIGHT", "0.5"))
                    
                    # Detect message type and respond with neural insights
                    if "how" in message.lower() or "why" in message.lower():
                        return (
                            f"Based on my neural analysis (NN weight: {nn_weight:.2f}):\n\n"
                            f"I detected several activation patterns in your query. "
                            f"The key concepts identified were {[f'C{idx}' for idx, _ in highest_activations[:3]]}, "
                            f"with resonance level of {neural_results['resonance']:.2f}.\n\n"
                            f"This suggests your question involves complex relationships that would require "
                            f"detailed analysis. In a full implementation, I would provide a comprehensive response "
                            f"integrating these neural patterns with language model outputs."
                        )
                    else:
                        # Simplified mock response
                        return f"I've processed your message with neural weight {nn_weight:.2f} and LLM weight {llm_weight:.2f}. In a full implementation, this would integrate neural insights with LLM-generated content for a complete response."
                elif rsen_results:
                    # Use RSEN results for response
                    return f"RSEN analysis complete with resonance score {rsen_results.get('resonance_metrics', {}).get('resonance_score', 0)}. Your message has been processed through quantum and relativistic analysis layers."
                else:
                    # Simple mock response
                    if "hello" in message.lower():
                        return "Hello! I'm the Memory-Enhanced Mistral Assistant with Neural Processing. How can I help you today?"
                    elif "help" in message.lower():
                        return "I can assist with various tasks. With memory enabled, I can remember our conversations and use neural processing to enhance my responses."
                    elif "?" in message:
                        return "That's an interesting question. In this demo, I provide neural-enhanced responses using both memory and neural network analysis."
                    else:
                        return f"I've received your message: '{message}'. This is a demo of the Neural-Enhanced Mistral Chat."
        
        def generate_response(self, message, context=None):
            """Generate a response (mock implementation)"""
            # In a real implementation, this would call the Mistral API
            # This is a simple mock implementation based on keywords
            
            # With context enhancement
            if context:
                if "hello" in message.lower():
                    return "Hello! I'm the Memory-Enhanced Mistral Assistant. I'm using information from our past conversations to assist you better today."
                elif "help" in message.lower():
                    return f"I can assist with various tasks. With memory enabled, I can remember our conversations and use them for context. {context}"
                elif "?" in message:
                    return f"That's an interesting question. Let me provide an enhanced answer based on our previous conversations.\n\n{context}"
                else:
                    return f"I've received your message and I'm using my memory to provide a more personalized response.\n\n{context}"
            # Without context enhancement
            else:
                if "hello" in message.lower():
                    return "Hello! I'm the Memory-Enhanced Mistral Assistant. How can I help you today?"
                elif "help" in message.lower():
                    return "I can assist with various tasks. With memory enabled, I can remember our conversations and use them for context in future responses."
                elif "?" in message:
                    return "That's an interesting question. In this demo, I can provide basic responses and remember our conversation with onsite memory."
                else:
                    return f"I've received your message: '{message}'. This is a demo of the Memory-Enhanced Mistral Chat."
        
        def on_llm_slider_changed(self, value):
            """Handle LLM weight slider value change"""
            weight = value / 100.0  # Convert to 0.0-1.0 range
            self.llm_value_label.setText(f"{weight:.2f}")
            
            # Update environment variable
            os.environ["LLM_WEIGHT"] = str(weight)
            
            # Update weights in the system
            self.update_language_weights()
        
        def on_nn_slider_changed(self, value):
            """Handle neural network weight slider value change"""
            weight = value / 100.0  # Convert to 0.0-1.0 range
            self.nn_value_label.setText(f"{weight:.2f}")
            
            # Update environment variable
            os.environ["NN_WEIGHT"] = str(weight)
            
            # Update weights in the system
            self.update_language_weights()
        
        def update_language_weights(self):
            """Update language system weights"""
            if not isinstance(self.memory, DatabaseMemory) or not self.memory.central_language_node:
                return
                
            try:
                # Get weights from sliders
                llm_weight = self.llm_slider.value() / 100.0
                nn_weight = self.nn_slider.value() / 100.0
                
                # Update central node weights
                central_node = self.memory.central_language_node
                
                if hasattr(central_node, 'set_llm_weight'):
                    central_node.set_llm_weight(llm_weight)
                    self.chat_display.append(f"<i>Updated LLM weight to {llm_weight:.2f}</i>")
                
                if hasattr(central_node, 'set_nn_weight'):
                    central_node.set_nn_weight(nn_weight)
                    self.chat_display.append(f"<i>Updated NN weight to {nn_weight:.2f}</i>")
                
                # Update weights in neural processor if available
                if hasattr(self, 'neural_processor') and self.neural_processor:
                    # Neural processors typically don't have weight setters directly
                    # but we can store the values as metadata for reference
                    self.neural_processor.temperature = 0.7 + (nn_weight * 0.5)  # Scale temperature by NN weight
                    
                    # Store the weights in the processor's speaker config
                    if not self.neural_processor.speaker_config:
                        self.neural_processor.speaker_config = {}
                    
                    self.neural_processor.speaker_config["llm_weight"] = llm_weight
                    self.neural_processor.speaker_config["nn_weight"] = nn_weight
                    
                    self.chat_display.append(f"<i>Neural processor parameters updated</i>")
                
                # Update RSEN if available in the global scope
                try:
                    import sys
                    if 'RSEN_node' in sys.modules:
                        rsen_module = sys.modules['RSEN_node']
                        if hasattr(rsen_module, 'RSEN') and hasattr(rsen_module.RSEN, 'get_instance'):
                            rsen_instance = rsen_module.RSEN.get_instance()
                            if rsen_instance and hasattr(rsen_instance, 'set_nn_weight'):
                                rsen_instance.set_nn_weight(nn_weight)
                                self.chat_display.append(f"<i>Updated RSEN NN weight to {nn_weight:.2f}</i>")
                except Exception as e:
                    print(f"Error updating RSEN weights: {e}")
                    
            except Exception as e:
                self.chat_display.append(f"<i>Error updating weights: {e}</i>")
                QMessageBox.warning(self, "Weight Update Error", f"Error updating language system weights: {e}")
        
        def verify_database(self):
            """Verify database connections and status"""
            if not isinstance(self.memory, DatabaseMemory) or not self.memory.db_connected:
                return
                
            try:
                # Check database bridge status
                if self.memory.db_bridge and hasattr(self.memory.db_bridge, 'get_status'):
                    bridge_status = self.memory.db_bridge.get_status()
                    
                    # Format status information
                    status_text = "Database Bridge Status:\n"
                    status_text += f"- Initialized: {bridge_status.get('initialized', False)}\n"
                    status_text += f"- Connected to Central DB: {bridge_status.get('connected_to_central_db', False)}\n"
                    status_text += f"- Connected to Language DB: {bridge_status.get('connected_to_language_db', False)}\n"
                    status_text += f"- Sync Thread Running: {bridge_status.get('sync_thread_running', False)}\n"
                    
                    # Add sync stats if available
                    sync_stats = bridge_status.get('sync_stats', {})
                    if sync_stats:
                        status_text += "\nSync Statistics:\n"
                        status_text += f"- Sync Count: {sync_stats.get('sync_count', 0)}\n"
                        status_text += f"- Conversation Sync Count: {sync_stats.get('conversation_sync_count', 0)}\n"
                        status_text += f"- Pattern Sync Count: {sync_stats.get('pattern_sync_count', 0)}\n"
                        status_text += f"- Learning Sync Count: {sync_stats.get('learning_sync_count', 0)}\n"
                        status_text += f"- Error Count: {sync_stats.get('error_count', 0)}\n"
                    
                    # Show status dialog
                    QMessageBox.information(
                        self,
                        "Database Verification",
                        status_text
                    )
                    
                    # Update in chat
                    self.chat_display.append("<i>Database verification completed. See dialog for details.</i>")
                else:
                    QMessageBox.warning(
                        self,
                        "Database Verification",
                        "Database bridge not available or does not support status checking."
                    )
            except Exception as e:
                self.chat_display.append(f"<i>Error verifying database: {e}</i>")
                QMessageBox.warning(self, "Database Verification Error", f"Error verifying database: {e}")
        
        def sync_with_database(self):
            """Manually trigger database synchronization"""
            if isinstance(self.memory, DatabaseMemory):
                try:
                    # First verify the database is healthy
                    if self.memory.db_bridge and hasattr(self.memory.db_bridge, 'get_status'):
                        bridge_status = self.memory.db_bridge.get_status()
                        if not bridge_status.get('initialized', False):
                            self.chat_display.append("<i>Database bridge not initialized. Cannot sync.</i>")
                            return
                    
                    # Sync from memory to database
                    self.memory._sync_to_database()
                    
                    # Sync from database to memory
                    self.memory._sync_from_database()
                    
                    # Refresh the UI
                    self.refresh_memory_lists()
                    
                    self.chat_display.append("<i>Synchronized with database</i>")
                except Exception as e:
                    self.chat_display.append(f"<i>Error synchronizing with database: {e}</i>")
                    QMessageBox.warning(self, "Synchronization Error", f"Error synchronizing with database: {e}")
        
        def toggle_memory(self, state):
            """Toggle memory usage"""
            self.use_memory = (state == Qt.Checked)
            status = "enabled" if self.use_memory else "disabled"
            self.chat_display.append(f"<i>Memory usage {status}</i>")
        
        def add_knowledge(self):
            """Add knowledge to the memory"""
            # Get topic
            topic, ok = QInputDialog.getText(self, "Add Knowledge", "Enter topic:")
            if not ok or not topic:
                return
            
            # Get content
            content, ok = QInputDialog.getMultiLineText(self, "Add Knowledge", "Enter content:")
            if not ok or not content:
                return
            
            # Get source (optional)
            source, ok = QInputDialog.getText(self, "Add Knowledge", "Enter source (optional):")
            if not ok:
                return
            
            # Add to memory
            self.memory.add_knowledge(topic, content, source)
            
            # Refresh memory lists
            self.refresh_memory_lists()
            
            # Confirmation
            self.chat_display.append(f"<i>Added knowledge: {topic}</i>")
        
        def view_knowledge(self, item):
            """View knowledge entry"""
            if not item:
                return
            
            topic = item.text()
            entry = self.memory.knowledge_base.get(topic)
            
            if entry:
                QMessageBox.information(
                    self,
                    f"Knowledge: {topic}",
                    f"Content: {entry['content']}\n\nSource: {entry.get('source', 'Unknown')}\n\nAdded: {entry.get('added', 'Unknown')}"
                )
        
        def delete_knowledge(self):
            """Delete knowledge entry"""
            item = self.knowledge_list.currentItem()
            if not item:
                return
            
            topic = item.text()
            
            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Delete Knowledge",
                f"Are you sure you want to delete the knowledge entry '{topic}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if topic in self.memory.knowledge_base:
                    del self.memory.knowledge_base[topic]
                    self.memory.save_memory()
                    self.refresh_memory_lists()
                    self.chat_display.append(f"<i>Deleted knowledge: {topic}</i>")
        
        def view_conversation(self, item):
            """View conversation"""
            if not item:
                return
            
            # Extract index from item text (format: "1. message...")
            try:
                index_str = item.text().split('.')[0]
                index = int(index_str) - 1
                
                if 0 <= index < len(self.memory.conversation_history):
                    conversation = self.memory.conversation_history[index]
                    QMessageBox.information(
                        self,
                        "Conversation",
                        f"User: {conversation['user_message']}\n\nAssistant: {conversation['assistant_response']}\n\nTimestamp: {conversation.get('timestamp', 'Unknown')}"
                    )
            except:
                pass
        
        def clear_history(self):
            """Clear conversation history"""
            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Clear History",
                "Are you sure you want to clear all conversation history?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.memory.conversation_history = []
                self.memory.save_memory()
                self.refresh_memory_lists()
                self.chat_display.append("<i>Cleared all conversation history</i>")
        
        def show_memory_stats(self):
            """Show memory statistics"""
            stats = self.memory.get_stats()
            
            # Format stats into readable text
            stats_text = f"Conversations: {stats['conversations']}\n" \
                         f"Knowledge Entries: {stats['knowledge_entries']}\n" \
                         f"Memory File: {stats['memory_file']}\n" \
                         f"Last Updated: {stats['last_updated']}"
            
            # Add database info if available
            if 'database_available' in stats:
                db_status = "Connected" if stats.get('database_connected', False) else "Not Connected"
                stats_text += f"\n\nDatabase Available: {stats['database_available']}\n" \
                              f"Database Status: {db_status}"
                
                # Add bridge stats if available
                if 'database_bridge_status' in stats and isinstance(stats['database_bridge_status'], dict):
                    bridge = stats['database_bridge_status']
                    sync_stats = bridge.get('sync_stats', {})
                    stats_text += f"\n\nDatabase Bridge Status:" \
                                 f"\n- Last Sync: {bridge.get('last_sync_time', 'Never')}" \
                                 f"\n- Sync Count: {sync_stats.get('sync_count', 0)}" \
                                 f"\n- Conversations Synced: {sync_stats.get('conversation_sync_count', 0)}" \
                                 f"\n- Knowledge Entries Synced: {sync_stats.get('learning_sync_count', 0)}"
            
            QMessageBox.information(
                self,
                "Memory Statistics",
                stats_text
            )
        
        def closeEvent(self, event):
            """Handle window close event"""
            # Save memory before closing
            self.memory.save_memory()
            
            # Sync to database if connected
            if isinstance(self.memory, DatabaseMemory) and self.memory.db_connected:
                try:
                    self.memory._sync_to_database()
                except:
                    pass  # Ignore sync errors on close
                    
            super().closeEvent(event)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = SimpleMistralWindow()
    window.show()
    
    # Run application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 