"""
Dictionary Socket for AutoWiki Integration

This module provides a socket interface for the Dictionary Manager,
enabling integration with the AutoWiki system and frontend components.
It handles:
- AutoWiki data reception via socket
- Frontend visualization connections
- PySide6 integration for UI updates
- V5/V7 compatibility
"""

import os
import logging
import threading
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union

# Import V5 socket components if available
try:
    from src.v5.node_socket import NodeSocket
    from src.v5.v5_plugin import V5Plugin
    HAS_V5 = True
except ImportError:
    HAS_V5 = False
    # Create dummy base class
    class V5Plugin:
        def __init__(self, plugin_id=None, plugin_type=None, name=None):
            self.node_id = plugin_id or f"dictionary_socket_{uuid.uuid4().hex[:8]}"
            self.node_type = plugin_type or "dictionary_socket"
            self.name = name or f"Dictionary Socket ({self.node_id})"
    
    class NodeSocket:
        def __init__(self, node_id, socket_type="service"):
            self.node_id = node_id
            self.socket_type = socket_type
            self.message_handlers = {}
        
        def register_message_handler(self, message_type, handler):
            self.message_handlers[message_type] = handler
        
        def send_message(self, message):
            pass
        
        def connect_to(self, target_socket):
            pass

# Try to import PySide6 components
try:
    from PySide6.QtCore import QObject, Signal, Slot, QTimer
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    # Create dummy classes
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
    
    class QTimer:
        def __init__(self):
            pass
        
        def start(self, interval):
            pass
        
        def stop(self):
            pass

# Setup logging
logger = logging.getLogger(__name__)

class DictionarySocketPlugin(V5Plugin):
    """
    V5-compatible plugin for Dictionary Manager socket integration
    
    This class provides a socket interface that follows the V5 plugin
    architecture, making it compatible with the frontend socket manager.
    """
    
    def __init__(self, dictionary_manager=None, plugin_id=None):
        """
        Initialize the Dictionary Socket Plugin
        
        Args:
            dictionary_manager: The DictionaryManager instance to integrate with
            plugin_id: Optional plugin ID (generated if not provided)
        """
        plugin_id = plugin_id or f"dictionary_socket_{uuid.uuid4().hex[:8]}"
        super().__init__(plugin_id=plugin_id, plugin_type="dictionary_socket", name="Dictionary Socket")
        
        self.dictionary_manager = dictionary_manager
        self.socket = NodeSocket(self.node_id, "dictionary_socket") if HAS_V5 else None
        
        # Register message handlers
        if self.socket:
            self.socket.register_message_handler("autowiki_entry", self._handle_autowiki_entry)
            self.socket.register_message_handler("query_dictionary", self._handle_query_dictionary)
            self.socket.register_message_handler("dictionary_stats", self._handle_dictionary_stats)
            self.socket.register_message_handler("add_entry", self._handle_add_entry)
            self.socket.register_message_handler("update_entry", self._handle_update_entry)
            self.socket.register_message_handler("verify_entry", self._handle_verify_entry)
        
        logger.info(f"Dictionary Socket Plugin initialized with ID {self.node_id}")
    
    def get_socket_descriptor(self):
        """Return socket descriptor for frontend integration"""
        return {
            "plugin_id": self.node_id,
            "plugin_type": "dictionary_socket",
            "name": "Dictionary Socket",
            "message_types": [
                "autowiki_entry", 
                "query_dictionary",
                "dictionary_stats",
                "add_entry",
                "update_entry",
                "verify_entry"
            ],
            "subscription_mode": "request-response",
            "ui_components": ["dictionary_panel", "autowiki_integration"]
        }
    
    def get_status(self):
        """Get plugin status"""
        if not self.dictionary_manager:
            return {
                "status": "inactive",
                "plugin_id": self.node_id,
                "plugin_type": "dictionary_socket",
                "name": "Dictionary Socket",
                "error": "Dictionary Manager not available"
            }
        
        try:
            stats = self.dictionary_manager.get_dictionary_statistics()
            
            return {
                "status": "active",
                "plugin_id": self.node_id,
                "plugin_type": "dictionary_socket",
                "name": "Dictionary Socket",
                "stats": {
                    "total_entries": stats.get("total_entries", 0),
                    "verified_entries": stats.get("verified_entries", 0),
                    "total_relationships": stats.get("total_relationships", 0)
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "plugin_id": self.node_id,
                "plugin_type": "dictionary_socket",
                "name": "Dictionary Socket",
                "error": str(e)
            }
    
    def _handle_autowiki_entry(self, message):
        """Handle AutoWiki entry message"""
        if not self.dictionary_manager:
            self._send_error_response(message, "Dictionary Manager not available")
            return
        
        try:
            content = message.get("content", {})
            term = content.get("term")
            definition = content.get("definition")
            
            if not term or not definition:
                self._send_error_response(message, "Missing term or definition")
                return
            
            # Prepare metadata
            metadata = content.get("metadata", {})
            metadata["message_id"] = message.get("id")
            metadata["plugin_id"] = message.get("source")
            
            # Add entry to dictionary
            entry_id = self.dictionary_manager.add_entry(
                term=term,
                definition=definition,
                source="autowiki",
                confidence=content.get("confidence", 0.7),
                metadata=metadata,
                verified=False
            )
            
            # Add relationships if provided
            if "relationships" in content and isinstance(content["relationships"], list):
                for rel in content["relationships"]:
                    target_term = rel.get("term")
                    rel_type = rel.get("type", "related")
                    
                    if not target_term:
                        continue
                    
                    # Find or create target entry
                    target_entries = self.dictionary_manager.lookup_term(target_term)
                    target_id = None
                    
                    if target_entries:
                        # Use existing entry
                        target_id = target_entries[0]["id"]
                    else:
                        # Create stub entry
                        target_id = self.dictionary_manager.add_entry(
                            term=target_term,
                            definition=f"Stub entry related to {term}",
                            source="autowiki_generated",
                            confidence=0.5,
                            metadata={"stub": True, "generated_for": entry_id},
                            verified=False
                        )
                    
                    if target_id:
                        self.dictionary_manager.add_relationship(
                            source_id=entry_id,
                            target_id=target_id,
                            relationship_type=rel_type,
                            strength=rel.get("strength", 0.5)
                        )
            
            # Send success response
            self._send_response(message, {
                "success": True,
                "entry_id": entry_id,
                "term": term
            })
            
            logger.info(f"Processed AutoWiki entry for '{term}' with ID {entry_id}")
            
        except Exception as e:
            logger.error(f"Error handling AutoWiki entry: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_query_dictionary(self, message):
        """Handle dictionary query message"""
        if not self.dictionary_manager:
            self._send_error_response(message, "Dictionary Manager not available")
            return
        
        try:
            content = message.get("content", {})
            query_type = content.get("type", "search")
            
            if query_type == "search":
                query = content.get("query", "")
                limit = content.get("limit", 10)
                results = self.dictionary_manager.search_dictionary(query, limit)
                
                self._send_response(message, {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results)
                })
                
                logger.info(f"Processed dictionary search for '{query}' with {len(results)} results")
                
            elif query_type == "lookup":
                term = content.get("term", "")
                results = self.dictionary_manager.lookup_term(term)
                
                self._send_response(message, {
                    "success": True,
                    "term": term,
                    "results": results,
                    "count": len(results)
                })
                
                logger.info(f"Processed dictionary lookup for '{term}' with {len(results)} results")
                
            elif query_type == "entry":
                entry_id = content.get("entry_id")
                
                if not entry_id:
                    self._send_error_response(message, "Missing entry_id parameter")
                    return
                
                entry = self.dictionary_manager.get_entry(entry_id)
                
                if not entry:
                    self._send_error_response(message, f"Entry with ID {entry_id} not found")
                    return
                
                # Get related entries if requested
                related = []
                if content.get("include_related", False):
                    related = self.dictionary_manager.get_related_terms(entry_id)
                
                self._send_response(message, {
                    "success": True,
                    "entry": entry,
                    "related": related
                })
                
                logger.info(f"Processed dictionary entry request for ID {entry_id}")
                
            else:
                self._send_error_response(message, f"Unsupported query type: {query_type}")
            
        except Exception as e:
            logger.error(f"Error handling dictionary query: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_dictionary_stats(self, message):
        """Handle dictionary statistics request"""
        if not self.dictionary_manager:
            self._send_error_response(message, "Dictionary Manager not available")
            return
        
        try:
            stats = self.dictionary_manager.get_dictionary_statistics()
            
            self._send_response(message, {
                "success": True,
                "statistics": stats
            })
            
            logger.info("Processed dictionary statistics request")
            
        except Exception as e:
            logger.error(f"Error handling dictionary statistics request: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_add_entry(self, message):
        """Handle add entry message"""
        if not self.dictionary_manager:
            self._send_error_response(message, "Dictionary Manager not available")
            return
        
        try:
            content = message.get("content", {})
            term = content.get("term")
            definition = content.get("definition")
            
            if not term or not definition:
                self._send_error_response(message, "Missing term or definition")
                return
            
            entry_id = self.dictionary_manager.add_entry(
                term=term,
                definition=definition,
                source=content.get("source", "manual"),
                confidence=content.get("confidence", 0.7),
                metadata=content.get("metadata"),
                verified=content.get("verified", False)
            )
            
            self._send_response(message, {
                "success": True,
                "entry_id": entry_id,
                "term": term
            })
            
            logger.info(f"Added dictionary entry for '{term}' with ID {entry_id}")
            
        except Exception as e:
            logger.error(f"Error handling add entry: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_update_entry(self, message):
        """Handle update entry message"""
        if not self.dictionary_manager:
            self._send_error_response(message, "Dictionary Manager not available")
            return
        
        try:
            content = message.get("content", {})
            entry_id = content.get("entry_id")
            
            if not entry_id:
                self._send_error_response(message, "Missing entry_id parameter")
                return
            
            success = self.dictionary_manager.update_entry(
                entry_id=entry_id,
                definition=content.get("definition"),
                confidence=content.get("confidence"),
                metadata=content.get("metadata")
            )
            
            if success:
                entry = self.dictionary_manager.get_entry(entry_id)
                
                self._send_response(message, {
                    "success": True,
                    "entry_id": entry_id,
                    "entry": entry
                })
                
                logger.info(f"Updated dictionary entry with ID {entry_id}")
            else:
                self._send_error_response(message, f"Failed to update entry with ID {entry_id}")
            
        except Exception as e:
            logger.error(f"Error handling update entry: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _handle_verify_entry(self, message):
        """Handle verify entry message"""
        if not self.dictionary_manager:
            self._send_error_response(message, "Dictionary Manager not available")
            return
        
        try:
            content = message.get("content", {})
            entry_id = content.get("entry_id")
            verified = content.get("verified", True)
            
            if not entry_id:
                self._send_error_response(message, "Missing entry_id parameter")
                return
            
            success = self.dictionary_manager.verify_entry(entry_id, verified)
            
            if success:
                self._send_response(message, {
                    "success": True,
                    "entry_id": entry_id,
                    "verified": verified
                })
                
                logger.info(f"Verified dictionary entry with ID {entry_id}: {verified}")
            else:
                self._send_error_response(message, f"Failed to verify entry with ID {entry_id}")
            
        except Exception as e:
            logger.error(f"Error handling verify entry: {str(e)}")
            self._send_error_response(message, str(e))
    
    def _send_response(self, request_message, content):
        """Send a response to a request message"""
        if not self.socket:
            logger.warning("Cannot send response: NodeSocket not available")
            return
        
        try:
            message = {
                "type": "dictionary_response",
                "request_id": request_message.get("id"),
                "source": self.node_id,
                "timestamp": time.time(),
                "content": content
            }
            
            self.socket.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
    
    def _send_error_response(self, request_message, error_message):
        """Send an error response to a request message"""
        if not self.socket:
            logger.warning("Cannot send error response: NodeSocket not available")
            return
        
        try:
            message = {
                "type": "dictionary_response",
                "request_id": request_message.get("id"),
                "source": self.node_id,
                "timestamp": time.time(),
                "content": {
                    "success": False,
                    "error": error_message
                }
            }
            
            self.socket.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error response: {str(e)}")

# PySide6 adapter for the Dictionary Socket
class DictionarySocketAdapter(QObject):
    """PySide6 adapter for the Dictionary Socket Plugin"""
    # Signals
    entry_added = Signal(dict)
    entry_updated = Signal(dict)
    entry_verified = Signal(dict)
    autowiki_received = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, dictionary_socket=None, dictionary_manager=None):
        """
        Initialize the Dictionary Socket adapter
        
        Args:
            dictionary_socket: Optional DictionarySocketPlugin to use
            dictionary_manager: Optional DictionaryManager to use if socket is not provided
        """
        super().__init__()
        
        self.has_qt = HAS_PYSIDE6
        
        # Set up the dictionary socket
        if dictionary_socket and isinstance(dictionary_socket, DictionarySocketPlugin):
            self.socket = dictionary_socket
            self.available = True
            logger.info("Using provided Dictionary Socket Plugin")
        elif dictionary_manager and HAS_V5:
            try:
                self.socket = DictionarySocketPlugin(dictionary_manager)
                self.available = True
                logger.info("Created new Dictionary Socket Plugin with provided Dictionary Manager")
            except Exception as e:
                self.socket = None
                self.available = False
                logger.error(f"Failed to create Dictionary Socket Plugin: {str(e)}")
        else:
            self.socket = None
            self.available = False
            logger.warning("Neither Dictionary Socket Plugin nor Dictionary Manager provided")
        
        # Monitor socket for messages if available
        if self.has_qt and self.available and hasattr(self.socket, 'socket') and self.socket.socket:
            self._setup_message_monitoring()
    
    def _setup_message_monitoring(self):
        """Set up monitoring for socket messages"""
        from src.v5.node_socket import QtSocketAdapter
        
        try:
            # Create Qt adapter for the socket
            self.qt_socket = QtSocketAdapter(self.socket.socket)
            
            # Connect to messageReceived signal
            self.qt_socket.messageReceived.connect(self._process_socket_message)
            
            logger.info("Set up message monitoring for Dictionary Socket")
            
        except Exception as e:
            logger.error(f"Error setting up message monitoring: {str(e)}")
    
    @Slot(dict)
    def _process_socket_message(self, message):
        """Process messages from the socket"""
        message_type = message.get("type")
        
        if message_type == "autowiki_entry":
            # Forward to the UI
            self.autowiki_received.emit(message)
        elif message_type == "dictionary_response":
            content = message.get("content", {})
            
            if not content.get("success", True):
                error = content.get("error", "Unknown error")
                self.error_occurred.emit(error)
                return
            
            # Determine the type of response based on content
            if "entry_id" in content and "verified" in content:
                # Verify entry response
                self.entry_verified.emit(content)
            elif "entry_id" in content and "entry" in content:
                # Update entry response
                self.entry_updated.emit(content)
            elif "entry_id" in content:
                # Add entry response
                self.entry_added.emit(content)
    
    def send_message(self, message_type, content):
        """
        Send a message to the dictionary socket
        
        Args:
            message_type: Type of message to send
            content: Message content
            
        Returns:
            bool: Success flag
        """
        if not self.available or not hasattr(self.socket, 'socket') or not self.socket.socket:
            self.error_occurred.emit("Dictionary Socket not available")
            return False
        
        try:
            message = {
                "type": message_type,
                "id": str(uuid.uuid4()),
                "source": "dictionary_socket_adapter",
                "timestamp": time.time(),
                "content": content
            }
            
            self.socket.socket.send_message(message)
            logger.info(f"Sent {message_type} message to Dictionary Socket")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to Dictionary Socket: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
    
    def add_entry(self, term, definition, source="manual", confidence=0.7, metadata=None):
        """
        Add an entry through the socket
        
        Args:
            term: Term to define
            definition: Definition text
            source: Source of the definition
            confidence: Confidence level
            metadata: Additional metadata
            
        Returns:
            bool: Success flag
        """
        content = {
            "term": term,
            "definition": definition,
            "source": source,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        return self.send_message("add_entry", content)
    
    def update_entry(self, entry_id, definition=None, confidence=None, metadata=None):
        """
        Update an entry through the socket
        
        Args:
            entry_id: ID of the entry to update
            definition: New definition
            confidence: New confidence level
            metadata: New metadata
            
        Returns:
            bool: Success flag
        """
        content = {
            "entry_id": entry_id
        }
        
        if definition is not None:
            content["definition"] = definition
            
        if confidence is not None:
            content["confidence"] = confidence
            
        if metadata is not None:
            content["metadata"] = metadata
        
        return self.send_message("update_entry", content)
    
    def verify_entry(self, entry_id, verified=True):
        """
        Verify an entry through the socket
        
        Args:
            entry_id: ID of the entry to verify
            verified: Verification status
            
        Returns:
            bool: Success flag
        """
        content = {
            "entry_id": entry_id,
            "verified": verified
        }
        
        return self.send_message("verify_entry", content)
    
    def search_dictionary(self, query, limit=10):
        """
        Search the dictionary through the socket
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            bool: Success flag
        """
        content = {
            "type": "search",
            "query": query,
            "limit": limit
        }
        
        return self.send_message("query_dictionary", content)
    
    def get_entry(self, entry_id, include_related=False):
        """
        Get an entry by ID through the socket
        
        Args:
            entry_id: ID of the entry to retrieve
            include_related: Whether to include related entries
            
        Returns:
            bool: Success flag
        """
        content = {
            "type": "entry",
            "entry_id": entry_id,
            "include_related": include_related
        }
        
        return self.send_message("query_dictionary", content)
    
    def get_statistics(self):
        """
        Get dictionary statistics through the socket
        
        Returns:
            bool: Success flag
        """
        return self.send_message("dictionary_stats", {}) 