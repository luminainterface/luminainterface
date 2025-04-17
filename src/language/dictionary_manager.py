"""
Dictionary Manager for Enhanced Language System

This module provides dictionary database capabilities for:
- Storing concept definitions and associations
- Supporting AutoWiki integration for knowledge acquisition
- Providing socket-based communication for UI integration
- Enhancing the learning capabilities of the system

Uses SQLAlchemy for ORM and integrates with the existing DatabaseManager.
"""

import os
import json
import logging
import datetime
import threading
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session

# Import NodeSocket for AutoWiki integration
try:
    from src.v5.node_socket import NodeSocket
    HAS_NODE_SOCKET = True
except ImportError:
    HAS_NODE_SOCKET = False
    
try:
    from PySide6.QtCore import QObject, Signal, Slot, QTimer
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    # Create dummy base class
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

# Import DatabaseManager if available
try:
    from .database_manager import DatabaseManager, Base
    HAS_DATABASE_MANAGER = True
except ImportError:
    HAS_DATABASE_MANAGER = False
    Base = declarative_base()

# Setup logging
logger = logging.getLogger(__name__)

# Define ORM models
class DictionaryEntry(Base):
    """Dictionary entry for concept definition"""
    __tablename__ = 'dictionary_entries'
    
    id = Column(Integer, primary_key=True)
    concept_id = Column(Integer, ForeignKey('concepts.id'), nullable=True)
    term = Column(String(255), nullable=False, index=True)
    definition = Column(Text, nullable=False)
    source = Column(String(100), nullable=True)  # manual, autowiki, derived
    confidence = Column(Float, default=0.7)
    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    verified = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    related_entries = relationship("EntryRelationship", 
                                 foreign_keys="EntryRelationship.source_id",
                                 back_populates="source_entry",
                                 cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "concept_id": self.concept_id,
            "term": self.term,
            "definition": self.definition,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "verified": self.verified,
            "metadata": self.metadata
        }

class EntryRelationship(Base):
    """Relationship between dictionary entries"""
    __tablename__ = 'entry_relationships'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('dictionary_entries.id'), nullable=False)
    target_id = Column(Integer, ForeignKey('dictionary_entries.id'), nullable=False)
    relationship_type = Column(String(50), nullable=False)  # synonym, antonym, broader, narrower, etc.
    strength = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.datetime.now)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    source_entry = relationship("DictionaryEntry", foreign_keys=[source_id], back_populates="related_entries")
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata
        }

class DictionaryManager:
    """
    Dictionary manager for the Enhanced Language System
    
    Provides:
    - Concept definition storage and retrieval
    - AutoWiki integration for knowledge acquisition
    - Semantic relationships between concepts
    - Socket interface for UI integration
    """
    
    def __init__(self, data_dir="data/dictionary", db_manager=None):
        """
        Initialize the dictionary manager
        
        Args:
            data_dir: Directory to store dictionary data
            db_manager: Optional DatabaseManager instance to reuse
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database connection through DatabaseManager if available
        if db_manager and isinstance(db_manager, DatabaseManager):
            self.db_manager = db_manager
            self.engine = db_manager.engine
            self.Session = db_manager.Session
            logger.info("Using existing DatabaseManager for dictionary")
        else:
            # Create a standalone database connection
            db_path = os.path.join(data_dir, "dictionary.db")
            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
            self.Session = scoped_session(sessionmaker(bind=self.engine))
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            logger.info(f"Created dictionary database at {db_path}")
        
        # Initialize socket for AutoWiki integration if available
        self.node_socket = None
        if HAS_NODE_SOCKET:
            self.node_socket = NodeSocket(f"dictionary_manager_{uuid.uuid4().hex[:8]}", "dictionary")
            self.node_socket.register_message_handler("autowiki_entry", self._handle_autowiki_entry)
            self.node_socket.register_message_handler("verify_entry", self._handle_verify_entry)
            self.node_socket.register_message_handler("query_dictionary", self._handle_query_dictionary)
            logger.info("Initialized NodeSocket for dictionary AutoWiki integration")
        
        # Initialize signal emitter for UI updates
        self.has_qt = HAS_PYSIDE6
        if self.has_qt:
            self._initialize_signals()
        
        logger.info("Dictionary Manager initialized")
    
    def _initialize_signals(self):
        """Initialize Qt signals for UI integration"""
        class SignalEmitter(QObject):
            entry_added = Signal(dict)
            entry_updated = Signal(dict)
            relationship_added = Signal(dict)
            autowiki_received = Signal(dict)
        
        self.signals = SignalEmitter()
    
    def add_entry(self, term, definition, source="manual", confidence=0.7, 
                 concept_id=None, metadata=None, verified=False):
        """
        Add a new dictionary entry
        
        Args:
            term: The term/concept to define
            definition: The text definition
            source: Source of definition (manual, autowiki, derived)
            confidence: Confidence level (0.0-1.0) in the definition
            concept_id: Optional link to a concept in the concepts table
            metadata: Additional metadata for the entry
            verified: Whether the entry is verified
            
        Returns:
            int: ID of the created entry
        """
        session = self.Session()
        
        try:
            # Create new entry
            entry = DictionaryEntry(
                term=term,
                definition=definition,
                source=source,
                confidence=confidence,
                concept_id=concept_id,
                metadata=metadata or {},
                verified=verified
            )
            
            session.add(entry)
            session.commit()
            
            # Emit signal if available
            if self.has_qt:
                self.signals.entry_added.emit(entry.to_dict())
            
            logger.info(f"Added dictionary entry for '{term}' with ID {entry.id}")
            return entry.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding dictionary entry: {str(e)}")
            raise
        finally:
            session.close()
    
    def add_relationship(self, source_id, target_id, relationship_type, strength=0.5, metadata=None):
        """
        Add a relationship between dictionary entries
        
        Args:
            source_id: ID of the source entry
            target_id: ID of the target entry
            relationship_type: Type of relationship (synonym, antonym, etc.)
            strength: Strength of relationship (0.0-1.0)
            metadata: Additional metadata for the relationship
            
        Returns:
            int: ID of the created relationship
        """
        session = self.Session()
        
        try:
            # Create new relationship
            relationship = EntryRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=strength,
                metadata=metadata or {}
            )
            
            session.add(relationship)
            session.commit()
            
            # Emit signal if available
            if self.has_qt:
                self.signals.relationship_added.emit(relationship.to_dict())
            
            logger.info(f"Added {relationship_type} relationship between entries {source_id} and {target_id}")
            return relationship.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding entry relationship: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_entry(self, entry_id):
        """
        Get dictionary entry by ID
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            dict: Dictionary entry data
        """
        session = self.Session()
        
        try:
            entry = session.query(DictionaryEntry).filter(DictionaryEntry.id == entry_id).first()
            
            if not entry:
                logger.warning(f"Dictionary entry with ID {entry_id} not found")
                return None
                
            return entry.to_dict()
            
        except Exception as e:
            logger.error(f"Error retrieving dictionary entry: {str(e)}")
            return None
        finally:
            session.close()
    
    def lookup_term(self, term, threshold=0.0):
        """
        Look up term in the dictionary
        
        Args:
            term: Term to look up
            threshold: Minimum confidence threshold
            
        Returns:
            list: List of matching entries
        """
        session = self.Session()
        
        try:
            # Exact match first
            entries = (session.query(DictionaryEntry)
                      .filter(DictionaryEntry.term == term)
                      .filter(DictionaryEntry.confidence >= threshold)
                      .all())
            
            # If no exact matches, try substring match
            if not entries:
                entries = (session.query(DictionaryEntry)
                          .filter(DictionaryEntry.term.like(f"%{term}%"))
                          .filter(DictionaryEntry.confidence >= threshold)
                          .all())
            
            return [entry.to_dict() for entry in entries]
            
        except Exception as e:
            logger.error(f"Error looking up term '{term}': {str(e)}")
            return []
        finally:
            session.close()
    
    def get_related_terms(self, entry_id, relationship_type=None):
        """
        Get terms related to a dictionary entry
        
        Args:
            entry_id: ID of the entry
            relationship_type: Optional relationship type filter
            
        Returns:
            list: List of related entries
        """
        session = self.Session()
        
        try:
            # Base query
            query = (session.query(DictionaryEntry)
                    .join(EntryRelationship, EntryRelationship.target_id == DictionaryEntry.id)
                    .filter(EntryRelationship.source_id == entry_id))
            
            # Add relationship type filter if provided
            if relationship_type:
                query = query.filter(EntryRelationship.relationship_type == relationship_type)
            
            entries = query.all()
            return [entry.to_dict() for entry in entries]
            
        except Exception as e:
            logger.error(f"Error getting related terms for entry {entry_id}: {str(e)}")
            return []
        finally:
            session.close()
    
    def search_dictionary(self, query, limit=10):
        """
        Search the dictionary for entries matching the query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            list: List of matching entries
        """
        session = self.Session()
        
        try:
            # Search in terms and definitions
            entries = (session.query(DictionaryEntry)
                      .filter(sa.or_(
                          DictionaryEntry.term.like(f"%{query}%"),
                          DictionaryEntry.definition.like(f"%{query}%")
                      ))
                      .order_by(DictionaryEntry.confidence.desc())
                      .limit(limit)
                      .all())
            
            return [entry.to_dict() for entry in entries]
            
        except Exception as e:
            logger.error(f"Error searching dictionary for '{query}': {str(e)}")
            return []
        finally:
            session.close()
    
    def verify_entry(self, entry_id, verified=True):
        """
        Mark a dictionary entry as verified
        
        Args:
            entry_id: ID of the entry to verify
            verified: Verification status
            
        Returns:
            bool: Success flag
        """
        session = self.Session()
        
        try:
            entry = session.query(DictionaryEntry).filter(DictionaryEntry.id == entry_id).first()
            
            if not entry:
                logger.warning(f"Dictionary entry with ID {entry_id} not found")
                return False
                
            entry.verified = verified
            entry.updated_at = datetime.datetime.now()
            
            session.commit()
            
            logger.info(f"Dictionary entry {entry_id} marked as {'verified' if verified else 'unverified'}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error verifying dictionary entry: {str(e)}")
            return False
        finally:
            session.close()
    
    def update_entry(self, entry_id, definition=None, confidence=None, metadata=None):
        """
        Update a dictionary entry
        
        Args:
            entry_id: ID of the entry to update
            definition: New definition (if None, keep existing)
            confidence: New confidence level (if None, keep existing)
            metadata: New metadata (if None, keep existing)
            
        Returns:
            bool: Success flag
        """
        session = self.Session()
        
        try:
            entry = session.query(DictionaryEntry).filter(DictionaryEntry.id == entry_id).first()
            
            if not entry:
                logger.warning(f"Dictionary entry with ID {entry_id} not found")
                return False
                
            if definition is not None:
                entry.definition = definition
                
            if confidence is not None:
                entry.confidence = confidence
                
            if metadata is not None:
                if entry.metadata is None:
                    entry.metadata = metadata
                else:
                    entry.metadata.update(metadata)
            
            entry.updated_at = datetime.datetime.now()
            
            session.commit()
            
            # Emit signal if available
            if self.has_qt:
                self.signals.entry_updated.emit(entry.to_dict())
            
            logger.info(f"Updated dictionary entry {entry_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating dictionary entry: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_dictionary_statistics(self):
        """
        Get statistics about the dictionary
        
        Returns:
            dict: Dictionary statistics
        """
        session = self.Session()
        
        try:
            # Get total entries
            total_entries = session.query(sa.func.count(DictionaryEntry.id)).scalar()
            
            # Get verified entries
            verified_entries = session.query(sa.func.count(DictionaryEntry.id)) \
                              .filter(DictionaryEntry.verified == True).scalar()
            
            # Get entries by source
            sources = {}
            source_counts = session.query(DictionaryEntry.source, sa.func.count(DictionaryEntry.id)) \
                                 .group_by(DictionaryEntry.source).all()
            
            for source, count in source_counts:
                sources[source] = count
            
            # Get total relationships
            total_relationships = session.query(sa.func.count(EntryRelationship.id)).scalar()
            
            # Get relationships by type
            relationships = {}
            relationship_counts = session.query(EntryRelationship.relationship_type, sa.func.count(EntryRelationship.id)) \
                                      .group_by(EntryRelationship.relationship_type).all()
            
            for rel_type, count in relationship_counts:
                relationships[rel_type] = count
            
            return {
                "total_entries": total_entries,
                "verified_entries": verified_entries,
                "source_distribution": sources,
                "total_relationships": total_relationships,
                "relationship_distribution": relationships
            }
            
        except Exception as e:
            logger.error(f"Error getting dictionary statistics: {str(e)}")
            return {}
        finally:
            session.close()
    
    # AutoWiki socket handler methods
    def _handle_autowiki_entry(self, message):
        """Handle AutoWiki entry message from socket"""
        if not self.node_socket:
            logger.warning("Received AutoWiki message but NodeSocket is not available")
            return
        
        try:
            content = message.get("content", {})
            term = content.get("term")
            definition = content.get("definition")
            
            if not term or not definition:
                logger.warning("Received AutoWiki entry with missing term or definition")
                self._send_response(message, success=False, error="Missing term or definition")
                return
            
            # Create entry with AutoWiki source and unverified status
            metadata = content.get("metadata", {})
            metadata["message_id"] = message.get("id")
            metadata["source_message"] = message
            
            entry_id = self.add_entry(
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
                    target_entries = self.lookup_term(target_term)
                    target_id = None
                    
                    if target_entries:
                        # Use existing entry
                        target_id = target_entries[0]["id"]
                    else:
                        # Create stub entry
                        target_id = self.add_entry(
                            term=target_term,
                            definition=f"Stub entry related to {term}",
                            source="autowiki_generated",
                            confidence=0.5,
                            metadata={"stub": True},
                            verified=False
                        )
                    
                    if target_id:
                        self.add_relationship(
                            source_id=entry_id,
                            target_id=target_id,
                            relationship_type=rel_type,
                            strength=rel.get("strength", 0.5)
                        )
            
            # Emit signal if available
            if self.has_qt:
                entry_data = self.get_entry(entry_id)
                self.signals.autowiki_received.emit(entry_data)
            
            self._send_response(message, success=True, entry_id=entry_id)
            logger.info(f"Processed AutoWiki entry for '{term}' with ID {entry_id}")
            
        except Exception as e:
            logger.error(f"Error handling AutoWiki entry: {str(e)}")
            self._send_response(message, success=False, error=str(e))
    
    def _handle_verify_entry(self, message):
        """Handle verify entry message from socket"""
        if not self.node_socket:
            return
        
        try:
            content = message.get("content", {})
            entry_id = content.get("entry_id")
            verified = content.get("verified", True)
            
            if not entry_id:
                logger.warning("Received verify entry with missing entry_id")
                self._send_response(message, success=False, error="Missing entry_id")
                return
            
            success = self.verify_entry(entry_id, verified)
            self._send_response(message, success=success, entry_id=entry_id, verified=verified)
            logger.info(f"Processed verification for entry {entry_id}: {verified}")
            
        except Exception as e:
            logger.error(f"Error handling verify entry: {str(e)}")
            self._send_response(message, success=False, error=str(e))
    
    def _handle_query_dictionary(self, message):
        """Handle query dictionary message from socket"""
        if not self.node_socket:
            return
        
        try:
            content = message.get("content", {})
            query_type = content.get("type", "search")
            
            if query_type == "search":
                query = content.get("query", "")
                limit = content.get("limit", 10)
                results = self.search_dictionary(query, limit)
                
                self._send_response(message, success=True, results=results)
                logger.info(f"Processed dictionary search for '{query}'")
                
            elif query_type == "lookup":
                term = content.get("term", "")
                results = self.lookup_term(term)
                
                self._send_response(message, success=True, results=results)
                logger.info(f"Processed dictionary lookup for '{term}'")
                
            elif query_type == "statistics":
                stats = self.get_dictionary_statistics()
                
                self._send_response(message, success=True, statistics=stats)
                logger.info("Processed dictionary statistics request")
                
            else:
                logger.warning(f"Unsupported query type: {query_type}")
                self._send_response(message, success=False, error=f"Unsupported query type: {query_type}")
            
        except Exception as e:
            logger.error(f"Error handling query dictionary: {str(e)}")
            self._send_response(message, success=False, error=str(e))
    
    def _send_response(self, message, success=True, **kwargs):
        """Send response to a request message"""
        if not self.node_socket:
            return
        
        try:
            response = {
                "type": "dictionary_response",
                "request_id": message.get("id"),
                "success": success,
                "timestamp": datetime.datetime.now().isoformat(),
                "content": kwargs
            }
            
            self.node_socket.send_message(response)
            
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
    
    def connect_to_autowiki(self, autowiki_socket):
        """
        Connect to an AutoWiki socket for data exchange
        
        Args:
            autowiki_socket: The NodeSocket instance of the AutoWiki component
            
        Returns:
            bool: Success flag
        """
        if not self.node_socket:
            logger.warning("Cannot connect to AutoWiki: NodeSocket not available")
            return False
        
        try:
            self.node_socket.connect_to(autowiki_socket)
            logger.info(f"Connected to AutoWiki socket {autowiki_socket.node_id}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to AutoWiki socket: {str(e)}")
            return False
    
    def close(self):
        """Clean up resources"""
        logger.info("Closing Dictionary Manager")
        # Nothing to do here as Sessions are managed by scoped_session

# PySide6 Adapter for the Dictionary Manager
class DictionaryManagerAdapter(QObject):
    """PySide6 adapter for the Dictionary Manager"""
    # Signals
    entry_added = Signal(dict)
    entry_updated = Signal(dict)
    relationship_added = Signal(dict)
    autowiki_received = Signal(dict)
    error_occurred = Signal(str)
    statistics_updated = Signal(dict)
    
    def __init__(self, dictionary_manager=None, data_dir="data/dictionary", db_manager=None):
        """
        Initialize the Dictionary Manager adapter
        
        Args:
            dictionary_manager: Optional existing DictionaryManager to use
            data_dir: Directory for dictionary storage
            db_manager: Optional DatabaseManager instance to use
        """
        super().__init__()
        
        # Create dictionary manager if not provided
        if dictionary_manager and isinstance(dictionary_manager, DictionaryManager):
            self.dictionary = dictionary_manager
            self.available = True
        elif HAS_PYSIDE6:
            try:
                self.dictionary = DictionaryManager(data_dir=data_dir, db_manager=db_manager)
                self.available = True
                
                # Connect signals
                if hasattr(self.dictionary, 'has_qt') and self.dictionary.has_qt:
                    self.dictionary.signals.entry_added.connect(self.entry_added.emit)
                    self.dictionary.signals.entry_updated.connect(self.entry_updated.emit)
                    self.dictionary.signals.relationship_added.connect(self.relationship_added.emit)
                    self.dictionary.signals.autowiki_received.connect(self.autowiki_received.emit)
                
                logger.info("Dictionary Manager adapter initialized")
            except Exception as e:
                self.dictionary = None
                self.available = False
                logger.error(f"Failed to initialize Dictionary Manager: {str(e)}")
        else:
            self.dictionary = None
            self.available = False
            logger.warning("PySide6 not available for Dictionary Manager adapter")
        
        # For background stats updates
        if HAS_PYSIDE6:
            self.stats_timer = QTimer()
            self.stats_timer.timeout.connect(self.update_statistics)
            self.stats_timer.start(20000)  # Update every 20 seconds
    
    @Slot()
    def update_statistics(self):
        """Update statistics periodically"""
        if not self.available:
            return
        
        try:
            stats = self.dictionary.get_dictionary_statistics()
            self.statistics_updated.emit(stats)
        except Exception as e:
            logger.error(f"Error updating dictionary statistics: {str(e)}")
            self.error_occurred.emit(str(e))
    
    def add_entry(self, term, definition, source="manual", confidence=0.7, metadata=None):
        """Add a dictionary entry in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Dictionary Manager not available")
            return
        
        from src.language.pyside6_adapter import LanguageWorker
        
        worker = LanguageWorker(
            target_function=self.dictionary.add_entry,
            kwargs={
                "term": term,
                "definition": definition,
                "source": source,
                "confidence": confidence,
                "metadata": metadata or {}
            }
        )
        worker.task_complete.connect(self._on_entry_added)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    def search_dictionary(self, query, limit=10):
        """Search the dictionary in a non-blocking way"""
        if not self.available:
            self.error_occurred.emit("Dictionary Manager not available")
            return
        
        from src.language.pyside6_adapter import LanguageWorker
        
        worker = LanguageWorker(
            target_function=self.dictionary.search_dictionary,
            kwargs={
                "query": query,
                "limit": limit
            }
        )
        worker.task_complete.connect(self._on_search_complete)
        worker.task_error.connect(self.error_occurred.emit)
        worker.start()
    
    @Slot(dict)
    def _on_entry_added(self, result):
        """Handle entry added result"""
        entry_id = result.get("result")
        if entry_id:
            try:
                entry = self.dictionary.get_entry(entry_id)
                self.entry_added.emit(entry)
            except Exception as e:
                logger.error(f"Error getting added entry: {str(e)}")
    
    @Slot(dict)
    def _on_search_complete(self, result):
        """Handle search complete result"""
        # This method would typically emit a signal with the search results
        # For now, we're not implementing a specific signal for this
        pass 