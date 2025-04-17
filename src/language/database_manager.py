"""
Database Manager for Enhanced Language System

This module provides database storage and retrieval capabilities for:
- Conversation history and exchange metadata
- Neural network learning patterns
- User interaction statistics
- System performance metrics
- Pattern recognition data

Uses SQLAlchemy for ORM and SQLite as the underlying database engine.
"""

import os
import json
import logging
import datetime
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, JSON, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool

# Setup logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy base class
Base = declarative_base()

# Define ORM models
class Conversation(Base):
    """Conversation entity representing a series of exchanges"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.datetime.now, index=True)
    end_time = Column(DateTime, nullable=True)
    title = Column(String(255), nullable=True)
    meta_data = Column(JSON, nullable=True)
    exchanges = relationship("Exchange", back_populates="conversation", cascade="all, delete-orphan")
    
    # Define indexes
    __table_args__ = (
        Index('idx_conversation_start_time', start_time),
        Index('idx_conversation_end_time', end_time),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "title": self.title,
            "meta_data": self.meta_data,
            "exchange_count": len(self.exchanges) if self.exchanges else 0
        }

class Exchange(Base):
    """Individual message exchange in a conversation"""
    __tablename__ = 'exchanges'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.now, index=True)
    user_message = Column(Text, nullable=False)
    system_response = Column(Text, nullable=True)
    llm_weight = Column(Float, default=0.5)
    nn_weight = Column(Float, default=0.5)
    learning_value = Column(Float, default=0.0, index=True)
    consciousness_level = Column(Float, default=0.0, index=True)
    neural_score = Column(Float, default=0.0)
    pattern_confidence = Column(Float, default=0.0)
    meta_data = Column(JSON, nullable=True)
    
    conversation = relationship("Conversation", back_populates="exchanges")
    patterns = relationship("PatternDetection", back_populates="exchange", cascade="all, delete-orphan")
    concepts = relationship("Concept", back_populates="exchange", cascade="all, delete-orphan")
    
    # Define composite indexes for common queries
    __table_args__ = (
        Index('idx_exchange_conversation_timestamp', conversation_id, timestamp),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_message": self.user_message,
            "system_response": self.system_response,
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "learning_value": self.learning_value,
            "consciousness_level": self.consciousness_level,
            "neural_score": self.neural_score,
            "pattern_confidence": self.pattern_confidence,
            "meta_data": self.meta_data,
            "patterns": [p.to_dict() for p in self.patterns] if self.patterns else [],
            "concepts": [c.to_dict() for c in self.concepts] if self.concepts else []
        }

class PatternDetection(Base):
    """Pattern detected in an exchange"""
    __tablename__ = 'pattern_detections'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), index=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    pattern_text = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0, index=True)
    detection_method = Column(String(50), nullable=True)  # rule_based, llm, neural
    
    exchange = relationship("Exchange", back_populates="patterns")
    
    def to_dict(self):
        return {
            "id": self.id,
            "exchange_id": self.exchange_id,
            "pattern_type": self.pattern_type,
            "pattern_text": self.pattern_text,
            "confidence": self.confidence,
            "detection_method": self.detection_method
        }

class Concept(Base):
    """Key concept extracted from an exchange"""
    __tablename__ = 'concepts'
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'))
    concept_text = Column(String(100), nullable=False)
    importance = Column(Float, default=0.5)
    related_concepts = Column(JSON, nullable=True)  # List of related concept IDs
    
    exchange = relationship("Exchange", back_populates="concepts")
    
    def to_dict(self):
        return {
            "id": self.id,
            "exchange_id": self.exchange_id,
            "concept_text": self.concept_text,
            "importance": self.importance,
            "related_concepts": self.related_concepts
        }

class UserPreference(Base):
    """User preferences and feedback learned through interactions"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    preference_key = Column(String(100), nullable=False)
    preference_value = Column(Float, default=0.5)
    confidence = Column(Float, default=0.0)
    examples = Column(JSON, nullable=True)  # List of exchange IDs supporting this preference
    last_updated = Column(DateTime, default=datetime.datetime.now)
    
    def to_dict(self):
        return {
            "id": self.id,
            "preference_key": self.preference_key,
            "preference_value": self.preference_value,
            "confidence": self.confidence,
            "examples": self.examples,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

class SystemMetric(Base):
    """System performance and learning metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=True)  # performance, learning, error
    details = Column(JSON, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_type": self.metric_type,
            "details": self.details
        }

class DatabaseManager:
    """
    Database manager for the Enhanced Language System
    
    Provides:
    - Conversation storage and retrieval
    - Pattern learning persistence
    - User preference tracking
    - System metrics collection
    - Query interface for conversation history
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, data_dir: str = "data/db", db_name: str = "language_system.db"):
        """Get singleton instance of DatabaseManager"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = DatabaseManager(data_dir, db_name)
            return cls._instance
    
    def __init__(self, data_dir: str = "data/db", db_name: str = "language_system.db"):
        """
        Initialize the database manager
        
        Args:
            data_dir: Directory to store the database file
            db_name: Name of the database file
        """
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, db_name)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", 
            echo=False,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600  # Recycle connections every hour
        )
        
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
        # Thread lock for database operations
        self.db_lock = threading.RLock()
        
        # Cache for frequently accessed data
        self.cache = {
            "recent_conversations": {"data": None, "timestamp": 0, "ttl": 60},  # 60 sec TTL
            "learning_statistics": {"data": None, "timestamp": 0, "ttl": 300},  # 5 min TTL
        }
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "failed_queries": 0,
            "query_time_total": 0,
            "last_optimization": 0
        }
        
        # Auto-optimization settings
        self.auto_optimize_interval = 3600  # 1 hour
        self.operations_since_optimization = 0
        self.optimization_threshold = 1000  # Number of operations before auto-optimization
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if valid data exists in cache"""
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if cache_entry["data"] is not None:
                # Check if cache is still valid
                if time.time() - cache_entry["timestamp"] < cache_entry["ttl"]:
                    return cache_entry["data"]
        return None
    
    def _update_cache(self, cache_key: str, data: Any):
        """Update cache with new data"""
        if cache_key in self.cache:
            self.cache[cache_key]["data"] = data
            self.cache[cache_key]["timestamp"] = time.time()
    
    def _record_query_metrics(self, success: bool, query_time: float):
        """Record query metrics for performance analysis"""
        self.metrics["total_queries"] += 1
        if not success:
            self.metrics["failed_queries"] += 1
        self.metrics["query_time_total"] += query_time
        self.operations_since_optimization += 1
        
        # Check if we should auto-optimize
        if (self.operations_since_optimization >= self.optimization_threshold or
            time.time() - self.metrics["last_optimization"] > self.auto_optimize_interval):
            self.optimize_database()
    
    def create_conversation(self, title: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """
        Create a new conversation
        
        Args:
            title: Optional title for the conversation
            metadata: Optional metadata for the conversation
            
        Returns:
            int: ID of the created conversation
        """
        with self.db_lock:
            session = self.Session()
            try:
                conversation = Conversation(title=title, meta_data=metadata)
                session.add(conversation)
                session.commit()
                conversation_id = conversation.id
                logger.info(f"Created new conversation with ID: {conversation_id}")
                return conversation_id
            except Exception as e:
                session.rollback()
                logger.error(f"Error creating conversation: {e}")
                raise
            finally:
                session.close()
    
    def store_exchange(self, 
                      conversation_id: int,
                      user_message: str,
                      system_response: str,
                      llm_weight: float,
                      nn_weight: float,
                      learning_value: float = 0.0,
                      consciousness_level: float = 0.0,
                      neural_score: float = 0.0, 
                      pattern_confidence: float = 0.0,
                      metadata: Optional[Dict] = None) -> int:
        """
        Store a conversation exchange
        
        Args:
            conversation_id: ID of the conversation
            user_message: Message from the user
            system_response: Response from the system
            llm_weight: Current LLM weight used
            nn_weight: Current neural network weight used
            learning_value: Value of this exchange for learning (0.0-1.0)
            consciousness_level: Consciousness level detected
            neural_score: Neural processing score
            pattern_confidence: Confidence in pattern detection
            metadata: Optional metadata for the exchange
            
        Returns:
            int: ID of the created exchange
        """
        with self.db_lock:
            session = self.Session()
            try:
                exchange = Exchange(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    system_response=system_response,
                    llm_weight=llm_weight,
                    nn_weight=nn_weight,
                    learning_value=learning_value,
                    consciousness_level=consciousness_level,
                    neural_score=neural_score,
                    pattern_confidence=pattern_confidence,
                    meta_data=metadata
                )
                session.add(exchange)
                session.commit()
                exchange_id = exchange.id
                
                # Update conversation end time
                conversation = session.query(Conversation).get(conversation_id)
                if conversation:
                    conversation.end_time = datetime.datetime.now()
                    session.commit()
                
                logger.info(f"Stored exchange with ID: {exchange_id} in conversation: {conversation_id}")
                return exchange_id
            except Exception as e:
                session.rollback()
                logger.error(f"Error storing exchange: {e}")
                raise
            finally:
                session.close()
    
    def add_pattern_detection(self, exchange_id: int, pattern_type: str, pattern_text: str, 
                             confidence: float, detection_method: str) -> int:
        """
        Add a pattern detection to an exchange
        
        Args:
            exchange_id: ID of the exchange
            pattern_type: Type of pattern detected
            pattern_text: Text of the pattern
            confidence: Confidence in the pattern detection
            detection_method: Method used to detect the pattern
            
        Returns:
            int: ID of the created pattern detection
        """
        with self.db_lock:
            session = self.Session()
            try:
                pattern = PatternDetection(
                    exchange_id=exchange_id,
                    pattern_type=pattern_type,
                    pattern_text=pattern_text,
                    confidence=confidence,
                    detection_method=detection_method
                )
                session.add(pattern)
                session.commit()
                pattern_id = pattern.id
                logger.info(f"Added pattern detection with ID: {pattern_id} to exchange: {exchange_id}")
                return pattern_id
            except Exception as e:
                session.rollback()
                logger.error(f"Error adding pattern detection: {e}")
                raise
            finally:
                session.close()
    
    def add_concept(self, exchange_id: int, concept_text: str, 
                   importance: float = 0.5, related_concepts: Optional[List] = None) -> int:
        """
        Add a concept to an exchange
        
        Args:
            exchange_id: ID of the exchange
            concept_text: Text of the concept
            importance: Importance of the concept (0.0-1.0)
            related_concepts: Optional list of related concept IDs
            
        Returns:
            int: ID of the created concept
        """
        with self.db_lock:
            session = self.Session()
            try:
                concept = Concept(
                    exchange_id=exchange_id,
                    concept_text=concept_text,
                    importance=importance,
                    related_concepts=related_concepts
                )
                session.add(concept)
                session.commit()
                concept_id = concept.id
                logger.info(f"Added concept with ID: {concept_id} to exchange: {exchange_id}")
                return concept_id
            except Exception as e:
                session.rollback()
                logger.error(f"Error adding concept: {e}")
                raise
            finally:
                session.close()
    
    def update_user_preference(self, preference_key: str, preference_value: float, 
                              confidence: float, example_exchange_id: Optional[int] = None) -> int:
        """
        Update a user preference
        
        Args:
            preference_key: Key for the preference
            preference_value: Value for the preference (typically 0.0-1.0)
            confidence: Confidence in this preference
            example_exchange_id: Optional exchange ID supporting this preference
            
        Returns:
            int: ID of the created or updated preference
        """
        with self.db_lock:
            session = self.Session()
            try:
                # Check if preference exists
                preference = session.query(UserPreference).filter_by(preference_key=preference_key).first()
                
                if preference:
                    # Update existing preference
                    preference.preference_value = preference_value
                    preference.confidence = confidence
                    preference.last_updated = datetime.datetime.now()
                    
                    # Add example if provided
                    if example_exchange_id:
                        if preference.examples is None:
                            preference.examples = [example_exchange_id]
                        else:
                            examples = preference.examples
                            if example_exchange_id not in examples:
                                examples.append(example_exchange_id)
                            preference.examples = examples
                else:
                    # Create new preference
                    examples = [example_exchange_id] if example_exchange_id else []
                    preference = UserPreference(
                        preference_key=preference_key,
                        preference_value=preference_value,
                        confidence=confidence,
                        examples=examples
                    )
                    session.add(preference)
                
                session.commit()
                preference_id = preference.id
                logger.info(f"Updated user preference: {preference_key} with ID: {preference_id}")
                return preference_id
            except Exception as e:
                session.rollback()
                logger.error(f"Error updating user preference: {e}")
                raise
            finally:
                session.close()
    
    def record_metric(self, metric_name: str, metric_value: float, 
                     metric_type: str, details: Optional[Dict] = None) -> int:
        """
        Record a system metric
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric (performance, learning, error)
            details: Optional details about the metric
            
        Returns:
            int: ID of the created metric
        """
        with self.db_lock:
            session = self.Session()
            try:
                metric = SystemMetric(
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_type=metric_type,
                    details=details
                )
                session.add(metric)
                session.commit()
                metric_id = metric.id
                return metric_id
            except Exception as e:
                session.rollback()
                logger.error(f"Error recording metric: {e}")
                raise
            finally:
                session.close()
    
    def get_conversation(self, conversation_id: int) -> Dict:
        """
        Get a conversation by ID
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dict: Conversation data
        """
        with self.db_lock:
            session = self.Session()
            try:
                conversation = session.query(Conversation).get(conversation_id)
                if conversation:
                    return conversation.to_dict()
                return None
            except Exception as e:
                logger.error(f"Error getting conversation: {e}")
                raise
            finally:
                session.close()
    
    def get_exchange(self, exchange_id: int) -> Dict:
        """
        Get an exchange by ID
        
        Args:
            exchange_id: ID of the exchange
            
        Returns:
            Dict: Exchange data
        """
        with self.db_lock:
            session = self.Session()
            try:
                exchange = session.query(Exchange).get(exchange_id)
                if exchange:
                    return exchange.to_dict()
                return None
            except Exception as e:
                logger.error(f"Error getting exchange: {e}")
                raise
            finally:
                session.close()
    
    def get_conversation_exchanges(self, conversation_id: int, limit: int = 10) -> List[Dict]:
        """
        Get exchanges for a conversation
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of exchanges to return
            
        Returns:
            List[Dict]: List of exchange data
        """
        with self.db_lock:
            session = self.Session()
            try:
                exchanges = session.query(Exchange).filter_by(conversation_id=conversation_id)\
                    .order_by(Exchange.timestamp.desc()).limit(limit).all()
                return [exchange.to_dict() for exchange in exchanges]
            except Exception as e:
                logger.error(f"Error getting conversation exchanges: {e}")
                raise
            finally:
                session.close()
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """
        Get recent conversations
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List[Dict]: List of conversation data
        """
        # Check cache first
        cache_key = "recent_conversations"
        cached_data = self._check_cache(cache_key)
        if cached_data is not None and len(cached_data) >= limit:
            return cached_data[:limit]
        
        with self.db_lock:
            start_time = time.time()
            session = self.Session()
            success = True
            try:
                conversations = session.query(Conversation)\
                    .order_by(Conversation.start_time.desc()).limit(limit).all()
                result = [conversation.to_dict() for conversation in conversations]
                
                # Update cache
                self._update_cache(cache_key, result)
                
                return result
            except Exception as e:
                success = False
                logger.error(f"Error getting recent conversations: {e}")
                raise
            finally:
                session.close()
                self._record_query_metrics(success, time.time() - start_time)
    
    def search_exchanges(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search exchanges for a query
        
        Args:
            query: Search query
            limit: Maximum number of exchanges to return
            
        Returns:
            List[Dict]: List of matching exchange data
        """
        with self.db_lock:
            session = self.Session()
            try:
                exchanges = session.query(Exchange)\
                    .filter(Exchange.user_message.like(f"%{query}%") | Exchange.system_response.like(f"%{query}%"))\
                    .order_by(Exchange.timestamp.desc()).limit(limit).all()
                return [exchange.to_dict() for exchange in exchanges]
            except Exception as e:
                logger.error(f"Error searching exchanges: {e}")
                raise
            finally:
                session.close()
    
    def get_learning_statistics(self) -> Dict:
        """
        Get learning statistics
        
        Returns:
            Dict: Learning statistics
        """
        # Check cache first
        cache_key = "learning_statistics"
        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        with self.db_lock:
            start_time = time.time()
            session = self.Session()
            success = True
            try:
                total_exchanges = session.query(Exchange).count()
                avg_learning_value = session.query(sa.func.avg(Exchange.learning_value)).scalar() or 0
                avg_consciousness = session.query(sa.func.avg(Exchange.consciousness_level)).scalar() or 0
                total_patterns = session.query(PatternDetection).count()
                total_concepts = session.query(Concept).count()
                total_preferences = session.query(UserPreference).count()
                
                result = {
                    "total_exchanges": total_exchanges,
                    "avg_learning_value": round(avg_learning_value, 3),
                    "avg_consciousness_level": round(avg_consciousness, 3),
                    "total_patterns_detected": total_patterns,
                    "total_concepts_extracted": total_concepts,
                    "total_user_preferences": total_preferences,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Update cache
                self._update_cache(cache_key, result)
                
                return result
            except Exception as e:
                success = False
                logger.error(f"Error getting learning statistics: {e}")
                raise
            finally:
                session.close()
                self._record_query_metrics(success, time.time() - start_time)
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Optimize database performance
        
        - Runs VACUUM to reclaim unused space
        - Runs ANALYZE to update query planner statistics
        - Clears caches
        
        Returns:
            Dict: Optimization statistics
        """
        with self.db_lock:
            start_time = time.time()
            session = self.Session()
            try:
                # Run VACUUM to reclaim unused space
                session.execute("VACUUM")
                
                # Run ANALYZE to update query planner statistics
                session.execute("ANALYZE")
                
                # Clear all caches
                for cache_key in self.cache:
                    self.cache[cache_key]["data"] = None
                    self.cache[cache_key]["timestamp"] = 0
                
                # Update optimization metrics
                self.metrics["last_optimization"] = time.time()
                self.operations_since_optimization = 0
                
                # Calculate optimization stats
                optimization_time = time.time() - start_time
                optimization_stats = {
                    "time_taken": optimization_time,
                    "caches_cleared": len(self.cache),
                    "operations_since_last": self.operations_since_optimization,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Database optimized in {optimization_time:.2f}s")
                return optimization_stats
                
            except Exception as e:
                logger.error(f"Error optimizing database: {e}")
                raise
            finally:
                session.close()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get database performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        with self.db_lock:
            # Calculate derived metrics
            avg_query_time = 0
            if self.metrics["total_queries"] > 0:
                avg_query_time = self.metrics["query_time_total"] / self.metrics["total_queries"]
                
            # Copy metrics to avoid modification during calculation
            metrics = dict(self.metrics)
            
            # Add additional stats
            metrics.update({
                "avg_query_time": round(avg_query_time, 5),
                "failure_rate": round(metrics["failed_queries"] / max(1, metrics["total_queries"]), 4),
                "cache_entries": len(self.cache),
                "time_since_optimization": round(time.time() - metrics["last_optimization"], 1),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return metrics
    
    def close(self):
        """Close the database connection"""
        self.Session.remove()
        logger.info("Database connection closed") 