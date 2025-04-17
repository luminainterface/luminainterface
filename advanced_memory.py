import numpy as np
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass, field
import logging
import json
import os
from datetime import datetime

@dataclass
class MemoryEntry:
    """Represents a memory entry with enhanced metadata"""
    content: Any
    timestamp: float
    confidence: float
    associations: List[str]
    breath_context: Optional[Dict[str, Any]] = None
    neural_pattern: Optional[np.ndarray] = None
    linguistic_pattern: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_id: str = field(default_factory=lambda: f"mem_{time.time():.6f}")

class AdvancedMemory:
    """Advanced memory system with breath-aware storage and retrieval"""
    
    def __init__(self, memory_dir: str = "data/memory"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_dir = memory_dir
        self.memory_index = {}
        self.recent_memories = []
        self.breath_detector = None
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
        
    def set_breath_detector(self, detector):
        """Set the breath detector for context-aware memory operations"""
        self.breath_detector = detector
        
    def store_memory(self, content: Any, confidence: float, associations: List[str],
                    neural_pattern: Optional[np.ndarray] = None,
                    linguistic_pattern: Optional[np.ndarray] = None) -> str:
        """Store a new memory with enhanced context"""
        try:
            # Get breath context if detector is available
            breath_context = None
            if self.breath_detector:
                breath_context = self.breath_detector.get_current_state()
                
            # Create memory entry
            memory = MemoryEntry(
                content=content,
                timestamp=time.time(),
                confidence=confidence,
                associations=associations,
                breath_context=breath_context,
                neural_pattern=neural_pattern,
                linguistic_pattern=linguistic_pattern,
                metadata={
                    'storage_time': datetime.now().isoformat(),
                    'memory_type': type(content).__name__
                }
            )
            
            # Store in memory
            self._save_memory(memory)
            self.memory_index[memory.memory_id] = memory
            self.recent_memories.append(memory)
            
            # Maintain recent memories list
            if len(self.recent_memories) > 100:
                self.recent_memories.pop(0)
                
            return memory.memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {str(e)}")
            raise
            
    def retrieve_memory(self, query: str, max_results: int = 5) -> List[MemoryEntry]:
        """Retrieve memories based on query with breath-aware context"""
        try:
            # Get current breath context
            current_breath_context = None
            if self.breath_detector:
                current_breath_context = self.breath_detector.get_current_state()
                
            # Score memories based on relevance and breath context
            scored_memories = []
            for memory in self.recent_memories:
                score = self._calculate_memory_score(memory, query, current_breath_context)
                scored_memories.append((memory, score))
                
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in scored_memories[:max_results]]
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {str(e)}")
            return []
            
    def _calculate_memory_score(self, memory: MemoryEntry, query: str,
                              current_breath_context: Optional[Dict[str, Any]]) -> float:
        """Calculate relevance score for memory retrieval"""
        score = 0.0
        
        # Content relevance
        if isinstance(memory.content, str):
            score += 0.5 if query.lower() in memory.content.lower() else 0.0
            
        # Association matching
        query_words = set(query.lower().split())
        memory_words = set(' '.join(memory.associations).lower().split())
        score += 0.3 * len(query_words.intersection(memory_words)) / len(query_words)
        
        # Breath context matching
        if current_breath_context and memory.breath_context:
            if (current_breath_context.get('current_pattern', {}).get('pattern_type') == 
                memory.breath_context.get('current_pattern', {}).get('pattern_type')):
                score += 0.2
                
        # Recency factor
        time_diff = time.time() - memory.timestamp
        recency_factor = np.exp(-time_diff / (24 * 3600))  # Decay over 24 hours
        score *= recency_factor
        
        return score
        
    def _save_memory(self, memory: MemoryEntry):
        """Save memory to disk"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            memory_dict = memory.__dict__.copy()
            if memory.neural_pattern is not None:
                memory_dict['neural_pattern'] = memory.neural_pattern.tolist()
            if memory.linguistic_pattern is not None:
                memory_dict['linguistic_pattern'] = memory.linguistic_pattern.tolist()
                
            # Save to file
            file_path = os.path.join(self.memory_dir, f"{memory.memory_id}.json")
            with open(file_path, 'w') as f:
                json.dump(memory_dict, f)
                
        except Exception as e:
            self.logger.error(f"Error saving memory to disk: {str(e)}")
            raise
            
    def _load_memories(self):
        """Load memories from disk"""
        try:
            for filename in os.listdir(self.memory_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.memory_dir, filename)
                    with open(file_path, 'r') as f:
                        memory_dict = json.load(f)
                        
                    # Convert lists back to numpy arrays
                    if 'neural_pattern' in memory_dict:
                        memory_dict['neural_pattern'] = np.array(memory_dict['neural_pattern'])
                    if 'linguistic_pattern' in memory_dict:
                        memory_dict['linguistic_pattern'] = np.array(memory_dict['linguistic_pattern'])
                        
                    memory = MemoryEntry(**memory_dict)
                    self.memory_index[memory.memory_id] = memory
                    self.recent_memories.append(memory)
                    
            # Sort recent memories by timestamp
            self.recent_memories.sort(key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error loading memories from disk: {str(e)}")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        return {
            'total_memories': len(self.memory_index),
            'recent_memories': len(self.recent_memories),
            'memory_types': self._get_memory_type_distribution(),
            'average_confidence': self._get_average_confidence(),
            'breath_context_enabled': self.breath_detector is not None
        }
        
    def _get_memory_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory types"""
        type_counts = {}
        for memory in self.memory_index.values():
            mem_type = memory.metadata.get('memory_type', 'unknown')
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        return type_counts
        
    def _get_average_confidence(self) -> float:
        """Get average confidence of stored memories"""
        if not self.memory_index:
            return 0.0
        return sum(memory.confidence for memory in self.memory_index.values()) / len(self.memory_index) 
 
 
from typing import Dict, List, Optional, Any
import time
from dataclasses import dataclass, field
import logging
import json
import os
from datetime import datetime

@dataclass
class MemoryEntry:
    """Represents a memory entry with enhanced metadata"""
    content: Any
    timestamp: float
    confidence: float
    associations: List[str]
    breath_context: Optional[Dict[str, Any]] = None
    neural_pattern: Optional[np.ndarray] = None
    linguistic_pattern: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_id: str = field(default_factory=lambda: f"mem_{time.time():.6f}")

class AdvancedMemory:
    """Advanced memory system with breath-aware storage and retrieval"""
    
    def __init__(self, memory_dir: str = "data/memory"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_dir = memory_dir
        self.memory_index = {}
        self.recent_memories = []
        self.breath_detector = None
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Load existing memories
        self._load_memories()
        
    def set_breath_detector(self, detector):
        """Set the breath detector for context-aware memory operations"""
        self.breath_detector = detector
        
    def store_memory(self, content: Any, confidence: float, associations: List[str],
                    neural_pattern: Optional[np.ndarray] = None,
                    linguistic_pattern: Optional[np.ndarray] = None) -> str:
        """Store a new memory with enhanced context"""
        try:
            # Get breath context if detector is available
            breath_context = None
            if self.breath_detector:
                breath_context = self.breath_detector.get_current_state()
                
            # Create memory entry
            memory = MemoryEntry(
                content=content,
                timestamp=time.time(),
                confidence=confidence,
                associations=associations,
                breath_context=breath_context,
                neural_pattern=neural_pattern,
                linguistic_pattern=linguistic_pattern,
                metadata={
                    'storage_time': datetime.now().isoformat(),
                    'memory_type': type(content).__name__
                }
            )
            
            # Store in memory
            self._save_memory(memory)
            self.memory_index[memory.memory_id] = memory
            self.recent_memories.append(memory)
            
            # Maintain recent memories list
            if len(self.recent_memories) > 100:
                self.recent_memories.pop(0)
                
            return memory.memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {str(e)}")
            raise
            
    def retrieve_memory(self, query: str, max_results: int = 5) -> List[MemoryEntry]:
        """Retrieve memories based on query with breath-aware context"""
        try:
            # Get current breath context
            current_breath_context = None
            if self.breath_detector:
                current_breath_context = self.breath_detector.get_current_state()
                
            # Score memories based on relevance and breath context
            scored_memories = []
            for memory in self.recent_memories:
                score = self._calculate_memory_score(memory, query, current_breath_context)
                scored_memories.append((memory, score))
                
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in scored_memories[:max_results]]
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {str(e)}")
            return []
            
    def _calculate_memory_score(self, memory: MemoryEntry, query: str,
                              current_breath_context: Optional[Dict[str, Any]]) -> float:
        """Calculate relevance score for memory retrieval"""
        score = 0.0
        
        # Content relevance
        if isinstance(memory.content, str):
            score += 0.5 if query.lower() in memory.content.lower() else 0.0
            
        # Association matching
        query_words = set(query.lower().split())
        memory_words = set(' '.join(memory.associations).lower().split())
        score += 0.3 * len(query_words.intersection(memory_words)) / len(query_words)
        
        # Breath context matching
        if current_breath_context and memory.breath_context:
            if (current_breath_context.get('current_pattern', {}).get('pattern_type') == 
                memory.breath_context.get('current_pattern', {}).get('pattern_type')):
                score += 0.2
                
        # Recency factor
        time_diff = time.time() - memory.timestamp
        recency_factor = np.exp(-time_diff / (24 * 3600))  # Decay over 24 hours
        score *= recency_factor
        
        return score
        
    def _save_memory(self, memory: MemoryEntry):
        """Save memory to disk"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            memory_dict = memory.__dict__.copy()
            if memory.neural_pattern is not None:
                memory_dict['neural_pattern'] = memory.neural_pattern.tolist()
            if memory.linguistic_pattern is not None:
                memory_dict['linguistic_pattern'] = memory.linguistic_pattern.tolist()
                
            # Save to file
            file_path = os.path.join(self.memory_dir, f"{memory.memory_id}.json")
            with open(file_path, 'w') as f:
                json.dump(memory_dict, f)
                
        except Exception as e:
            self.logger.error(f"Error saving memory to disk: {str(e)}")
            raise
            
    def _load_memories(self):
        """Load memories from disk"""
        try:
            for filename in os.listdir(self.memory_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.memory_dir, filename)
                    with open(file_path, 'r') as f:
                        memory_dict = json.load(f)
                        
                    # Convert lists back to numpy arrays
                    if 'neural_pattern' in memory_dict:
                        memory_dict['neural_pattern'] = np.array(memory_dict['neural_pattern'])
                    if 'linguistic_pattern' in memory_dict:
                        memory_dict['linguistic_pattern'] = np.array(memory_dict['linguistic_pattern'])
                        
                    memory = MemoryEntry(**memory_dict)
                    self.memory_index[memory.memory_id] = memory
                    self.recent_memories.append(memory)
                    
            # Sort recent memories by timestamp
            self.recent_memories.sort(key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error loading memories from disk: {str(e)}")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        return {
            'total_memories': len(self.memory_index),
            'recent_memories': len(self.recent_memories),
            'memory_types': self._get_memory_type_distribution(),
            'average_confidence': self._get_average_confidence(),
            'breath_context_enabled': self.breath_detector is not None
        }
        
    def _get_memory_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory types"""
        type_counts = {}
        for memory in self.memory_index.values():
            mem_type = memory.metadata.get('memory_type', 'unknown')
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        return type_counts
        
    def _get_average_confidence(self) -> float:
        """Get average confidence of stored memories"""
        if not self.memory_index:
            return 0.0
        return sum(memory.confidence for memory in self.memory_index.values()) / len(self.memory_index) 
 