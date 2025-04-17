from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import json
import os
from datetime import datetime

class MemoryNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.memory_store = {}
        self.memory_index = {}
        self.memory_path = "data/memory"
        
    def initialize(self) -> bool:
        """Initialize the memory node"""
        try:
            # Create memory directory if it doesn't exist
            os.makedirs(self.memory_path, exist_ok=True)
            
            # Load existing memories
            self._load_memories()
            
            self.active = True
            logging.info("MemoryNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize MemoryNode: {str(e)}")
            return False
            
    def _load_memories(self):
        """Load existing memories from storage"""
        try:
            memory_files = [f for f in os.listdir(self.memory_path) if f.endswith('.json')]
            for file in memory_files:
                with open(os.path.join(self.memory_path, file), 'r') as f:
                    memory_data = json.load(f)
                    self.memory_store.update(memory_data)
                    
            # Build memory index
            self._build_index()
        except Exception as e:
            logging.error(f"Error loading memories: {str(e)}")
            
    def _build_index(self):
        """Build search index for memories"""
        self.memory_index = {}
        for memory_id, memory in self.memory_store.items():
            # Index by timestamp
            timestamp = memory.get('timestamp', '')
            if timestamp:
                if timestamp not in self.memory_index:
                    self.memory_index[timestamp] = []
                self.memory_index[timestamp].append(memory_id)
                
            # Index by tags
            for tag in memory.get('tags', []):
                if tag not in self.memory_index:
                    self.memory_index[tag] = []
                self.memory_index[tag].append(memory_id)
                
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for memory operations"""
        try:
            operation = input_data.get('operation')
            if operation == 'store':
                return self._store_memory(input_data.get('data', {}))
            elif operation == 'retrieve':
                return self._retrieve_memory(input_data.get('query', {}))
            else:
                return {'error': 'Invalid operation'}
        except Exception as e:
            logging.error(f"Error processing memory operation: {str(e)}")
            return {'error': str(e)}
            
    def _store_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store new memory"""
        try:
            memory_id = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            memory_data['timestamp'] = datetime.now().isoformat()
            
            # Store memory
            self.memory_store[memory_id] = memory_data
            
            # Update index
            self._build_index()
            
            # Save to file
            self._save_memory(memory_id, memory_data)
            
            return {'status': 'success', 'memory_id': memory_id}
        except Exception as e:
            logging.error(f"Error storing memory: {str(e)}")
            return {'error': str(e)}
            
    def _retrieve_memory(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memories based on query"""
        try:
            results = []
            
            # Search by tags
            if 'tags' in query:
                for tag in query['tags']:
                    if tag in self.memory_index:
                        for memory_id in self.memory_index[tag]:
                            results.append(self.memory_store[memory_id])
                            
            # Search by timestamp range
            if 'start_time' in query and 'end_time' in query:
                start = datetime.fromisoformat(query['start_time'])
                end = datetime.fromisoformat(query['end_time'])
                
                for memory_id, memory in self.memory_store.items():
                    memory_time = datetime.fromisoformat(memory['timestamp'])
                    if start <= memory_time <= end:
                        results.append(memory)
                        
            return {'status': 'success', 'results': results}
        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            return {'error': str(e)}
            
    def _save_memory(self, memory_id: str, memory_data: Dict[str, Any]):
        """Save memory to file"""
        try:
            file_path = os.path.join(self.memory_path, f"{memory_id}.json")
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving memory to file: {str(e)}")
            
    def get_status(self) -> str:
        """Get current status of the memory node"""
        if not self.active:
            return "inactive"
        return f"active (memories: {len(self.memory_store)})"
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 