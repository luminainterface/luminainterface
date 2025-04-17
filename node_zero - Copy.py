import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import os
import traceback
from datetime import datetime
import logging
import sqlite3
import json

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Database Functionality ---
def _init_node_zero_database():
    """Initializes the SQLite database for Node Zero."""
    db_path = 'node_zero.db'
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Table for received data via broadcast
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS node_zero_received_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            content_snippet TEXT,
            metadata TEXT -- Store full data dict as JSON string
        )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"Initialized Node Zero database at {db_path}")
        return db_path
    except Exception as e:
        logger.error(f"Error initializing Node Zero database: {e}")
        logger.error(traceback.format_exc())
        return None

node_zero_db_path = _init_node_zero_database()

# --- Node Zero Logic ---

class ComplexLayer(nn.Module):
    def __init__(self, dimension, device):
        super().__init__()
        self.dimension = dimension
        self.real = nn.Parameter(torch.randn(dimension, device=device))
        self.imag = nn.Parameter(torch.randn(dimension, device=device))
        self.device = device
        
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        # Treat input as real component
        real_out = x * self.real - torch.zeros_like(x) * self.imag
        imag_out = x * self.imag + torch.zeros_like(x) * self.real
        return torch.complex(real_out, imag_out)

class NodeZero:
    def __init__(self):
        self._initialized = False
        self._active = False
        self.initialize()
        
    def initialize(self):
        """Initialize Node Zero"""
        try:
            # Add initialization logic here
            self._initialized = True
            self.model = nn.Sequential(
                nn.Linear(72, 36),  # 64 + 8 from hybrid and RESN
                nn.ReLU(),
                nn.Linear(36, 18),
                nn.ReLU(),
                nn.Linear(18, 9)
            )
            self.learning_rate = 0.001
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.integration_history = []
            self.max_history = 50
            self.hybrid_node = None
            logger.info("Initialized NodeZero")
            return True
        except Exception as e:
            self._initialized = False
            return False
            
    def activate(self):
        """Activate Node Zero"""
        if self._initialized:
            self._active = True
            return True
        return False
        
    def deactivate(self):
        """Deactivate Node Zero"""
        self._active = False
        
    def is_initialized(self):
        """Check if the node is initialized"""
        return self._initialized
        
    def is_active(self):
        """Check if the node is active"""
        return self._active and self._initialized
        
    def connect_to_hybrid(self, hybrid_node):
        """Connect to hybrid node for direct communication"""
        self.hybrid_node = hybrid_node
        logger.info("Connected to HybridNode")
        
    def get_status(self) -> Dict[str, Any]:
        """Return the current status and basic metrics of Node Zero."""
        return {
            'active': self.is_active(),
            'history_size': len(self.integration_history),
            'last_integration': self.integration_history[-1]['timestamp'].isoformat() if self.integration_history else None
        }
        
    def integrate_results(self, hybrid_result, resn_result):
        """Integrate results from hybrid and RESN nodes"""
        try:
            if not hybrid_result or not resn_result:
                return None
                
            # Concatenate embeddings
            hybrid_embedding = torch.FloatTensor(hybrid_result['embedding'])
            resn_embedding = torch.FloatTensor(resn_result['embedding'])
            combined = torch.cat([hybrid_embedding, resn_embedding])
            
            # Process through integration network
            with torch.no_grad():
                output = self.model(combined)
            
            # Merge and enhance relationships
            merged_relationships = self._merge_relationships(
                hybrid_result['relationships'],
                resn_result['relationships']
            )
            
            # Combine confidence scores
            merged_confidence = self._merge_confidence(
                hybrid_result['confidence_scores'],
                resn_result['confidence_scores']
            )
            
            # Update integration history
            self._update_history(output, merged_relationships)
            
            return {
                'key_concepts': list(set(hybrid_result['key_concepts'])),
                'relationships': merged_relationships,
                'confidence_scores': merged_confidence,
                'embedding': output.numpy(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error integrating results in NodeZero: {str(e)}")
            return None
            
    def _merge_relationships(self, hybrid_rels, resn_rels):
        """Merge and enhance relationships from both nodes"""
        merged = {}
        
        # Process both sets of relationships
        for rels in [hybrid_rels, resn_rels]:
            for rel in rels:
                key = (rel['source'], rel['target'])
                if key not in merged:
                    merged[key] = rel['confidence']
                else:
                    # Take the higher confidence score
                    merged[key] = max(merged[key], rel['confidence'])
        
        # Convert back to list format
        return [
            {'source': src, 'target': tgt, 'confidence': conf}
            for (src, tgt), conf in merged.items()
        ]
        
    def _merge_confidence(self, hybrid_conf, resn_conf):
        """Merge confidence scores with weighting"""
        merged = {}
        
        # Get all unique concepts
        all_concepts = set(hybrid_conf.keys()) | set(resn_conf.keys())
        
        for concept in all_concepts:
            # Get scores from both sources
            hybrid_score = hybrid_conf.get(concept, 0)
            resn_score = resn_conf.get(concept, 0)
            
            # Weight RESN scores slightly higher for temporal stability
            merged[concept] = (hybrid_score * 0.4 + resn_score * 0.6)
            
        return merged
        
    def _update_history(self, output, relationships):
        """Update integration history"""
        self.integration_history.append({
            'output': output,
            'relationships': relationships,
            'timestamp': datetime.now()
        })
        if len(self.integration_history) > self.max_history:
            self.integration_history.pop(0)

# --- Broadcast Receiver Function ---
def receive_data(data: dict):
    """Receives data broadcasted by quantum_infection."""
    timestamp = datetime.now().isoformat()
    source = data.get('source', 'unknown')
    content_snippet = str(data.get('text', data.get('content', '')))[:200]
    logger.info(f"Node Zero received data from {source} at {timestamp}")

    # Store received data
    if node_zero_db_path:
        try:
            metadata_json = json.dumps(data)
            conn = sqlite3.connect(node_zero_db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO node_zero_received_data (source, content_snippet, metadata) VALUES (?, ?, ?)",
                           (source, content_snippet, metadata_json))
            conn.commit()
            conn.close()
        except Exception as db_e:
            logger.error(f"Error storing received data in Node Zero DB: {db_e}")
    else:
         logger.warning("Node Zero DB path not set. Cannot store received data.")

    # Node Zero specific processing (if any)
    # For now, just acknowledge receipt
    logger.debug(f"Node Zero processed (logged) data from {source}.")

# --- Initialization (if needed globally) ---
# node_zero_instance = NodeZero(...) # If a global instance is needed

# Example usage (if run standalone for testing)
if __name__ == "__main__":
    logger.info("Node Zero Module - Standalone Test")
    test_data_packet = {
        'source': 'test_broadcast',
        'content': 'Testing Node Zero reception.',
        'timestamp': datetime.now().isoformat()
    }
    receive_data(test_data_packet)
    print("\n--- Test reception complete. Check logs and node_zero.db ---") 