#!/usr/bin/env python3
"""
V8 to V7 Knowledge Bridge 

This module provides specific functionality to transfer knowledge between the v8 
Knowledge CI/CD system and the v7 Holographic system. It extends the basic bridge
with specialized knowledge transfer capabilities.
"""

import os
import sys
import json
import uuid
import time
import logging
import threading
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/v8_v7_knowledge_bridge_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v8_v7_knowledge_bridge")

class V8ToV7KnowledgeBridge:
    """
    Specialized bridge for transferring knowledge between v8 Knowledge CI/CD system
    and v7 Holographic system.
    """
    
    def __init__(self, v8_health_port=8765, v7_knowledge_path=None):
        """Initialize the knowledge bridge"""
        self.v8_health_url = f"http://localhost:{v8_health_port}/health"
        self.v8_metrics_url = f"http://localhost:{v8_health_port}/metrics"
        self.v7_knowledge_path = v7_knowledge_path or os.path.join(project_root, "data", "consciousness", "knowledge")
        self.running = False
        self.bridge_thread = None
        self.last_sync = None
        self.v7_seed = None
        self.sync_interval = 30  # seconds
        self.v7_concept_mapping = {}  # Maps v8 concept IDs to v7 concept IDs
        
        # Create v7 knowledge directory if it doesn't exist
        os.makedirs(self.v7_knowledge_path, exist_ok=True)
        
        # Try to import v7 seed system
        try:
            from src.seed import get_neural_seed
            self.v7_seed = get_neural_seed()
            logger.info("Successfully connected to v7 Neural Seed system")
        except ImportError:
            logger.warning("Cannot import v7 Neural Seed system - will use file-based knowledge transfer")
            
        logger.info(f"V8 to V7 Knowledge Bridge initialized with sync interval of {self.sync_interval}s")
    
    def start(self):
        """Start the knowledge bridge"""
        if self.running:
            logger.info("Knowledge bridge is already running")
            return
            
        self.running = True
        self.bridge_thread = threading.Thread(target=self._bridge_loop)
        self.bridge_thread.daemon = True
        self.bridge_thread.start()
        logger.info("V8 to V7 Knowledge Bridge started")
        
    def stop(self):
        """Stop the knowledge bridge"""
        self.running = False
        if self.bridge_thread:
            self.bridge_thread.join(timeout=2.0)
        logger.info("V8 to V7 Knowledge Bridge stopped")
    
    def _bridge_loop(self):
        """Main bridge loop for knowledge transfer"""
        while self.running:
            try:
                # Check v8 system health
                v8_health = self._check_v8_health()
                if v8_health and v8_health.get("status") != "critical":
                    # Transfer concepts from v8 to v7
                    self._transfer_concepts_to_v7()
                    
                    # Transfer neural seed data from v7 to v8
                    self._transfer_neural_seed_to_v8()
                    
                    self.last_sync = datetime.now()
                    logger.info(f"Knowledge synchronization completed at {self.last_sync.isoformat()}")
            except Exception as e:
                logger.error(f"Error in knowledge bridge loop: {e}")
                
            # Wait for next sync
            time.sleep(self.sync_interval)
    
    def _check_v8_health(self):
        """Check the health of the v8 system"""
        try:
            response = requests.get(self.v8_health_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"V8 health check returned status code {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error checking V8 health: {e}")
            return None
    
    def _transfer_concepts_to_v7(self):
        """Transfer concepts from v8 knowledge database to v7 system"""
        try:
            # Get metrics first to check number of concepts
            response = requests.get(self.v8_metrics_url, timeout=5)
            if response.status_code != 200:
                logger.warning(f"V8 metrics returned status code {response.status_code}")
                return
                
            v8_metrics = response.json()
            if not v8_metrics or not isinstance(v8_metrics, list) or not v8_metrics:
                logger.warning("No metrics data received from v8 system")
                return
                
            latest_metrics = v8_metrics[0]
            total_concepts = latest_metrics.get('total_concepts', 0)
            
            if total_concepts == 0:
                logger.info("No concepts available in v8 system to transfer")
                return
                
            logger.info(f"Found {total_concepts} concepts in v8 system")
            
            # In real implementation, would query v8 API for all concepts
            # For now, implement a simplified version with placeholder concepts
            v8_concepts = self._get_sample_v8_concepts()
            
            # Transfer to v7 seed if available
            if self.v7_seed:
                self._update_v7_seed(v8_concepts)
            else:
                # Otherwise write to v7 knowledge files
                self._write_v7_knowledge_files(v8_concepts)
                
            logger.info(f"Transferred {len(v8_concepts)} concepts from v8 to v7")
            
        except Exception as e:
            logger.error(f"Error transferring concepts to v7: {e}")
    
    def _get_sample_v8_concepts(self):
        """
        Get a sample of concepts from v8 system
        
        In a real implementation, this would query the v8 API to get real concepts
        """
        # Placeholder implementation with sample concepts
        return [
            {
                "id": f"v8_concept_{i}",
                "name": f"V8 Concept {i}",
                "description": f"Knowledge concept generated by v8 system #{i}",
                "weight": 0.7,
                "connections": [
                    {"target_id": f"v8_concept_{i+1 if i < 5 else 0}", "type": "related", "weight": 0.6}
                ]
            }
            for i in range(5)
        ]
    
    def _update_v7_seed(self, v8_concepts):
        """Update v7 neural seed with concepts from v8"""
        if not self.v7_seed:
            return
            
        # Access the neural seed dictionary
        try:
            seed_dict = getattr(self.v7_seed, "dictionary", {})
            
            # Convert and add v8 concepts to v7 seed
            for concept in v8_concepts:
                v7_id = f"v7_{concept['id']}"
                self.v7_concept_mapping[concept['id']] = v7_id
                
                # Create v7-compatible concept
                v7_concept = {
                    "name": concept['name'],
                    "description": concept['description'],
                    "weight": concept['weight'],
                    "source": "v8_knowledge_system",
                    "connections": []
                }
                
                # Add connections if any
                for conn in concept.get('connections', []):
                    target_v7_id = f"v7_{conn['target_id']}"
                    v7_concept['connections'].append({
                        "target": target_v7_id,
                        "type": conn.get('type', 'related'),
                        "weight": conn.get('weight', 0.5)
                    })
                
                # Add to seed dictionary
                seed_dict[v7_id] = v7_concept
            
            # Trigger seed update if needed
            if hasattr(self.v7_seed, "update"):
                self.v7_seed.update()
                
            logger.info(f"Updated v7 seed with {len(v8_concepts)} concepts")
            
        except Exception as e:
            logger.error(f"Error updating v7 seed: {e}")
    
    def _write_v7_knowledge_files(self, v8_concepts):
        """Write v8 concepts to v7 knowledge files"""
        try:
            # Create a map of existing files to avoid duplicates
            existing_files = set()
            if os.path.exists(self.v7_knowledge_path):
                existing_files = set(os.listdir(self.v7_knowledge_path))
            
            # Write each concept to a file
            for concept in v8_concepts:
                # Create a filename from the concept name
                base_filename = concept['name'].lower().replace(' ', '_')
                filename = f"{base_filename}.json"
                
                # Ensure filename is unique
                counter = 1
                while filename in existing_files:
                    filename = f"{base_filename}_{counter}.json"
                    counter += 1
                
                # Add to tracking set
                existing_files.add(filename)
                
                # Convert v8 concept to v7 format
                v7_concept = {
                    "name": concept['name'],
                    "description": concept['description'],
                    "weight": concept['weight'],
                    "source": "v8_knowledge_system",
                    "connections": []
                }
                
                # Add connections
                for conn in concept.get('connections', []):
                    v7_concept['connections'].append({
                        "target": conn['target_id'],
                        "type": conn.get('type', 'related'),
                        "weight": conn.get('weight', 0.5)
                    })
                
                # Write to file
                file_path = os.path.join(self.v7_knowledge_path, filename)
                with open(file_path, 'w') as f:
                    json.dump(v7_concept, f, indent=2)
                    
            logger.info(f"Wrote {len(v8_concepts)} concepts to v7 knowledge files")
            
        except Exception as e:
            logger.error(f"Error writing v7 knowledge files: {e}")
    
    def _transfer_neural_seed_to_v8(self):
        """Transfer neural seed data from v7 to v8 system"""
        try:
            if not self.v7_seed:
                # Try to read from v7 knowledge files instead
                v7_concepts = self._read_v7_knowledge_files()
            else:
                # Get concepts from v7 neural seed
                seed_dict = getattr(self.v7_seed, "dictionary", {})
                v7_concepts = []
                
                # Convert seed dictionary to list of concepts
                for concept_id, concept_data in seed_dict.items():
                    # Skip concepts that originated from v8
                    if concept_id.startswith("v7_v8_"):
                        continue
                        
                    v7_concepts.append({
                        "id": concept_id,
                        "name": concept_data.get("name", "Unknown"),
                        "description": concept_data.get("description", ""),
                        "weight": concept_data.get("weight", 0.5),
                        "connections": concept_data.get("connections", [])
                    })
            
            # In a real implementation, would use v8 API to add these concepts
            logger.info(f"Would transfer {len(v7_concepts)} concepts from v7 to v8")
            
        except Exception as e:
            logger.error(f"Error transferring neural seed to v8: {e}")
    
    def _read_v7_knowledge_files(self):
        """Read v7 knowledge files and convert to concepts"""
        concepts = []
        try:
            if not os.path.exists(self.v7_knowledge_path):
                return concepts
                
            # Read all JSON files in the knowledge directory
            for filename in os.listdir(self.v7_knowledge_path):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(self.v7_knowledge_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        concept_data = json.load(f)
                        
                    # Create concept from file data
                    concept_id = os.path.splitext(filename)[0]
                    concepts.append({
                        "id": concept_id,
                        "name": concept_data.get("name", "Unknown"),
                        "description": concept_data.get("description", ""),
                        "weight": concept_data.get("weight", 0.5),
                        "connections": concept_data.get("connections", [])
                    })
                except Exception as e:
                    logger.error(f"Error reading knowledge file {filename}: {e}")
                    
            logger.info(f"Read {len(concepts)} concepts from v7 knowledge files")
            
        except Exception as e:
            logger.error(f"Error reading v7 knowledge files: {e}")
            
        return concepts
    
    def get_status(self):
        """Get the current status of the knowledge bridge"""
        return {
            "running": self.running,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "v7_seed_connected": self.v7_seed is not None,
            "v7_knowledge_path": self.v7_knowledge_path,
            "v8_health": self._check_v8_health(),
            "concepts_mapped": len(self.v7_concept_mapping)
        }

def main():
    """Main function to run the v8 to v7 knowledge bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V8 to V7 Knowledge Bridge")
    parser.add_argument("--v8-port", type=int, default=8765, help="V8 health check port")
    parser.add_argument("--knowledge-path", type=str, default=None, help="V7 knowledge directory path")
    parser.add_argument("--sync-interval", type=int, default=30, help="Sync interval in seconds")
    args = parser.parse_args()
    
    bridge = V8ToV7KnowledgeBridge(
        v8_health_port=args.v8_port,
        v7_knowledge_path=args.knowledge_path
    )
    bridge.sync_interval = args.sync_interval
    bridge.start()
    
    try:
        logger.info("V8 to V7 Knowledge Bridge running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping bridge...")
        
    finally:
        bridge.stop()

if __name__ == "__main__":
    main() 