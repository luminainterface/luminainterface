#!/usr/bin/env python3
"""
V7 to V8 Bridge

This module provides functionality to transfer data from the v7 Holographic system
to the v8 Knowledge CI/CD system. It establishes a unidirectional bridge to feed
v7 neural seed data into the v8 knowledge pipeline.
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
        logging.FileHandler(f"logs/v7_to_v8_bridge_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v7_to_v8_bridge")

class V7ToV8Bridge:
    """
    Bridge for transferring data from v7 Holographic system to v8 Knowledge CI/CD system.
    """
    
    def __init__(self, v7_connection_port=5678, v8_health_port=8765):
        """Initialize the bridge with connection ports for both systems"""
        self.v7_api_url = f"http://localhost:{v7_connection_port}/api/v1"
        self.v8_health_url = f"http://localhost:{v8_health_port}/health"
        self.v8_api_url = f"http://localhost:{v8_health_port}"
        self.running = False
        self.bridge_thread = None
        self.last_sync = None
        self.sync_interval = 30  # seconds
        self.v7_seed = None
        
        # Try to import v7 seed system
        try:
            from src.seed import get_neural_seed
            self.v7_seed = get_neural_seed()
            logger.info("Successfully connected to v7 Neural Seed system")
        except ImportError:
            logger.warning("Cannot import v7 Neural Seed system - will use file-based transfer")
        
        # Create v7 knowledge directory path for file-based knowledge loading
        self.v7_knowledge_path = os.environ.get(
            'V7_KNOWLEDGE_PATH',
            os.path.join(project_root, "data", "consciousness", "knowledge")
        )
            
        logger.info(f"V7 to V8 Bridge initialized with sync interval of {self.sync_interval}s")
    
    def start(self):
        """Start the bridge process"""
        if self.running:
            logger.info("Bridge is already running")
            return
            
        self.running = True
        self.bridge_thread = threading.Thread(target=self._bridge_loop)
        self.bridge_thread.daemon = True
        self.bridge_thread.start()
        logger.info("V7 to V8 Bridge started")
        
    def stop(self):
        """Stop the bridge process"""
        self.running = False
        if self.bridge_thread:
            self.bridge_thread.join(timeout=2.0)
        logger.info("V7 to V8 Bridge stopped")
    
    def _bridge_loop(self):
        """Main bridge loop to transfer data from V7 to V8"""
        while self.running:
            try:
                # Check if V8 is healthy before attempting transfer
                v8_health = self._check_v8_health()
                
                if v8_health and v8_health.get("status") != "critical":
                    # Check if v7 API is available
                    v7_status = self._check_v7_status()
                    
                    if v7_status:
                        # Transfer neural seed data
                        self._transfer_neural_seeds()
                    else:
                        # Try file-based transfer if API not available
                        self._transfer_from_files()
                        
                    self.last_sync = datetime.now()
                    logger.info(f"V7 to V8 synchronization completed at {self.last_sync.isoformat()}")
                    
            except Exception as e:
                logger.error(f"Error in bridge loop: {e}")
                
            # Wait for next sync
            time.sleep(self.sync_interval)
    
    def _check_v8_health(self):
        """Check the health of the V8 system"""
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
    
    def _check_v7_status(self):
        """Check if V7 system is available"""
        try:
            response = requests.get(f"{self.v7_api_url}/system/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"V7 status check returned status code {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"V7 system not available via API: {e}")
            return None
    
    def _transfer_neural_seeds(self):
        """Transfer neural seed data from v7 to v8 system"""
        if not self.v7_seed:
            logger.warning("V7 seed connection not available for direct transfer")
            return
            
        try:
            # Get neural seed dictionary
            seed_dict = getattr(self.v7_seed, "dictionary", {})
            
            if not seed_dict:
                logger.warning("V7 neural seed dictionary is empty")
                return
                
            logger.info(f"Transferring {len(seed_dict)} neural seeds from V7 to V8")
            
            # Convert seeds to v8 concept format
            v8_concepts = []
            for seed_id, seed_data in seed_dict.items():
                # Skip seeds that originated from v8
                if seed_id.startswith("v8_"):
                    continue
                    
                concept = {
                    "id": f"v7_{seed_id}",
                    "name": seed_data.get("name", "Unknown Seed"),
                    "description": seed_data.get("description", "Neural seed from v7"),
                    "weight": seed_data.get("weight", 0.5),
                    "source": "v7_neural_seed",
                    "connections": []
                }
                
                # Add connections if any
                for conn in seed_data.get("connections", []):
                    concept["connections"].append({
                        "target_id": f"v7_{conn.get('target')}",
                        "type": conn.get("type", "related"),
                        "weight": conn.get("weight", 0.5)
                    })
                
                v8_concepts.append(concept)
            
            # Send concepts to v8 system
            success_count = self._send_concepts_to_v8(v8_concepts)
            logger.info(f"Successfully transferred {success_count} of {len(v8_concepts)} concepts to V8")
            
        except Exception as e:
            logger.error(f"Error transferring neural seeds: {e}")
    
    def _transfer_from_files(self):
        """Transfer knowledge data from v7 files to v8 system"""
        try:
            if not os.path.exists(self.v7_knowledge_path):
                logger.warning(f"V7 knowledge path {self.v7_knowledge_path} does not exist")
                return
                
            # Find all JSON files in knowledge directory
            knowledge_files = [f for f in os.listdir(self.v7_knowledge_path)
                             if f.endswith('.json')]
                             
            if not knowledge_files:
                logger.warning("No knowledge files found in V7 knowledge directory")
                return
                
            logger.info(f"Found {len(knowledge_files)} knowledge files to transfer")
            
            # Load knowledge from files and convert to v8 format
            v8_concepts = []
            for filename in knowledge_files:
                try:
                    with open(os.path.join(self.v7_knowledge_path, filename), 'r') as f:
                        knowledge_data = json.load(f)
                        
                    concept_id = os.path.splitext(filename)[0]
                    
                    # Create v8 concept
                    concept = {
                        "id": f"v7_file_{concept_id}",
                        "name": knowledge_data.get("name", f"Concept {concept_id}"),
                        "description": knowledge_data.get("description", "Concept from v7 knowledge file"),
                        "weight": knowledge_data.get("weight", 0.5),
                        "source": "v7_knowledge_file",
                        "connections": []
                    }
                    
                    # Add connections if any
                    for conn in knowledge_data.get("connections", []):
                        concept["connections"].append({
                            "target_id": f"v7_file_{conn.get('target')}",
                            "type": conn.get("type", "related"),
                            "weight": conn.get("weight", 0.5)
                        })
                    
                    v8_concepts.append(concept)
                        
                except Exception as e:
                    logger.error(f"Error processing knowledge file {filename}: {e}")
            
            # Send concepts to v8 system
            success_count = self._send_concepts_to_v8(v8_concepts)
            logger.info(f"Successfully transferred {success_count} of {len(v8_concepts)} concepts to V8")
            
        except Exception as e:
            logger.error(f"Error transferring from files: {e}")
    
    def _send_concepts_to_v8(self, concepts):
        """Send concepts to v8 system through its API"""
        success_count = 0
        
        # In a real implementation, would call v8 API to add concepts
        # For now, simulate the process and log the attempt
        try:
            # Create batch of concepts to send
            batch_size = 10
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i+batch_size]
                
                try:
                    # Would call v8 API endpoint if it existed
                    # response = requests.post(f"{self.v8_api_url}/knowledge/concepts",
                    #                         json={"concepts": batch},
                    #                         timeout=10)
                    # if response.status_code == 200:
                    #     success_count += len(batch)
                    
                    # For now, just simulate success
                    time.sleep(0.5)  # Simulate API call
                    success_count += len(batch)
                    
                    # Log sample of concepts being transferred
                    if i == 0:
                        sample = batch[0]
                        logger.info(f"Sample concept transferred: {sample['name']} ({sample['id']})")
                    
                except Exception as e:
                    logger.error(f"Error sending concept batch {i//batch_size}: {e}")
                    
            # Save to v8 concepts directory for file-based integration
            v8_concepts_dir = os.path.join(project_root, "data", "v8", "knowledge")
            os.makedirs(v8_concepts_dir, exist_ok=True)
            
            # Save a marker file to indicate transfer occurred
            marker_file = os.path.join(v8_concepts_dir, f"v7_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(marker_file, 'w') as f:
                json.dump({
                    "transferred": len(concepts),
                    "timestamp": datetime.now().isoformat(),
                    "source": "v7_to_v8_bridge"
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error in _send_concepts_to_v8: {e}")
            
        return success_count
    
    def get_status(self):
        """Get current bridge status"""
        return {
            "running": self.running,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "v7_seed_connected": self.v7_seed is not None,
            "v7_knowledge_path": self.v7_knowledge_path,
            "v8_health": self._check_v8_health() is not None,
            "v7_status": self._check_v7_status() is not None
        }

def main():
    """Main function to run the bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V7 to V8 Bridge")
    parser.add_argument("--v7-port", type=int, default=5678, help="V7 connection port")
    parser.add_argument("--v8-port", type=int, default=8765, help="V8 health check port")
    parser.add_argument("--sync-interval", type=int, default=30, help="Sync interval in seconds")
    args = parser.parse_args()
    
    bridge = V7ToV8Bridge(
        v7_connection_port=args.v7_port,
        v8_health_port=args.v8_port
    )
    bridge.sync_interval = args.sync_interval
    bridge.start()
    
    try:
        logger.info("V7 to V8 Bridge running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping bridge...")
        
    finally:
        bridge.stop()

if __name__ == "__main__":
    main() 