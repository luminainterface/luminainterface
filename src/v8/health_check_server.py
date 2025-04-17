#!/usr/bin/env python3
"""
V8 Health Check Server

A simple Flask-based server that provides health check endpoints
for the V8 Knowledge CI/CD system and metrics collection.
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from flask import Flask, jsonify, request
except ImportError:
    print("Flask is required. Please install it using: pip install flask")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/v8_health_check_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v8_health_check")

# Create Flask app
app = Flask(__name__)

# Global variables for system status
system_status = {
    "status": "initializing",  # initializing, healthy, degraded, critical
    "components": {
        "knowledge_db": {
            "status": "unknown",
            "message": "Not initialized",
            "last_check": None
        },
        "metrics_db": {
            "status": "unknown",
            "message": "Not initialized",
            "last_check": None
        },
        "temple_mapper": {
            "status": "unknown",
            "message": "Not initialized",
            "last_check": None
        },
        "pipeline": {
            "status": "unknown",
            "message": "Not initialized",
            "last_check": None
        }
    },
    "last_updated": datetime.now().isoformat()
}

# Metrics storage
metrics_history = []

class V8HealthCheck:
    """
    Health check manager for V8 Knowledge CI/CD system components
    """
    def __init__(self):
        self.running = False
        self.check_thread = None
        self.check_interval = 30  # seconds
        
        # Paths to component data
        self.knowledge_db_path = os.environ.get(
            'V8_KNOWLEDGE_DB_PATH', 
            os.path.join(project_root, "data", "v8", "knowledge")
        )
        self.metrics_db_path = os.environ.get(
            'V8_METRICS_DB_PATH',
            os.path.join(project_root, "data", "v8", "metrics")
        )
        self.temple_path = os.environ.get(
            'V8_TEMPLE_PATH',
            os.path.join(project_root, "data", "v8", "temple")
        )
        
        # Ensure directories exist
        os.makedirs(self.knowledge_db_path, exist_ok=True)
        os.makedirs(self.metrics_db_path, exist_ok=True)
        os.makedirs(self.temple_path, exist_ok=True)
        
        logger.info("V8 Health Check initialized with check interval of %s seconds", self.check_interval)
        
    def start(self):
        """Start the health check background thread"""
        if self.running:
            logger.info("Health check is already running")
            return
            
        self.running = True
        self.check_thread = threading.Thread(target=self._check_loop)
        self.check_thread.daemon = True
        self.check_thread.start()
        
        # Initial check to set status
        self._check_components()
        logger.info("V8 Health Check started")
        
    def stop(self):
        """Stop the health check background thread"""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=2.0)
        logger.info("V8 Health Check stopped")
        
    def _check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                self._check_components()
                self._collect_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error("Error in health check loop: %s", e)
                time.sleep(5)  # Shorter interval on error
    
    def _check_components(self):
        """Check the health of all V8 components"""
        global system_status
        
        # Check knowledge database
        self._check_knowledge_db()
        
        # Check metrics database
        self._check_metrics_db()
        
        # Check temple mapper
        self._check_temple_mapper()
        
        # Check pipeline
        self._check_pipeline()
        
        # Determine overall system status based on component statuses
        component_statuses = [c["status"] for c in system_status["components"].values()]
        
        if "critical" in component_statuses:
            system_status["status"] = "critical"
        elif "degraded" in component_statuses:
            system_status["status"] = "degraded"
        elif all(s == "healthy" for s in component_statuses):
            system_status["status"] = "healthy"
        else:
            system_status["status"] = "degraded"
            
        system_status["last_updated"] = datetime.now().isoformat()
        logger.debug("Updated system status: %s", system_status["status"])
    
    def _check_knowledge_db(self):
        """Check the health of the knowledge database"""
        try:
            # Check if directory exists and count concept files
            if not os.path.exists(self.knowledge_db_path):
                system_status["components"]["knowledge_db"] = {
                    "status": "critical",
                    "message": "Knowledge database directory does not exist",
                    "last_check": datetime.now().isoformat()
                }
                return
                
            # Count JSON files (concepts)
            concept_files = [f for f in os.listdir(self.knowledge_db_path) 
                            if f.endswith('.json')]
            
            if len(concept_files) == 0:
                status = "degraded"
                message = "Knowledge database contains no concepts"
            else:
                status = "healthy"
                message = f"Knowledge database contains {len(concept_files)} concepts"
                
            system_status["components"]["knowledge_db"] = {
                "status": status,
                "message": message,
                "count": len(concept_files),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error checking knowledge database: %s", e)
            system_status["components"]["knowledge_db"] = {
                "status": "critical",
                "message": f"Error checking knowledge database: {str(e)}",
                "last_check": datetime.now().isoformat()
            }
    
    def _check_metrics_db(self):
        """Check the health of the metrics database"""
        try:
            # Check if directory exists and metrics file
            if not os.path.exists(self.metrics_db_path):
                system_status["components"]["metrics_db"] = {
                    "status": "critical",
                    "message": "Metrics database directory does not exist",
                    "last_check": datetime.now().isoformat()
                }
                return
                
            # Check for metrics file
            metrics_file = os.path.join(self.metrics_db_path, "metrics.json")
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    if isinstance(metrics_data, list) and len(metrics_data) > 0:
                        status = "healthy"
                        message = f"Metrics database contains {len(metrics_data)} records"
                    else:
                        status = "degraded"
                        message = "Metrics database file exists but contains no records"
                except Exception as e:
                    status = "degraded"
                    message = f"Error reading metrics data: {str(e)}"
            else:
                status = "degraded"
                message = "Metrics file does not exist yet"
                
            system_status["components"]["metrics_db"] = {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error checking metrics database: %s", e)
            system_status["components"]["metrics_db"] = {
                "status": "critical",
                "message": f"Error checking metrics database: {str(e)}",
                "last_check": datetime.now().isoformat()
            }
    
    def _check_temple_mapper(self):
        """Check the health of the temple mapper"""
        try:
            # Check if directory exists and has mapping files
            if not os.path.exists(self.temple_path):
                system_status["components"]["temple_mapper"] = {
                    "status": "critical",
                    "message": "Temple mapper directory does not exist",
                    "last_check": datetime.now().isoformat()
                }
                return
                
            # Check for mapping files
            mapping_files = [f for f in os.listdir(self.temple_path) 
                           if f.endswith('.json') or f.endswith('.yaml')]
            
            if len(mapping_files) == 0:
                status = "degraded"
                message = "Temple mapper contains no mapping files"
            else:
                status = "healthy"
                message = f"Temple mapper contains {len(mapping_files)} mapping files"
                
            system_status["components"]["temple_mapper"] = {
                "status": status,
                "message": message,
                "count": len(mapping_files),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error checking temple mapper: %s", e)
            system_status["components"]["temple_mapper"] = {
                "status": "critical",
                "message": f"Error checking temple mapper: {str(e)}",
                "last_check": datetime.now().isoformat()
            }
    
    def _check_pipeline(self):
        """Check the health of the pipeline"""
        try:
            # For the pipeline, we'll check if a lock file exists indicating it's running
            pipeline_lock = os.path.join(project_root, "data", "v8", "pipeline.lock")
            
            if os.path.exists(pipeline_lock):
                # Check if lock file is stale (older than 10 minutes)
                mod_time = os.path.getmtime(pipeline_lock)
                current_time = time.time()
                
                if current_time - mod_time > 600:  # 10 minutes
                    status = "degraded"
                    message = "Pipeline lock file exists but may be stale (>10 minutes old)"
                else:
                    status = "healthy"
                    message = "Pipeline is running"
            else:
                # No lock file means pipeline is not running
                status = "degraded"
                message = "Pipeline is not currently running"
                
            system_status["components"]["pipeline"] = {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error checking pipeline: %s", e)
            system_status["components"]["pipeline"] = {
                "status": "critical",
                "message": f"Error checking pipeline: {str(e)}",
                "last_check": datetime.now().isoformat()
            }
    
    def _collect_metrics(self):
        """Collect system metrics and store them"""
        global metrics_history
        
        try:
            # Collect basic metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_status": system_status["status"],
                "total_concepts": system_status["components"]["knowledge_db"].get("count", 0),
                "total_mappings": system_status["components"]["temple_mapper"].get("count", 0),
                "components_healthy": sum(1 for c in system_status["components"].values() 
                                       if c["status"] == "healthy"),
                "components_degraded": sum(1 for c in system_status["components"].values() 
                                        if c["status"] == "degraded"),
                "components_critical": sum(1 for c in system_status["components"].values() 
                                        if c["status"] == "critical")
            }
            
            # Add to metrics history, keeping last 100 records
            metrics_history.append(metrics)
            if len(metrics_history) > 100:
                metrics_history = metrics_history[-100:]
                
            # Save metrics to file
            try:
                metrics_file = os.path.join(self.metrics_db_path, "metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_history, f, indent=2)
            except Exception as e:
                logger.error("Error saving metrics: %s", e)
                
        except Exception as e:
            logger.error("Error collecting metrics: %s", e)

# Initialize health check
health_check = V8HealthCheck()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify(system_status)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint"""
    limit = request.args.get('limit', '10')
    try:
        limit = int(limit)
    except ValueError:
        limit = 10
        
    return jsonify(metrics_history[-limit:])

@app.route('/component/<component_name>', methods=['GET'])
def component_status(component_name):
    """Get status of a specific component"""
    if component_name in system_status["components"]:
        return jsonify(system_status["components"][component_name])
    else:
        return jsonify({"error": f"Component '{component_name}' not found"}), 404

@app.route('/knowledge/concepts', methods=['GET'])
def list_concepts():
    """List available knowledge concepts"""
    try:
        knowledge_path = health_check.knowledge_db_path
        if not os.path.exists(knowledge_path):
            return jsonify({"error": "Knowledge database directory does not exist"}), 404
            
        concept_files = [f for f in os.listdir(knowledge_path) if f.endswith('.json')]
        concepts = []
        
        for file in concept_files[:10]:  # Limit to 10 for API response
            try:
                with open(os.path.join(knowledge_path, file), 'r') as f:
                    concept = json.load(f)
                concept_id = os.path.splitext(file)[0]
                concepts.append({
                    "id": concept_id,
                    "name": concept.get("name", "Unknown"),
                    "description": concept.get("description", "")[:100]  # Truncate description
                })
            except Exception as e:
                logger.error(f"Error reading concept file {file}: {e}")
        
        return jsonify({
            "total": len(concept_files),
            "concepts": concepts
        })
    except Exception as e:
        logger.error(f"Error listing concepts: {e}")
        return jsonify({"error": f"Error listing concepts: {str(e)}"}), 500

def start_server(host='0.0.0.0', port=8765):
    """Start the health check server"""
    # Start the health check background thread
    health_check.start()
    
    try:
        logger.info(f"Starting V8 Health Check server on {host}:{port}")
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    finally:
        health_check.stop()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V8 Health Check Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--check-interval", type=int, default=30, 
                        help="Health check interval in seconds")
    args = parser.parse_args()
    
    health_check.check_interval = args.check_interval
    start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main() 