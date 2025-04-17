#!/usr/bin/env python3
"""
Knowledge CI/CD Integrated System for v8

This module integrates the Knowledge Database with the CI/CD pipeline, providing
a complete knowledge management system with automated discovery, processing,
and deployment capabilities.
"""

import os
import sys
import time
import uuid
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import v8 components
from src.v8.knowledge_ci_cd import KnowledgePipeline
from src.v8.knowledge_database import KnowledgeDatabase
from src.v8.root_connection_system import RootConnectionSystem
from src.v8.spatial_temple_mapper import SpatialTempleMapper
from src.v8.temple_to_seed_bridge import ConceptSeed
from src.v8.auto_seed_growth import AutoSeedGrowthSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/knowledge_ci_cd_integrated_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v8.knowledge_ci_cd_integrated")

class IntegratedKnowledgeSystem:
    """
    Integrated Knowledge System that combines the database with CI/CD pipeline.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the integrated knowledge system"""
        self.config_path = config_path or os.path.join(project_root, "config", "knowledge_ci_cd.json")
        
        # Initialize components
        self.database = None
        self.pipeline = None
        self.temple_mapper = None
        self.run_id = None
        self.health_status = {"status": "initializing", "last_check": datetime.now().isoformat()}
        
        logger.info("Initializing Integrated Knowledge System")
        
    def initialize(self):
        """Initialize all components of the system"""
        try:
            # Initialize database
            logger.info("Initializing Knowledge Database")
            self.database = KnowledgeDatabase(self.config_path)
            
            # Initialize temple mapper
            logger.info("Initializing Spatial Temple Mapper")
            self.temple_mapper = SpatialTempleMapper()
            
            # Initialize pipeline with the database
            logger.info("Initializing Knowledge Pipeline")
            self.pipeline = KnowledgePipeline()
            
            # Connect components
            success = self._connect_components()
            
            # Health check
            self.health_status = {
                "status": "healthy" if success else "degraded",
                "last_check": datetime.now().isoformat(),
                "components": {
                    "database": "connected" if self.database else "failed",
                    "pipeline": "initialized" if self.pipeline else "failed",
                    "temple_mapper": "initialized" if self.temple_mapper else "failed"
                }
            }
            
            logger.info("Integrated Knowledge System initialized")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize Integrated Knowledge System: {e}")
            self.health_status = {
                "status": "failed",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
            return False
            
    def _connect_components(self):
        """Connect all components together"""
        if not self.database or not self.pipeline or not self.temple_mapper:
            logger.error("Cannot connect components: some components are not initialized")
            return False
        
        # Initialize the pipeline components
        self.pipeline.initialize()
        
        # Add database hooks to the pipeline stages
        self._add_database_hooks()
        
        logger.info("Components connected successfully")
        return True
    
    def _add_database_hooks(self):
        """Add database hooks to pipeline stages for recording metrics and data"""
        if not self.pipeline:
            return
        
        # Store original methods to wrap
        original_run_discovery = self.pipeline.run_discovery
        original_run_attachment = self.pipeline.run_attachment
        original_run_growth = self.pipeline.run_growth
        original_run_linking = self.pipeline.run_linking
        original_run_deployment = self.pipeline.run_deployment
        
        # Hook into discovery stage
        def discovery_hook(*args, **kwargs):
            stage_start = datetime.now()
            logger.info(f"[{self.run_id}] Starting DISCOVERY stage with database integration")
            
            # Run original method
            results = original_run_discovery(*args, **kwargs)
            
            # Record metrics
            stage_metrics = {
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "items_processed": len(results) if results else 0,
                "items_succeeded": len(results) if results else 0,
                "items_failed": 0,
                "status": "completed"
            }
            
            # Store discovery sources in database
            if self.database and results:
                for source in results:
                    self.database.add_knowledge_source(
                        source_id=source.get("source_id"),
                        name=source.get("name", "Unknown"),
                        source_type=source.get("source_type", "unknown"),
                        description=f"Discovered through {source.get('seed_concept')}"
                    )
            
            # Record stage metrics
            if self.database:
                self.database.record_stage_metrics(self.run_id, "discovery", stage_metrics)
            
            return results
        
        # Hook into attachment stage
        def attachment_hook(discoveries, *args, **kwargs):
            stage_start = datetime.now()
            logger.info(f"[{self.run_id}] Starting ATTACHMENT stage with database integration")
            
            # Run original method
            results = original_run_attachment(discoveries, *args, **kwargs)
            
            # Count successes
            successes = 0
            for attachment in results:
                if attachment.get("attachment_points"):
                    successes += 1
                    
                    # Store attachments in database
                    if self.database:
                        source_id = attachment.get("source_id")
                        for ap in attachment.get("attachment_points", []):
                            attachment_id = f"att_{uuid.uuid4().hex[:8]}"
                            node_id = ap.get("node_id")
                            
                            # Add attachment to database
                            self.database.add_attachment(
                                attachment_id=attachment_id,
                                concept_id=node_id,
                                source_id=source_id,
                                similarity_score=ap.get("similarity")
                            )
            
            # Record metrics
            stage_metrics = {
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "items_processed": len(discoveries) if discoveries else 0,
                "items_succeeded": successes,
                "items_failed": len(discoveries) - successes if discoveries else 0,
                "status": "completed"
            }
            
            # Record stage metrics
            if self.database:
                self.database.record_stage_metrics(self.run_id, "attachment", stage_metrics)
            
            return results
        
        # Hook into growth stage
        def growth_hook(attachments, *args, **kwargs):
            stage_start = datetime.now()
            logger.info(f"[{self.run_id}] Starting GROWTH stage with database integration")
            
            # Run original method
            results = original_run_growth(attachments, *args, **kwargs)
            
            # Store new concepts and connections
            if self.database and results:
                for concept in results.get("new_concepts", []):
                    # Add concept to database
                    self.database.add_concept(
                        concept_id=concept.get("id"),
                        name=concept.get("name"),
                        description=concept.get("description", ""),
                        weight=concept.get("weight", 0.5)
                    )
                
                for conn in results.get("new_connections", []):
                    # Add connection to database
                    self.database.add_connection(
                        connection_id=conn.get("id"),
                        source_id=conn.get("source_id"),
                        target_id=conn.get("target_id"),
                        weight=conn.get("weight", 0.5),
                        connection_type=conn.get("type", "related"),
                        bidirectional=conn.get("bidirectional", False)
                    )
            
            # Record metrics
            stage_metrics = {
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "items_processed": len(attachments) if attachments else 0,
                "items_succeeded": len(results.get("new_concepts", [])) if results else 0,
                "items_failed": 0,
                "status": "completed"
            }
            
            # Record stage metrics
            if self.database:
                self.database.record_stage_metrics(self.run_id, "growth", stage_metrics)
            
            return results
        
        # Hook into linking stage
        def linking_hook(*args, **kwargs):
            stage_start = datetime.now()
            logger.info(f"[{self.run_id}] Starting LINKING stage with database integration")
            
            # Run original method
            results = original_run_linking(*args, **kwargs)
            
            # Record metrics
            stage_metrics = {
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "items_processed": results.get("processed", 0) if results else 0,
                "items_succeeded": results.get("linked", 0) if results else 0,
                "items_failed": results.get("failed", 0) if results else 0,
                "status": "completed"
            }
            
            # Record stage metrics
            if self.database:
                self.database.record_stage_metrics(self.run_id, "linking", stage_metrics)
            
            return results
        
        # Hook into deployment stage
        def deployment_hook(*args, **kwargs):
            stage_start = datetime.now()
            logger.info(f"[{self.run_id}] Starting DEPLOYMENT stage with database integration")
            
            # Run original method
            results = original_run_deployment(*args, **kwargs)
            
            # Record metrics
            stage_metrics = {
                "start_time": stage_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "items_processed": results.get("processed", 0) if results else 0,
                "items_succeeded": results.get("deployed", 0) if results else 0,
                "items_failed": results.get("failed", 0) if results else 0,
                "status": "completed"
            }
            
            # Record stage metrics
            if self.database:
                self.database.record_stage_metrics(self.run_id, "deployment", stage_metrics)
                
                # Update knowledge metrics
                self.database.update_knowledge_metrics()
            
            return results
        
        # Replace pipeline methods with hooks
        self.pipeline.run_discovery = discovery_hook
        self.pipeline.run_attachment = attachment_hook
        self.pipeline.run_growth = growth_hook
        self.pipeline.run_linking = linking_hook
        self.pipeline.run_deployment = deployment_hook
        
        logger.info("Database hooks added to pipeline stages")
    
    def run(self):
        """Run the integrated knowledge system"""
        if not self.database or not self.pipeline:
            logger.error("Cannot run: system not initialized")
            return False
        
        # Generate run ID
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting Integrated Knowledge System run [{self.run_id}]")
        
        # Set up run_id propagation to hooks
        self._propagate_run_id_to_hooks()
        
        # Perform health check before run
        health = self.check_health()
        if health["status"] == "critical":
            logger.error(f"System health is critical, cannot proceed with run: {health}")
            self._record_error_run("critical_health", health)
            return False
            
        try:
            # Start time
            start_time = datetime.now()
            
            # Ensure database connection is ready
            if hasattr(self.database, '_ensure_connected'):
                db_ready = self.database._ensure_connected()
                if not db_ready:
                    logger.error("Database connection failed, cannot proceed with run")
                    self._record_error_run("database_connection_failed")
                    return False
                    
            # Run pipeline
            success = self.pipeline.run_pipeline(run_id=self.run_id)
            
            # End time
            end_time = datetime.now()
            
            # Record pipeline metrics
            pipeline_metrics = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "discoveries": self.pipeline.metrics.get("discoveries", 0),
                "attachments": self.pipeline.metrics.get("attachments", 0),
                "growth_points": self.pipeline.metrics.get("growth_points", 0),
                "bidirectional_links": self.pipeline.metrics.get("bidirectional_links", 0),
                "deployments": self.pipeline.metrics.get("deployments", 0),
                "errors": 0,
                "status": "completed" if success else "partial_success"
            }
            
            # Record pipeline run
            self.database.record_pipeline_run(self.run_id, pipeline_metrics)
            
            # Backup database after successful run
            self.database.backup_database()
            
            logger.info(f"Integrated Knowledge System run completed [{self.run_id}]")
            return True
        except Exception as e:
            logger.error(f"Error in integrated system run: {e}")
            
            # Try recovery
            recovery_success = self._attempt_recovery(e)
            if recovery_success:
                logger.info(f"Recovery was successful, continuing with run [{self.run_id}]")
                return self.run()  # Try running again after recovery
                
            # Record error pipeline metrics
            self._record_error_run(str(e))
            return False
            
    def _record_error_run(self, error_message, additional_data=None):
        """Record information about a failed run"""
        try:
            # Record error pipeline metrics
            pipeline_metrics = {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "errors": 1,
                "status": f"error: {error_message}"
            }
            
            # If there's additional data, include in metrics
            if additional_data:
                pipeline_metrics["error_details"] = json.dumps(additional_data)
            
            # Record pipeline run error
            if self.database:
                self.database.record_pipeline_run(self.run_id or f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                                                pipeline_metrics)
        except Exception as e:
            logger.error(f"Failed to record error run: {e}")
                
    def _attempt_recovery(self, exception) -> bool:
        """Attempt to recover from an error during pipeline run"""
        logger.info(f"Attempting to recover from error: {exception}")
        
        try:
            # Check if database-related exception
            if "database" in str(exception).lower() or "sqlite" in str(exception).lower():
                logger.info("Trying to reconnect to database...")
                if self.database:
                    # Try to reconnect
                    if hasattr(self.database, '_ensure_connected'):
                        reconnected = self.database._ensure_connected()
                        if reconnected:
                            logger.info("Successfully reconnected to database")
                            return True
                            
            # Check for pipeline-specific issues
            if "pipeline" in str(exception).lower():
                logger.info("Trying to reinitialize pipeline...")
                if self.pipeline:
                    # Try to restart the pipeline
                    self.pipeline.stop()
                    time.sleep(1)  # Give it time to clean up
                    initialized = self.pipeline.initialize()
                    if initialized:
                        logger.info("Successfully reinitialized pipeline")
                        return True
            
            # Default recovery: reinitialize components
            logger.info("Attempting full component reconnection...")
            success = self._connect_components()
            return success
                
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
    
    def stop(self):
        """Stop all components"""
        logger.info("Stopping Integrated Knowledge System")
        
        try:
            # Stop pipeline
            if self.pipeline:
                self.pipeline.stop()
            
            # Close database connections
            if self.database:
                self.database.close()
            
            logger.info("Integrated Knowledge System stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping integrated system: {e}")
            return False

    def _propagate_run_id_to_hooks(self):
        """Ensure run_id is available in pipeline stage hooks"""
        if not self.pipeline or not self.run_id:
            return False
            
        # Store original run discovery hook
        original_discovery_hook = self.pipeline.run_discovery
        
        # Define new discovery hook with run_id capturing
        def discovery_hook_with_run_id(*args, **kwargs):
            logger.info(f"[{self.run_id}] Starting DISCOVERY stage with database integration")
            kwargs['run_id'] = self.run_id
            return original_discovery_hook(*args, **kwargs)
            
        # Update the pipeline's hook
        self.pipeline.run_discovery = discovery_hook_with_run_id
        
        # Apply similar changes to other hooks
        for hook_name in ['run_attachment', 'run_growth', 'run_linking', 'run_deployment']:
            if hasattr(self.pipeline, hook_name):
                original_hook = getattr(self.pipeline, hook_name)
                def hook_with_run_id(original=original_hook, name=hook_name):
                    def wrapped_hook(*args, **kwargs):
                        logger.info(f"[{self.run_id}] Starting {name.upper()} stage with database integration")
                        kwargs['run_id'] = self.run_id
                        return original(*args, **kwargs)
                    return wrapped_hook
                setattr(self.pipeline, hook_name, hook_with_run_id())
                
        return True
        
    def check_health(self) -> Dict[str, Any]:
        """Perform health check of all components"""
        status = "healthy"
        components = {}
        
        # Check database
        db_status = "unknown"
        if self.database:
            try:
                # Test database connection
                if hasattr(self.database, '_ensure_connected'):
                    db_connected = self.database._ensure_connected()
                    db_status = "connected" if db_connected else "disconnected"
                else:
                    # Fallback if method doesn't exist
                    try:
                        self.database.main_conn.execute("SELECT 1")
                        db_status = "connected"
                    except Exception:
                        db_status = "disconnected"
            except Exception:
                db_status = "error"
        else:
            db_status = "missing"
            
        components["database"] = db_status
        
        # Check pipeline
        pipeline_status = "unknown"
        if self.pipeline:
            pipeline_status = "running" if getattr(self.pipeline, "running", False) else "stopped"
        else:
            pipeline_status = "missing"
            
        components["pipeline"] = pipeline_status
        
        # Check temple mapper
        temple_status = "unknown"
        if self.temple_mapper:
            temple_status = f"active with {len(getattr(self.temple_mapper, 'nodes', []))} nodes"
        else:
            temple_status = "missing"
            
        components["temple_mapper"] = temple_status
        
        # Determine overall status
        if "missing" in components.values() or "error" in components.values():
            status = "degraded"
        if components["database"] == "disconnected":
            status = "critical"
            
        # Update health status
        self.health_status = {
            "status": status,
            "last_check": datetime.now().isoformat(),
            "components": components,
            "run_id": self.run_id
        }
        
        return self.health_status

def main():
    """Main function for running the integrated knowledge system"""
    parser = argparse.ArgumentParser(description="Integrated Knowledge CI/CD System")
    parser.add_argument(
        "--config", type=str, 
        default=os.path.join(project_root, "config", "knowledge_ci_cd.json"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--run-once", action="store_true",
        help="Run the pipeline once and exit"
    )
    parser.add_argument(
        "--health-check-port", type=int, default=0,
        help="Enable health check endpoint on specified port (0 to disable)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = IntegratedKnowledgeSystem(args.config)
    success = system.initialize()
    
    if not success:
        logger.error("Failed to initialize system, exiting")
        sys.exit(1)
    
    # Start health check server if port is specified
    if args.health_check_port > 0:
        try:
            from flask import Flask, jsonify
            
            # Create Flask app for health checks
            app = Flask("ci_cd_healthcheck")
            
            @app.route('/health')
            def health_check():
                status = system.check_health()
                response_code = 200
                if status["status"] == "degraded":
                    response_code = 429  # Too Many Requests
                elif status["status"] == "critical":
                    response_code = 503  # Service Unavailable
                return jsonify(status), response_code
                
            @app.route('/metrics')
            def metrics():
                metrics_data = {}
                if system.database:
                    try:
                        metrics_data = system.database.get_recent_metrics(limit=10)
                    except Exception as e:
                        metrics_data = {"error": str(e)}
                return jsonify(metrics_data)
            
            # Start Flask in a separate thread
            import threading
            def run_health_server():
                app.run(host='0.0.0.0', port=args.health_check_port)
                
            health_thread = threading.Thread(target=run_health_server)
            health_thread.daemon = True
            health_thread.start()
            logger.info(f"Health check endpoint started on port {args.health_check_port}")
            
        except ImportError:
            logger.warning("Flask not installed, health check endpoint disabled")
    
    try:
        if args.run_once:
            # Run once and exit
            system.run()
        else:
            # Start pipeline
            system.pipeline.start()
            logger.info("Press Ctrl+C to stop the system")
            
            # Keep running until interrupted, periodically check health
            while True:
                time.sleep(60)  # Check every minute
                try:
                    health = system.check_health()
                    if health["status"] == "critical":
                        logger.critical(f"System health is critical: {health}")
                        # Try recovery
                        logger.info("Attempting recovery...")
                        success = system._connect_components()
                        if not success:
                            logger.error("Recovery failed, restarting system...")
                            system.stop()
                            time.sleep(5)
                            success = system.initialize()
                            if success:
                                system.pipeline.start()
                                logger.info("System restarted successfully")
                except Exception as e:
                    logger.error(f"Error during health check: {e}")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping system...")
    finally:
        # Stop the system
        system.stop()

if __name__ == "__main__":
    main() 