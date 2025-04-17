#!/usr/bin/env python3
"""
Knowledge CI/CD System for v8

This module implements a continuous integration and deployment system for 
knowledge growth in the v8 seed system. Like a mold that finds and connects
to other mold sources to grow, this system:

1. DISCOVERY: Finds new knowledge sources automatically
2. ATTACHMENT: Connects discoveries to existing knowledge
3. GROWTH: Grows new connections from attached knowledge
4. LINKING: Creates bidirectional links between v1-v7 (DNA) and v8 (seed)
5. DEPLOYMENT: Makes the enhanced knowledge available to the system

The system runs as a scheduled process and can also be triggered manually.
"""

import os
import sys
import time
import logging
import json
import random
import threading
import argparse
import schedule
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import v8 components
from src.v8.root_connection_system import RootConnectionSystem, NutrientPacket
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode, TempleZone
from src.v8.temple_to_seed_bridge import ConceptSeed
from src.v8.auto_seed_growth import AutoSeedGrowthSystem, KnowledgeSource
from src.v8.seed_dispersal_system import KnowledgeFruit, SeedDispersalWindow
from src.v8.spatial_temple_connector import SpatialTempleConnector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/knowledge_ci_cd_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v8.knowledge_ci_cd")

class KnowledgePipeline:
    """
    Implements the CI/CD pipeline for knowledge growth and integration.
    Like a mold's mycelial network that spreads and connects.
    """
    
    def __init__(self):
        """Initialize the knowledge pipeline"""
        self.temple_mapper = None
        self.root_system = None
        self.auto_growth = None
        self.connector = None
        self.running = False
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        
        # Pipeline metrics
        self.metrics = {
            "discoveries": 0,
            "attachments": 0,
            "growth_points": 0,
            "bidirectional_links": 0,
            "deployments": 0,
            "last_run": None,
            "total_runs": 0,
            "knowledge_size": 0
        }
        
        # Pipeline configuration
        self.config = {
            "schedule_interval": 6,  # Hours between scheduled runs
            "discovery_sources": ["wiki", "web", "local", "api"],
            "attachment_threshold": 0.65,  # Minimum similarity for attachment
            "growth_iterations": 3,  # Number of growth iterations per run
            "max_discoveries_per_run": 10,
            "bidirectional_timeout": 30,  # Seconds to wait for bidirectional flow
            "auto_deploy": True
        }
        
        # Load config from file if exists
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file"""
        config_path = os.path.join(project_root, "config", "knowledge_ci_cd.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def initialize(self):
        """Initialize all required components"""
        logger.info("Initializing Knowledge CI/CD Pipeline")
        
        # Initialize temple mapper
        self.temple_mapper = SpatialTempleMapper()
        
        # Initialize auto growth system
        self.auto_growth = AutoSeedGrowthSystem(self.temple_mapper)
        
        # Initialize root connection system
        self.root_system = RootConnectionSystem(self.temple_mapper, self.auto_growth)
        
        # Initialize spatial temple connector
        self.connector = SpatialTempleConnector(self.temple_mapper)
        
        logger.info("Knowledge CI/CD Pipeline initialized")
        return True
    
    def start(self):
        """Start the scheduled pipeline"""
        if self.running:
            logger.info("Pipeline already running")
            return False
        
        # Initialize components if needed
        if not self.temple_mapper:
            self.initialize()
        
        # Start the background thread
        self.running = True
        self.stop_event.clear()
        self.pipeline_thread = threading.Thread(target=self._scheduler_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        
        logger.info(f"Knowledge CI/CD Pipeline started, scheduled every {self.config['schedule_interval']} hours")
        return True
    
    def stop(self):
        """Stop the scheduled pipeline"""
        if not self.running:
            return False
        
        self.running = False
        self.stop_event.set()
        
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=10.0)
        
        logger.info("Knowledge CI/CD Pipeline stopped")
        return True
    
    def _scheduler_loop(self):
        """Background thread for scheduled runs"""
        # Schedule the pipeline to run at the configured interval
        schedule.every(self.config['schedule_interval']).hours.do(self.run_pipeline)
        
        # Run immediately once on startup
        self.run_pipeline()
        
        # Keep checking the schedule
        while not self.stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_pipeline(self):
        """Run the complete pipeline once"""
        logger.info("Starting Knowledge CI/CD Pipeline run")
        
        try:
            # Record start time
            start_time = datetime.now()
            
            # Start component systems
            if self.auto_growth:
                self.auto_growth.start()
            if self.root_system:
                self.root_system.start()
            
            # 1. DISCOVERY stage
            discovery_results = self.run_discovery()
            
            # 2. ATTACHMENT stage
            attachment_results = self.run_attachment(discovery_results)
            
            # 3. GROWTH stage
            growth_results = self.run_growth(attachment_results)
            
            # 4. LINKING stage
            linking_results = self.run_linking()
            
            # 5. DEPLOYMENT stage
            if self.config["auto_deploy"]:
                deployment_results = self.run_deployment()
            
            # Update metrics
            self.metrics["last_run"] = datetime.now().isoformat()
            self.metrics["total_runs"] += 1
            self.metrics["knowledge_size"] = len(self.temple_mapper.nodes) if self.temple_mapper else 0
            
            # Record elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed Knowledge CI/CD Pipeline run in {elapsed:.2f} seconds")
            
            # Save metrics
            self._save_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pipeline run: {e}")
            return False
        finally:
            # Stop component systems only if they were started by us
            if self.auto_growth and self.auto_growth.growth_engine.running:
                self.auto_growth.stop()
    
    def run_discovery(self) -> List[Dict[str, Any]]:
        """Run the discovery stage to find new knowledge sources"""
        logger.info("STAGE 1: DISCOVERY - Finding new knowledge sources")
        
        discovered_sources = []
        
        # Use the auto growth system to discover knowledge sources
        if self.auto_growth and self.auto_growth.growth_engine:
            growth_engine = self.auto_growth.growth_engine
            
            # Get seed concepts from temple mapper
            seed_concepts = []
            for node in self.temple_mapper.nodes.values():
                seed_concepts.append(node.concept)
            
            # If no concepts exist yet, add some starter concepts
            if not seed_concepts:
                seed_concepts = ["neural_network", "artificial_intelligence", "consciousness", "memory_system"]
                for concept in seed_concepts:
                    node = SpatialNode(concept=concept, node_type="concept", weight=0.8)
                    self.temple_mapper.nodes[node.id] = node
            
            # Select a subset of concepts to use as seeds
            selected_seeds = random.sample(
                seed_concepts, 
                min(self.config["max_discoveries_per_run"], len(seed_concepts))
            )
            
            # For each source type, discover related sources
            for source_type in self.config["discovery_sources"]:
                for concept in selected_seeds:
                    # Create a temporary seed
                    temp_seed = ConceptSeed(
                        concept=concept,
                        weight=0.8,
                        node_type="concept",
                        connections=set(),
                        attributes={"discovery_source": True}
                    )
                    
                    # Use the growth engine's discovery method
                    if source_type == "wiki":
                        sources = growth_engine._find_wiki_sources(concept)
                    elif source_type == "web":
                        sources = growth_engine._find_web_sources(concept)
                    else:
                        sources = []
                    
                    # Add valid sources to the discoveries
                    for source in sources:
                        discovered_sources.append({
                            "source_id": source.id,
                            "name": source.name,
                            "source_type": source.source_type,
                            "seed_concept": concept
                        })
                        
                        # Store in growth engine for later use
                        growth_engine.knowledge_sources[source.id] = source
            
            # Update metrics
            self.metrics["discoveries"] += len(discovered_sources)
            
            logger.info(f"Discovered {len(discovered_sources)} new knowledge sources")
        else:
            logger.warning("Auto growth system not available for discovery")
        
        return discovered_sources
    
    def run_attachment(self, discoveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the attachment stage to connect new knowledge to existing knowledge"""
        logger.info("STAGE 2: ATTACHMENT - Connecting discoveries to existing knowledge")
        
        attachments = []
        
        if not self.temple_mapper or not discoveries:
            logger.warning("Temple mapper not available or no discoveries to attach")
            return attachments
        
        # For each discovery, find attachment points in the temple
        for discovery in discoveries:
            # Get the source from growth engine
            source_id = discovery["source_id"]
            source = self.auto_growth.growth_engine.knowledge_sources.get(source_id)
            
            if not source:
                continue
            
            # Find potential attachment points based on concept similarity
            attachment_points = []
            seed_concept = discovery["seed_concept"]
            
            for node_id, node in self.temple_mapper.nodes.items():
                # Compare concepts to find similarity
                # In a real system, this would use vector embeddings or semantic similarity
                # Here we use a simplified approach based on string overlap
                similarity = self._calculate_similarity(seed_concept, node.concept)
                
                if similarity >= self.config["attachment_threshold"]:
                    attachment_points.append({
                        "node_id": node_id,
                        "concept": node.concept,
                        "similarity": similarity
                    })
            
            # Create attachment if attachment points found
            if attachment_points:
                # Sort by similarity
                attachment_points.sort(key=lambda x: x["similarity"], reverse=True)
                
                # Take the top 3 attachment points
                top_attachments = attachment_points[:3]
                
                # Record the attachment
                attachment = {
                    "source_id": source_id,
                    "source_name": source.name,
                    "seed_concept": seed_concept,
                    "attachment_points": top_attachments
                }
                attachments.append(attachment)
                
                # Create connections in the temple mapper
                for ap in top_attachments:
                    node_id = ap["node_id"]
                    node = self.temple_mapper.nodes.get(node_id)
                    if node:
                        # Add an attribute marking this as an attachment point
                        node.attributes["attached_to_source"] = source.name
                        node.attributes["attached_at"] = datetime.now().isoformat()
                        node.attributes["source_id"] = source_id
        
        # Update metrics
        self.metrics["attachments"] += len(attachments)
        
        logger.info(f"Created {len(attachments)} attachments between discoveries and existing knowledge")
        return attachments
    
    def _calculate_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concepts"""
        # In a real system, this would use vector embeddings or NLP
        # Here we use a simplified approach based on string overlap
        c1 = concept1.lower().replace("_", " ")
        c2 = concept2.lower().replace("_", " ")
        
        # Check for exact match
        if c1 == c2:
            return 1.0
        
        # Check for substring
        if c1 in c2 or c2 in c1:
            return 0.8
        
        # Check for word overlap
        words1 = set(c1.split())
        words2 = set(c2.split())
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        if total == 0:
            return 0.0
            
        return overlap / total
    
    def run_growth(self, attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the growth stage to grow new connections from attached knowledge"""
        logger.info("STAGE 3: GROWTH - Growing new connections from attached knowledge")
        
        growth_results = {
            "new_nodes": 0,
            "new_connections": 0,
            "growth_iterations": 0
        }
        
        if not self.auto_growth or not attachments:
            logger.warning("Auto growth system not available or no attachments to grow from")
            return growth_results
        
        # Run multiple growth iterations
        for i in range(self.config["growth_iterations"]):
            # For each attachment, extract concepts and create nodes
            for attachment in attachments:
                source_id = attachment["source_id"]
                source = self.auto_growth.growth_engine.knowledge_sources.get(source_id)
                
                if not source:
                    continue
                
                # Extract concepts from the source
                concepts = self.auto_growth.growth_engine._extract_concepts_from_source(
                    source, 
                    ConceptSeed(
                        concept=attachment["seed_concept"],
                        weight=0.8,
                        node_type="concept",
                        connections=set()
                    )
                )
                
                # Incorporate extracted concepts into the temple
                if concepts:
                    self.auto_growth.growth_engine._incorporate_concepts(
                        concepts,
                        ConceptSeed(
                            concept=attachment["seed_concept"],
                            weight=0.8,
                            node_type="concept",
                            connections=set()
                        ),
                        source
                    )
                    
                    # Update results
                    growth_results["new_nodes"] += len(concepts)
                    growth_results["new_connections"] += len(concepts) * 2  # Estimate connections
            
            # Update iteration count
            growth_results["growth_iterations"] += 1
            
            # Allow time for growth to occur
            time.sleep(1)
        
        # Update metrics
        self.metrics["growth_points"] += growth_results["new_nodes"]
        
        logger.info(f"Created {growth_results['new_nodes']} new nodes from growth")
        return growth_results
    
    def run_linking(self) -> Dict[str, Any]:
        """Run the linking stage to create bidirectional links between v1-v7 and v8"""
        logger.info("STAGE 4: LINKING - Creating bidirectional links between v1-v7 and v8")
        
        linking_results = {
            "v8_to_v7_links": 0,
            "v7_to_v8_links": 0,
            "bidirectional_links": 0
        }
        
        if not self.root_system:
            logger.warning("Root system not available for linking")
            return linking_results
        
        # Test bidirectional flow using the root connection system
        start_time = time.time()
        timeout = self.config["bidirectional_timeout"]
        
        # Record initial state
        initial_stats = self.root_system.get_statistics()
        initial_roots = initial_stats.get("root_concepts", 0)
        initial_temple_nodes = len(self.temple_mapper.nodes) if self.temple_mapper else 0
        
        # Flow from v8 to v1-v7 (seed to DNA)
        if self.root_system.vascular_system:
            # Manually harvest nutrients
            self.root_system.vascular_system._harvest_from_temple()
            # Process them
            self.root_system.vascular_system._process_nutrient_queue()
            
            # Wait and check results
            time.sleep(2)
            mid_stats = self.root_system.get_statistics()
            mid_roots = mid_stats.get("root_concepts", 0)
            
            # Calculate v8 to v7 links
            v8_to_v7_links = mid_roots - initial_roots
            linking_results["v8_to_v7_links"] = max(0, v8_to_v7_links)
        
        # Flow from v1-v7 to v8 (DNA to seed)
        if self.root_system.vascular_system and self.root_system.vascular_system.root_nexus:
            # Manually apply root knowledge to temple
            updates = self.root_system.vascular_system.root_nexus.apply_to_temple(self.temple_mapper)
            
            # Calculate v7 to v8 links
            linking_results["v7_to_v8_links"] = updates
            
            # Wait and check results
            time.sleep(2)
            final_temple_nodes = len(self.temple_mapper.nodes) if self.temple_mapper else 0
            
            # Additional verification of v7 to v8 flow
            v7_to_v8_flow = final_temple_nodes - initial_temple_nodes
            if v7_to_v8_flow > 0:
                linking_results["v7_to_v8_links"] = max(linking_results["v7_to_v8_links"], v7_to_v8_flow)
        
        # Calculate bidirectional links
        linking_results["bidirectional_links"] = min(
            linking_results["v8_to_v7_links"],
            linking_results["v7_to_v8_links"]
        )
        
        # Update metrics
        self.metrics["bidirectional_links"] += linking_results["bidirectional_links"]
        
        logger.info(f"Created {linking_results['bidirectional_links']} bidirectional links")
        return linking_results
    
    def run_deployment(self) -> Dict[str, Any]:
        """Run the deployment stage to make enhanced knowledge available"""
        logger.info("STAGE 5: DEPLOYMENT - Making enhanced knowledge available")
        
        deployment_results = {
            "fruits_packaged": 0,
            "temple_updated": False,
            "api_updated": False
        }
        
        # 1. Package knowledge fruits
        if self.temple_mapper and hasattr(self.temple_mapper, 'nodes'):
            # Get top nodes by weight
            nodes = list(self.temple_mapper.nodes.values())
            nodes.sort(key=lambda n: n.weight, reverse=True)
            top_nodes = nodes[:10]  # Package top 10 nodes
            
            fruits = []
            for node in top_nodes:
                # Create knowledge fruit
                fruit = KnowledgeFruit(
                    content=f"Concept: {node.concept}",
                    patterns={
                        "weight": node.weight,
                        "node_type": node.node_type
                    },
                    source_version=8.0,
                    consciousness_imprint=node.weight,
                    metadata={
                        "created_by": "Knowledge CI/CD",
                        "created_at": datetime.now().isoformat(),
                        "node_id": node.id
                    }
                )
                fruits.append(fruit)
            
            # Save fruits to deployment directory
            deployment_dir = os.path.join(project_root, "data", "deployment")
            os.makedirs(deployment_dir, exist_ok=True)
            
            with open(os.path.join(deployment_dir, f"knowledge_fruits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w') as f:
                json.dump([fruit.to_dict() for fruit in fruits], f, indent=2)
            
            deployment_results["fruits_packaged"] = len(fruits)
        
        # 2. Update spatial temple
        try:
            if self.connector:
                self.connector.update_temple()
                deployment_results["temple_updated"] = True
        except Exception as e:
            logger.error(f"Error updating temple: {e}")
        
        # 3. Update API (if available)
        try:
            api_path = os.path.join(project_root, "src", "v8", "api_integration.py")
            if os.path.exists(api_path):
                # Execute the API integration module
                api_result = os.system(f"python {api_path} --deploy")
                deployment_results["api_updated"] = api_result == 0
        except Exception as e:
            logger.error(f"Error updating API: {e}")
        
        # Update metrics
        self.metrics["deployments"] += 1
        
        logger.info(f"Deployed {deployment_results['fruits_packaged']} knowledge fruits")
        return deployment_results
    
    def _save_metrics(self):
        """Save pipeline metrics to file"""
        try:
            metrics_dir = os.path.join(project_root, "data", "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_path = os.path.join(metrics_dir, "knowledge_ci_cd_metrics.json")
            
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics about the pipeline"""
        stats = self.metrics.copy()
        
        # Add current component stats
        if self.temple_mapper:
            stats["temple_nodes"] = len(self.temple_mapper.nodes)
            stats["temple_zones"] = len(self.temple_mapper.zones)
        
        if self.root_system:
            root_stats = self.root_system.get_statistics()
            stats.update(root_stats)
        
        if self.auto_growth:
            growth_stats = self.auto_growth.get_statistics()
            stats["growth_active"] = growth_stats.get("growth_active", False)
            stats["knowledge_sources"] = growth_stats.get("knowledge_sources", 0)
        
        stats["running"] = self.running
        
        return stats

def run_knowledge_ci_cd():
    """Run the knowledge CI/CD pipeline"""
    parser = argparse.ArgumentParser(description="Knowledge CI/CD Pipeline for v8")
    parser.add_argument("--initialize", action="store_true", help="Initialize the pipeline")
    parser.add_argument("--start", action="store_true", help="Start the scheduled pipeline")
    parser.add_argument("--stop", action="store_true", help="Stop the scheduled pipeline")
    parser.add_argument("--run-once", action="store_true", help="Run the pipeline once")
    parser.add_argument("--stage", type=str, help="Run a specific stage (discovery, attachment, growth, linking, deployment)")
    parser.add_argument("--stats", action="store_true", help="Print current statistics")
    args = parser.parse_args()
    
    # Create the pipeline
    pipeline = KnowledgePipeline()
    
    if args.initialize:
        pipeline.initialize()
    
    if args.start:
        pipeline.start()
        
        # Keep running until interrupted
        try:
            while pipeline.running:
                time.sleep(10)
        except KeyboardInterrupt:
            pipeline.stop()
    
    if args.stop:
        pipeline.stop()
    
    if args.run_once:
        if not pipeline.temple_mapper:
            pipeline.initialize()
        pipeline.run_pipeline()
    
    if args.stage:
        if not pipeline.temple_mapper:
            pipeline.initialize()
            
        if args.stage == "discovery":
            pipeline.run_discovery()
        elif args.stage == "attachment":
            discoveries = pipeline.run_discovery()
            pipeline.run_attachment(discoveries)
        elif args.stage == "growth":
            discoveries = pipeline.run_discovery()
            attachments = pipeline.run_attachment(discoveries)
            pipeline.run_growth(attachments)
        elif args.stage == "linking":
            pipeline.run_linking()
        elif args.stage == "deployment":
            pipeline.run_deployment()
        else:
            print(f"Unknown stage: {args.stage}")
    
    if args.stats:
        stats = pipeline.get_statistics()
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    run_knowledge_ci_cd() 