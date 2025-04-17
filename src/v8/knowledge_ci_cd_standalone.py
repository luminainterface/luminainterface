#!/usr/bin/env python3
"""
Standalone Knowledge CI/CD System for v8

This simplified version implements the mold-like growth behavior without depending
on other v8 components, avoiding circular imports. It demonstrates how the system:

1. DISCOVERY: Finds new knowledge sources automatically
2. ATTACHMENT: Connects discoveries to existing knowledge
3. GROWTH: Grows new connections from attached knowledge
4. LINKING: Creates bidirectional links between v1-v7 (DNA) and v8 (seed)
5. DEPLOYMENT: Makes the enhanced knowledge available to the system
"""

import os
import sys
import time
import logging
import json
import random
import threading
import argparse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

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

# Simple mock classes to represent v8 components
class SpatialNode:
    """Mock of SpatialNode from spatial_temple_mapper"""
    def __init__(self, concept: str, node_type: str = "concept", weight: float = 0.5):
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.position = (random.random(), random.random(), random.random())
        self.node_type = node_type
        self.weight = weight
        self.connections = set()
        self.attributes = {}

class SpatialTempleMapper:
    """Mock of SpatialTempleMapper from spatial_temple_mapper"""
    def __init__(self):
        self.nodes = {}
        self.zones = {}
    
    def add_node(self, concept: str, node_type: str = "concept", weight: float = 0.5) -> SpatialNode:
        node = SpatialNode(concept, node_type, weight)
        self.nodes[node.id] = node
        return node

class NutrientPacket:
    """Mock of NutrientPacket from root_connection_system"""
    def __init__(self, source_concept: str, source_id: str, enriched_data: Dict[str, Any], strength: float = 1.0):
        self.id = str(uuid.uuid4())
        self.source_concept = source_concept
        self.source_id = source_id
        self.enriched_data = enriched_data
        self.strength = strength
        self.created_at = datetime.now().isoformat()
        self.status = "created"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_concept": self.source_concept,
            "source_id": self.source_id,
            "enriched_data": self.enriched_data,
            "strength": self.strength,
            "created_at": self.created_at,
            "status": self.status
        }

class RootNexus:
    """Mock of RootNexus from root_connection_system"""
    def __init__(self):
        self.core_concepts = {}
        self.connections = {}
        self.nutrient_history = []
        self.last_update = datetime.now().isoformat()
    
    def add_nutrient(self, nutrient: NutrientPacket) -> bool:
        concept_key = nutrient.source_concept.lower()
        
        # Add to core concepts
        self.core_concepts[concept_key] = {
            "concept": nutrient.source_concept,
            "strength": nutrient.strength,
            "data": nutrient.enriched_data.copy(),
            "last_update": datetime.now().isoformat()
        }
        
        # Add connections if any
        if "connections" in nutrient.enriched_data:
            for conn in nutrient.enriched_data["connections"]:
                target_concept = conn[0].lower()
                conn_key = tuple(sorted([concept_key, target_concept]))
                self.connections[conn_key] = {
                    "concepts": [concept_key, target_concept],
                    "strength": conn[1],
                    "last_update": datetime.now().isoformat()
                }
        
        # Record in history
        nutrient.status = "absorbed"
        self.nutrient_history.append(nutrient.to_dict())
        self.last_update = datetime.now().isoformat()
        
        return True
    
    def apply_to_temple(self, temple_mapper: SpatialTempleMapper) -> int:
        """Apply v1-v7 knowledge to v8 temple (DNA to seed flow)"""
        updates = 0
        
        # For each core concept, create a node in the temple if it doesn't exist
        for concept_key, data in self.core_concepts.items():
            # Check if concept already exists in temple
            exists = False
            for node in temple_mapper.nodes.values():
                if node.concept.lower() == concept_key:
                    exists = True
                    break
                    
            if not exists:
                # Create new node in temple
                node = SpatialNode(
                    concept=data["concept"],
                    node_type="knowledge",
                    weight=data["strength"]
                )
                temple_mapper.nodes[node.id] = node
                updates += 1
                
        return updates

class KnowledgeSource:
    """Mock of KnowledgeSource from auto_seed_growth"""
    def __init__(self, name: str, source_type: str = "wiki"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.source_type = source_type
        self.accessed_at = datetime.now().isoformat()
        self.content_summary = ""
        self.extracted_concepts = []
        self.metadata = {}

class KnowledgePipeline:
    """
    Implements the CI/CD pipeline for knowledge growth and integration.
    Like a mold's mycelial network that spreads and connects.
    """
    
    def __init__(self):
        """Initialize the knowledge pipeline"""
        self.temple_mapper = SpatialTempleMapper()
        self.root_nexus = RootNexus()
        self.knowledge_sources = {}
        self.running = False
        
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
            "discovery_sources": ["wiki", "web", "database", "api"],
            "attachment_threshold": 0.65,
            "growth_iterations": 3,
            "max_discoveries_per_run": 5,
            "auto_deploy": True
        }
    
    def initialize(self):
        """Initialize the pipeline with starter knowledge"""
        logger.info("Initializing Knowledge CI/CD Pipeline")
        
        # Add some starter concepts
        starter_concepts = [
            "neural_network", 
            "artificial_intelligence", 
            "consciousness", 
            "memory_system",
            "language_processing"
        ]
        
        for concept in starter_concepts:
            node = self.temple_mapper.add_node(
                concept=concept,
                node_type="concept",
                weight=random.uniform(0.7, 0.9)
            )
            logger.info(f"Added starter concept: {concept}")
        
        logger.info(f"Knowledge CI/CD Pipeline initialized with {len(self.temple_mapper.nodes)} concepts")
        return True
    
    def run_pipeline(self):
        """Run the complete pipeline once"""
        logger.info("Starting Knowledge CI/CD Pipeline run")
        
        try:
            # Record start time
            start_time = datetime.now()
            
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
            self.metrics["knowledge_size"] = len(self.temple_mapper.nodes)
            
            # Record elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed Knowledge CI/CD Pipeline run in {elapsed:.2f} seconds")
            
            # Save metrics
            self._save_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pipeline run: {e}")
            return False
    
    def run_discovery(self) -> List[Dict[str, Any]]:
        """Run the discovery stage to find new knowledge sources"""
        logger.info("STAGE 1: DISCOVERY - Finding new knowledge sources")
        
        discovered_sources = []
        
        # Get seed concepts from temple mapper
        seed_concepts = []
        for node in self.temple_mapper.nodes.values():
            seed_concepts.append(node.concept)
        
        # Select a subset of concepts to use as seeds
        selected_seeds = random.sample(
            seed_concepts, 
            min(self.config["max_discoveries_per_run"], len(seed_concepts))
        )
        
        # For each seed, discover related sources (simulated)
        for concept in selected_seeds:
            # For each source type, discover related sources
            for source_type in self.config["discovery_sources"]:
                # Simulate discovering 1-3 sources
                source_count = random.randint(1, 3)
                
                for i in range(source_count):
                    # Create a simulated source
                    source = KnowledgeSource(
                        name=f"{concept}_{source_type}_{i}",
                        source_type=source_type
                    )
                    
                    # Add to discoveries
                    discovered_sources.append({
                        "source_id": source.id,
                        "name": source.name,
                        "source_type": source.source_type,
                        "seed_concept": concept
                    })
                    
                    # Store for later use
                    self.knowledge_sources[source.id] = source
        
        # Update metrics
        self.metrics["discoveries"] += len(discovered_sources)
        
        logger.info(f"Discovered {len(discovered_sources)} new knowledge sources")
        return discovered_sources
    
    def run_attachment(self, discoveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the attachment stage to connect new knowledge to existing knowledge"""
        logger.info("STAGE 2: ATTACHMENT - Connecting discoveries to existing knowledge")
        
        attachments = []
        
        if not discoveries:
            logger.warning("No discoveries to attach")
            return attachments
        
        # For each discovery, find attachment points in the temple
        for discovery in discoveries:
            # Get the source
            source_id = discovery["source_id"]
            source = self.knowledge_sources.get(source_id)
            
            if not source:
                continue
            
            # Find potential attachment points based on concept similarity
            attachment_points = []
            seed_concept = discovery["seed_concept"]
            
            for node_id, node in self.temple_mapper.nodes.items():
                # Compare concepts to find similarity
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
        
        if not attachments:
            logger.warning("No attachments to grow from")
            return growth_results
        
        # Run multiple growth iterations
        for i in range(self.config["growth_iterations"]):
            # For each attachment, extract concepts and create nodes
            for attachment in attachments:
                source_id = attachment["source_id"]
                source = self.knowledge_sources.get(source_id)
                
                if not source:
                    continue
                
                # Simulate extracting concepts from the source
                # In a real system, this would analyze the source content
                
                # Get base concept
                seed_concept = attachment["seed_concept"]
                
                # Generate related concepts based on the seed concept
                related_concepts = self._generate_related_concepts(seed_concept, 3)
                
                # Add these as extracted concepts on the source
                source.extracted_concepts = related_concepts
                
                # Create new nodes for each extracted concept
                for concept in related_concepts:
                    # Check if this concept already exists
                    exists = False
                    for node in self.temple_mapper.nodes.values():
                        if node.concept.lower() == concept.lower():
                            exists = True
                            break
                    
                    if not exists:
                        # Create new node
                        new_node = self.temple_mapper.add_node(
                            concept=concept,
                            node_type=random.choice(["concept", "knowledge", "memory"]),
                            weight=random.uniform(0.5, 0.8)
                        )
                        
                        # Connect to attachment points
                        for ap in attachment["attachment_points"]:
                            ap_id = ap["node_id"]
                            if ap_id in self.temple_mapper.nodes:
                                # Create bidirectional connection
                                new_node.connections.add(ap_id)
                                self.temple_mapper.nodes[ap_id].connections.add(new_node.id)
                                growth_results["new_connections"] += 2
                        
                        growth_results["new_nodes"] += 1
            
            # Update iteration count
            growth_results["growth_iterations"] += 1
            
            # Allow time for growth to occur
            time.sleep(0.1)
        
        # Update metrics
        self.metrics["growth_points"] += growth_results["new_nodes"]
        
        logger.info(f"Created {growth_results['new_nodes']} new nodes from growth")
        return growth_results
    
    def _generate_related_concepts(self, seed: str, count: int) -> List[str]:
        """Generate related concepts based on a seed concept"""
        # This would use NLP in a real system
        # Here we'll use a simple approach
        
        # Dictionary of related terms for common concepts
        related_terms = {
            "neural_network": ["deep_learning", "backpropagation", "activation_function", "weights", "bias"],
            "artificial_intelligence": ["machine_learning", "natural_language_processing", "computer_vision", "reinforcement_learning"],
            "consciousness": ["awareness", "sentience", "qualia", "self_reflection", "cognition"],
            "memory_system": ["working_memory", "long_term_memory", "encoding", "retrieval", "storage"],
            "language_processing": ["parsing", "tokenization", "embedding", "transformer", "attention_mechanism"]
        }
        
        # Find if we have related terms for this seed
        seed_lower = seed.lower()
        concepts = []
        
        # Look for exact match
        if seed_lower in related_terms:
            # Return a random selection from related terms
            available = related_terms[seed_lower]
            selection = random.sample(available, min(count, len(available)))
            concepts.extend(selection)
        
        # Look for partial match
        else:
            for key, values in related_terms.items():
                if key in seed_lower or seed_lower in key:
                    # Return a random selection
                    selection = random.sample(values, min(count - len(concepts), len(values)))
                    concepts.extend(selection)
                    if len(concepts) >= count:
                        break
        
        # If still not enough, generate some by adding prefixes/suffixes
        prefixes = ["enhanced_", "advanced_", "automated_", "integrated_"]
        suffixes = ["_system", "_algorithm", "_model", "_framework"]
        
        while len(concepts) < count:
            if random.choice([True, False]):
                # Add prefix
                prefix = random.choice(prefixes)
                concepts.append(f"{prefix}{seed}")
            else:
                # Add suffix
                suffix = random.choice(suffixes)
                concepts.append(f"{seed}{suffix}")
        
        return concepts
    
    def run_linking(self) -> Dict[str, Any]:
        """Run the linking stage to create bidirectional links between v1-v7 and v8"""
        logger.info("STAGE 4: LINKING - Creating bidirectional links between v1-v7 and v8")
        
        linking_results = {
            "v8_to_v7_links": 0,
            "v7_to_v8_links": 0,
            "bidirectional_links": 0
        }
        
        # Record initial state
        initial_roots = len(self.root_nexus.core_concepts)
        initial_temple_nodes = len(self.temple_mapper.nodes)
        
        # 1. Flow from v8 to v1-v7 (seed to DNA)
        # Select nodes to harvest nutrients from
        nodes = list(self.temple_mapper.nodes.values())
        harvest_nodes = random.sample(nodes, min(5, len(nodes)))
        
        # Create and add nutrient packets
        for node in harvest_nodes:
            # Create enriched data
            enriched_data = {
                "weight": node.weight,
                "node_type": node.node_type,
                "connections": []
            }
            
            # Add connection information
            for conn_id in node.connections:
                if conn_id in self.temple_mapper.nodes:
                    target_node = self.temple_mapper.nodes[conn_id]
                    enriched_data["connections"].append((
                        target_node.concept,
                        min(node.weight, target_node.weight)
                    ))
            
            # Create nutrient packet
            nutrient = NutrientPacket(
                source_concept=node.concept,
                source_id=node.id,
                enriched_data=enriched_data,
                strength=node.weight
            )
            
            # Add to root nexus
            self.root_nexus.add_nutrient(nutrient)
            
            logger.info(f"Sent nutrient from '{node.concept}' to root system")
        
        # Check updated root state
        mid_roots = len(self.root_nexus.core_concepts)
        linking_results["v8_to_v7_links"] = mid_roots - initial_roots
        
        # 2. Flow from v1-v7 to v8 (DNA to seed)
        # Apply root knowledge to temple
        updates = self.root_nexus.apply_to_temple(self.temple_mapper)
        linking_results["v7_to_v8_links"] = updates
        
        # Check final temple state
        final_temple_nodes = len(self.temple_mapper.nodes)
        v7_to_v8_flow = final_temple_nodes - initial_temple_nodes
        if v7_to_v8_flow > linking_results["v7_to_v8_links"]:
            linking_results["v7_to_v8_links"] = v7_to_v8_flow
        
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
            "artifacts_packaged": 0,
            "visualizations": 0,
            "api_updates": 0
        }
        
        # 1. Package knowledge artifacts
        if self.temple_mapper and hasattr(self.temple_mapper, 'nodes'):
            # Get top nodes by weight
            nodes = list(self.temple_mapper.nodes.values())
            nodes.sort(key=lambda n: n.weight, reverse=True)
            top_nodes = nodes[:10]  # Package top 10 nodes
            
            # Create serializable knowledge artifacts
            artifacts = []
            for node in top_nodes:
                artifact = {
                    "id": node.id,
                    "concept": node.concept,
                    "weight": node.weight,
                    "node_type": node.node_type,
                    "connections": list(node.connections),
                    "attributes": node.attributes,
                    "created_by": "Knowledge CI/CD",
                    "created_at": datetime.now().isoformat()
                }
                artifacts.append(artifact)
            
            # Save artifacts to deployment directory
            deployment_dir = os.path.join("data", "deployment")
            os.makedirs(deployment_dir, exist_ok=True)
            
            artifact_path = os.path.join(deployment_dir, f"knowledge_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(artifact_path, 'w') as f:
                json.dump(artifacts, f, indent=2)
            
            deployment_results["artifacts_packaged"] = len(artifacts)
        
        # 2. Create visualizations
        try:
            # Create temple visualization data
            temple_data = {
                "nodes": {},
                "connections": [],
                "updated_at": datetime.now().isoformat()
            }
            
            # Add nodes to visualization
            for node_id, node in self.temple_mapper.nodes.items():
                temple_data["nodes"][node_id] = {
                    "id": node_id,
                    "concept": node.concept,
                    "position": node.position,
                    "node_type": node.node_type,
                    "weight": node.weight
                }
                
                # Add connections
                for conn_id in node.connections:
                    if conn_id in self.temple_mapper.nodes:
                        connection = [node_id, conn_id]
                        # Avoid duplicates by sorting
                        connection.sort()
                        if connection not in temple_data["connections"]:
                            temple_data["connections"].append(connection)
            
            # Save visualization data
            temple_dir = os.path.join("data", "temple")
            os.makedirs(temple_dir, exist_ok=True)
            
            vis_path = os.path.join(temple_dir, f"temple_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(vis_path, 'w') as f:
                json.dump(temple_data, f, indent=2)
            
            # Also save to latest for visualization tools
            latest_path = os.path.join(temple_dir, "temple_visualization_latest.json")
            with open(latest_path, 'w') as f:
                json.dump(temple_data, f, indent=2)
                
            deployment_results["visualizations"] = 1
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        # 3. Update API
        try:
            # Simulate API update
            api_data = {
                "knowledge_count": len(self.temple_mapper.nodes),
                "root_concepts": len(self.root_nexus.core_concepts),
                "top_concepts": [node.concept for node in top_nodes[:5]],
                "updated_at": datetime.now().isoformat()
            }
            
            # Save API data
            api_dir = os.path.join("data", "api")
            os.makedirs(api_dir, exist_ok=True)
            
            api_path = os.path.join(api_dir, "knowledge_api_latest.json")
            with open(api_path, 'w') as f:
                json.dump(api_data, f, indent=2)
                
            deployment_results["api_updates"] = 1
                
        except Exception as e:
            logger.error(f"Error updating API: {e}")
        
        # Update metrics
        self.metrics["deployments"] += 1
        
        logger.info(f"Deployed {deployment_results['artifacts_packaged']} knowledge artifacts")
        return deployment_results
    
    def _save_metrics(self):
        """Save pipeline metrics to file"""
        try:
            metrics_dir = os.path.join("data", "metrics")
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
        stats["temple_nodes"] = len(self.temple_mapper.nodes)
        stats["root_concepts"] = len(self.root_nexus.core_concepts)
        stats["root_connections"] = len(self.root_nexus.connections)
        stats["knowledge_sources"] = len(self.knowledge_sources)
        
        return stats

def run_knowledge_ci_cd():
    """Run the knowledge CI/CD pipeline"""
    parser = argparse.ArgumentParser(description="Knowledge CI/CD Pipeline for v8")
    parser.add_argument("--initialize", action="store_true", help="Initialize the pipeline")
    parser.add_argument("--run", action="store_true", help="Run the pipeline once")
    parser.add_argument("--stage", type=str, help="Run a specific stage (discovery, attachment, growth, linking, deployment)")
    parser.add_argument("--stats", action="store_true", help="Print current statistics")
    args = parser.parse_args()
    
    # Create the pipeline
    pipeline = KnowledgePipeline()
    
    if args.initialize:
        pipeline.initialize()
    
    if args.run:
        if not pipeline.temple_mapper.nodes:
            pipeline.initialize()
        pipeline.run_pipeline()
    
    if args.stage:
        if not pipeline.temple_mapper.nodes:
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
    
    if args.stats or not any([args.initialize, args.run, args.stage]):
        stats = pipeline.get_statistics()
        print("\nKnowledge CI/CD Pipeline Statistics:")
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    run_knowledge_ci_cd() 