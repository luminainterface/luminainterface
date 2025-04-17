#!/usr/bin/env python3
"""
Automated Seed Growth System (v8)

This module implements an automated process for knowledge expansion where
concepts from the temple/seed system can autonomously discover, attach to,
and grow from external knowledge sources (like wiki articles).

The system works like mold spreading to new nutrients or seeds finding fertile soil:
1. Seeds from the temple seek out relevant external knowledge
2. When a match is found, they attach and begin extracting new concepts
3. These new concepts are automatically incorporated into the knowledge graph
4. The process repeats organically, allowing continuous knowledge expansion
"""

import os
import sys
import logging
import json
import random
import time
import threading
import re
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, TYPE_CHECKING
import uuid

# Add parent directory to path for imports
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
from src.v8.spatial_temple_mapper import SpatialTempleMapper, SpatialNode

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from src.v8.temple_to_seed_bridge import ConceptSeed
else:
    # At runtime, we'll import ConceptSeed dynamically when needed
    ConceptSeed = Any

from src.v8.demo_data_generator import generate_demo_nodes, NODE_TYPES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v8.auto_seed_growth")

class KnowledgeSource:
    """Represents an external knowledge source that seeds can connect to"""
    
    def __init__(self, name: str, source_type: str = "wiki"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.source_type = source_type  # "wiki", "database", "text", etc.
        self.accessed_at = datetime.now().isoformat()
        self.content_summary = ""
        self.extracted_concepts = []
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "accessed_at": self.accessed_at,
            "content_summary": self.content_summary,
            "extracted_concepts": self.extracted_concepts,
            "metadata": self.metadata
        }

class SeedGrowthEngine:
    """
    Engine that manages automated growth of seeds by connecting to
    external knowledge sources and extracting new concepts.
    """
    
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None):
        self.temple_mapper = temple_mapper
        self.seed_pool = []  # Current active seeds
        self.knowledge_sources = {}  # External sources that have been accessed
        self.growth_history = []  # Record of growth events
        self.running = False
        self.growth_thread = None
        self.discovery_rate = 10  # Seconds between discovery attempts
        
        # Load API keys and config
        self.load_config()
        
    def load_config(self):
        """Load configuration for external APIs"""
        self.config = {
            "wiki_enabled": True,
            "web_enabled": True,
            "max_sources_per_cycle": 3,
            "concept_extraction_threshold": 0.6,
            "max_concepts_per_source": 10
        }
        
        # Try to load API keys from file
        try:
            api_keys_path = os.path.join(project_root, "config", "api_keys.json")
            if os.path.exists(api_keys_path):
                with open(api_keys_path, 'r') as f:
                    self.api_keys = json.load(f)
            else:
                self.api_keys = {}
        except Exception as e:
            logger.warning(f"Failed to load API keys: {e}")
            self.api_keys = {}
    
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper to harvest seeds from"""
        self.temple_mapper = temple_mapper
        
    def start_growth(self):
        """Start the automated growth process"""
        if self.running:
            logger.info("Growth engine already running")
            return
            
        self.running = True
        self.growth_thread = threading.Thread(target=self._growth_loop)
        self.growth_thread.daemon = True
        self.growth_thread.start()
        logger.info("Started automated seed growth engine")
        
    def stop_growth(self):
        """Stop the automated growth process"""
        self.running = False
        if self.growth_thread:
            self.growth_thread.join(timeout=2.0)
        logger.info("Stopped automated seed growth engine")
        
    def _growth_loop(self):
        """Main loop for the growth process"""
        while self.running:
            try:
                # 1. Ensure we have seeds in the pool
                if len(self.seed_pool) < 5:
                    self._harvest_seeds_from_temple()
                
                # 2. Select seeds for growth
                growth_candidates = self._select_seeds_for_growth()
                
                if not growth_candidates:
                    logger.info("No suitable growth candidates found")
                    time.sleep(self.discovery_rate)
                    continue
                
                # 3. For each candidate, find relevant external knowledge
                for seed in growth_candidates:
                    sources = self._discover_knowledge_sources(seed)
                    
                    # 4. Connect to sources and extract concepts
                    for source in sources:
                        try:
                            new_concepts = self._extract_concepts_from_source(source, seed)
                            
                            # 5. Create new nodes from extracted concepts
                            if new_concepts:
                                self._incorporate_concepts(new_concepts, seed, source)
                                
                                # Record growth event
                                self.growth_history.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "seed_concept": seed.concept,
                                    "source_name": source.name,
                                    "concepts_added": len(new_concepts)
                                })
                                
                                logger.info(f"Added {len(new_concepts)} new concepts from {source.name}")
                        except Exception as e:
                            logger.error(f"Error processing source {source.name}: {e}")
                
                # Wait before next cycle
                time.sleep(self.discovery_rate)
                
            except Exception as e:
                logger.error(f"Error in growth loop: {e}")
                time.sleep(self.discovery_rate * 2)  # Wait longer after error
    
    def _harvest_seeds_from_temple(self):
        """Harvest seeds from the temple mapper to use for growth"""
        if not self.temple_mapper or not self.temple_mapper.nodes:
            logger.warning("No temple mapper available for seed harvesting")
            return
            
        # Import ConceptSeed dynamically to avoid circular imports
        from src.v8.temple_to_seed_bridge import ConceptSeed
            
        # Get nodes sorted by weight
        nodes = list(self.temple_mapper.nodes.values())
        nodes.sort(key=lambda n: n.weight, reverse=True)
        
        # Take top nodes plus some random ones for diversity
        top_count = min(10, len(nodes))
        seed_nodes = nodes[:top_count]
        
        if len(nodes) > top_count:
            seed_nodes.extend(random.sample(nodes[top_count:], min(5, len(nodes) - top_count)))
            
        # Convert to seeds and add to pool
        for node in seed_nodes:
            seed = ConceptSeed.from_spatial_node(node)
            
            # Check if already in pool
            if not any(s.concept == seed.concept for s in self.seed_pool):
                self.seed_pool.append(seed)
                
        logger.info(f"Harvested {len(seed_nodes)} seeds from temple, pool now has {len(self.seed_pool)} seeds")
    
    def _select_seeds_for_growth(self, max_count: int = 3) -> List[ConceptSeed]:
        """Select seeds from the pool for the current growth cycle"""
        if not self.seed_pool:
            return []
            
        # Use weighted random selection based on seed weight
        weights = [seed.weight for seed in self.seed_pool]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.sample(self.seed_pool, min(max_count, len(self.seed_pool)))
            
        # Normalize weights
        normalized_weights = [w/total_weight for w in weights]
        
        # Select seeds
        selected_indices = []
        for _ in range(min(max_count, len(self.seed_pool))):
            if not normalized_weights:
                break
                
            # Weighted selection
            r = random.random()
            cumulative = 0
            for i, weight in enumerate(normalized_weights):
                if i in selected_indices:
                    continue
                    
                cumulative += weight
                if r <= cumulative:
                    selected_indices.append(i)
                    break
        
        # Get selected seeds
        return [self.seed_pool[i] for i in selected_indices]
    
    def _discover_knowledge_sources(self, seed: ConceptSeed) -> List[KnowledgeSource]:
        """Discover relevant knowledge sources for a seed concept"""
        sources = []
        
        # Limit number of sources per cycle
        max_sources = self.config.get("max_sources_per_cycle", 3)
        
        # Try wiki sources if enabled
        if self.config.get("wiki_enabled", True):
            wiki_sources = self._find_wiki_sources(seed.concept)
            sources.extend(wiki_sources[:max_sources])
            
        # Try web sources if enabled and we need more
        if self.config.get("web_enabled", True) and len(sources) < max_sources:
            web_sources = self._find_web_sources(seed.concept)
            sources.extend(web_sources[:max_sources - len(sources)])
            
        return sources
    
    def _find_wiki_sources(self, concept: str) -> List[KnowledgeSource]:
        """Find relevant Wikipedia articles for a concept"""
        sources = []
        
        try:
            # Simple Wikipedia API search
            query = urllib.parse.quote(concept)
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=3&namespace=0&format=json"
            
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                
                # Format: [search_term, [titles], [descriptions], [urls]]
                if len(data) >= 4:
                    titles = data[1]
                    descriptions = data[2]
                    urls = data[3]
                    
                    for i in range(len(titles)):
                        source = KnowledgeSource(titles[i], "wiki")
                        source.content_summary = descriptions[i] if i < len(descriptions) else ""
                        source.metadata = {"url": urls[i] if i < len(urls) else ""}
                        sources.append(source)
                        
                        # Store in knowledge sources dictionary
                        self.knowledge_sources[source.id] = source
        except Exception as e:
            logger.error(f"Error finding wiki sources for {concept}: {e}")
            
        return sources
    
    def _find_web_sources(self, concept: str) -> List[KnowledgeSource]:
        """Find relevant web pages for a concept"""
        # This would typically use a search API
        # For now, we'll return a simulated result
        
        sources = []
        fake_sources = [
            (f"{concept} - Overview", f"An overview of {concept} and related topics"),
            (f"Learning about {concept}", f"Educational resources about {concept}"),
            (f"{concept} in context", f"How {concept} relates to other domains")
        ]
        
        for title, desc in fake_sources:
            source = KnowledgeSource(title, "web")
            source.content_summary = desc
            source.metadata = {"url": f"https://example.com/{urllib.parse.quote(title.lower().replace(' ', '-'))}"}
            sources.append(source)
            
            # Store in knowledge sources dictionary
            self.knowledge_sources[source.id] = source
            
        return sources
    
    def _extract_concepts_from_source(self, source: KnowledgeSource, seed: ConceptSeed) -> List[Dict[str, Any]]:
        """Extract new concepts from a knowledge source based on a seed"""
        # Import ConceptSeed dynamically to avoid circular imports
        from src.v8.temple_to_seed_bridge import ConceptSeed
        
        concepts = []
        source_type = source.source_type
        
        logger.info(f"Extracting concepts from {source.name} ({source_type})")
        
        # Different extraction methods based on source type
        if source_type == "wiki":
            concepts = self._extract_from_wiki(source, seed)
        elif source_type == "web":
            concepts = self._extract_from_web(source, seed)
        elif source_type == "text":
            concepts = self._extract_from_text(source, seed)
        else:
            logger.warning(f"Unsupported source type: {source_type}")
            return []
            
        # Limit number of concepts
        max_concepts = self.config.get("max_concepts_per_source", 10)
        if len(concepts) > max_concepts:
            # Sort by relevance and take top ones
            concepts.sort(key=lambda c: c.get("relevance", 0), reverse=True)
            concepts = concepts[:max_concepts]
            
        return concepts
    
    def _incorporate_concepts(self, concepts: List[Dict[str, Any]], parent_seed: ConceptSeed, source: KnowledgeSource):
        """Incorporate new concepts into the temple mapper"""
        # Import ConceptSeed dynamically to avoid circular imports
        from src.v8.temple_to_seed_bridge import ConceptSeed
        
        if not self.temple_mapper:
            logger.warning("No temple mapper available to incorporate concepts")
            return
        
        for concept_data in concepts:
            # Create a new node in the temple
            concept_name = concept_data.get("name", "")
            if not concept_name:
                continue
                
            # Check if node already exists
            existing_nodes = [n for n in self.temple_mapper.nodes.values() 
                              if n.concept.lower() == concept_name.lower()]
            
            if existing_nodes:
                # Update existing node
                node = existing_nodes[0]
                
                # Increase weight
                node.weight = min(1.0, node.weight + 0.1)
                
                # Add connection to parent if not already connected
                if parent_seed.source_node_id and parent_seed.source_node_id not in node.connections:
                    node.connections.add(parent_seed.source_node_id)
                    
                    # Also update the parent node connections
                    if parent_seed.source_node_id in self.temple_mapper.nodes:
                        parent_node = self.temple_mapper.nodes[parent_seed.source_node_id]
                        parent_node.connections.add(node.id)
                
                logger.info(f"Updated existing node: {node.concept}")
            else:
                # Create new node
                node_type = concept_data.get("node_type", random.choice(NODE_TYPES))
                weight = concept_data.get("weight", random.uniform(0.3, 0.7))
                
                # Create connections
                connections = set()
                if parent_seed.source_node_id:
                    connections.add(parent_seed.source_node_id)
                
                # Create attributes
                attributes = {
                    "source": source.name,
                    "discovery_date": datetime.now().isoformat(),
                    "description": concept_data.get("description", ""),
                    "relevance": concept_data.get("relevance", 0.5)
                }
                
                # Add any additional attributes from concept data
                if "attributes" in concept_data:
                    attributes.update(concept_data["attributes"])
                
                # Create the node
                node_id = str(uuid.uuid4())
                new_node = SpatialNode(
                    id=node_id,
                    concept=concept_name,
                    weight=weight,
                    node_type=node_type,
                    connections=connections,
                    attributes=attributes
                )
                
                # Add to temple mapper
                self.temple_mapper.add_node(new_node)
                
                # Update parent connections
                if parent_seed.source_node_id in self.temple_mapper.nodes:
                    parent_node = self.temple_mapper.nodes[parent_seed.source_node_id]
                    parent_node.connections.add(node_id)
                
                logger.info(f"Added new node: {concept_name}")
    
    def get_growth_statistics(self) -> Dict[str, Any]:
        """Get statistics about the growth process"""
        stats = {
            "total_seeds": len(self.seed_pool),
            "knowledge_sources_accessed": len(self.knowledge_sources),
            "growth_events": len(self.growth_history),
            "concepts_added": sum(event["concepts_added"] for event in self.growth_history),
            "running": self.running,
            "last_growth": self.growth_history[-1]["timestamp"] if self.growth_history else None
        }
        return stats

class AutoSeedGrowthSystem:
    """Main class for the automated seed growth system"""
    
    def __init__(self, temple_mapper: Optional[SpatialTempleMapper] = None):
        self.growth_engine = SeedGrowthEngine(temple_mapper)
    
    def start(self):
        """Start the growth engine"""
        self.growth_engine.start_growth()
    
    def stop(self):
        """Stop the growth engine"""
        self.growth_engine.stop_growth()
    
    def set_temple_mapper(self, temple_mapper: SpatialTempleMapper):
        """Set the temple mapper to use for growth"""
        self.growth_engine.set_temple_mapper(temple_mapper)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get growth statistics"""
        return self.growth_engine.get_growth_statistics()

def create_demo_temple_mapper() -> SpatialTempleMapper:
    """Create a demo temple mapper for testing"""
    mapper = SpatialTempleMapper()
    
    # Generate some demo nodes
    nodes = generate_demo_nodes(30)
    
    # Add to mapper
    for node in nodes:
        mapper.add_node(node)
    
    return mapper

# We need to define the missing extraction methods that are now referenced
def _extract_from_wiki(self, source: KnowledgeSource, seed: Any) -> List[Dict[str, Any]]:
    """Extract concepts from a wiki source"""
    logger.info(f"Extracting concepts from Wiki: {source.name}")
    # Simplified version for demo purposes
    concepts = []
    
    # Generate some plausible concepts based on the seed
    concept_base = seed.concept.lower()
    
    variations = [
        f"Advanced {concept_base.capitalize()}",
        f"{concept_base.capitalize()} Theory",
        f"{concept_base.capitalize()} Applications",
        f"{concept_base.capitalize()} Systems",
        f"{concept_base.capitalize()} Framework",
        f"Modern {concept_base.capitalize()}"
    ]
    
    for variant in variations:
        concepts.append({
            "name": variant,
            "description": f"Extracted from wiki about {source.name}",
            "weight": random.uniform(0.5, 0.8),
            "node_type": random.choice(NODE_TYPES),
            "relevance": random.uniform(0.7, 1.0)
        })
    
    return concepts

def _extract_from_web(self, source: KnowledgeSource, seed: Any) -> List[Dict[str, Any]]:
    """Extract concepts from a web source"""
    logger.info(f"Extracting concepts from Web: {source.name}")
    # Simplified version for demo purposes
    concepts = []
    
    # Generate some plausible concepts based on the seed
    concept_base = seed.concept.lower()
    
    variations = [
        f"{concept_base.capitalize()} Platform",
        f"{concept_base.capitalize()} Service",
        f"Online {concept_base.capitalize()}",
        f"{concept_base.capitalize()} API",
        f"Digital {concept_base.capitalize()}",
        f"{concept_base.capitalize()} Interface"
    ]
    
    for variant in variations:
        concepts.append({
            "name": variant,
            "description": f"Extracted from web source {source.name}",
            "weight": random.uniform(0.4, 0.7),
            "node_type": random.choice(NODE_TYPES),
            "relevance": random.uniform(0.6, 0.9)
        })
    
    return concepts

def _extract_from_text(self, source: KnowledgeSource, seed: Any) -> List[Dict[str, Any]]:
    """Extract concepts from a text source"""
    logger.info(f"Extracting concepts from Text: {source.name}")
    # Simplified version for demo purposes
    concepts = []
    
    # Generate some plausible concepts based on the seed
    concept_base = seed.concept.lower()
    
    variations = [
        f"{concept_base.capitalize()} Analysis",
        f"{concept_base.capitalize()} Patterns",
        f"{concept_base.capitalize()} Structure",
        f"Textual {concept_base.capitalize()}",
        f"{concept_base.capitalize()} Content",
        f"{concept_base.capitalize()} Documentation"
    ]
    
    for variant in variations:
        concepts.append({
            "name": variant,
            "description": f"Extracted from text source {source.name}",
            "weight": random.uniform(0.3, 0.6),
            "node_type": random.choice(NODE_TYPES),
            "relevance": random.uniform(0.5, 0.8)
        })
    
    return concepts

# Add the extraction methods to SeedGrowthEngine
SeedGrowthEngine._extract_from_wiki = _extract_from_wiki
SeedGrowthEngine._extract_from_web = _extract_from_web
SeedGrowthEngine._extract_from_text = _extract_from_text

if __name__ == "__main__":
    # Create a demo temple mapper
    temple_mapper = create_demo_temple_mapper()
    
    # Initialize auto-growth system
    growth_system = AutoSeedGrowthSystem(temple_mapper)
    
    # Start growth
    growth_system.start()
    
    try:
        # Run for a while to simulate growth
        print("Auto seed growth system running...")
        print("Press Ctrl+C to stop")
        
        # Check stats periodically
        for _ in range(10):
            time.sleep(5)
            stats = growth_system.get_statistics()
            print(f"Seeds: {stats['total_seeds']}, Sources accessed: {stats['knowledge_sources_accessed']}")
            print(f"Growth events: {stats['growth_events']}, Concepts added: {stats['concepts_added']}")
    except KeyboardInterrupt:
        print("\nStopping growth system...")
    finally:
        # Stop system
        growth_system.stop()
        print("Growth system stopped") 