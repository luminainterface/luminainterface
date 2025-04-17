#!/usr/bin/env python
"""
Knowledge Source Mock Module

This is a mock implementation of the Knowledge Source module to enable
central_node.py to import successfully. It provides stub implementations
of the required classes and methods.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_source")
logger.info("Mock Knowledge Source module loaded")

class KnowledgeEntry:
    """Represents a single knowledge entry in the knowledge base"""
    
    def __init__(self, concept: str, description: str, relations: Optional[Dict[str, List[str]]] = None):
        self.concept = concept
        self.description = description
        self.relations = relations or {}
        self.metadata = {"source": "mock", "confidence": 0.8}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge entry to a dictionary"""
        return {
            "concept": self.concept,
            "description": self.description,
            "relations": self.relations,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Create a knowledge entry from a dictionary"""
        entry = cls(data["concept"], data["description"], data.get("relations", {}))
        if "metadata" in data:
            entry.metadata = data["metadata"]
        return entry
        
    def __str__(self) -> str:
        return f"KnowledgeEntry({self.concept}: {self.description[:30]}...)"

class KnowledgeSource:
    """Mock Knowledge Source class that provides basic knowledge functionality"""
    
    def __init__(self):
        self.knowledge_base = {}
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize with some mock knowledge"""
        concepts = [
            ("neural_network", "A computational model inspired by the human brain", 
                {"related_to": ["artificial_intelligence", "machine_learning"]}),
            ("machine_learning", "A field of AI focused on learning from data",
                {"subtypes": ["supervised_learning", "unsupervised_learning", "reinforcement_learning"]}),
            ("artificial_intelligence", "The simulation of human intelligence in machines",
                {"applications": ["natural_language_processing", "computer_vision", "robotics"]})
        ]
        
        for concept, description, relations in concepts:
            self.add_knowledge(concept, description, relations)
            
    def query(self, concept: str) -> Optional[KnowledgeEntry]:
        """Query for information about a concept"""
        logger.info(f"Querying knowledge base for: {concept}")
        return self.knowledge_base.get(concept.lower())
        
    def get_related_concepts(self, concept: str, relation_type: Optional[str] = None) -> List[str]:
        """Get concepts related to the given concept"""
        entry = self.query(concept)
        if not entry:
            return []
            
        if relation_type and relation_type in entry.relations:
            return entry.relations[relation_type]
            
        # If no specific relation type is requested, return all related concepts
        related = []
        for rel_type, concepts in entry.relations.items():
            related.extend(concepts)
        return related
        
    def add_knowledge(self, concept: str, description: str, 
                     relations: Optional[Dict[str, List[str]]] = None) -> KnowledgeEntry:
        """Add knowledge to the knowledge base"""
        entry = KnowledgeEntry(concept, description, relations)
        self.knowledge_base[concept.lower()] = entry
        return entry
        
    def save_knowledge_base(self, filepath: str = "knowledge_base.json"):
        """Save the knowledge base to a file"""
        data = {concept: entry.to_dict() for concept, entry in self.knowledge_base.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_knowledge_base(self, filepath: str = "knowledge_base.json"):
        """Load the knowledge base from a file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.knowledge_base = {}
            for concept, entry_data in data.items():
                self.knowledge_base[concept] = KnowledgeEntry.from_dict(entry_data)
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading knowledge base: {e}") 