import asyncio
import logging
from typing import Dict, List, Optional, Set
import spacy
from collections import defaultdict
from pydantic import BaseModel
import aiohttp
from datetime import datetime
import re
import requests
import os

from shared.log_config import setup_logging

logger = setup_logging('concept-extractor')

CONCEPT_DICT_API_KEY = os.getenv("CONCEPT_DICT_API_KEY", "changeme")
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8000")

class Concept(BaseModel):
    """Represents an extracted concept with its metadata and relationships."""
    label: str
    type: str  # entity, technical_term, abstract_concept
    confidence: float
    source_chunks: List[str]
    related_concepts: List[str]
    metadata: Dict[str, str]
    timestamp: datetime

class ConceptExtractor:
    """Extracts and manages concepts from training content."""
    
    def __init__(self, config: Dict):
        self.config = config
        # Use environment variable or fallback to config, ensuring correct port
        self.graph_api_url = os.getenv("GRAPH_API_URL", "http://graph-api:8200")
        if 'graph_api_url' in config:
            # Extract host and use correct port
            host = config['graph_api_url'].split(':')[1]
            self.graph_api_url = f"http:{host}:8200"
        
        self.nlp = spacy.load("en_core_web_lg")
        self.concept_patterns = config['concept_patterns']
        self.min_confidence = config['min_confidence']
        self.concept_cache = {}  # Cache for processed concepts
        
    async def extract_concepts(self, text: str, metadata: Dict) -> List[Concept]:
        """Extract concepts from a text chunk."""
        try:
            doc = self.nlp(text)
            concepts = []
            
            # Extract named entities
            for ent in doc.ents:
                if self._is_relevant_entity(ent):
                    concept = await self._create_concept(
                        label=ent.text,
                        type="entity",
                        confidence=ent._.confidence if hasattr(ent._, 'confidence') else 0.8,
                        source_chunks=[metadata.get('chunk_id', '')],
                        metadata={
                            "entity_type": ent.label_,
                            "start_char": str(ent.start_char),
                            "end_char": str(ent.end_char)
                        }
                    )
                    concepts.append(concept)
            
            # Extract technical terms using patterns
            for pattern in self.concept_patterns['technical_terms']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    term = match.group(0)
                    if self._is_technical_term(term):
                        concept = await self._create_concept(
                            label=term,
                            type="technical_term",
                            confidence=0.7,
                            source_chunks=[metadata.get('chunk_id', '')],
                            metadata={
                                "pattern": pattern,
                                "start_char": str(match.start()),
                                "end_char": str(match.end())
                            }
                        )
                        concepts.append(concept)
            
            # Extract abstract concepts using semantic similarity
            for abstract in self.concept_patterns['abstract_concepts']:
                doc_abstract = self.nlp(abstract)
                if doc.similarity(doc_abstract) > self.min_confidence:
                    concept = await self._create_concept(
                        label=abstract,
                        type="abstract_concept",
                        confidence=doc.similarity(doc_abstract),
                        source_chunks=[metadata.get('chunk_id', '')],
                        metadata={
                            "similarity_score": str(doc.similarity(doc_abstract))
                        }
                    )
                    concepts.append(concept)
            
            # Find relationships between concepts
            await self._link_related_concepts(concepts)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {str(e)}", exc_info=True)
            return []
    
    def _is_relevant_entity(self, ent) -> bool:
        """Check if an entity is relevant for concept extraction."""
        relevant_types = {'ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'}
        return ent.label_ in relevant_types
    
    def _is_technical_term(self, term: str) -> bool:
        """Check if a term is likely to be technical."""
        # Check against common technical term patterns
        patterns = [
            r'[A-Z][a-z]+[A-Z][a-z]+',  # CamelCase
            r'[a-z]+_[a-z]+',  # snake_case
            r'[A-Z]{2,}',  # Acronyms
        ]
        return any(re.match(pattern, term) for pattern in patterns)
    
    async def _create_concept(self, label: str, type: str, confidence: float,
                            source_chunks: List[str], metadata: Dict) -> Concept:
        """Create a new concept and store it in the graph."""
        try:
            concept = Concept(
                label=label,
                type=type,
                confidence=confidence,
                source_chunks=source_chunks,
                related_concepts=[],
                metadata=metadata,
                timestamp=datetime.utcnow()
            )
            
            # Post to concept-dictionary
            try:
                concept_dict = {
                    "term": concept.label,
                    "definition": concept.metadata.get("definition"),
                    "embedding": concept.metadata.get("embedding", []),
                    "sources": concept.source_chunks,
                    "usage_count": 1,
                    "mistral_explanation": None,
                    "nn_response": None,
                    "drift_score": 0.0
                }
                requests.post(f"{CONCEPT_DICT_URL}/concepts", json=concept_dict, timeout=2, headers={"X-API-Key": CONCEPT_DICT_API_KEY})
            except Exception as e:
                logger.warning(f"Could not post concept to dictionary: {e}")
            
            # Store in graph
            await self._store_concept(concept)
            
            return concept
            
        except Exception as e:
            logger.error(f"Error creating concept: {str(e)}", exc_info=True)
            return None
    
    async def _store_concept(self, concept: Concept):
        """Store a concept in the knowledge graph."""
        try:
            async with aiohttp.ClientSession() as session:
                # Generate concept node ID from label
                concept_id = f"concept_{concept.label.lower().replace(' ', '_')}"
                node_payload = {
                    "id": concept_id,
                    "type": "concept",
                    "properties": {
                        "label": concept.label,
                        "concept_type": concept.type,
                        "confidence": concept.confidence,
                        "source_chunks": concept.source_chunks,
                        "metadata": concept.metadata,
                        "created_at": concept.timestamp.isoformat()
                    },
                    "metadata": {
                        "source": "concept_extractor",
                        "extracted_at": concept.timestamp.isoformat()
                    }
                }
                # Check if concept node exists
                async with session.get(f"{self.graph_api_url}/nodes/{concept_id}") as check_response:
                    if check_response.status == 200:
                        logger.info(f"Concept node {concept_id} already exists")
                    else:
                        async with session.post(f"{self.graph_api_url}/nodes", json=node_payload) as response:
                            logger.info(f"POST /nodes {response.status} {await response.text()}")
                    if response.status != 200:
                        logger.error(f"Error storing concept: {await response.text()}")
                        return
                # For each chunk, ensure chunk node exists and create relationship
                    for chunk_id in concept.source_chunks:
                    chunk_payload = {
                        "id": chunk_id,
                        "type": "chunk",
                        "properties": {
                            "source": "concept_extractor",
                            "created_at": datetime.utcnow().isoformat()
                        },
                        "metadata": {
                            "source": "concept_extractor",
                            "extracted_at": datetime.utcnow().isoformat()
                        }
                    }
                    # Check if chunk node exists
                    async with session.get(f"{self.graph_api_url}/nodes/{chunk_id}") as check_response:
                        if check_response.status != 200:
                            async with session.post(f"{self.graph_api_url}/nodes", json=chunk_payload) as chunk_response:
                                logger.info(f"POST /nodes (chunk) {chunk_response.status} {await chunk_response.text()}")
                                if chunk_response.status != 200:
                                    logger.error(f"Error creating chunk node: {await chunk_response.text()}")
                                    continue
                    # Create relationship if not exists
                    rel_id = f"{chunk_id}_{concept_id}"
                        rel_payload = {
                        "id": rel_id,
                        "source": chunk_id,
                        "target": concept_id,
                            "type": "contains",
                            "properties": {
                            "confidence": concept.confidence,
                            "created_at": datetime.utcnow().isoformat()
                            }
                        }
                    async with session.get(f"{self.graph_api_url}/edges/{rel_id}") as rel_check:
                        if rel_check.status == 200:
                            logger.info(f"Relationship {rel_id} already exists")
                            continue
                    async with session.post(f"{self.graph_api_url}/edges", json=rel_payload) as rel_response:
                        logger.info(f"POST /edges {rel_response.status} {await rel_response.text()}")
                            if rel_response.status != 200:
                                logger.error(f"Error creating relationship: {await rel_response.text()}")
                        else:
                            logger.info(f"Created relationship: {rel_id}")
            logger.info(f"Stored concept: {concept.label}")
        except Exception as e:
            logger.error(f"Error storing concept: {str(e)}", exc_info=True)
    
    async def _link_related_concepts(self, concepts: List[Concept]):
        """Find and create relationships between related concepts."""
        try:
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    # Calculate semantic similarity
                    doc1 = self.nlp(concept1.label)
                    doc2 = self.nlp(concept2.label)
                    similarity = doc1.similarity(doc2)
                    
                    if similarity > self.min_confidence:
                        concept1.related_concepts.append(concept2.label)
                        concept2.related_concepts.append(concept1.label)
                        
                        # Store relationship in graph
                        await self._store_concept_relationship(concept1, concept2, similarity)
        
        except Exception as e:
            logger.error(f"Error linking concepts: {str(e)}", exc_info=True)
    
    async def _store_concept_relationship(self, concept1: Concept, concept2: Concept, similarity: float):
        """Store a relationship between two concepts in the graph."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get node IDs for both concepts
                concept1_id = f"concept_{concept1.label.lower().replace(' ', '_')}"
                concept2_id = f"concept_{concept2.label.lower().replace(' ', '_')}"
                
                rel_payload = {
                    "id": f"{concept1_id}_{concept2_id}",
                    "source": concept1_id,
                    "target": concept2_id,
                    "type": "relates_to",
                    "properties": {
                        "similarity": similarity,
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
                
                async with session.post(
                    f"{self.graph_api_url}/edges",
                    json=rel_payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error storing concept relationship: {await response.text()}")
                    else:
                        logger.info(f"Created relationship between {concept1.label} and {concept2.label}")
        
        except Exception as e:
            logger.error(f"Error storing concept relationship: {str(e)}", exc_info=True)

# Example usage:
# config = {
#     'graph_api_url': 'http://graph-api:8000',
#     'min_confidence': 0.7,
#     'concept_patterns': {
#         'technical_terms': [
#             r'\b[A-Z][a-z]+[A-Z][a-z]+\b',  # CamelCase
#             r'\b[a-z]+_[a-z]+\b',  # snake_case
#             r'\b[A-Z]{2,}\b'  # Acronyms
#         ],
#         'abstract_concepts': [
#             'Vector Search',
#             'Knowledge Graph',
#             'Semantic Understanding',
#             'Self Learning'
#         ]
#     }
# }
# extractor = ConceptExtractor(config) 