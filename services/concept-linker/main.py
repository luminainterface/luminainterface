from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram
import asyncio
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
import spacy
from datetime import datetime
import numpy as np
from lumina_core.common.bus import BusClient
from lumina_core.common.retry import with_retry
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-linker")

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Initialize sentence transformer for vector generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Prometheus metrics
LINK_CREATED = Counter("link_created_total", "Number of links created", ["type"])
LINK_SKIP = Counter("link_skip_total", "Number of links skipped", ["reason"])
LINK_FAIL = Counter("link_fail_total", "Number of link creation failures")
LINK_SECONDS = Histogram("link_seconds", "Time spent creating links",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
# New metrics
LINK_QUALITY = Histogram("link_quality", "Link quality score",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
LINK_DISTANCE = Histogram("link_distance", "Vector distance between linked concepts",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ENTITY_TYPES = Counter("entity_types_total", "Distribution of entity types", ["type"])
RELATION_TYPES = Counter("relation_types_total", "Distribution of relation types", ["type"])

class ConceptLinker:
    def __init__(self, redis_url: str, qdrant_url: str):
        self.bus = BusClient(redis_url=redis_url)
        self.qdrant = QdrantClient(url=qdrant_url)
        self.similarity_threshold = 0.85
        self.max_neighbors = 10
        self.processed_concepts: Set[str] = set()
        self.collection_name = "concepts"
        self.vector_size = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize Qdrant collection if it doesn't exist
        self._init_collection()
        
    def _init_collection(self):
        """Initialize Qdrant collection with proper schema"""
        try:
            collections = self.qdrant.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise
            
    async def connect(self):
        """Connect to Redis and create consumer group"""
        await self.bus.connect()
        try:
            await self.bus.create_group("concept.new", "linker")
        except Exception as e:
            logger.info(f"Group may exist: {e}")
            
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using spaCy"""
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            # Generate embedding using sentence transformer
            entity_embedding = embedding_model.encode(ent.text).tolist()
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "vector": entity_embedding  # Use sentence transformer embedding
            })
            ENTITY_TYPES.labels(type=ent.label_).inc()
        return entities
        
    def find_similar_concepts(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar concepts using Qdrant vector search"""
        try:
            # Search for similar vectors
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                score_threshold=self.similarity_threshold
            )
            
            # Process results
            similar_concepts = []
            for scored_point in search_result:
                # Get concept metadata from payload
                payload = scored_point.payload
                similar_concepts.append({
                    "id": payload.get("id"),
                    "text": payload.get("text", ""),
                    "score": scored_point.score,
                    "type": payload.get("type", "unknown")
                })
                # Record metrics
                LINK_DISTANCE.observe(1.0 - scored_point.score)  # Convert similarity to distance
                
            return similar_concepts
            
        except Exception as e:
            logger.error(f"Failed to find similar concepts: {e}")
            return []
            
    def calculate_link_quality(self, src_type: str, dst_type: str, score: float) -> float:
        """Calculate link quality score based on various factors"""
        quality = 0.0
        
        # Base score from vector similarity (0-0.4)
        quality += score * 0.4
        
        # Entity type compatibility (0-0.3)
        compatible_types = {
            "PERSON": {"PERSON", "ORG", "GPE"},
            "ORG": {"PERSON", "ORG", "GPE"},
            "GPE": {"PERSON", "ORG", "GPE", "LOC"},
            "LOC": {"GPE", "LOC"},
            "DATE": {"EVENT", "WORK_OF_ART"},
            "EVENT": {"DATE", "PERSON", "ORG"},
            "WORK_OF_ART": {"PERSON", "ORG", "DATE"}
        }
        if src_type in compatible_types and dst_type in compatible_types[src_type]:
            quality += 0.3
            
        # Relation type strength (0-0.3)
        strong_relations = {"similar", "mentions", "part_of", "located_in"}
        if any(rel in strong_relations for rel in [src_type, dst_type]):
            quality += 0.3
            
        return min(quality, 1.0)
        
    def create_link(self, src_id: str, dst_id: str, rel_type: str, score: float,
                   src_type: str = "unknown", dst_type: str = "unknown") -> Dict[str, Any]:
        """Create a link between concepts with quality metrics"""
        # Calculate link quality
        quality_score = self.calculate_link_quality(src_type, dst_type, score)
        LINK_QUALITY.observe(quality_score)
        RELATION_TYPES.labels(type=rel_type).inc()
        
        return {
            "src": src_id,
            "dst": dst_id,
            "rel": rel_type,
            "score": score,
            "quality_score": quality_score,
            "src_type": src_type,
            "dst_type": dst_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    @with_retry("concept.new", max_attempts=3, dead_letter_stream="concept.dlq")
    async def process_concept(self, msg: Dict[str, Any]):
        """Process a new concept and create links with DLQ support"""
        start_time = datetime.now()
        try:
            concept_id = msg["cid"]
            if concept_id in self.processed_concepts:
                raise Skip("already_processed")
                
            text = msg.get("text", "")
            if not text:
                raise Skip("no_text")
                
            # Extract entities
            entities = self.extract_entities(text)
            if not entities:
                raise Skip("no_entities")
                
            # Find similar concepts
            embedding = msg.get("embedding")
            if not embedding:
                raise Skip("no_embedding")
                
            # Store concept in Qdrant
            try:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=concept_id,
                            vector=embedding,
                            payload={
                                "id": concept_id,
                                "text": text,
                                "type": msg.get("type", "unknown"),
                                "entities": [e["label"] for e in entities],
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    ]
                )
            except Exception as e:
                logger.error(f"Failed to store concept in Qdrant: {e}")
                raise
                
            similar_concepts = self.find_similar_concepts(embedding, self.max_neighbors)
            
            # Create links
            links = []
            
            # Entity-based links
            for entity in entities:
                # Create entity-based links
                if entity["label"] in ["PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "WORK_OF_ART"]:
                    # Find similar entities
                    similar_entities = self.find_similar_concepts(entity["vector"], limit=5)
                    for similar in similar_entities:
                        if similar["score"] >= self.similarity_threshold:
                            links.append(self.create_link(
                                src_id=concept_id,
                                dst_id=similar["id"],
                                rel_type=f"mentions_{entity['label'].lower()}",
                                score=similar["score"],
                                src_type=msg.get("type", "unknown"),
                                dst_type=similar["type"]
                            ))
                            
            # Similarity-based links
            for similar in similar_concepts:
                if similar["score"] >= self.similarity_threshold:
                    links.append(self.create_link(
                        src_id=concept_id,
                        dst_id=similar["id"],
                        rel_type="similar",
                        score=similar["score"],
                        src_type=msg.get("type", "unknown"),
                        dst_type=similar["type"]
                    ))
                    
            # Publish links
            for link in links:
                await self.bus.publish("concept.link", link)
                LINK_CREATED.labels(type=link["rel"]).inc()
                
            # Mark as processed
            self.processed_concepts.add(concept_id)
            
            # Record metrics
            LINK_SECONDS.observe((datetime.now() - start_time).total_seconds())
            
        except Skip as s:
            LINK_SKIP.labels(reason=str(s)).inc()
            raise
        except Exception as e:
            logger.error(f"Error processing concept: {e}")
            LINK_FAIL.inc()
            raise
            
    async def start(self):
        """Start consuming from concept.new stream"""
        while True:
            try:
                await self.bus.consume(
                    stream="concept.new",
                    group="linker",
                    consumer="worker",
                    handler=self.process_concept,
                    block_ms=1000,
                    count=10
                )
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

# FastAPI app
app = FastAPI(title="Concept Linker Service")

@app.on_event("startup")
async def startup():
    """Initialize linker on startup"""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    linker = ConceptLinker(redis_url, qdrant_url)
    await linker.connect()
    app.state.linker = linker
    # Start consumer loop
    asyncio.create_task(linker.start())

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 