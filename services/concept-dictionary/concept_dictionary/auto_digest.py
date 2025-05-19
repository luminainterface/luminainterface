import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from .db import ConceptDB, ConceptMetadata, SIMILARITY_THRESHOLD, MIN_USAGE_COUNT
from .models import Concept

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-dictionary-auto-digest")

class AutoDigest:
    """Handles automatic concept digestion and maintenance."""
    
    def __init__(self, db: ConceptDB):
        self.db = db
        self.digest_interval = 300  # 5 minutes
        self.quality_check_interval = 3600  # 1 hour
        self.merge_interval = 1800  # 30 minutes
        self.min_concept_length = 10
        self.max_concept_length = 1000
        self.min_quality_score = 0.5
        self._running = False
        self._last_digest = None
        self._last_quality_check = None
        self._last_merge = None

    async def start(self):
        """Start the auto-digestion process."""
        self._running = True
        while self._running:
            try:
                current_time = time.time()
                
                # Run periodic digest
                if not self._last_digest or (current_time - self._last_digest) >= self.digest_interval:
                    await self._run_digest()
                    self._last_digest = current_time
                
                # Run quality checks
                if not self._last_quality_check or (current_time - self._last_quality_check) >= self.quality_check_interval:
                    await self._run_quality_checks()
                    self._last_quality_check = current_time
                
                # Run auto-merging
                if not self._last_merge or (current_time - self._last_merge) >= self.merge_interval:
                    await self._run_auto_merge()
                    self._last_merge = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-digest loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def stop(self):
        """Stop the auto-digestion process."""
        self._running = False

    async def _run_digest(self):
        """Run the main digestion process."""
        try:
            logger.info("Starting concept digestion...")
            stats = {
                "processed": 0,
                "improved": 0,
                "errors": 0
            }
            
            # Get all concepts
            concepts = []
            for key in self.db.redis.scan_iter("concept:*"):
                if hasattr(self.db, 'is_concept_key'):
                    if not self.db.is_concept_key(key):
                        continue
                else:
                    if not self.db.is_concept_key(key):
                        continue
                data = self.db.redis.get(key)
                if data:
                    concept = ConceptMetadata.from_dict(json.loads(data))
                    concepts.append(concept)
            
            for concept in concepts:
                try:
                    stats["processed"] += 1
                    
                    # Improve concept quality
                    improved = await self._improve_concept(concept)
                    if improved:
                        stats["improved"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing concept {concept.term}: {e}")
                    stats["errors"] += 1
                    continue
            
            logger.info(f"Digestion complete. Stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in digestion process: {e}")

    async def _improve_concept(self, concept: ConceptMetadata) -> bool:
        """Improve a concept's quality."""
        improved = False
        
        # Check and improve definition
        if concept.definition:
            # Truncate if too long
            if len(concept.definition) > self.max_concept_length:
                concept.definition = concept.definition[:self.max_concept_length] + "..."
                improved = True
            
            # Remove excessive whitespace
            cleaned_def = " ".join(concept.definition.split())
            if cleaned_def != concept.definition:
                concept.definition = cleaned_def
                improved = True
        
        # Update metadata
        if not concept.metadata.get("last_improved"):
            concept.metadata["last_improved"] = datetime.utcnow().isoformat()
            improved = True
        
        if improved:
            # Save improvements
            self.db.redis.set(
                self.db._get_redis_key(concept.term),
                json.dumps(concept.to_dict())
            )
            
            # Update Qdrant if embedding exists
            if concept.embedding:
                self.db.qdrant.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": concept.term,
                        "vector": concept.embedding,
                        "payload": concept.to_dict()
                    }]
                )
        
        return improved

    async def _run_quality_checks(self):
        """Run quality checks on concepts."""
        try:
            logger.info("Starting concept quality checks...")
            stats = {
                "checked": 0,
                "flagged": 0,
                "errors": 0
            }
            
            concepts = []
            for key in self.db.redis.scan_iter("concept:*"):
                if hasattr(self.db, 'is_concept_key'):
                    if not self.db.is_concept_key(key):
                        continue
                else:
                    if not self.db.is_concept_key(key):
                        continue
                data = self.db.redis.get(key)
                if data:
                    concept = ConceptMetadata.from_dict(json.loads(data))
                    concepts.append(concept)
            
            for concept in concepts:
                try:
                    stats["checked"] += 1
                    
                    # Check concept quality
                    quality_score = await self._check_concept_quality(concept)
                    if quality_score < self.min_quality_score:
                        stats["flagged"] += 1
                        concept.metadata["quality_score"] = quality_score
                        concept.metadata["quality_checked"] = datetime.utcnow().isoformat()
                        
                        # Save quality check results
                        self.db.redis.set(
                            self.db._get_redis_key(concept.term),
                            json.dumps(concept.to_dict())
                        )
                        
                except Exception as e:
                    logger.error(f"Error checking concept {concept.term}: {e}")
                    stats["errors"] += 1
                    continue
            
            logger.info(f"Quality checks complete. Stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in quality check process: {e}")

    async def _check_concept_quality(self, concept: ConceptMetadata) -> float:
        """Check the quality of a concept."""
        score = 1.0
        
        # Check definition length
        if not concept.definition:
            score *= 0.5
        elif len(concept.definition) < self.min_concept_length:
            score *= 0.7
        elif len(concept.definition) > self.max_concept_length:
            score *= 0.8
        
        # Check term quality
        if not concept.term or len(concept.term) < 3:
            score *= 0.6
        
        # Check metadata completeness
        if not concept.metadata:
            score *= 0.9
        else:
            required_fields = ["source", "timestamp"]
            for field in required_fields:
                if field not in concept.metadata:
                    score *= 0.95
        
        # Check embedding presence
        if not concept.embedding:
            score *= 0.8
        
        return score

    async def _run_auto_merge(self):
        """Run automatic merging of similar concepts."""
        try:
            logger.info("Starting automatic concept merging...")
            stats = {
                "checked": 0,
                "merged": 0,
                "skipped": 0,
                "errors": 0
            }
            
            # Get concepts sorted by usage count
            concepts = []
            for key in self.db.redis.scan_iter("concept:*"):
                if hasattr(self.db, 'is_concept_key'):
                    if not self.db.is_concept_key(key):
                        continue
                else:
                    if not self.db.is_concept_key(key):
                        continue
                data = self.db.redis.get(key)
                if data:
                    concept = ConceptMetadata.from_dict(json.loads(data))
                    if concept.usage_count >= MIN_USAGE_COUNT:
                        concepts.append(concept)
            
            concepts.sort(key=lambda x: x.usage_count, reverse=True)
            
            for concept in concepts:
                try:
                    stats["checked"] += 1
                    
                    if not concept.embedding:
                        stats["skipped"] += 1
                        continue
                    
                    # Find similar concepts
                    similar = await self.db.find_similar_concepts(
                        concept.embedding,
                        SIMILARITY_THRESHOLD,
                        limit=5
                    )
                    
                    for similar_term, score in similar:
                        if similar_term == concept.term:
                            continue
                        
                        similar_concept = self.db.find(similar_term)
                        if not similar_concept:
                            continue
                        
                        # Merge if similar concept has lower usage
                        if similar_concept.usage_count < concept.usage_count:
                            success = await self.db.merge_concepts(
                                similar_term,
                                concept.term,
                                merge_metadata=True
                            )
                            if success:
                                stats["merged"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing concept {concept.term}: {e}")
                    stats["errors"] += 1
                    continue
            
            logger.info(f"Auto-merge complete. Stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in auto-merge process: {e}") 