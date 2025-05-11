import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ops')
import asyncio
import logging
from typing import Dict, List, Optional
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server, make_asgi_app
from prometheus_api_client import PrometheusConnect
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for semantic health
CONCEPT_DRIFT = Gauge('concept_drift_score', 'Semantic drift score per concept', ['concept'])
RELATIONSHIP_STRENGTH = Gauge('concept_relationship_strength', 'Relationship strength between concepts', ['source', 'target'])
CONCEPT_IMPORTANCE = Gauge('concept_importance_score', 'Importance score per concept', ['concept'])
CONCEPT_USAGE = Counter('concept_usage_total', 'Usage count per concept', ['concept'])
SEMANTIC_FIDELITY = Histogram('semantic_fidelity_score', 'Overall semantic memory fidelity',
                            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Initialize FastAPI app
app = FastAPI(title="Lumina Concept Analyzer")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Prometheus connection
prom = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)

class ConceptMetrics(BaseModel):
    """Model for concept health metrics"""
    concept_id: str
    importance_score: float
    usage_count: int
    last_accessed: float
    embedding_quality: float
    relationship_count: int

class ConceptAnalyzer:
    def __init__(self):
        self.drift_threshold = 0.3
        self.importance_decay_rate = 0.1
        self.relationship_threshold = 0.5
        
    async def calculate_concept_drift(self, concept: str, window: str = "7d") -> float:
        """Calculate semantic drift for a concept over time"""
        try:
            # Query historical embedding quality
            query = f'embedding_quality{{concept="{concept}"}}[{window}]'
            result = prom.custom_query(query)
            
            if not result:
                return 0.0
                
            values = [float(point[1]) for point in result[0]['values']]
            
            # Calculate drift as variance from baseline
            baseline = values[0] if values else 1.0
            current = values[-1] if values else baseline
            drift_score = abs(current - baseline) / baseline
            
            CONCEPT_DRIFT.labels(concept=concept).set(drift_score)
            return drift_score
            
        except Exception as e:
            logger.error(f"Error calculating drift for concept {concept}: {e}")
            return 0.0

    async def analyze_relationship_strength(self, source: str, target: str) -> float:
        """Analyze strength of relationship between concepts"""
        try:
            # Query co-occurrence and individual usage
            cooccurrence = await redis_client.get(f"cooccurrence:{source}:{target}")
            source_usage = await redis_client.get(f"usage:{source}")
            target_usage = await redis_client.get(f"usage:{target}")
            
            if not all([cooccurrence, source_usage, target_usage]):
                return 0.0
                
            # Calculate normalized relationship strength
            strength = float(cooccurrence) / (float(source_usage) * float(target_usage)) ** 0.5
            
            RELATIONSHIP_STRENGTH.labels(source=source, target=target).set(strength)
            return strength
            
        except Exception as e:
            logger.error(f"Error analyzing relationship {source}->{target}: {e}")
            return 0.0

    async def update_concept_importance(self, concept: str):
        """Update concept importance based on usage and relationships"""
        try:
            # Get concept metrics
            usage = await redis_client.get(f"usage:{concept}")
            relationships = await redis_client.smembers(f"relationships:{concept}")
            embedding_quality = await redis_client.get(f"embedding_quality:{concept}")
            
            if not usage or not embedding_quality:
                return
                
            # Calculate importance score
            base_importance = float(usage) * float(embedding_quality)
            relationship_factor = len(relationships) / 100  # Normalize relationship count
            
            importance_score = base_importance * (1 + relationship_factor)
            
            CONCEPT_IMPORTANCE.labels(concept=concept).set(importance_score)
            
        except Exception as e:
            logger.error(f"Error updating importance for concept {concept}: {e}")

    async def calculate_semantic_fidelity(self) -> float:
        """Calculate overall semantic memory fidelity"""
        try:
            # Get all concepts
            concepts = await redis_client.smembers("concepts")
            if not concepts:
                return 1.0
                
            total_drift = 0.0
            for concept in concepts:
                drift = await self.calculate_concept_drift(concept)
                total_drift += drift
                
            # Calculate fidelity score (inverse of average drift)
            fidelity = 1.0 - (total_drift / len(concepts))
            SEMANTIC_FIDELITY.observe(fidelity)
            
            return fidelity
            
        except Exception as e:
            logger.error(f"Error calculating semantic fidelity: {e}")
            return 1.0

    async def monitor_concepts(self):
        """Main monitoring loop"""
        while True:
            try:
                # Get all concepts
                concepts = await redis_client.smembers("concepts")
                
                for concept in concepts:
                    # Update metrics
                    await self.calculate_concept_drift(concept)
                    await self.update_concept_importance(concept)
                    
                    # Check relationships
                    relationships = await redis_client.smembers(f"relationships:{concept}")
                    for related in relationships:
                        await self.analyze_relationship_strength(concept, related)
                        
                # Calculate overall semantic fidelity
                await self.calculate_semantic_fidelity()
                
                # Alert on significant drift
                for concept in concepts:
                    drift = float(await redis_client.get(f"drift:{concept}") or 0)
                    importance = float(await redis_client.get(f"importance:{concept}") or 0)
                    
                    if drift > self.drift_threshold and importance > 0.7:
                        alert = {
                            'type': 'semantic_drift',
                            'concept': concept,
                            'drift': drift,
                            'importance': importance
                        }
                        await redis_client.publish('concept_alerts', str(alert))
                        
            except Exception as e:
                logger.error(f"Error in concept monitoring loop: {e}")
                
            await asyncio.sleep(300)  # Check every 5 minutes

# Initialize analyzer
analyzer = ConceptAnalyzer()

@app.on_event("startup")
async def startup_event():
    """Start concept analysis on startup"""
    asyncio.create_task(analyzer.monitor_concepts())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500) 