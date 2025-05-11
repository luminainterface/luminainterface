"""
Learning Graph Service for Lumina
Tracks concept evolution, learning patterns, and semantic growth over time.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
import networkx as nx
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ops')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
CONCEPT_MATURITY = Gauge('concept_maturity_score', 'Maturity level of concepts', ['concept'])
LEARNING_RATE = Gauge('concept_learning_rate', 'Rate of concept understanding improvement', ['concept'])
CONCEPT_CONNECTIONS = Gauge('concept_connection_count', 'Number of strong relationships', ['concept'])
KNOWLEDGE_GAPS = Gauge('knowledge_gap_score', 'Identified knowledge gaps', ['domain'])
LEARNING_EVENTS = Counter('learning_events_total', 'Total number of learning events', ['type'])

class ConceptState(BaseModel):
    """Current state of a concept in the learning graph"""
    concept_id: str
    maturity: float  # 0-1 score of concept understanding
    last_updated: datetime
    embedding_quality: float
    usage_count: int
    relationship_strengths: Dict[str, float]
    learning_events: List[Dict]
    knowledge_gaps: List[str]

class LearningEvent(BaseModel):
    """Record of a concept learning or evolution event"""
    timestamp: datetime
    concept_id: str
    event_type: str  # 'usage', 'drift', 'reinforcement', 'connection'
    old_state: Dict
    new_state: Dict
    source: str  # 'crawler', 'user_feedback', 'inference'
    confidence: float

class LearningGraph:
    def __init__(self, redis_client: redis.Redis):
        """Initialize the learning graph"""
        self.redis = redis_client
        self.graph = nx.DiGraph()
        self.maturity_threshold = 0.7
        self.connection_threshold = 0.5
        
    async def get_concept_state(self, concept_id: str) -> ConceptState:
        """Get current state of a concept"""
        # Get base metrics
        embedding_quality = float(await self.redis.get(f"embedding_quality:{concept_id}") or 1.0)
        usage_count = int(await self.redis.get(f"usage:{concept_id}") or 0)
        
        # Get relationships
        relationships = await self.redis.smembers(f"relationships:{concept_id}")
        strengths = {}
        for rel in relationships:
            cooc_key = f"cooccurrence:{concept_id}:{rel}"
            cooc_count = int(await self.redis.get(cooc_key) or 0)
            rel_usage = int(await self.redis.get(f"usage:{rel}") or 1)
            strength = cooc_count / (usage_count * rel_usage) ** 0.5 if usage_count else 0
            strengths[rel] = strength
            
        # Calculate maturity score
        maturity = self._calculate_maturity(embedding_quality, usage_count, strengths)
        CONCEPT_MATURITY.labels(concept=concept_id).set(maturity)
        
        # Get recent learning events
        events = await self._get_recent_events(concept_id)
        
        # Identify knowledge gaps
        gaps = await self._identify_gaps(concept_id, strengths, maturity)
        
        return ConceptState(
            concept_id=concept_id,
            maturity=maturity,
            last_updated=datetime.now(),
            embedding_quality=embedding_quality,
            usage_count=usage_count,
            relationship_strengths=strengths,
            learning_events=events,
            knowledge_gaps=gaps
        )
        
    def _calculate_maturity(self, quality: float, usage: int, relationships: Dict[str, float]) -> float:
        """Calculate concept maturity score"""
        # Quality component (30%)
        quality_score = quality * 0.3
        
        # Usage component (20%)
        usage_score = min(1.0, usage / 1000) * 0.2
        
        # Relationship component (50%)
        strong_rels = len([s for s in relationships.values() if s > self.connection_threshold])
        rel_score = min(1.0, strong_rels / 5) * 0.5
        
        return quality_score + usage_score + rel_score
        
    async def _get_recent_events(self, concept_id: str, days: int = 7) -> List[Dict]:
        """Get recent learning events for a concept"""
        events_key = f"learning_events:{concept_id}"
        event_ids = await self.redis.zrange(
            events_key,
            datetime.now() - timedelta(days=days),
            datetime.now(),
            byscore=True
        )
        
        events = []
        for event_id in event_ids:
            event_data = await self.redis.get(f"event:{event_id}")
            if event_data:
                events.append(eval(event_data))
        return events
        
    async def _identify_gaps(
        self, concept_id: str, relationships: Dict[str, float], maturity: float
    ) -> List[str]:
        """Identify knowledge gaps for a concept"""
        gaps = []
        
        # Check for weak relationships
        expected_rels = await self.redis.smembers(f"expected_relationships:{concept_id}")
        for rel in expected_rels:
            if rel not in relationships or relationships[rel] < self.connection_threshold:
                gaps.append(f"weak_relationship:{rel}")
                
        # Check for low usage if mature
        if maturity > self.maturity_threshold:
            usage = int(await self.redis.get(f"usage:{concept_id}") or 0)
            if usage < 100:
                gaps.append("low_usage")
                
        # Check for concept staleness
        last_update = await self.redis.get(f"last_update:{concept_id}")
        if last_update:
            days_old = (datetime.now() - datetime.fromisoformat(last_update)).days
            if days_old > 30:
                gaps.append("stale_concept")
                
        return gaps
        
    async def record_learning_event(self, event: LearningEvent):
        """Record a learning event in the graph"""
        # Store event
        event_id = f"{event.concept_id}:{event.timestamp.isoformat()}"
        await self.redis.set(f"event:{event_id}", str(event.dict()))
        
        # Add to concept timeline
        await self.redis.zadd(
            f"learning_events:{event.concept_id}",
            {event_id: event.timestamp.timestamp()}
        )
        
        # Update metrics
        LEARNING_EVENTS.labels(type=event.event_type).inc()
        
        # Calculate learning rate
        old_quality = event.old_state.get('embedding_quality', 0)
        new_quality = event.new_state.get('embedding_quality', 0)
        if old_quality and new_quality:
            learning_rate = (new_quality - old_quality) / old_quality if old_quality > 0 else 0
            LEARNING_RATE.labels(concept=event.concept_id).set(learning_rate)
            
    async def get_learning_path(self, concept_id: str) -> List[Dict]:
        """Get the learning evolution path of a concept"""
        events = await self._get_recent_events(concept_id, days=30)
        
        path = []
        current_state = None
        
        for event in sorted(events, key=lambda e: e['timestamp']):
            if not current_state:
                current_state = event['old_state']
            
            path.append({
                'timestamp': event['timestamp'],
                'event_type': event['event_type'],
                'state_change': {
                    'from': current_state,
                    'to': event['new_state']
                },
                'source': event['source'],
                'confidence': event['confidence']
            })
            
            current_state = event['new_state']
            
        return path
        
    async def suggest_learning_actions(self, concept_id: str) -> List[Dict]:
        """Suggest actions to improve concept understanding"""
        state = await self.get_concept_state(concept_id)
        suggestions = []
        
        # Check maturity
        if state.maturity < self.maturity_threshold:
            if state.embedding_quality < 0.8:
                suggestions.append({
                    'action': 'retrain_embeddings',
                    'reason': 'Low embedding quality',
                    'priority': 'high'
                })
                
            weak_rels = [r for r, s in state.relationship_strengths.items() if s < self.connection_threshold]
            if weak_rels:
                suggestions.append({
                    'action': 'strengthen_relationships',
                    'targets': weak_rels,
                    'reason': 'Weak concept relationships',
                    'priority': 'medium'
                })
                
        # Check for knowledge gaps
        if state.knowledge_gaps:
            for gap in state.knowledge_gaps:
                if gap.startswith('weak_relationship:'):
                    rel = gap.split(':')[1]
                    suggestions.append({
                        'action': 'crawl_relationship',
                        'target': rel,
                        'reason': f'Missing relationship data for {rel}',
                        'priority': 'medium'
                    })
                elif gap == 'stale_concept':
                    suggestions.append({
                        'action': 'refresh_concept',
                        'reason': 'Concept data is stale',
                        'priority': 'low'
                    })
                    
        return suggestions

# Initialize FastAPI app
app = FastAPI(title="Lumina Learning Graph")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Initialize learning graph
graph = LearningGraph(redis_client)

@app.get("/concept/{concept_id}/state")
async def get_concept_state(concept_id: str):
    """Get current state of a concept"""
    return await graph.get_concept_state(concept_id)

@app.get("/concept/{concept_id}/path")
async def get_learning_path(concept_id: str):
    """Get learning evolution path of a concept"""
    return await graph.get_learning_path(concept_id)

@app.get("/concept/{concept_id}/suggestions")
async def get_learning_suggestions(concept_id: str):
    """Get suggestions for concept improvement"""
    return await graph.suggest_learning_actions(concept_id)

@app.post("/event")
async def record_event(event: LearningEvent):
    """Record a learning event"""
    await graph.record_learning_event(event)
    return {"status": "recorded"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600) 