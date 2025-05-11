"""
Test scenarios for semantic validation.
Each scenario represents a specific aspect of semantic health to test.
"""

import asyncio
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from abc import ABC, abstractmethod

class BaseScenario(ABC):
    """Base class for test scenarios"""
    def __init__(self, redis_client: redis.Redis, concepts: Dict):
        self.redis_client = redis_client
        self.concepts = concepts
        self.results = []
        
    @abstractmethod
    async def run(self, duration: int = 300) -> List[Dict]:
        """Run the scenario for specified duration in seconds"""
        pass
        
    def get_results(self) -> List[Dict]:
        """Get scenario results"""
        return self.results

class DriftScenario(BaseScenario):
    """Tests concept drift patterns"""
    async def run(self, duration: int = 300):
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        
        while datetime.now() < end_time:
            # Select concepts to drift
            drifting_concepts = random.sample(list(self.concepts.keys()), 2)
            
            for concept in drifting_concepts:
                # Get current quality
                current_quality = float(await self.redis_client.get(f"embedding_quality:{concept}") or 1.0)
                
                # Apply random drift
                drift = random.uniform(-0.1, 0.1)
                new_quality = max(0.1, min(1.0, current_quality + drift))
                
                # Update quality
                await self.redis_client.set(f"embedding_quality:{concept}", str(new_quality))
                
                # Record result
                self.results.append({
                    'timestamp': datetime.now().isoformat(),
                    'concept': concept,
                    'old_quality': current_quality,
                    'new_quality': new_quality,
                    'drift_amount': drift
                })
            
            await asyncio.sleep(10)
            
        return self.results

class UsageScenario(BaseScenario):
    """Tests concept usage patterns"""
    async def run(self, duration: int = 300):
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        
        while datetime.now() < end_time:
            # Select concepts to use
            used_concepts = random.sample(list(self.concepts.keys()), 3)
            
            for concept in used_concepts:
                # Get current usage
                current_usage = int(await self.redis_client.get(f"usage:{concept}") or 0)
                usage_increment = random.randint(1, 5)
                
                # Update usage
                await self.redis_client.set(f"usage:{concept}", str(current_usage + usage_increment))
                
                # Update co-occurrences
                related = self.concepts[concept]["related"]
                for rel in random.sample(related, min(2, len(related))):
                    cooc_key = f"cooccurrence:{concept}:{rel}"
                    current_cooc = int(await self.redis_client.get(cooc_key) or 0)
                    cooc_increment = random.randint(1, 3)
                    await self.redis_client.set(cooc_key, str(current_cooc + cooc_increment))
                
                # Record result
                self.results.append({
                    'timestamp': datetime.now().isoformat(),
                    'concept': concept,
                    'old_usage': current_usage,
                    'new_usage': current_usage + usage_increment,
                    'related_updates': [rel for rel in related]
                })
            
            await asyncio.sleep(5)
            
        return self.results

class RelationshipScenario(BaseScenario):
    """Tests concept relationship strength changes"""
    async def run(self, duration: int = 300):
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        
        while datetime.now() < end_time:
            # Select a concept and its relationship to test
            concept = random.choice(list(self.concepts.keys()))
            related = random.choice(self.concepts[concept]["related"])
            
            # Get current relationship strength
            cooc_key = f"cooccurrence:{concept}:{related}"
            current_cooc = int(await self.redis_client.get(cooc_key) or 0)
            
            # Calculate normalized strength
            source_usage = int(await self.redis_client.get(f"usage:{concept}") or 1)
            target_usage = int(await self.redis_client.get(f"usage:{related}") or 1)
            strength = current_cooc / ((source_usage * target_usage) ** 0.5)
            
            # Record result
            self.results.append({
                'timestamp': datetime.now().isoformat(),
                'source_concept': concept,
                'target_concept': related,
                'relationship_strength': strength,
                'cooccurrence_count': current_cooc
            })
            
            await asyncio.sleep(15)
            
        return self.results 