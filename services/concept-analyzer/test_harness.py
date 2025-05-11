import asyncio
import random
import time
import redis.asyncio as redis
from datetime import datetime, timedelta
import json

# Test concepts and their relationships
TEST_CONCEPTS = {
    "machine_learning": {
        "related": ["neural_networks", "deep_learning", "training_data"],
        "importance": 0.9
    },
    "neural_networks": {
        "related": ["machine_learning", "backpropagation", "activation_functions"],
        "importance": 0.8
    },
    "deep_learning": {
        "related": ["machine_learning", "neural_networks", "gpu_computing"],
        "importance": 0.85
    },
    "training_data": {
        "related": ["machine_learning", "data_preprocessing", "validation"],
        "importance": 0.75
    },
    "backpropagation": {
        "related": ["neural_networks", "gradient_descent", "optimization"],
        "importance": 0.7
    }
}

class ConceptTestHarness:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
    async def setup_test_data(self):
        """Initialize test concepts in Redis"""
        print("Setting up test concepts...")
        
        # Clear existing data
        await self.redis_client.flushdb()
        
        # Add concepts to set
        await self.redis_client.sadd("concepts", *TEST_CONCEPTS.keys())
        
        # Initialize concept metrics
        for concept, data in TEST_CONCEPTS.items():
            # Set initial embedding quality
            await self.redis_client.set(f"embedding_quality:{concept}", "1.0")
            
            # Set initial usage count
            await self.redis_client.set(f"usage:{concept}", "100")
            
            # Set relationships
            await self.redis_client.sadd(f"relationships:{concept}", *data["related"])
            
            # Set initial co-occurrences
            for related in data["related"]:
                await self.redis_client.set(
                    f"cooccurrence:{concept}:{related}", 
                    str(int(100 * random.uniform(0.5, 0.9)))
                )
                
        print("Test data setup complete!")

    async def simulate_concept_drift(self):
        """Simulate gradual concept drift"""
        print("Simulating concept drift...")
        
        while True:
            # Randomly select concepts to drift
            drifting_concepts = random.sample(list(TEST_CONCEPTS.keys()), 2)
            
            for concept in drifting_concepts:
                # Get current quality
                current_quality = float(await self.redis_client.get(f"embedding_quality:{concept}") or 1.0)
                
                # Apply random drift
                drift = random.uniform(-0.1, 0.1)
                new_quality = max(0.1, min(1.0, current_quality + drift))
                
                # Update quality
                await self.redis_client.set(f"embedding_quality:{concept}", str(new_quality))
                
                print(f"Concept {concept} drifted from {current_quality:.2f} to {new_quality:.2f}")
            
            await asyncio.sleep(10)  # Drift every 10 seconds

    async def simulate_concept_usage(self):
        """Simulate concept usage patterns"""
        print("Simulating concept usage...")
        
        while True:
            # Randomly select concepts to use
            used_concepts = random.sample(list(TEST_CONCEPTS.keys()), 3)
            
            for concept in used_concepts:
                # Increment usage counter
                current_usage = int(await self.redis_client.get(f"usage:{concept}") or 0)
                await self.redis_client.set(f"usage:{concept}", str(current_usage + random.randint(1, 5)))
                
                # Update co-occurrences
                related = TEST_CONCEPTS[concept]["related"]
                for rel in random.sample(related, min(2, len(related))):
                    cooc_key = f"cooccurrence:{concept}:{rel}"
                    current_cooc = int(await self.redis_client.get(cooc_key) or 0)
                    await self.redis_client.set(cooc_key, str(current_cooc + random.randint(1, 3)))
                
                print(f"Concept {concept} used, new usage count: {current_usage + 1}")
            
            await asyncio.sleep(5)  # Usage events every 5 seconds

async def main():
    harness = ConceptTestHarness()
    
    # Setup test data
    await harness.setup_test_data()
    
    # Run simulations
    await asyncio.gather(
        harness.simulate_concept_drift(),
        harness.simulate_concept_usage()
    )

if __name__ == "__main__":
    print("Starting concept test harness...")
    asyncio.run(main()) 