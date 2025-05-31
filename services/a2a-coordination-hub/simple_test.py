#!/usr/bin/env python3
"""
Simple diagnostic test for the Enhanced A2A Coordination Hub
"""

import asyncio
import aiohttp
import json

async def test_simple_query():
    async with aiohttp.ClientSession() as session:
        # Test health first
        print("🏥 Testing health endpoint...")
        async with session.get("http://localhost:8891/health") as response:
            health = await response.json()
            print(f"Health: {health}")
        
        # Test service health
        print("\n🔍 Testing service health...")
        async with session.get("http://localhost:8891/service_health") as response:
            service_health = await response.json()
            print(f"Service Health Summary:")
            for tier, services in service_health["service_health"].items():
                available = sum(1 for status in services.values() if status == "healthy")
                total = len(services)
                print(f"  {tier}: {available}/{total} services healthy")
        
        # Test simple math query
        print("\n🧮 Testing simple math query...")
        payload = {
            "query": "What is 2 + 2?",
            "enable_mathematical_validation": True
        }
        
        async with session.post("http://localhost:8891/intelligent_query", json=payload) as response:
            result = await response.json()
            print(f"Response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_simple_query()) 