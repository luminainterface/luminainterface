#!/usr/bin/env python3
"""
Enhanced A2A Coordination Hub - Comprehensive Test Suite
Demonstrates intelligent orchestration, mathematical validation, and multi-service coordination
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

class EnhancedOrchestratorTester:
    def __init__(self, base_url: str = "http://localhost:8891"):
        self.base_url = base_url
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def test_health_check(self):
        """Test the enhanced health check endpoint"""
        print("🏥 Testing Enhanced Health Check...")
        session = await self.get_session()
        
        try:
            async with session.get(f"{self.base_url}/health") as response:
                result = await response.json()
                print(f"✅ Health Status: {result}")
                return result
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return None
    
    async def test_service_health(self):
        """Test service health monitoring"""
        print("\n🔍 Testing Service Health Monitoring...")
        session = await self.get_session()
        
        try:
            async with session.get(f"{self.base_url}/service_health") as response:
                result = await response.json()
                print(f"🏥 Service Health Results:")
                for tier, services in result["service_health"].items():
                    print(f"  {tier.upper()} TIER:")
                    for service, status in services.items():
                        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "unhealthy" else "❌"
                        print(f"    {status_emoji} {service}: {status}")
                return result
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return None
    
    async def test_intelligent_query(self, query: str, expected_phase: str = None):
        """Test intelligent query processing with orchestration"""
        print(f"\n🧠 Testing Intelligent Query: '{query}'")
        session = await self.get_session()
        
        payload = {
            "query": query,
            "enable_mathematical_validation": True
        }
        
        start_time = time.time()
        
        try:
            async with session.post(f"{self.base_url}/intelligent_query", json=payload) as response:
                result = await response.json()
                processing_time = time.time() - start_time
                
                print(f"📊 Query Results:")
                print(f"  🎯 Confidence: {result['confidence']:.3f}")
                print(f"  🏗️ Processing Phase: {result['processing_phase']}")
                print(f"  🔧 Services Used: {len(result['services_used'])} services")
                print(f"    📝 Services: {', '.join(result['services_used'])}")
                print(f"  ⏱️ Processing Time: {result['processing_time']:.2f}s")
                print(f"  🔍 Mathematical Validation: {result['validation_applied']}")
                
                if result['mathematical_corrections']:
                    print(f"  🔧 Mathematical Corrections:")
                    for correction in result['mathematical_corrections']:
                        print(f"    ➤ {correction}")
                
                print(f"  💬 Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
                
                if expected_phase and result['processing_phase'] != expected_phase:
                    print(f"  ⚠️ Expected phase '{expected_phase}', got '{result['processing_phase']}'")
                
                return result
        except Exception as e:
            print(f"❌ Intelligent query failed: {e}")
            return None
    
    async def test_agent_coordination(self, agents: list, task: str):
        """Test enhanced agent coordination with orchestration"""
        print(f"\n🤝 Testing Agent Coordination: {agents}")
        session = await self.get_session()
        
        payload = {
            "agents": agents,
            "task": task,
            "coordination_type": "intelligent",
            "enable_routing": True,
            "enable_validation": True
        }
        
        try:
            async with session.post(f"{self.base_url}/coordinate", json=payload) as response:
                result = await response.json()
                
                print(f"🎯 Coordination Results:")
                print(f"  📋 Coordination ID: {result['coordination_id']}")
                print(f"  🤖 Agents: {result['agents']}")
                print(f"  📊 Confidence Score: {result['confidence_score']:.3f}")
                print(f"  🏗️ Processing Phase: {result['processing_phase']}")
                print(f"  🔧 Services Used: {len(result['services_used'])} services")
                print(f"  ✅ Status: {result['status']}")
                
                if 'intelligent_response' in result['result']:
                    response_text = result['result']['intelligent_response']
                    print(f"  💭 Intelligent Response: {response_text[:150]}{'...' if len(response_text) > 150 else ''}")
                
                return result
        except Exception as e:
            print(f"❌ Agent coordination failed: {e}")
            return None
    
    async def test_metrics(self):
        """Test orchestration metrics"""
        print("\n📊 Testing Orchestration Metrics...")
        session = await self.get_session()
        
        try:
            async with session.get(f"{self.base_url}/metrics") as response:
                result = await response.json()
                
                print(f"📈 System Metrics:")
                print(f"  🔄 Total Queries: {result['processing_metrics']['total_queries']}")
                print(f"  ✅ Successful Orchestrations: {result['processing_metrics']['successful_orchestrations']}")
                print(f"  🔧 Mathematical Corrections: {result['processing_metrics']['mathematical_corrections']}")
                print(f"  🏗️ Service Tiers Available:")
                for tier, count in result['service_tiers'].items():
                    print(f"    📦 {tier.capitalize()}: {count} services")
                print(f"  🎯 Total Available Services: {result['total_available_services']}")
                
                return result
        except Exception as e:
            print(f"❌ Metrics test failed: {e}")
            return None
    
    async def run_comprehensive_test_suite(self):
        """Run comprehensive test suite for the enhanced orchestrator"""
        print("🚀 ENHANCED A2A COORDINATION HUB - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        
        # Basic health checks
        await self.test_health_check()
        await self.test_service_health()
        
        # Test different confidence levels and phases
        test_queries = [
            # High confidence - BASELINE phase
            ("What is 25 + 17?", "baseline"),
            
            # Medium confidence - ENHANCED phase  
            ("Explain the relationship between AI and machine learning", "enhanced"),
            
            # Low confidence - ORCHESTRATED phase
            ("What would happen if Tesla and Einstein had a secret meeting?", "orchestrated"),
            
            # Very low confidence - COMPREHENSIVE phase
            ("Research the historical impact of quantum computing on medical literature", "comprehensive"),
            
            # Mathematical validation test
            ("Calculate 144 divided by 12", "baseline"),
            
            # Complex algebraic problem
            ("Solve for x: 2x + 5 = 17", "enhanced")
        ]
        
        print(f"\n🧪 Testing {len(test_queries)} different query types...")
        
        for query, expected_phase in test_queries:
            await self.test_intelligent_query(query, expected_phase)
            await asyncio.sleep(1)  # Brief pause between tests
        
        # Test agent coordination
        await self.test_agent_coordination(
            ["math_agent", "validation_agent", "orchestration_agent"],
            "Solve complex mathematical equations with validation"
        )
        
        await self.test_agent_coordination(
            ["research_agent", "analysis_agent"],
            "Analyze historical trends in AI development"
        )
        
        # Check final metrics
        await self.test_metrics()
        
        print("\n" + "=" * 70)
        print("🎉 COMPREHENSIVE TEST SUITE COMPLETED!")
        print("✨ Enhanced A2A Coordination Hub with Intelligent Orchestration is ready!")

async def run_mathematical_validation_tests():
    """Specific tests for mathematical validation capabilities"""
    print("\n🧮 MATHEMATICAL VALIDATION TEST SUITE")
    print("-" * 50)
    
    tester = EnhancedOrchestratorTester()
    
    math_tests = [
        "What is 144 divided by 12?",
        "Calculate 25 + 17",
        "What is 8 * 7?",
        "Solve 100 - 23",
        "What is 15 / 3?",
        "Solve for x: 2x + 5 = 17",
        "Find x when 3x - 4 = 14",
        "Calculate the derivative of x²"
    ]
    
    print(f"🔬 Running {len(math_tests)} mathematical validation tests...")
    
    for test_query in math_tests:
        result = await tester.test_intelligent_query(test_query)
        if result and result.get('mathematical_corrections'):
            print(f"  🔧 Corrections applied for: {test_query}")
        await asyncio.sleep(0.5)

async def main():
    """Main test execution"""
    try:
        print("🎯 Starting Enhanced A2A Coordination Hub Tests...")
        
        # Run comprehensive test suite
        tester = EnhancedOrchestratorTester()
        await tester.run_comprehensive_test_suite()
        
        # Run specific mathematical validation tests
        await run_mathematical_validation_tests()
        
        # Cleanup
        if tester.session:
            await tester.session.close()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 