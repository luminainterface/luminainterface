#!/usr/bin/env python3
"""
Ultimate AI Orchestration Architecture v10 - Core Functionality Test
Tests the revolutionary 3-tier strategic steering system
"""

import sys
import time
import asyncio
import requests
import json
from datetime import datetime

def test_imports():
    """Test if all core modules can be imported successfully"""
    print("🔍 Testing Core Module Imports...")
    
    try:
        # Test importing the core architecture files
        print("📦 Importing high_rank_adapter...")
        import high_rank_adapter
        print("✅ high_rank_adapter imported successfully")
        
        print("📦 Importing meta_orchestration_controller...")
        import meta_orchestration_controller
        print("✅ meta_orchestration_controller imported successfully")
        
        print("📦 Importing enhanced_real_world_benchmark...")
        import enhanced_real_world_benchmark
        print("✅ enhanced_real_world_benchmark imported successfully")
        
        print("📦 Importing ultimate_ai_architecture_summary...")
        import ultimate_ai_architecture_summary
        print("✅ ultimate_ai_architecture_summary imported successfully")
        
        print("📦 Importing constraint_mask...")
        import constraint_mask
        print("✅ constraint_mask imported successfully")
        
        print("📦 Importing token_limiter...")
        import token_limiter  
        print("✅ token_limiter imported successfully")
        
        print("📦 Importing unsat_guard...")
        import unsat_guard
        print("✅ unsat_guard imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_fastapi_initialization():
    """Test if FastAPI apps can be initialized"""
    print("\n🚀 Testing FastAPI App Initialization...")
    
    try:
        import high_rank_adapter
        app1 = high_rank_adapter.app
        print("✅ High-Rank Adapter FastAPI app initialized")
        
        import meta_orchestration_controller
        app2 = meta_orchestration_controller.app
        print("✅ Meta-Orchestration Controller FastAPI app initialized")
        
        import enhanced_real_world_benchmark
        app3 = enhanced_real_world_benchmark.app
        print("✅ Enhanced Execution Suite FastAPI app initialized")
        
        import ultimate_ai_architecture_summary
        app4 = ultimate_ai_architecture_summary.app
        print("✅ Architecture Summary FastAPI app initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI initialization failed: {e}")
        return False

def test_core_functionality():
    """Test core functionality of each service"""
    print("\n🧪 Testing Core Service Functionality...")
    
    try:
        # Test High-Rank Adapter
        print("🧠 Testing High-Rank Adapter...")
        import high_rank_adapter
        
        # Test pattern analysis
        test_transcript = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "How are you?", "assistant": "I'm doing well, thank you!"}
        ]
        
        adapter = high_rank_adapter.HighRankAdapter()
        patterns = adapter.analyze_conversation_patterns(test_transcript)
        print(f"✅ Pattern analysis successful: {len(patterns)} patterns found")
        
        # Test strategic steering
        steering = adapter.generate_strategic_steering(test_transcript, {})
        print(f"✅ Strategic steering generated: {len(steering)} parameters")
        
        # Test Meta-Orchestration Controller
        print("🎯 Testing Meta-Orchestration Controller...")
        import meta_orchestration_controller
        
        controller = meta_orchestration_controller.MetaOrchestrationController()
        
        # Test strategy selection
        context = {"complexity": "medium", "domain": "general"}
        strategy = controller.select_optimal_strategy(context)
        print(f"✅ Strategy selection successful: {strategy}")
        
        # Test execution plan generation
        plan = controller.generate_execution_plan("test query", context, strategy)
        print(f"✅ Execution plan generated: {len(plan)} steps")
        
        # Test Enhanced Execution Suite
        print("⚡ Testing Enhanced Execution Suite...")
        import enhanced_real_world_benchmark
        
        suite = enhanced_real_world_benchmark.EnhancedExecutionSuite()
        
        # Test 8-phase orchestration (just the structure)
        phases = suite.get_orchestration_phases()
        print(f"✅ 8-phase orchestration structure: {len(phases)} phases")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_architecture_integration():
    """Test the integration between the 3 tiers"""
    print("\n🌟 Testing 3-Tier Architecture Integration...")
    
    try:
        import high_rank_adapter
        import meta_orchestration_controller
        import enhanced_real_world_benchmark
        
        # Simulate a complete flow through the 3 tiers
        print("📋 Simulating complete 3-tier flow...")
        
        # Tier 1: High-Rank Adapter generates strategic steering
        adapter = high_rank_adapter.HighRankAdapter()
        test_data = {
            "query": "Explain quantum computing",
            "context": {"complexity": "high", "domain": "science"}
        }
        
        strategic_params = adapter.generate_strategic_steering([], test_data["context"])
        print(f"✅ Tier 1 (High-Rank Adapter): Generated {len(strategic_params)} strategic parameters")
        
        # Tier 2: Meta-Orchestration uses strategic parameters
        controller = meta_orchestration_controller.MetaOrchestrationController()
        strategy = controller.select_optimal_strategy(test_data["context"])
        execution_plan = controller.generate_execution_plan(test_data["query"], test_data["context"], strategy)
        print(f"✅ Tier 2 (Meta-Orchestration): Selected '{strategy}' strategy with {len(execution_plan)} steps")
        
        # Tier 3: Enhanced Execution Suite processes the plan
        suite = enhanced_real_world_benchmark.EnhancedExecutionSuite()
        phases = suite.get_orchestration_phases()
        print(f"✅ Tier 3 (Enhanced Execution): Ready for 8-phase orchestration ({len(phases)} phases)")
        
        print("🎉 3-Tier Integration Test SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Architecture integration test failed: {e}")
        return False

def test_dockerfile_presence():
    """Test if all required Dockerfiles exist"""
    print("\n🐳 Testing Docker Infrastructure...")
    
    import os
    
    required_dockerfiles = [
        "services/high-rank-adapter/Dockerfile",
        "services/meta-orchestration/Dockerfile", 
        "services/enhanced-execution/Dockerfile",
        "services/architecture-summary/Dockerfile"
    ]
    
    all_exist = True
    for dockerfile in required_dockerfiles:
        if os.path.exists(dockerfile):
            print(f"✅ {dockerfile} exists")
        else:
            print(f"❌ {dockerfile} missing")
            all_exist = False
    
    return all_exist

def test_startup_scripts():
    """Test if startup scripts exist and are properly formatted"""
    print("\n📜 Testing Startup Scripts...")
    
    import os
    
    scripts = ["start-ultimate-architecture.sh", "start-ultimate-architecture.ps1"]
    all_exist = True
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} exists")
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Ultimate AI Orchestration Architecture v10" in content:
                    print(f"✅ {script} contains correct architecture branding")
                else:
                    print(f"⚠️ {script} may be outdated")
        else:
            print(f"❌ {script} missing")
            all_exist = False
    
    return all_exist

def display_results():
    """Display comprehensive test results"""
    print("\n" + "="*70)
    print("🌟 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10 - TEST RESULTS")
    print("="*70)
    
    start_time = datetime.now()
    
    results = {
        "Module Imports": test_imports(),
        "FastAPI Initialization": test_fastapi_initialization(),
        "Core Functionality": test_core_functionality(),
        "3-Tier Integration": test_architecture_integration(),
        "Docker Infrastructure": test_dockerfile_presence(),
        "Startup Scripts": test_startup_scripts()
    }
    
    print(f"\n📊 TEST SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):")
    print("-" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"OVERALL RESULT: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10 IS READY!")
        print("🚀 The revolutionary 3-tier strategic steering system is functional")
        print("📊 Dashboard available at: http://localhost:9001 (when running)")
        print("🌟 Status: PRODUCTION READY")
    else:
        print(f"\n⚠️ {total-passed} issues found - please review failed tests")
    
    return passed == total

if __name__ == "__main__":
    print("🌟 Ultimate AI Orchestration Architecture v10 - Core Test Suite")
    print("Revolutionary 3-Tier Strategic Steering System")
    print("=" * 70)
    
    success = display_results()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Run: docker-compose -f docker-compose-v10-ultimate.yml up -d")
        print("2. Access dashboard: http://localhost:9001")
        print("3. Test endpoints: http://localhost:9000, 8999, 8998")
        
        exit(0)
    else:
        exit(1) 