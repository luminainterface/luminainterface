#!/usr/bin/env python3
"""
V5 and Language Memory Integration Script

This script connects the V5 Fractal Echo Visualization system with 
the Language Memory System and demonstrates the integration.
"""

import os
import sys
import time
import logging
import json
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/v5_language_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v5_language_integration")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))


def initialize_components(args):
    """
    Initialize all required components
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing initialized components
    """
    logger.info("Initializing integration components")
    components = {}
    
    # Initialize language memory V5 bridge
    try:
        from src.language_memory_v5_bridge import get_language_memory_v5_bridge
        components["bridge"] = get_language_memory_v5_bridge(mock_mode=args.mock)
        logger.info("✅ Language Memory V5 Bridge initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Language Memory V5 Bridge: {str(e)}")
        if not args.mock:
            logger.info("Trying again with mock mode enabled")
            try:
                from src.language_memory_v5_bridge import get_language_memory_v5_bridge
                components["bridge"] = get_language_memory_v5_bridge(mock_mode=True)
                logger.info("✅ Language Memory V5 Bridge initialized (mock mode)")
            except Exception as e2:
                logger.error(f"❌ Failed to initialize Language Memory V5 Bridge in mock mode: {str(e2)}")
                return components
    
    # Initialize central language node
    try:
        from src.central_language_node import CentralLanguageNode
        components["central_node"] = CentralLanguageNode()
        logger.info("✅ Central Language Node initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Central Language Node: {str(e)}")
    
    # Initialize neural linguistic processor
    try:
        from src.neural_linguistic_processor import get_linguistic_processor
        components["processor"] = get_linguistic_processor(config={"mock_mode": args.mock})
        logger.info("✅ Neural Linguistic Processor initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Neural Linguistic Processor: {str(e)}")
    
    # Initialize memory API socket
    try:
        from src.memory_api_socket import get_bridge
        components["memory_socket"] = get_bridge(mock_mode=args.mock)
        logger.info("✅ Memory API Socket initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Memory API Socket: {str(e)}")
    
    return components


def test_integration(components, args):
    """
    Test the integration between V5 and Language Memory
    
    Args:
        components: Dictionary of initialized components
        args: Command line arguments
        
    Returns:
        Dictionary containing test results
    """
    logger.info("Testing V5 and Language Memory integration")
    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "test_details": []
    }
    
    # Test 1: Bridge availability
    logger.info("Test 1: Bridge availability")
    results["tests_run"] += 1
    test_result = {"name": "Bridge availability", "passed": False, "details": ""}
    
    if "bridge" in components:
        bridge = components["bridge"]
        vis_available = bridge.is_visualization_available()
        test_result["details"] = f"V5 visualization available: {vis_available}"
        test_result["passed"] = True
        results["tests_passed"] += 1
        logger.info(f"✅ Bridge availability test passed, V5 available: {vis_available}")
    else:
        test_result["details"] = "Bridge component not initialized"
        results["tests_failed"] += 1
        logger.error("❌ Bridge availability test failed: Bridge component not initialized")
    
    results["test_details"].append(test_result)
    
    # Test 2: Memory synthesis
    logger.info("Test 2: Memory synthesis")
    results["tests_run"] += 1
    test_result = {"name": "Memory synthesis", "passed": False, "details": ""}
    
    if "bridge" in components:
        bridge = components["bridge"]
        try:
            start_time = time.time()
            topic = args.topic or "neural networks"
            synthesis = bridge.synthesize_topic(topic, depth=args.depth)
            processing_time = time.time() - start_time
            
            if "error" in synthesis:
                test_result["details"] = f"Error synthesizing topic: {synthesis['error']}"
                results["tests_failed"] += 1
                logger.error(f"❌ Memory synthesis test failed: {synthesis['error']}")
            else:
                test_result["details"] = f"Successfully synthesized topic '{topic}' in {processing_time:.3f} seconds"
                test_result["passed"] = True
                results["tests_passed"] += 1
                logger.info(f"✅ Memory synthesis test passed: Topic '{topic}' synthesized in {processing_time:.3f}s")
        except Exception as e:
            test_result["details"] = f"Exception during synthesis: {str(e)}"
            results["tests_failed"] += 1
            logger.error(f"❌ Memory synthesis test failed: {str(e)}")
    else:
        test_result["details"] = "Bridge component not initialized"
        results["tests_failed"] += 1
        logger.error("❌ Memory synthesis test failed: Bridge component not initialized")
    
    results["test_details"].append(test_result)
    
    # Test 3: Language processing
    logger.info("Test 3: Language processing")
    results["tests_run"] += 1
    test_result = {"name": "Language processing", "passed": False, "details": ""}
    
    if "processor" in components:
        processor = components["processor"]
        try:
            start_time = time.time()
            text = "Neural networks create fractal patterns representing consciousness"
            processed = processor.process_text(text)
            processing_time = time.time() - start_time
            
            if not processed or not isinstance(processed, dict):
                test_result["details"] = "Invalid response from processor"
                results["tests_failed"] += 1
                logger.error("❌ Language processing test failed: Invalid response from processor")
            else:
                test_result["details"] = f"Successfully processed text in {processing_time:.3f} seconds"
                test_result["passed"] = True
                results["tests_passed"] += 1
                logger.info(f"✅ Language processing test passed: Text processed in {processing_time:.3f}s")
        except Exception as e:
            test_result["details"] = f"Exception during processing: {str(e)}"
            results["tests_failed"] += 1
            logger.error(f"❌ Language processing test failed: {str(e)}")
    else:
        test_result["details"] = "Processor component not initialized"
        results["tests_failed"] += 1
        logger.error("❌ Language processing test failed: Processor component not initialized")
    
    results["test_details"].append(test_result)
    
    # Test 4: Central node integration
    logger.info("Test 4: Central node integration")
    results["tests_run"] += 1
    test_result = {"name": "Central node integration", "passed": False, "details": ""}
    
    if "central_node" in components:
        central_node = components["central_node"]
        try:
            status = central_node.get_status()
            v5_integration = status.get("component_status", {}).get("v5_language_integration")
            v5_bridge = status.get("component_status", {}).get("v5_bridge")
            
            test_result["details"] = f"V5 integration: {v5_integration}, V5 bridge: {v5_bridge}"
            
            if v5_integration == "active" or v5_bridge == "active":
                test_result["passed"] = True
                results["tests_passed"] += 1
                logger.info(f"✅ Central node integration test passed: {test_result['details']}")
            else:
                results["tests_failed"] += 1
                logger.error(f"❌ Central node integration test failed: {test_result['details']}")
        except Exception as e:
            test_result["details"] = f"Exception during central node test: {str(e)}"
            results["tests_failed"] += 1
            logger.error(f"❌ Central node integration test failed: {str(e)}")
    else:
        test_result["details"] = "Central node component not initialized"
        results["tests_failed"] += 1
        logger.error("❌ Central node integration test failed: Central node component not initialized")
    
    results["test_details"].append(test_result)
    
    # Test 5: Memory API socket
    logger.info("Test 5: Memory API socket")
    results["tests_run"] += 1
    test_result = {"name": "Memory API socket", "passed": False, "details": ""}
    
    if "memory_socket" in components:
        socket = components["memory_socket"]
        try:
            status = socket.get_status()
            
            # Create test message
            test_message = {
                "type": "get_stats",
                "request_id": f"test_{int(time.time())}",
                "content": {}
            }
            
            # Process message
            response = socket.socket.process_message(test_message)
            
            if response and "data" in response:
                test_result["details"] = "Socket successfully processed test message"
                test_result["passed"] = True
                results["tests_passed"] += 1
                logger.info("✅ Memory API socket test passed: Socket processed test message")
            else:
                test_result["details"] = f"Socket returned invalid response: {response}"
                results["tests_failed"] += 1
                logger.error(f"❌ Memory API socket test failed: Invalid response: {response}")
        except Exception as e:
            test_result["details"] = f"Exception during socket test: {str(e)}"
            results["tests_failed"] += 1
            logger.error(f"❌ Memory API socket test failed: {str(e)}")
    else:
        test_result["details"] = "Memory socket component not initialized"
        results["tests_failed"] += 1
        logger.error("❌ Memory API socket test failed: Memory socket component not initialized")
    
    results["test_details"].append(test_result)
    
    return results


def run_integration_demo(components, args):
    """
    Run a demonstration of the integration
    
    Args:
        components: Dictionary of initialized components
        args: Command line arguments
    """
    logger.info("Running V5 and Language Memory integration demo")
    
    if "bridge" not in components:
        logger.error("❌ Cannot run demo: Bridge component not initialized")
        return
    
    bridge = components["bridge"]
    
    # Demo 1: Synthesize topics
    topics = args.demo_topics.split(",") if args.demo_topics else ["neural networks", "consciousness", "language"]
    
    logger.info(f"Demo 1: Synthesizing {len(topics)} topics")
    for topic in topics:
        try:
            logger.info(f"Synthesizing topic: {topic}")
            start_time = time.time()
            result = bridge.synthesize_topic(topic, depth=args.depth)
            processing_time = time.time() - start_time
            
            if "error" in result:
                logger.error(f"❌ Failed to synthesize topic '{topic}': {result['error']}")
            else:
                # Extract and display some interesting data
                synthesis_results = result.get("synthesis_results", {})
                memory = synthesis_results.get("synthesized_memory", {})
                core_understanding = memory.get("core_understanding", "")
                insights = memory.get("novel_insights", [])
                related = synthesis_results.get("related_topics", [])
                
                logger.info(f"✅ Successfully synthesized topic '{topic}' in {processing_time:.3f}s")
                
                if core_understanding:
                    logger.info(f"Core understanding: {core_understanding[:150]}...")
                
                if insights:
                    logger.info(f"Found {len(insights)} insights")
                    for i, insight in enumerate(insights[:3]):
                        logger.info(f"  Insight {i+1}: {insight[:100]}...")
                
                if related:
                    logger.info(f"Found {len(related)} related topics")
                    for i, rel in enumerate(related[:5]):
                        if isinstance(rel, dict):
                            rel_topic = rel.get("topic", "unknown")
                            rel_relevance = rel.get("relevance", 0)
                            logger.info(f"  Related {i+1}: {rel_topic} (relevance: {rel_relevance:.2f})")
        except Exception as e:
            logger.error(f"❌ Exception synthesizing topic '{topic}': {str(e)}")
    
    # Demo 2: Process text with neural linguistic processor
    if "processor" in components:
        processor = components["processor"]
        
        texts = [
            "Neural networks encode patterns through weighted connections",
            "Consciousness emerges from integrated information processing",
            "Language serves as a bridge between individual consciousness"
        ]
        
        logger.info(f"Demo 2: Processing {len(texts)} texts with neural linguistic processor")
        
        for i, text in enumerate(texts):
            try:
                logger.info(f"Processing text: {text}")
                start_time = time.time()
                result = processor.process_text(text)
                processing_time = time.time() - start_time
                
                # Extract and display some interesting data
                analysis = result.get("analysis", {})
                features = analysis.get("features", {})
                complexity = features.get("complexity_score", 0)
                unique_ratio = features.get("unique_word_ratio", 0)
                
                pattern = result.get("pattern", {})
                node_count = len(pattern.get("nodes", []))
                
                logger.info(f"✅ Successfully processed text in {processing_time:.3f}s")
                logger.info(f"  Complexity: {complexity:.2f}, Unique word ratio: {unique_ratio:.2f}")
                logger.info(f"  Generated pattern with {node_count} nodes")
            except Exception as e:
                logger.error(f"❌ Exception processing text: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="V5 and Language Memory Integration Script")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--topic", type=str, help="Topic to synthesize for testing")
    parser.add_argument("--depth", type=int, default=3, help="Synthesis depth (1-5)")
    parser.add_argument("--demo", action="store_true", help="Run integration demo")
    parser.add_argument("--demo-topics", type=str, help="Comma-separated list of topics for demo")
    parser.add_argument("--output", type=str, help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    logger.info("Starting V5 and Language Memory integration")
    logger.info(f"Mock mode: {args.mock}")
    
    # Initialize components
    components = initialize_components(args)
    
    # Log component initialization summary
    component_count = len(components)
    logger.info(f"Initialized {component_count} components: {', '.join(components.keys())}")
    
    # Test integration
    test_results = test_integration(components, args)
    
    # Log test summary
    tests_run = test_results["tests_run"]
    tests_passed = test_results["tests_passed"]
    tests_failed = test_results["tests_failed"]
    
    logger.info(f"Integration tests complete: {tests_passed}/{tests_run} passed, {tests_failed}/{tests_run} failed")
    
    # Save test results if output file specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(test_results, f, indent=2)
            logger.info(f"Test results saved to {args.output}")
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")
    
    # Run demo if requested
    if args.demo:
        run_integration_demo(components, args)
    
    logger.info("V5 and Language Memory integration complete")


if __name__ == "__main__":
    main() 