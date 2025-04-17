#!/usr/bin/env python3
"""
Test LLM Integration

This script demonstrates the integration of Mistral AI with the
Enhanced Language System in Lumina.
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/llm_integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_integration_test")

def test_integration(weight: float = 0.7, test_prompt: Optional[str] = None) -> None:
    """
    Test LLM integration with the Enhanced Language System
    
    Args:
        weight: LLM weight for processing
        test_prompt: Optional test prompt (uses default if None)
    """
    from src.v7.enhanced_language_mistral_integration import get_enhanced_language_integration
    
    print("\n=== Testing Enhanced Language System with Mistral ===\n")
    
    # Get API key from environment
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Check if API key is valid
    mock_mode = False
    if not api_key or api_key == "your_mistral_api_key_here":
        logger.warning("No valid Mistral API key found in .env file")
        print("\n⚠️ Warning: No valid Mistral API key found in .env file")
        print("Using simulated LLM responses instead of real Mistral AI.")
        print("To use real LLM, update your .env file with a valid MISTRAL_API_KEY value.\n")
        mock_mode = True
    
    # Initialize the system
    config = {
        "api_key": api_key,
        "model": os.getenv("LLM_MODEL", "mistral-medium"),
        "llm_weight": weight,
        "nn_weight": 0.6
    }
    
    print(f"LLM Weight: {weight}")
    print(f"Model: {config['model']}")
    print(f"Mock Mode: {mock_mode}")
    
    # Create the integration
    integration = get_enhanced_language_integration(mock_mode=mock_mode, config=config)
    
    # Use default test prompt if none provided
    if not test_prompt:
        test_prompt = "Explain how neural networks and language models can work together to create more comprehensive AI systems."
    
    print("\nProcessing prompt...")
    print(f"Prompt: {test_prompt}")
    
    if mock_mode:
        print("\n⚠️ Using simulated LLM responses instead of real Mistral AI.")
    
    # Process the text and measure time
    start_time = time.time()
    results = integration.process_text(test_prompt)
    processing_time = time.time() - start_time
    
    # Print the results
    print("\n=== Results ===")
    print(f"Processing Time: {processing_time:.2f} seconds")
    
    if "consciousness_level" in results:
        print(f"Consciousness Level: {results['consciousness_level']:.2f}")
    
    if "neural_linguistic_score" in results:
        print(f"Neural-Linguistic Score: {results['neural_linguistic_score']:.2f}")
    
    # Print the response
    print("\n=== Response ===\n")
    if "response" in results:
        print(results["response"])
    else:
        print(json.dumps(results, indent=2))
    
    # Print system metrics
    print("\n=== System Metrics ===")
    metrics = integration.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Clean up resources
    integration.shutdown()
    print("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Enhanced Language System with Mistral AI")
    parser.add_argument("--weight", type=float, default=0.7, help="Weight for LLM influence (0.0-1.0)")
    parser.add_argument("--prompt", type=str, help="Test prompt to use")
    args = parser.parse_args()
    
    test_integration(args.weight, args.prompt) 