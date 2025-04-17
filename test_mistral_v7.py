#!/usr/bin/env python3
"""
Test V7 Mistral Integration with AutoWiki

This script demonstrates the Enhanced Language Mistral Integration system
with the entries we've added to the autowiki.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the integration
from src.v7.enhanced_language_mistral_integration import get_enhanced_language_integration

# Test queries that should use our autowiki knowledge
TEST_QUERIES = [
    "What is the V7 System Architecture?",
    "How does consciousness work in the V7 system?",
    "Explain the Neural-Linguistic Score and its importance",
    "How are neural networks implemented in V7?",
    "What is the Mistral Integration in V7?",
    "Compare consciousness levels and neural-linguistic scores in V7"
]

def print_separator():
    """Print a separator line"""
    print("\n" + "=" * 80 + "\n")

def process_query(integration, query):
    """Process a query and print the results"""
    print(f"Query: {query}")
    
    # Process the query
    start_time = time.time()
    result = integration.process_text(query)
    process_time = time.time() - start_time
    
    # Print results
    print(f"\nResponse (processed in {process_time:.2f}s):")
    print(result["response"])
    
    # Print metrics
    print(f"\nConsciousness Level: {result['consciousness_level']:.2f}")
    print(f"Neural-Linguistic Score: {result['neural_linguistic_score']:.2f}")
    print(f"Recursive Pattern Depth: {result['recursive_pattern_depth']}")
    
    print_separator()
    
    return result

def main():
    """Main test function"""
    # API key - replace with your own if needed
    api_key = "2AyKmqCkChQ75bseJTLK9QF2AK0aefJP"
    
    # Create configuration
    config = {
        "api_key": api_key,
        "model": "mistral-medium",
        "learning_enabled": True,
        "learning_dict_path": "data/demo/enhanced_mistral_integration/learning_dict.json",
        "llm_weight": 0.7,
        "nn_weight": 0.6
    }
    
    # Initialize integration
    print("Initializing Mistral V7 integration...")
    integration = get_enhanced_language_integration(config=config)
    
    # Get initial metrics
    metrics = integration.get_metrics()
    print(f"Autowiki entries: {metrics['autowiki_entries']}")
    print(f"Learning dictionary size: {metrics['learning_dict_size']} bytes")
    
    print_separator()
    
    # Process each test query
    for query in TEST_QUERIES:
        process_query(integration, query)
    
    # Get final metrics
    metrics = integration.get_metrics()
    print("Final Metrics:")
    print(f"API calls: {metrics['api_calls']}")
    print(f"Tokens used: {metrics['tokens_used']}")
    
    # Shutdown
    integration.shutdown()
    print("Test completed successfully.")

if __name__ == "__main__":
    main() 