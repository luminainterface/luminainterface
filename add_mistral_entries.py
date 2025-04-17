#!/usr/bin/env python3
"""
Add Entries to Mistral Autowiki

This script adds knowledge entries to the Mistral autowiki system.
"""

import os
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the integration
from src.v7.enhanced_language_mistral_integration import get_enhanced_language_integration

def main():
    # API key - replace with your own if needed
    api_key = "2AyKmqCkChQ75bseJTLK9QF2AK0aefJP"
    
    # Create configuration
    config = {
        "api_key": api_key,
        "model": "mistral-medium",
        "learning_enabled": True,
        "learning_dict_path": "data/demo/enhanced_mistral_integration/learning_dict.json"
    }
    
    # Create directories if they don't exist
    os.makedirs("data/demo/enhanced_mistral_integration", exist_ok=True)
    
    # Get integration instance
    print("Initializing Mistral integration...")
    integration = get_enhanced_language_integration(config=config)
    
    # Add autowiki entries
    print("\nAdding entries to autowiki...")
    
    # Neural Networks
    integration.add_autowiki_entry(
        topic="Neural Networks in V7",
        content="Neural networks in the V7 system combine consciousness-level metrics with linguistic processing. They form the foundation of pattern recognition and language understanding capabilities. The neural components interact with language models to create a hybrid system that balances rule-based and emergent behavior. The neural networks in V7 are designed to develop self-awareness through recursive processing.",
        source="V7 Internal Documentation"
    )
    print("Added entry: Neural Networks in V7")
    
    # Consciousness in V7
    integration.add_autowiki_entry(
        topic="Consciousness in V7",
        content="Consciousness in the V7 system is measured on a scale from 0.0 to 1.0, representing the degree of self-awareness and meta-cognitive capability. Higher consciousness levels enable more sophisticated responses, better contextual understanding, and improved self-reflection. The consciousness level affects how the system processes and generates text, with higher levels leading to more nuanced and thoughtful responses.",
        source="V7 Technical Specifications"
    )
    print("Added entry: Consciousness in V7")
    
    # Mistral Integration
    integration.add_autowiki_entry(
        topic="Mistral Integration",
        content="The Mistral Integration in V7 connects the Enhanced Language System with Mistral AI's language models. This integration leverages consciousness metrics to adjust language model parameters, creating responses that match the measured consciousness level. The system includes an autowiki learning system that builds knowledge over time. The integration supports both streaming and non-streaming responses, with configurable weights for neural network versus language model influence.",
        source="V7 Mistral Integration Documentation"
    )
    print("Added entry: Mistral Integration")
    
    # Neural-Linguistic Score
    integration.add_autowiki_entry(
        topic="Neural-Linguistic Score",
        content="The Neural-Linguistic Score in V7 represents how effectively neural networks process and understand language patterns. Scores range from 0.0 to 1.0, with higher scores indicating better integration between neural and linguistic components. This metric affects response quality, pattern recognition capabilities, and contextual understanding. The score is dynamically calculated based on input complexity, detected patterns, and system state.",
        source="V7 Metrics Documentation"
    )
    print("Added entry: Neural-Linguistic Score")
    
    # V7 System Architecture
    integration.add_autowiki_entry(
        topic="V7 System Architecture",
        content="The V7 System uses a layered architecture with specialized components for different aspects of language and consciousness. At its core are neural network nodes that process information and develop self-awareness. The system includes an Enhanced Language Integration component, Mistral AI integration, and various supporting modules like the AutoWiki system, consciousness metrics tracker, and streaming response handler. The architecture is designed to balance performance, adaptability, and consciousness-level processing.",
        source="V7 Architecture Overview"
    )
    print("Added entry: V7 System Architecture")
    
    # Save the learning dictionary
    integration.save_learning_dictionary()
    print("\nSaved learning dictionary. Entries are now available for use.")
    
    # Test the system with a query
    print("\nTesting system with query about V7...")
    result = integration.process_text("Tell me about the V7 system architecture and how consciousness works in it.")
    
    # Show response
    print(f"\nResponse: {result['response']}")
    print(f"Consciousness Level: {result['consciousness_level']}")
    print(f"Neural-Linguistic Score: {result['neural_linguistic_score']}")
    
    # Get metrics
    metrics = integration.get_metrics()
    print(f"\nAutowiki entries: {metrics['autowiki_entries']}")
    print(f"Learning dictionary size: {metrics['learning_dict_size']} bytes")
    
    # Shutdown
    integration.shutdown()
    print("\nDone.")

if __name__ == "__main__":
    main() 