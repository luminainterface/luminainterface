#!/usr/bin/env python3
"""
Mistral AI Integration Demo Script

This script demonstrates the usage of the MistralIntegration class,
focusing on the autowiki learning functionality.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the MistralIntegration class
from src.v7.mistral_integration import MistralIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MistralDemo")

def run_demo():
    """Run the Mistral integration demo with autowiki learning."""
    
    # Check for API key in environment variables
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    
    # Initialize the Mistral integration
    # Using mock_mode=True if no API key is provided
    mock_mode = not bool(api_key)
    if mock_mode:
        logger.warning("No API key found. Running in mock mode.")
        logger.info("Set the MISTRAL_API_KEY environment variable to use the actual API.")
    
    mistral = MistralIntegration(
        api_key=api_key,
        model="mistral-medium",  # Can be changed to other models
        mock_mode=mock_mode,
        learning_enabled=True,
        max_memory_entries=100
    )
    
    logger.info(f"Initialized Mistral integration with model: {mistral.model}")
    logger.info(f"Mock mode: {mistral.mock_mode}")
    logger.info(f"Learning enabled: {mistral.learning_enabled}")
    
    # Example 1: Add information to the autowiki
    logger.info("\n--- Example 1: Adding information to autowiki ---")
    
    # Add some information about neural networks
    mistral.add_autowiki_entry(
        topic="Neural Networks",
        content="Neural networks are computing systems inspired by biological neural networks in animal brains.",
        source="https://en.wikipedia.org/wiki/Neural_network"
    )
    
    # Add additional information on the same topic
    mistral.add_autowiki_entry(
        topic="Neural Networks",
        content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        source="https://en.wikipedia.org/wiki/Deep_learning"
    )
    
    # Add information on a different topic
    mistral.add_autowiki_entry(
        topic="Consciousness",
        content="Consciousness is the state of being awake and aware of one's surroundings.",
        source="https://en.wikipedia.org/wiki/Consciousness"
    )
    
    # Example 2: Retrieve information from autowiki
    logger.info("\n--- Example 2: Retrieving information from autowiki ---")
    
    nn_info = mistral.retrieve_autowiki(topic="Neural Networks")
    if nn_info:
        logger.info(f"Information about Neural Networks:")
        logger.info(f"Content: {nn_info['content']}")
        logger.info(f"Sources: {nn_info['sources']}")
    
    consciousness_info = mistral.retrieve_autowiki(topic="Consciousness")
    if consciousness_info:
        logger.info(f"\nInformation about Consciousness:")
        logger.info(f"Content: {consciousness_info['content']}")
        logger.info(f"Sources: {consciousness_info['sources']}")
    
    # Example 3: Process a query related to the learning dictionary
    logger.info("\n--- Example 3: Processing a query with learning ---")
    
    query = "Explain how neural networks relate to consciousness"
    logger.info(f"Query: '{query}'")
    
    response = mistral.process_message(
        message=query,
        system_prompt="You are a helpful AI assistant with expertise in neuroscience and AI.",
        temperature=0.7,
        max_tokens=300
    )
    
    logger.info(f"Response: {response['response']}")
    logger.info(f"Model used: {response['model']}")
    logger.info(f"Is cached: {response['is_cached']}")
    
    # Example 4: Get usage metrics
    logger.info("\n--- Example 4: Getting usage metrics ---")
    
    metrics = mistral.get_metrics()
    logger.info(f"API calls: {metrics['api_calls']}")
    logger.info(f"Tokens used: {metrics['tokens_used']}")
    logger.info(f"Learning dictionary size: {metrics['learning_dict_size']}")
    logger.info(f"Autowiki entries: {metrics['autowiki_entries']}")
    
    # Save the learning dictionary before exiting
    mistral.save_learning_dictionary()
    logger.info("\nLearning dictionary saved.")

if __name__ == "__main__":
    logger.info("Starting Mistral integration demo...")
    run_demo()
    logger.info("Demo completed.") 