#!/usr/bin/env python3
"""
Mistral Autowiki Demo

This script demonstrates the usage of the MistralIntegration class
with an autowiki learning example. The script shows how to add entries
to the autowiki, retrieve information, and use the autowiki with queries.
"""

import os
import logging
import time
from typing import Dict, Any

from mistral_integration import MistralIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MistralAutowikiDemo")

def run_demo() -> None:
    """
    Run the Mistral Autowiki demonstration.
    
    This function:
    1. Initializes the Mistral integration
    2. Adds information to the autowiki
    3. Retrieves information from the autowiki
    4. Processes queries using the autowiki as context
    5. Displays usage metrics
    """
    logger.info("Starting Mistral Autowiki Demo")
    
    # Get API key from environment variable if available
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    
    # Initialize Mistral integration
    # If no API key is provided, it will run in mock mode
    mistral = MistralIntegration(
        api_key=api_key,
        model="mistral-medium",
        learning_enabled=True,
        mock_mode=not api_key,
        learning_dict_path="data/mistral_autowiki_demo.json"
    )
    
    # Check if we're running in mock mode
    if mistral.mock_mode:
        logger.warning(
            "Running in mock mode. For real API responses, set the MISTRAL_API_KEY environment variable."
        )
    
    logger.info(f"Using model: {mistral.model}")

    # Example 1: Add entries to the autowiki
    logger.info("EXAMPLE 1: Adding entries to the autowiki")
    
    # Add information about neural networks
    neural_networks_content = """
    Neural networks are computational systems inspired by the biological neural networks in animal brains.
    They consist of artificial neurons that can learn from and make decisions based on input data.
    Key types include feedforward networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs).
    Deep learning involves neural networks with many layers that can learn hierarchical representations.
    """
    
    success = mistral.add_autowiki_entry(
        topic="Neural Networks",
        content=neural_networks_content,
        source="Introduction to Deep Learning, MIT Press, 2021"
    )
    logger.info(f"Added Neural Networks entry: {success}")
    
    # Add information about consciousness
    consciousness_content = """
    Consciousness is the state of being aware of and able to perceive one's surroundings, thoughts, and sensations.
    It involves subjective experiences and self-awareness. The neural correlates of consciousness (NCCs) are the neural mechanisms
    associated with conscious experience. Some theories of consciousness include Global Workspace Theory, Integrated Information Theory,
    and Higher-Order Theories.
    """
    
    success = mistral.add_autowiki_entry(
        topic="Consciousness",
        content=consciousness_content,
        source="The Conscious Mind, Oxford University Press, 2018"
    )
    logger.info(f"Added Consciousness entry: {success}")
    
    # Save the learning dictionary
    mistral.save_learning_dictionary()
    
    # Example 2: Retrieve information from the autowiki
    logger.info("\nEXAMPLE 2: Retrieving information from the autowiki")
    
    # Retrieve neural networks information
    neural_info = mistral.retrieve_autowiki("Neural Networks")
    if neural_info:
        logger.info(f"Retrieved Neural Networks information:")
        logger.info(f"- Content (excerpt): {neural_info['content'][:100]}...")
        logger.info(f"- Sources: {neural_info['sources']}")
    else:
        logger.error("Failed to retrieve Neural Networks information")
    
    # Retrieve consciousness information
    consciousness_info = mistral.retrieve_autowiki("consciousness")  # Testing case-insensitivity
    if consciousness_info:
        logger.info(f"Retrieved Consciousness information:")
        logger.info(f"- Content (excerpt): {consciousness_info['content'][:100]}...")
        logger.info(f"- Sources: {consciousness_info['sources']}")
    else:
        logger.error("Failed to retrieve Consciousness information")
    
    # Example 3: Process a query with autowiki context
    logger.info("\nEXAMPLE 3: Processing queries with autowiki context")
    
    query = "How are neural networks related to consciousness research?"
    
    logger.info(f"Processing query: '{query}'")
    response = mistral.process_message(
        message=query,
        system_prompt="You are an AI assistant specializing in neuroscience and artificial intelligence.",
        temperature=0.7,
        max_tokens=500,
        include_autowiki=True  # This will include relevant autowiki entries as context
    )
    
    logger.info(f"Response from {response['model']}:")
    logger.info(response['response'])
    
    # Example 4: Get usage metrics
    logger.info("\nEXAMPLE 4: Retrieving usage metrics")
    
    metrics = mistral.get_metrics()
    logger.info(f"API Calls: {metrics['api_calls']}")
    logger.info(f"Total Tokens Used: {metrics['tokens_used']}")
    logger.info(f"Prompt Tokens: {metrics['tokens_prompt']}")
    logger.info(f"Completion Tokens: {metrics['tokens_completion']}")
    logger.info(f"Learning Dictionary Size: {metrics['learning_dict_size']}")
    logger.info(f"Autowiki Entries: {metrics['autowiki_entries']}")
    
    # Save final state of the learning dictionary
    mistral.save_learning_dictionary()
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    run_demo() 