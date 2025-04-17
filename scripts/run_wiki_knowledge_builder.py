#!/usr/bin/env python3
"""
Wiki Knowledge Builder

This script automatically builds a knowledge base by reading Wikipedia articles
on specified topics. It integrates with the LUMINA V7 memory system and Mistral AutoWiki.
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure we can import from project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(project_root / "logs" / "wiki_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WikiKnowledgeBuilder")

# Default topics for knowledge building
DEFAULT_TOPICS = [
    "artificial intelligence", "neural networks", "machine learning",
    "consciousness", "cognition", "linguistics", "natural language processing",
    "deep learning", "reinforcement learning", "language models",
    "natural language understanding", "cognitive science", "AGI",
    "semantic networks", "knowledge representation", "ontology", 
    "vector spaces", "embedding models", "transformer models",
    "language acquisition", "information theory", "Bayesian inference"
]

def init_directories():
    """Initialize necessary directories"""
    dirs = [
        project_root / "data",
        project_root / "data" / "auto_wiki",
        project_root / "data" / "chat_memory",
        project_root / "logs"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def load_topics(topics_file: Optional[str] = None) -> List[str]:
    """
    Load topics from a file or use defaults
    
    Args:
        topics_file: Optional path to a file with topics (one per line)
        
    Returns:
        List of topics to process
    """
    if topics_file and os.path.exists(topics_file):
        try:
            with open(topics_file, 'r') as f:
                topics = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(topics)} topics from {topics_file}")
            return topics
        except Exception as e:
            logger.error(f"Error loading topics from {topics_file}: {e}")
    
    logger.info(f"Using {len(DEFAULT_TOPICS)} default topics")
    return DEFAULT_TOPICS

def save_progress(processed_topics: List[str], output_file: str):
    """
    Save progress to a file
    
    Args:
        processed_topics: List of topics that have been processed
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "processed_topics": processed_topics,
                "count": len(processed_topics)
            }, f, indent=2)
        logger.info(f"Saved progress to {output_file}")
    except Exception as e:
        logger.error(f"Error saving progress to {output_file}: {e}")

def import_components():
    """Import required components and return them"""
    try:
        # Import wiki utilities
        from plugins.plugin_utils import fetch_wiki_article, batch_process_wiki_topics
        
        # Try to import memory system
        try:
            from src.v7.onsite_memory import OnsiteMemory
            memory_available = True
        except ImportError:
            logger.warning("OnsiteMemory not available, proceeding without memory integration")
            OnsiteMemory = None
            memory_available = False
        
        # Try to import Mistral integration
        try:
            from src.v7.mistral_integration import MistralIntegration
            mistral_available = True
        except ImportError:
            logger.warning("MistralIntegration not available, proceeding without Mistral")
            MistralIntegration = None
            mistral_available = False
            
        return {
            "fetch_wiki_article": fetch_wiki_article,
            "batch_process_wiki_topics": batch_process_wiki_topics,
            "OnsiteMemory": OnsiteMemory,
            "MistralIntegration": MistralIntegration,
            "memory_available": memory_available,
            "mistral_available": mistral_available
        }
    except Exception as e:
        logger.error(f"Error importing components: {e}")
        return None

def run_knowledge_builder(args, components):
    """
    Run the knowledge builder process
    
    Args:
        args: Command line arguments
        components: Dictionary of imported components
    """
    if not components:
        logger.error("Missing required components, cannot proceed")
        return
    
    # Initialize components
    memory = None
    mistral = None
    
    # Create memory if available
    if components["memory_available"] and components["OnsiteMemory"]:
        try:
            memory_path = project_root / "data" / "chat_memory"
            memory = components["OnsiteMemory"](data_dir=str(memory_path))
            logger.info("Initialized memory system")
        except Exception as e:
            logger.error(f"Error initializing memory: {e}")
    
    # Create Mistral if available and API key provided
    if components["mistral_available"] and components["MistralIntegration"] and args.mistral_api_key:
        try:
            mistral = components["MistralIntegration"](
                api_key=args.mistral_api_key,
                model=args.mistral_model,
                learning_enabled=True
            )
            logger.info(f"Initialized Mistral with model: {args.mistral_model}")
        except Exception as e:
            logger.error(f"Error initializing Mistral: {e}")
    
    # Load topics
    topics = load_topics(args.topics_file)
    
    if args.max_topics > 0:
        topics = topics[:args.max_topics]
    
    logger.info(f"Preparing to process {len(topics)} topics")
    
    # Set up progress tracking
    processed_topics = []
    
    def progress_callback(topic, status, current, total):
        """Callback to track progress and log/save"""
        logger.info(f"[{current+1}/{total}] {topic}: {status}")
        if status == "Success":
            processed_topics.append(topic)
            if args.progress_file:
                save_progress(processed_topics, args.progress_file)
    
    # Process topics
    logger.info("Starting Wikipedia knowledge building process...")
    
    try:
        processed = components["batch_process_wiki_topics"](
            topics=topics,
            memory=memory,
            mistral=mistral,
            callback=progress_callback,
            max_topics=len(topics)
        )
        
        logger.info(f"Knowledge building complete. Processed {processed}/{len(topics)} topics.")
        
        # Save final progress
        if args.progress_file:
            save_progress(processed_topics, args.progress_file)
            
        # Save Mistral learning dictionary if available
        if mistral and hasattr(mistral, "save_learning_dictionary"):
            try:
                mistral.save_learning_dictionary()
                logger.info("Saved Mistral learning dictionary")
            except Exception as e:
                logger.error(f"Error saving Mistral learning dictionary: {e}")
                
        # Save memory if available
        if memory:
            try:
                memory.save()
                logger.info("Saved memory system")
            except Exception as e:
                logger.error(f"Error saving memory: {e}")
                
    except Exception as e:
        logger.error(f"Error in knowledge building process: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Wiki Knowledge Builder")
    parser.add_argument("--topics-file", type=str, help="Path to file with topics (one per line)")
    parser.add_argument("--progress-file", type=str, default="data/wiki_builder_progress.json", help="Path to save progress")
    parser.add_argument("--max-topics", type=int, default=0, help="Maximum number of topics to process (0 for all)")
    parser.add_argument("--mistral-api-key", type=str, help="Mistral API key")
    parser.add_argument("--mistral-model", type=str, default="mistral-medium", help="Mistral model to use")
    
    args = parser.parse_args()
    
    # Initialize directories
    init_directories()
    
    # Import components
    components = import_components()
    
    if not components:
        logger.error("Failed to import required components")
        return 1
    
    # Run knowledge builder
    run_knowledge_builder(args, components)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 