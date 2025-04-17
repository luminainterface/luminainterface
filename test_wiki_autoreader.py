#!/usr/bin/env python3
"""
Test script for Wiki Auto-Reading functionality

This script tests the wiki auto-reading functionality without requiring the full UI.
It verifies that articles can be fetched from Wikipedia and stored in the knowledge base.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WikiAutoReader")

# Ensure we can import from current directory
sys.path.insert(0, str(Path(__file__).parent.absolute()))

def test_wiki_utils_import():
    """Test importing the wiki utilities"""
    logger.info("Testing wiki utilities import...")
    try:
        from plugins.plugin_utils import fetch_wiki_article, add_to_knowledge_base, batch_process_wiki_topics
        logger.info("✓ Successfully imported wiki utilities")
        return (fetch_wiki_article, add_to_knowledge_base, batch_process_wiki_topics)
    except Exception as e:
        logger.error(f"✗ Failed to import wiki utilities: {e}")
        return None

def test_fetch_article(fetch_wiki_article):
    """Test fetching a Wikipedia article"""
    logger.info("Testing article fetching...")
    try:
        test_topic = "artificial intelligence"
        logger.info(f"Fetching article on '{test_topic}'...")
        
        summary, content = fetch_wiki_article(test_topic)
        
        if summary and content:
            logger.info(f"✓ Successfully fetched article on '{test_topic}'")
            logger.info(f"  Summary length: {len(summary)} characters")
            logger.info(f"  Content length: {len(content)} characters")
            logger.info(f"  Summary preview: {summary[:100]}...")
            return True
        else:
            logger.error(f"✗ Failed to fetch article - no content returned")
            return False
    except Exception as e:
        logger.error(f"✗ Error fetching article: {e}")
        return False

def test_memory_import():
    """Test importing the memory system"""
    logger.info("Testing memory system import...")
    try:
        sys.path.append(os.path.dirname(__file__))
        from src.v7.onsite_memory import OnsiteMemory
        logger.info("✓ Successfully imported OnsiteMemory")
        return OnsiteMemory
    except Exception as e:
        logger.error(f"✗ Failed to import OnsiteMemory: {e}")
        return None

def test_mistral_import():
    """Test importing the Mistral integration"""
    logger.info("Testing Mistral integration import...")
    try:
        sys.path.append(os.path.dirname(__file__))
        from src.v7.mistral_integration import MistralIntegration
        logger.info("✓ Successfully imported MistralIntegration")
        return MistralIntegration
    except Exception as e:
        logger.error(f"✗ Failed to import MistralIntegration: {e}")
        return None

def test_knowledge_storage(fetch_wiki_article, add_to_knowledge_base, OnsiteMemory):
    """Test storing knowledge in the memory system"""
    logger.info("Testing knowledge storage...")
    
    if not OnsiteMemory:
        logger.warning("⚠ Skipping knowledge storage test due to missing OnsiteMemory")
        return False
    
    try:
        # Create test memory instance
        test_dir = Path("data/test_memory")
        test_dir.mkdir(exist_ok=True, parents=True)
        
        memory = OnsiteMemory(data_dir=str(test_dir))
        
        # Fetch test article
        test_topic = "neural networks"
        summary, content = fetch_wiki_article(test_topic)
        
        if summary and content:
            # Store in memory
            success = add_to_knowledge_base(test_topic, summary, memory=memory)
            
            if success:
                logger.info(f"✓ Successfully stored article on '{test_topic}' in memory")
                
                # Verify it was stored
                try:
                    knowledge = memory.get_knowledge(test_topic)
                    if knowledge:
                        logger.info(f"✓ Successfully retrieved knowledge from memory")
                        return True
                    else:
                        logger.error(f"✗ Knowledge was not retrievable from memory")
                        return False
                except Exception as e:
                    logger.error(f"✗ Error retrieving knowledge: {e}")
                    return False
            else:
                logger.error(f"✗ Failed to store article in memory")
                return False
        else:
            logger.error(f"✗ No content to store in memory")
            return False
    except Exception as e:
        logger.error(f"✗ Error in knowledge storage test: {e}")
        return False

def test_batch_processing(batch_process_wiki_topics, OnsiteMemory):
    """Test batch processing of topics"""
    logger.info("Testing batch processing...")
    
    if not batch_process_wiki_topics or not OnsiteMemory:
        logger.warning("⚠ Skipping batch processing test due to missing components")
        return False
    
    try:
        # Create test memory instance
        test_dir = Path("data/test_memory")
        test_dir.mkdir(exist_ok=True, parents=True)
        
        memory = OnsiteMemory(data_dir=str(test_dir))
        
        # Define test topics
        test_topics = [
            "machine learning",
            "deep learning",
            "reinforcement learning"
        ]
        
        # Define callback for logging
        def progress_callback(topic, status, current, total):
            logger.info(f"Processing {current+1}/{total}: {topic} - {status}")
        
        # Process topics
        logger.info(f"Processing {len(test_topics)} topics...")
        processed = batch_process_wiki_topics(
            test_topics, 
            memory=memory, 
            callback=progress_callback,
            max_topics=3
        )
        
        logger.info(f"✓ Successfully processed {processed}/{len(test_topics)} topics")
        return processed > 0
    except Exception as e:
        logger.error(f"✗ Error in batch processing test: {e}")
        return False

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Wiki Auto-Reading functionality")
    parser.add_argument("--skip-mistral", action="store_true", help="Skip Mistral integration tests")
    parser.add_argument("--topics", type=str, help="Comma-separated list of topics to test")
    
    args = parser.parse_args()
    
    logger.info("Starting Wiki Auto-Reading tests...")
    
    # Test wiki utilities import
    wiki_utils = test_wiki_utils_import()
    if not wiki_utils:
        return 1
    
    fetch_wiki_article, add_to_knowledge_base, batch_process_wiki_topics = wiki_utils
    
    # Test fetching article
    fetch_ok = test_fetch_article(fetch_wiki_article)
    
    # Test memory import
    OnsiteMemory = test_memory_import()
    
    # Test Mistral import
    if not args.skip_mistral:
        MistralIntegration = test_mistral_import()
    else:
        MistralIntegration = None
        logger.info("Skipping Mistral integration tests")
    
    # Test knowledge storage
    if OnsiteMemory:
        storage_ok = test_knowledge_storage(fetch_wiki_article, add_to_knowledge_base, OnsiteMemory)
    else:
        storage_ok = False
        logger.warning("⚠ Skipping knowledge storage test due to missing OnsiteMemory")
    
    # Test batch processing
    if OnsiteMemory:
        batch_ok = test_batch_processing(batch_process_wiki_topics, OnsiteMemory)
    else:
        batch_ok = False
        logger.warning("⚠ Skipping batch processing test due to missing OnsiteMemory")
    
    # Process custom topics if provided
    if args.topics and OnsiteMemory:
        custom_topics = [topic.strip() for topic in args.topics.split(",")]
        logger.info(f"Testing with custom topics: {custom_topics}")
        
        test_dir = Path("data/test_memory")
        test_dir.mkdir(exist_ok=True, parents=True)
        memory = OnsiteMemory(data_dir=str(test_dir))
        
        def progress_callback(topic, status, current, total):
            logger.info(f"Custom topic {current+1}/{total}: {topic} - {status}")
        
        processed = batch_process_wiki_topics(
            custom_topics, 
            memory=memory, 
            callback=progress_callback,
            max_topics=len(custom_topics)
        )
        
        logger.info(f"✓ Processed {processed}/{len(custom_topics)} custom topics")
    
    # Print summary
    logger.info("\n----- TEST SUMMARY -----")
    logger.info(f"Wiki Utilities Import:  {'✓ PASS' if wiki_utils else '✗ FAIL'}")
    logger.info(f"Article Fetching:       {'✓ PASS' if fetch_ok else '✗ FAIL'}")
    logger.info(f"Memory Import:          {'✓ PASS' if OnsiteMemory else '✗ FAIL'}")
    logger.info(f"Mistral Import:         {'✓ PASS' if MistralIntegration else '⚠ SKIPPED' if args.skip_mistral else '✗ FAIL'}")
    logger.info(f"Knowledge Storage:      {'✓ PASS' if storage_ok else '⚠ SKIPPED' if not OnsiteMemory else '✗ FAIL'}")
    logger.info(f"Batch Processing:       {'✓ PASS' if batch_ok else '⚠ SKIPPED' if not OnsiteMemory else '✗ FAIL'}")
    
    if fetch_ok and (storage_ok or not OnsiteMemory) and (batch_ok or not OnsiteMemory):
        logger.info("Overall result: PASS (with some tests potentially skipped)")
        return 0
    else:
        logger.info("Overall result: FAIL")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 