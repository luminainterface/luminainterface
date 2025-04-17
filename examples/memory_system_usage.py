#!/usr/bin/env python3
"""
Memory System Usage Example

This script demonstrates how to use the language memory system programmatically,
without using the GUI interface.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import memory system components
from src.conversation_memory import ConversationMemory
from src.english_language_trainer import EnglishLanguageTrainer
from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration


def demonstrate_conversation_memory():
    """Demonstrate use of the Conversation Memory component"""
    print("\n===== Conversation Memory Demonstration =====")
    
    # Initialize conversation memory
    memory = ConversationMemory()
    
    # Store some example memories
    topics = ["language", "learning", "memory", "consciousness", "technology"]
    emotions = ["curious", "excited", "neutral", "contemplative", "enthusiastic"]
    
    print("Storing example memories...")
    for i in range(5):
        topic = topics[i % len(topics)]
        emotion = emotions[i % len(emotions)]
        content = f"This is a test memory about {topic}. It contains specific information related to {topic} systems."
        
        memory.store(
            content=content,
            metadata={
                "topic": topic,
                "emotion": emotion,
                "keywords": [topic, "example", "demonstration"],
                "source": "example_script"
            }
        )
    
    # Retrieve memories
    time.sleep(1)  # Short pause for better output formatting
    
    print("\nRetrieval by topic:")
    for topic in topics[:2]:  # Just check the first two topics
        results = memory.retrieve_by_topic(topic)
        print(f"- Found {len(results)} memories about '{topic}'")
        if results:
            print(f"  First memory: {results[0]['content'][:50]}...")
    
    print("\nRetrieval by emotion:")
    for emotion in emotions[:2]:  # Just check the first two emotions
        results = memory.retrieve_by_emotion(emotion)
        print(f"- Found {len(results)} memories with emotion '{emotion}'")
    
    print("\nRetrieval by keyword:")
    results = memory.retrieve_by_keyword("example")
    print(f"- Found {len(results)} memories with keyword 'example'")
    
    print("\nRetrieval by text search:")
    results = memory.search_text("specific information")
    print(f"- Found {len(results)} memories containing 'specific information'")
    
    # Display memory statistics
    time.sleep(1)
    print("\nMemory statistics:")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        if key != "top_topics":
            print(f"- {key}: {value}")
    
    # Display top topics
    print("- Top topics:")
    for topic, count in stats.get("top_topics", []):
        print(f"  * {topic}: {count} memories")


def demonstrate_language_trainer():
    """Demonstrate use of the English Language Trainer component"""
    print("\n===== Language Trainer Demonstration =====")
    
    # Initialize language trainer
    trainer = EnglishLanguageTrainer()
    
    # Generate training examples for different topics
    topics = ["learning", "memory", "language"]
    
    print("Generating training examples for different topics:")
    for topic in topics:
        examples = trainer.generate_training_data(topic=topic, count=2)
        print(f"\nTopic: {topic}")
        for i, example in enumerate(examples):
            print(f"- Example {i+1}: {example}")
    
    # Generate and evaluate some text
    print("\nGenerating and evaluating text:")
    text = " ".join(trainer.generate_training_data(topic="technology", count=3))
    print(f"Generated text:\n{text}\n")
    
    evaluation = trainer.evaluate_text(text)
    print("Text evaluation:")
    for key, value in evaluation.items():
        if key != "evaluation_id":
            print(f"- {key}: {value}")
    
    # Get a random idiom
    idiom = trainer.teach_idiom()
    print(f"\nRandom idiom: '{idiom['idiom']}'")
    print(f"Meaning: {idiom['meaning']}")
    
    # Display training statistics
    stats = trainer.get_training_stats()
    print("\nTraining statistics:")
    for key, value in stats.items():
        if key != "start_time":
            print(f"- {key}: {value}")


def demonstrate_memory_synthesis():
    """Demonstrate the Memory Synthesis Integration"""
    print("\n===== Memory Synthesis Demonstration =====")
    
    # Initialize memory synthesis integration
    synthesis = LanguageMemorySynthesisIntegration()
    
    # Synthesize memories for different topics
    topics = ["learning", "memory", "language"]
    
    print("Synthesizing memories across components:")
    for topic in topics:
        print(f"\nSynthesizing topic: '{topic}'")
        results = synthesis.synthesize_topic(topic, depth=3)
        
        if "synthesis_results" in results and results["synthesis_results"]:
            memory = results["synthesis_results"]["synthesized_memory"]
            print(f"Synthesis ID: {memory['id']}")
            print(f"Core understanding: {memory['core_understanding']}")
            print("Novel insights:")
            for insight in memory["novel_insights"][:3]:  # Show first 3 insights
                print(f"- {insight}")
    
    # Get memory synthesis statistics
    stats = synthesis.get_stats()
    print("\nSynthesis statistics:")
    print(f"- Synthesis count: {stats['synthesis_stats']['synthesis_count']}")
    print(f"- Topics synthesized: {stats['synthesis_stats']['topics_synthesized']}")


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("LANGUAGE MEMORY SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis script demonstrates how to use the language memory system programmatically.")
    
    # Demonstrate individual components
    demonstrate_conversation_memory()
    demonstrate_language_trainer()
    demonstrate_memory_synthesis()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main() 