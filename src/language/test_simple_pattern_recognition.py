#!/usr/bin/env python3
"""
Test the enhanced pattern recognition capabilities of the Neural Linguistic Processor.
This simplified test focuses on the core pattern detection methods.
"""

import re
from neural_linguistic_processor import NeuralLinguisticProcessor

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def test_pattern_detection():
    """Test the enhanced pattern detection method directly."""
    print_section("TESTING ENHANCED PATTERN DETECTION")
    
    # Create a processor instance
    nlp = NeuralLinguisticProcessor()
    
    # Sample text with various patterns
    test_text = """
    Neural networks can recognize complex patterns in data.
    These patterns help systems understand language relationships.
    The network processes text to identify semantic connections.
    Patterns within patterns create recursive understanding structures.
    Neural networks and pattern recognition form the foundation of advanced AI.
    """
    
    # Extract words using the same method as in the processor
    text = test_text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    print(f"Analyzing text with {len(words)} words and {len(set(words))} unique words...\n")
    
    # Call the pattern detection method directly
    patterns = nlp._detect_patterns(words, text)
    
    # Print results
    if patterns:
        print(f"Found {len(patterns)} pattern types:")
        for i, pattern in enumerate(patterns):
            print(f"\n{i+1}. Type: {pattern.get('type', 'unknown')}")
            print(f"   Confidence: {pattern.get('confidence', 0):.3f}")
            
            # Different ways to print pattern details based on type
            if pattern.get('type') == 'repetition' and 'words' in pattern:
                word_scores = pattern.get('significance_scores', {})
                top_words = [(w, word_scores.get(w, 0)) for w in pattern.get('words', [])]
                print(f"   Repeated words:")
                for word, score in top_words:
                    print(f"     - '{word}' (significance: {score:.3f})")
            
            elif pattern.get('type').startswith('repeated_') and 'ngrams' in pattern:
                print(f"   Repeated n-grams:")
                for ng in pattern.get('ngrams', [])[:3]:  # Show top 3
                    print(f"     - '{ng.get('text', '')}' (count: {ng.get('count', 0)})")
            
            elif pattern.get('type') == 'semantic_cluster' and 'clusters' in pattern:
                print(f"   Semantic clusters:")
                for cluster in pattern.get('clusters', [])[:2]:  # Show top 2
                    print(f"     - Central word: '{cluster.get('central_word', '')}'")
                    related = cluster.get('related_words', [])[:3]
                    if related:
                        related_info = [f"'{w.get('word', '')}'" for w in related]
                        print(f"       Related: {', '.join(related_info)}")
    else:
        print("No patterns detected.")

def test_thematic_clusters():
    """Test the enhanced thematic cluster detection method."""
    print_section("TESTING THEMATIC CLUSTER DETECTION")
    
    # Create a processor instance
    nlp = NeuralLinguisticProcessor()
    
    # Sample text with clear themes
    test_text = """
    Neural networks learn from data to recognize patterns.
    The learning process involves training the network with examples.
    Data processing is essential for effective network training.
    Pattern recognition enables networks to classify new data accurately.
    Learning algorithms optimize the model during the training phase.
    """
    
    # Extract words using the same method as in the processor
    text = test_text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    print(f"Analyzing text for thematic clusters...\n")
    
    # Call the thematic cluster detection method directly
    thematic_clusters = nlp._identify_thematic_clusters(words)
    
    # Print results
    if thematic_clusters:
        print(f"Found {len(thematic_clusters)} thematic clusters:")
        for i, cluster in enumerate(thematic_clusters[:3]):  # Show top 3
            print(f"\n{i+1}. Core word: '{cluster.get('core_word', '')}'")
            print(f"   Core count: {cluster.get('core_count', 0)}")
            print(f"   Theme strength: {cluster.get('theme_strength', 0):.3f}")
            
            related = cluster.get('related_words', [])[:5]  # Show top 5 related words
            if related:
                print(f"   Related words:")
                for word_info in related:
                    word = word_info.get('word', '')
                    count = word_info.get('count', 0)
                    strength = word_info.get('strength', 0)
                    print(f"     - '{word}' (count: {count}, strength: {strength:.3f})")
    else:
        print("No thematic clusters detected.")

if __name__ == "__main__":
    test_pattern_detection()
    test_thematic_clusters() 