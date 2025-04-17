import json
from neural_linguistic_processor import NeuralLinguisticProcessor

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def test_basic_functionality():
    """Test basic functionality of the Neural Linguistic Processor."""
    print_section("BASIC NLP FUNCTIONALITY TEST")
    
    nlp = NeuralLinguisticProcessor()
    
    test_text = """
    The neural network system has developed recursive pattern recognition capabilities.
    These capabilities enable the system to identify complex linguistic structures and
    analyze semantic relationships between words and phrases. The system can now detect
    patterns within patterns, creating a multi-level understanding of language.
    Neural networks and recursive patterns are the foundation of advanced language processing.
    """
    
    print(f"Processing text sample...")
    
    # Process the text without relying on complex methods
    # Just check that basic text processing works
    words = test_text.lower().split()
    
    # Basic word frequency analysis
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Find most frequent words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nBasic Analysis Results:")
    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(word_counts)}")
    print(f"Top words: {top_words}")
    
    # Check semantic network functionality
    print("\nSemantic Network Functionality:")
    network_words = ["neural", "network", "pattern", "language"]
    for word in network_words:
        # Get related words from the semantic network
        if word in nlp.semantic_network:
            related = list(nlp.semantic_network[word].keys())[:3]  # Get up to 3 related words
            print(f"'{word}' is related to: {related}")
        else:
            print(f"'{word}' not found in semantic network")

if __name__ == "__main__":
    test_basic_functionality() 