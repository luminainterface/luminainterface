#!/usr/bin/env python3
"""
Test the advanced pattern recognition capabilities of the Neural Linguistic Processor.
This test focuses on chiasmus and rhetorical pattern detection.
"""

import re
from neural_linguistic_processor import NeuralLinguisticProcessor

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def test_chiasmus_detection():
    """Test the chiasmus pattern detection method."""
    print_section("TESTING CHIASMUS PATTERN DETECTION")
    
    # Create a processor instance
    nlp = NeuralLinguisticProcessor()
    
    # Sample text with various types of chiasmus patterns
    test_text = """
    Ask not what your country can do for you ask what you can do for your country.
    In war prepare for peace in peace prepare for war.
    Fair is foul and foul is fair in this strange world.
    When the going gets tough the tough get going.
    Easy come easy go as they always say.
    The first shall be last and the last shall be first as it is written.
    We live to eat but some people eat to live.
    To stop too soon is to stop too late.
    I wasted time and now time wastes me.
    All for one and one for all.
    """
    
    # Extract words using the same method as in the processor
    text = test_text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    print(f"Analyzing text for chiasmus patterns...\n")
    
    # Call the chiasmus detection method directly
    chiasmus_patterns = nlp._detect_chiasmus(words, text)
    
    # Print results
    if chiasmus_patterns:
        print(f"Found {len(chiasmus_patterns)} chiasmus patterns:")
        
        # Group patterns by type for better display
        patterns_by_type = {}
        for pattern in chiasmus_patterns:
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append(pattern)
        
        # Display patterns grouped by type
        for pattern_type, patterns in patterns_by_type.items():
            print(f"\n--- {pattern_type.upper().replace('_', ' ')} PATTERNS ({len(patterns)}) ---")
            for i, pattern in enumerate(patterns):
                print(f"\n{i+1}. Pattern: {pattern.get('pattern', '')}")
                print(f"   Text: '{pattern.get('text', '')}'")
                print(f"   Confidence: {pattern.get('confidence', 0):.3f}")
                if pattern.get('semantic_relation', False):
                    print(f"   Semantic relation: Yes")
    else:
        print("No chiasmus patterns detected.")

def test_rhetorical_patterns():
    """Test the rhetorical pattern detection method."""
    print_section("TESTING RHETORICAL PATTERN DETECTION")
    
    # Create a processor instance
    nlp = NeuralLinguisticProcessor()
    
    # Sample text with rhetorical patterns
    test_text = """
    I have a dream that one day this nation will rise up. I have a dream that one day on the red hills of Georgia.
    I have a dream that one day even the state of Mississippi will be transformed. I have a dream today!
    
    We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields, 
    we shall fight in the hills; we shall never surrender.
    
    Is this the best we can do? Is this the path we want to follow? Is this the legacy we want to leave?
    
    The beautiful mountains, the beautiful valleys, the beautiful rivers all create a beautiful landscape.
    """
    
    # Extract words using the same method as in the processor
    text = test_text.lower()
    
    print(f"Analyzing text for rhetorical patterns...\n")
    
    # Call the rhetorical pattern detection method directly
    rhetorical_patterns = nlp._detect_rhetorical_patterns(text)
    
    # Print results
    if rhetorical_patterns:
        print(f"Found {len(rhetorical_patterns)} rhetorical patterns:")
        for i, pattern in enumerate(rhetorical_patterns):
            print(f"\n{i+1}. Type: {pattern.get('type', 'unknown')}")
            
            if pattern.get('type') == 'anaphora':
                print(f"   Phrase: '{pattern.get('phrase', '')}'")
                print(f"   Count: {pattern.get('count', 0)}")
                print(f"   Examples:")
                for example in pattern.get('examples', []):
                    print(f"     - '{example[:50]}...'")
            
            elif pattern.get('type') == 'epistrophe':
                print(f"   Phrase: '{pattern.get('phrase', '')}'")
                print(f"   Count: {pattern.get('count', 0)}")
            
            elif pattern.get('type') == 'rhetorical_question':
                print(f"   Text: '{pattern.get('text', '')[:50]}...'")
            
            elif pattern.get('type') == 'alliteration':
                print(f"   Letter: '{pattern.get('letter', '')}'")
                print(f"   Count: {pattern.get('count', 0)}")
                print(f"   Examples: {', '.join(pattern.get('examples', []))}")
            
            print(f"   Confidence: {pattern.get('confidence', 0):.3f}")
    else:
        print("No rhetorical patterns detected.")

if __name__ == "__main__":
    test_chiasmus_detection()
    test_rhetorical_patterns() 