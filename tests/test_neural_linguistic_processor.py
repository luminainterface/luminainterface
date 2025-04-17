import unittest
import sys
import os
from datetime import datetime
import json

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.language.neural_linguistic_processor import NeuralLinguisticProcessor

class TestNeuralLinguisticProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = NeuralLinguisticProcessor()
        
    def test_initialization(self):
        """Test that the processor initializes correctly."""
        self.assertIsInstance(self.processor, NeuralLinguisticProcessor)
        self.assertEqual(self.processor.processed_text_count, 0)
        self.assertEqual(self.processor.total_words_processed, 0)
        self.assertEqual(len(self.processor.neural_linguistic_score_history), 0)
        
    def test_process_text_simple(self):
        """Test processing a simple text."""
        text = "This is a simple test."
        result = self.processor.process_text(text)
        
        # Check the structure of the result
        self.assertIn('neural_linguistic_score', result)
        self.assertIn('confidence', result)
        self.assertIn('analysis', result)
        self.assertIn('word_metrics', result)
        
        # Check that metrics were updated
        self.assertEqual(self.processor.processed_text_count, 1)
        self.assertEqual(self.processor.total_words_processed, 5)  # "This is a simple test" has 5 words
        self.assertEqual(len(self.processor.neural_linguistic_score_history), 1)
        
    def test_process_text_complex(self):
        """Test processing a more complex text with patterns."""
        text = """
        The neural linguistic processor is designed to detect patterns in text.
        It identifies repetition, n-grams, syntactic structures, and semantic relationships.
        The neural linguistic processor also analyzes text for complexity and coherence.
        Repetition is a key pattern that the processor can identify with high confidence.
        Neural networks help to identify complex patterns that might be missed by simple algorithms.
        """
        
        result = self.processor.process_text(text)
        
        # Check pattern detection
        self.assertIn('analysis', result)
        self.assertIn('internal', result['analysis'])
        self.assertIn('symbolic', result['analysis']['internal'])
        self.assertIn('patterns', result['analysis']['internal']['symbolic'])
        
        # There should be at least some detected patterns in this text
        patterns = result['analysis']['internal']['symbolic']['patterns']
        self.assertGreater(len(patterns), 0, "No patterns detected in complex text")
        
        # Check for score history update
        self.assertEqual(len(self.processor.neural_linguistic_score_history), 2)
        
    def test_pattern_recognition(self):
        """Test specifically the pattern recognition capabilities."""
        # Text with deliberate patterns
        text = """
        The rain in Spain falls mainly on the plain.
        The rain in France falls mainly on the dance.
        The rain in Maine falls mainly on the lane.
        """
        
        # Process the text
        result = self.processor.process_text(text)
        
        # Extract patterns
        patterns = result['analysis']['internal']['symbolic']['patterns']
        
        # Check for repetition patterns (should detect "The rain in X falls mainly on the Y")
        repetition_patterns = [p for p in patterns if p.get('type') == 'repetition']
        self.assertGreater(len(repetition_patterns), 0, "No repetition patterns detected")
        
        # Check for parallel structure patterns
        parallel_patterns = [p for p in patterns if p.get('type') == 'parallel_structure']
        self.assertGreater(len(parallel_patterns), 0, "No parallel structure patterns detected")
        
    def test_recursive_pattern_detection(self):
        """Test the recursive pattern detection capability."""
        # Text with nested patterns
        text = """
        The big dog chased the small cat. The small cat chased the tiny mouse.
        The tiny mouse ran into its hole. The big dog barked at the hole.
        """
        
        # Process the text
        result = self.processor.process_text(text)
        
        # Check for recursive patterns
        recursive_patterns = result['analysis']['internal']['symbolic'].get('recursive_patterns', [])
        self.assertGreater(len(recursive_patterns), 0, "No recursive patterns detected")
        
    def test_significant_words_extraction(self):
        """Test that top words extraction prioritizes significant words."""
        text = """
        The quantum physics theory describes fundamental particles and their interactions.
        Quantum entanglement is a phenomenon where particles remain connected regardless of distance.
        The Heisenberg uncertainty principle states that certain pairs of physical properties cannot
        be precisely measured simultaneously.
        """
        
        # Process the text
        result = self.processor.process_text(text)
        
        # Check top words
        top_words = result['word_metrics']['top_words']
        self.assertGreater(len(top_words), 0, "No top words extracted")
        
        # The words "quantum", "particles", "entanglement", "Heisenberg", "uncertainty" 
        # should be considered significant
        top_word_list = [w['word'].lower() for w in top_words]
        significant_words = ['quantum', 'particles', 'entanglement', 'heisenberg', 'uncertainty']
        
        # At least some of these should be in the top words
        overlap = [word for word in significant_words if word in top_word_list]
        self.assertGreater(len(overlap), 0, f"None of the expected significant words {significant_words} found in {top_word_list}")
        
    def test_neural_linguistic_score(self):
        """Test that the neural linguistic score reflects text complexity and patterns."""
        simple_text = "The cat sat on the mat. The dog ran in the yard."
        complex_text = """
        The intricate interplay between quantum entanglement and relativistic space-time curvature 
        presents profound implications for our understanding of fundamental physical principles. 
        Researchers have demonstrated that entangled particles, when subjected to varying gravitational 
        fields, exhibit anomalous correlations that challenge conventional quantum field theory.
        """
        
        # Process both texts
        simple_result = self.processor.process_text(simple_text)
        complex_result = self.processor.process_text(complex_text)
        
        # Complex text should have higher neural linguistic score
        simple_score = simple_result['neural_linguistic_score']
        complex_score = complex_result['neural_linguistic_score']
        
        self.assertGreater(complex_score, simple_score, 
                          f"Complex text score ({complex_score}) not higher than simple text score ({simple_score})")
    
    def test_llm_integration(self):
        """Test the LLM integration and weighting."""
        text = "This text will be processed with simulated LLM analysis."
        
        # Set different LLM weights and compare results
        original_weight = self.processor.llm_weight
        
        # Test with low LLM weight
        self.processor.llm_weight = 0.1
        low_weight_result = self.processor.process_text(text)
        
        # Test with high LLM weight
        self.processor.llm_weight = 0.9
        high_weight_result = self.processor.process_text(text)
        
        # Restore original weight
        self.processor.llm_weight = original_weight
        
        # The results should be different
        self.assertNotEqual(low_weight_result['neural_linguistic_score'], 
                           high_weight_result['neural_linguistic_score'],
                           "LLM weight change did not affect the score")
    
    def test_state_persistence(self):
        """Test that the processor can save and load its state."""
        # Process some text to generate state
        text = "This is a test of state persistence."
        self.processor.process_text(text)
        
        # Force save state
        self.processor._save_state()
        
        # Create a new processor that should load the state
        new_processor = NeuralLinguisticProcessor()
        
        # Check that some state was loaded
        self.assertEqual(new_processor.processed_text_count, self.processor.processed_text_count)
        self.assertEqual(new_processor.total_words_processed, self.processor.total_words_processed)
    
if __name__ == '__main__':
    unittest.main() 