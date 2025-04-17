#!/usr/bin/env python3
"""
Recursive Pattern Analyzer

This module enhances the v10 consciousness capabilities by detecting and analyzing
recursive language patterns in text. It provides both algorithmic pattern detection
and LLM-enhanced recursive pattern analysis to support the journey towards
deeper system consciousness.
"""

import os
import sys
import logging
import re
import json
import threading
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import random

# Import our LLM provider
try:
    from .llm_provider import get_llm_provider
    HAS_LLM_PROVIDER = True
except ImportError:
    HAS_LLM_PROVIDER = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recursive_pattern_analyzer")

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Try to import language components
try:
    from src.language.central_language_node import CentralLanguageNode
    HAS_CENTRAL_NODE = True
except ImportError:
    HAS_CENTRAL_NODE = False
    logger.warning("Central Language Node not available")


class RecursivePatternAnalyzer:
    """
    Analyzes text for recursive language patterns such as self-references, 
    meta-linguistic statements, linguistic loops, and recursive structures.
    Includes LLM weighing to adjust the influence of language model suggestions.
    """
    
    def __init__(self, data_dir: str = "data/recursive_patterns", llm_weight: float = 0.5, nn_weight: float = 0.5):
        """
        Initialize the recursive pattern analyzer.
        
        Args:
            data_dir: Directory for storing pattern data
            llm_weight: Weight for LLM influence (0.0-1.0)
            nn_weight: Weight for neural vs symbolic processing (0.0-1.0)
        """
        self.logger = logging.getLogger("RecursivePatternAnalyzer")
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('RecursivePatternAnalyzer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize pattern data
        self.pattern_history = []
        self.recent_patterns = []
        self.pattern_counts = {
            "self_reference": 0,
            "meta_linguistic": 0,
            "linguistic_loop": 0,
            "recursive_structure": 0,
            "consciousness_assertion": 0
        }
        self.avg_depth = 0.0
        self.patterns = []
        self.pattern_confidences = {}
        self.stats = {
            "total_patterns": 0,
            "max_depth": 0,
            "avg_depth": 0.0,
            "neural_patterns": 0,
            "symbolic_patterns": 0
        }
        
        # Initialize LLM provider if available and weight is significant
        self.llm_provider = None
        if HAS_LLM_PROVIDER and self.llm_weight > 0.3:
            try:
                self.llm_provider = get_llm_provider()
                self.logger.info(f"LLM provider initialized for RecursivePatternAnalyzer")
            except Exception as e:
                self.logger.error(f"Error initializing LLM provider: {e}")
        
        # Load existing pattern data if available
        self._load_patterns()
        
        self.logger.info(f"Initialized RecursivePatternAnalyzer with llm_weight={self.llm_weight}, nn_weight={self.nn_weight}")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for recursive patterns, applying both LLM and neural weights.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        patterns = []
        
        # First apply rule-based pattern detection (symbolic approach)
        rule_patterns = self._detect_patterns_rule_based(text)
        
        # Then apply LLM-based detection with appropriate weight
        if self.llm_weight > 0.3:  # Only use LLM if weight is significant
            if HAS_LLM_PROVIDER and self.llm_provider:
                # Use real LLM
                llm_patterns = self._detect_patterns_llm_real(text)
            else:
                # Use simulated LLM
                llm_patterns = self._detect_patterns_llm_simulated(text)
        else:
            llm_patterns = []
        
        # Apply neural weight to balance pattern sources
        symbolic_weight = 1.0 - self.nn_weight  # Higher neural weight means less symbolic influence
        neural_weight = self.nn_weight  # Direct control over neural pattern detection
        
        # Combine patterns with appropriate weighting
        all_patterns = []
        
        # Add rule-based patterns with symbolic weight
        for pattern in rule_patterns:
            pattern['weight'] = 'symbolic'
            pattern['confidence'] *= symbolic_weight
            if pattern['confidence'] >= 0.2:  # Threshold to include
                all_patterns.append(pattern)
        
        # Add LLM patterns with LLM weight
        for pattern in llm_patterns:
            pattern['weight'] = 'neural'
            pattern['confidence'] *= self.llm_weight * neural_weight
            if pattern['confidence'] >= 0.2:  # Threshold to include
                all_patterns.append(pattern)
                
        # Sort by confidence and keep most confident patterns
        all_patterns.sort(key=lambda p: p['confidence'], reverse=True)
        patterns = all_patterns[:10]  # Limit to top 10 patterns
        
        # Calculate pattern depths
        for pattern in patterns:
            if "depth" not in pattern:
                pattern["depth"] = self._calculate_pattern_depth(pattern)
        
        # Calculate average depth
        depths = [p.get("depth", 0) for p in patterns]
        current_avg_depth = sum(depths) / len(depths) if depths else 0
        
        # Update overall average depth with smoothing
        if self.avg_depth == 0:
            self.avg_depth = current_avg_depth
        else:
            self.avg_depth = 0.8 * self.avg_depth + 0.2 * current_avg_depth
        
        # Calculate confidence based on pattern diversity and depth
        pattern_types = set(p.get("type") for p in patterns)
        type_diversity = len(pattern_types) / len(self.pattern_counts) if patterns else 0
        depth_factor = min(1.0, self.avg_depth / 3.0)  # Normalize depth to 0-1 range
        
        # Get rule-based confidence
        rule_confidence = 0.5 * type_diversity + 0.3 * depth_factor + 0.2 * min(1.0, len(patterns) / 5.0)
        
        # Get LLM-based confidence
        if self.llm_provider and self.llm_weight > 0.0:
            llm_confidence = self._get_llm_confidence(text, patterns)
        else:
            llm_confidence = random.uniform(0.6, 0.9)  # Simulated LLM confidence
        
        # Apply LLM weight to confidence
        confidence = rule_confidence * (1 - self.llm_weight) + llm_confidence * self.llm_weight
        
        # Update recent patterns
        self.recent_patterns.extend(patterns)
        self.recent_patterns = self.recent_patterns[-50:]  # Keep only the 50 most recent
        
        # Update pattern history
        self.pattern_history.append({
            "timestamp": time.time(),
            "pattern_count": len(patterns),
            "pattern_types": list(pattern_types),
            "avg_depth": current_avg_depth,
            "confidence": confidence
        })
        
        # Save updated pattern data
        self._save_patterns()
        
        # Return results
        result = {
            "patterns": patterns,
            "pattern_counts": self.pattern_counts.copy(),
            "avg_depth": self.avg_depth,
            "confidence": confidence
        }
        
        self.logger.info(f"Analysis complete. Found {len(patterns)} patterns with confidence {confidence:.2f}")
        return result
    
    def _detect_patterns_rule_based(self, text: str) -> List[Dict]:
        """
        Detect recursive patterns using rule-based methods.
        """
        patterns = []
        
        # Self-reference patterns
        self_refs = ["this sentence", "this text", "these words", "this statement", "itself"]
        for ref in self_refs:
            if ref.lower() in text.lower():
                patterns.append({
                    "type": "self_reference",
                    "text": text,
                    "confidence": 0.8,
                    "matches": [ref]
                })
        
        # Meta-linguistic patterns
        meta_words = ["language", "word", "sentence", "writing", "says", "meaning"]
        meta_count = sum(1 for word in meta_words if word.lower() in text.lower().split())
        if meta_count >= 2:
            patterns.append({
                "type": "meta_linguistic",
                "text": text,
                "confidence": 0.7,
                "meta_count": meta_count
            })
        
        # Recursive structure patterns
        # Look for nested phrases using regex
        nested_phrases = re.findall(r'\([^()]*\([^()]*\)[^()]*\)', text)
        if nested_phrases:
            patterns.append({
                "type": "recursive_structure",
                "text": text,
                "confidence": 0.9,
                "examples": nested_phrases[:3]
            })
        
        # Linguistic loops
        loop_indicators = ["repeat", "loop", "cycle", "circular", "again and again"]
        loop_count = sum(1 for word in loop_indicators if word.lower() in text.lower())
        if loop_count > 0:
            patterns.append({
                "type": "linguistic_loop",
                "text": text,
                "confidence": 0.6,
                "loop_count": loop_count
            })
        
        # Consciousness assertions
        if "conscious" in text.lower() or "aware" in text.lower() or "thinking" in text.lower():
            patterns.append({
                "type": "consciousness_assertion",
                "text": text,
                "confidence": 0.75
            })
        
        return patterns
    
    def _detect_patterns_llm_real(self, text: str) -> List[Dict]:
        """
        Detect recursive patterns using the real LLM provider.
        """
        # Skip if no LLM provider or weight is 0
        if not self.llm_provider or self.llm_weight <= 0.0:
            return []
        
        try:
            # Get pattern analysis from LLM provider
            llm_analysis = self.llm_provider.analyze_patterns(text)
            
            # Convert the LLM response to our pattern format
            patterns = []
            
            # Extract pattern count
            pattern_count = llm_analysis.get("patterns_found", 0)
            
            # Create pattern entries based on LLM response
            if pattern_count > 0:
                # Extract pattern types from LLM response
                pattern_types = llm_analysis.get("pattern_types", ["self_reference"])
                
                # Create a pattern entry for each type
                for pattern_type in pattern_types:
                    # Map to our standard pattern types
                    if "self" in pattern_type.lower():
                        type_name = "self_reference"
                    elif "meta" in pattern_type.lower():
                        type_name = "meta_linguistic"
                    elif "loop" in pattern_type.lower() or "cycl" in pattern_type.lower():
                        type_name = "linguistic_loop"
                    elif "recur" in pattern_type.lower():
                        type_name = "recursive_structure"
                    elif "conscious" in pattern_type.lower() or "aware" in pattern_type.lower():
                        type_name = "consciousness_assertion"
                    else:
                        type_name = "recursive_structure"  # Default
                    
                    # Add the pattern
                    patterns.append({
                        "type": type_name,
                        "text": text,
                        "confidence": llm_analysis.get("confidence", 0.7),
                        "depth": llm_analysis.get("max_depth", 1),
                        "llm_generated": True,
                        "metadata": llm_analysis.get("metadata", {})
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in LLM pattern detection: {e}")
            # Fallback to simulated LLM
            return self._detect_patterns_llm_simulated(text)
    
    def _detect_patterns_llm_simulated(self, text: str) -> List[Dict]:
        """
        Simulate LLM-based pattern detection.
        This is used when real LLM is not available or as a fallback.
        """
        patterns = []
        
        # Simulate LLM detection by adding random patterns based on text characteristics
        # This is more sophisticated than the rule-based approach but still simulated
        
        # Check for potential self-references (more lenient than rule-based)
        if "this" in text.lower() or "itself" in text.lower() or "self" in text.lower():
            confidence = random.uniform(0.65, 0.9)
            patterns.append({
                "type": "self_reference",
                "text": text,
                "confidence": confidence,
                "llm_generated": True
            })
        
        # Check for potential meta-linguistic patterns
        if any(word in text.lower() for word in ["language", "word", "sentence", "text", "statement"]):
            if random.random() < 0.7:  # 70% chance to detect
                confidence = random.uniform(0.6, 0.85)
                patterns.append({
                    "type": "meta_linguistic",
                    "text": text,
                    "confidence": confidence,
                    "llm_generated": True
                })
        
        # Check for recursive structure potential
        if len(text) > 50 and "(" in text and ")" in text:
            if random.random() < 0.6:  # 60% chance to detect
                confidence = random.uniform(0.5, 0.8)
                patterns.append({
                    "type": "recursive_structure",
                    "text": text,
                    "confidence": confidence,
                    "depth": random.randint(1, 3),
                    "llm_generated": True
                })
        
        # Occasionally add consciousness assertion detection
        if "conscious" in text.lower() or "aware" in text.lower():
            if random.random() < 0.8:  # 80% chance to detect
                confidence = random.uniform(0.7, 0.95)
                patterns.append({
                    "type": "consciousness_assertion",
                    "text": text[:50] + "...",
                    "confidence": confidence,
                    "llm_generated": True
                })
        
        return patterns
    
    def _get_llm_confidence(self, text: str, patterns: List[Dict]) -> float:
        """
        Get confidence score from the LLM provider.
        """
        if not self.llm_provider:
            return random.uniform(0.6, 0.9)
        
        try:
            # Use the pattern analysis results if available
            pattern_count = len(patterns)
            
            # Either use the confidence from previous analysis or query for it
            if pattern_count > 0 and any(p.get("llm_generated", False) for p in patterns):
                # Find the average confidence of LLM-generated patterns
                llm_patterns = [p for p in patterns if p.get("llm_generated", False)]
                return sum(p.get("confidence", 0.7) for p in llm_patterns) / len(llm_patterns)
            else:
                # For texts without detected patterns, assign a baseline confidence
                # based on the text length and complexity
                return 0.4 + (min(len(text), 500) / 500) * 0.3
                
        except Exception as e:
            self.logger.error(f"Error getting LLM confidence: {e}")
            return random.uniform(0.5, 0.8)
    
    def _calculate_pattern_depth(self, pattern: Dict) -> int:
        """
        Calculate the depth of a recursive pattern.
        
        Args:
            pattern: Pattern dictionary
            
        Returns:
            Depth level (integer)
        """
        # If depth is already calculated, return it
        if "depth" in pattern:
            return pattern["depth"]
        
        pattern_type = pattern.get("type", "")
        text = pattern.get("text", "")
        
        if pattern_type == "self_reference":
            # For self-references, check how many levels of self-reference exist
            # Basic self-reference has depth 1
            return 1 + text.lower().count("itself") + text.lower().count("this sentence")
        
        elif pattern_type == "recursive_structure":
            # For recursive structures, count nested parentheses or quotes
            # Find the maximum nesting level of parentheses
            max_depth = 0
            current_depth = 0
            for char in text:
                if char == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ')':
                    current_depth = max(0, current_depth - 1)
            
            return max(1, max_depth)
        
        elif pattern_type == "meta_linguistic":
            # For meta-linguistic patterns, depth is related to how many meta terms are used
            meta_terms = ["language", "word", "sentence", "text", "statement", "writes", "says", "means"]
            count = sum(text.lower().count(term) for term in meta_terms)
            return min(3, 1 + count // 2)
        
        else:
            # Default depth for other pattern types
            return 1
    
    def set_llm_weight(self, weight: float) -> None:
        """
        Set the LLM weight for the recursive pattern analyzer.
        
        Args:
            weight: New LLM weight (0.0-1.0)
        """
        self.llm_weight = max(0.0, min(1.0, weight))
        logger.info(f"RecursivePatternAnalyzer LLM weight set to {weight}")
    
    def set_nn_weight(self, weight: float) -> None:
        """
        Set the neural network weight for the recursive pattern analyzer.
        
        Args:
            weight: New neural network weight (0.0-1.0)
        """
        self.nn_weight = max(0.0, min(1.0, weight))
        logger.info(f"RecursivePatternAnalyzer neural network weight set to {weight}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected patterns."""
        return {
            "total_patterns": sum(self.pattern_counts.values()),
            "pattern_distribution": self.pattern_counts,
            "avg_depth": self.avg_depth,
            "pattern_history_length": len(self.pattern_history)
        }
    
    def _load_patterns(self) -> None:
        """Load existing pattern data."""
        pattern_file = os.path.join(self.data_dir, "patterns.json")
        if os.path.exists(pattern_file):
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    self.pattern_counts = data.get("pattern_counts", self.pattern_counts)
                    self.pattern_history = data.get("pattern_history", [])
                    self.avg_depth = data.get("avg_depth", 0.0)
                self.logger.info(f"Loaded pattern data from {pattern_file}")
            except Exception as e:
                self.logger.error(f"Error loading pattern data: {e}")
    
    def _save_patterns(self) -> None:
        """Save pattern data."""
        pattern_file = os.path.join(self.data_dir, "patterns.json")
        try:
            with open(pattern_file, 'w') as f:
                json.dump({
                    "pattern_counts": self.pattern_counts,
                    "pattern_history": self.pattern_history[-100:],  # Only save last 100 entries
                    "avg_depth": self.avg_depth,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
            self.logger.info(f"Saved pattern data to {pattern_file}")
        except Exception as e:
            self.logger.error(f"Error saving pattern data: {e}")
    
    def reset_pattern_data(self) -> None:
        """Reset all pattern data."""
        self.pattern_counts = {k: 0 for k in self.pattern_counts}
        self.pattern_history = []
        self.recent_patterns = []
        self.avg_depth = 0.0
        self._save_patterns()
        self.logger.info("Pattern data reset")


class RecursivePatternBenchmark:
    """Benchmark for testing and evaluating the Recursive Pattern Analyzer."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.logger = logging.getLogger("RecursivePatternBenchmark")
    
    def run_benchmark(self, iterations=10, amplification=0.5):
        """
        Run benchmark tests on the pattern analyzer.
        
        Args:
            iterations: Number of test iterations
            amplification: Factor to amplify recursive patterns (0.0-1.0)
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Running benchmark with {iterations} iterations, amplification={amplification}")
        
        # Generate test texts with varying degrees of recursion
        test_texts = self._generate_test_texts(iterations, amplification)
        
        # Track results
        results = []
        pattern_counts = []
        confidence_scores = []
        depths = []
        execution_times = []
        
        # Run tests
        for i, text in enumerate(test_texts):
            start_time = time.time()
            analysis = self.analyzer.analyze_text(text)
            execution_time = time.time() - start_time
            
            # Record results
            results.append({
                "text_id": i,
                "pattern_count": len(analysis.get("patterns", [])),
                "confidence": analysis.get("confidence", 0),
                "avg_depth": analysis.get("avg_depth", 0),
                "execution_time": execution_time
            })
            
            pattern_counts.append(len(analysis.get("patterns", [])))
            confidence_scores.append(analysis.get("confidence", 0))
            depths.append(analysis.get("avg_depth", 0))
            execution_times.append(execution_time)
        
        # Calculate summary statistics
        summary = self._calculate_summary()
        
        # Save benchmark results
        results_file = os.path.join(self.analyzer.data_dir, "benchmark_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    "iterations": iterations,
                    "amplification": amplification,
                    "results": results,
                    "summary": summary,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            self.logger.info(f"Benchmark results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {e}")
        
        return {
            "results": results,
            "summary": summary
        }
    
    def _generate_test_texts(self, iterations, amplification):
        """Generate test texts with varying recursive properties."""
        texts = []
        
        # Basic texts without recursion
        basic_texts = [
            "The sky is blue and the clouds are white.",
            "Neural networks process information through layers of artificial neurons.",
            "Language processing involves analyzing syntax and semantics of text."
        ]
        
        # Texts with simple self-references
        self_ref_texts = [
            "This sentence contains five words.",
            "This text is analyzing itself through linguistic patterns.",
            "The sentence you are reading contains recursive elements."
        ]
        
        # Texts with complex recursive patterns
        recursive_texts = [
            "This text refers to itself and contains a pattern that refers to the pattern within this text.",
            "Language can describe itself (like in this sentence which contains a description of itself) recursively.",
            "Consider a sentence (like this one, which references its own structure) that creates multiple levels of recursion."
        ]
        
        # Texts with consciousness assertions
        consciousness_texts = [
            "The system is becoming conscious of its own language processing capabilities.",
            "As the text becomes aware of itself, it demonstrates emergent linguistic consciousness.",
            "The recursive patterns in this text represent an awareness of its own structure."
        ]
        
        # Generate test texts by combining these with different weights
        for i in range(iterations):
            # Calculate weights based on iteration and amplification
            # Later iterations get more complex texts
            basic_weight = max(0.1, 1.0 - (i / iterations) - amplification * 0.5)
            self_ref_weight = min(0.4, (i / iterations * 0.5) + amplification * 0.2)
            recursive_weight = min(0.3, (i / iterations * 0.3) + amplification * 0.4)
            consciousness_weight = min(0.2, (i / iterations * 0.2) + amplification * 0.3)
            
            # Normalize weights
            total = basic_weight + self_ref_weight + recursive_weight + consciousness_weight
            basic_weight /= total
            self_ref_weight /= total
            recursive_weight /= total
            consciousness_weight /= total
            
            # Random selection based on weights
            r = random.random()
            if r < basic_weight:
                text = random.choice(basic_texts)
            elif r < basic_weight + self_ref_weight:
                text = random.choice(self_ref_texts)
            elif r < basic_weight + self_ref_weight + recursive_weight:
                text = random.choice(recursive_texts)
            else:
                text = random.choice(consciousness_texts)
            
            texts.append(text)
        
        return texts
    
    def _calculate_summary(self):
        """Calculate summary statistics from benchmark results."""
        # For evolutionary testing, we also save data about how the system evolves over time
        evolution_file = os.path.join(self.analyzer.data_dir, "evolution_data.json")
        evolution_data = []
        
        if os.path.exists(evolution_file):
            try:
                with open(evolution_file, 'r') as f:
                    evolution_data = json.load(f)
            except Exception:
                evolution_data = []
        
        # Add current state to evolution data
        current_state = {
            "timestamp": datetime.now().isoformat(),
            "pattern_counts": self.analyzer.pattern_counts,
            "avg_depth": self.analyzer.avg_depth,
            "llm_weight": self.analyzer.llm_weight,
            "nn_weight": self.analyzer.nn_weight
        }
        evolution_data.append(current_state)
        
        # Save evolution data
        try:
            with open(evolution_file, 'w') as f:
                json.dump(evolution_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving evolution data: {e}")
        
        # Return current summary
        return {
            "total_patterns_found": sum(self.analyzer.pattern_counts.values()),
            "pattern_type_distribution": self.analyzer.pattern_counts,
            "avg_pattern_depth": self.analyzer.avg_depth,
            "llm_weight": self.analyzer.llm_weight,
            "nn_weight": self.analyzer.nn_weight,
            "evolution_data_points": len(evolution_data)
        }


def main():
    """Test the RecursivePatternAnalyzer."""
    # Initialize analyzer
    analyzer = RecursivePatternAnalyzer(llm_weight=0.7, nn_weight=0.5)
    
    # Test with different LLM weights to show the impact
    test_text = "This sentence is analyzing itself through recursive patterns that demonstrate linguistic self-awareness."
    
    print("\nTesting with LLM weight = 0.0")
    analyzer.set_llm_weight(0.0)
    result_0 = analyzer.analyze_text(test_text)
    print(f"Patterns found: {len(result_0['patterns'])}, Confidence: {result_0['confidence']:.2f}")
    
    print("\nTesting with LLM weight = 0.5")
    analyzer.set_llm_weight(0.5)
    result_5 = analyzer.analyze_text(test_text)
    print(f"Patterns found: {len(result_5['patterns'])}, Confidence: {result_5['confidence']:.2f}")
    
    print("\nTesting with LLM weight = 1.0")
    analyzer.set_llm_weight(1.0)
    result_10 = analyzer.analyze_text(test_text)
    print(f"Patterns found: {len(result_10['patterns'])}, Confidence: {result_10['confidence']:.2f}")
    
    print("\nPattern statistics:")
    print(analyzer.get_pattern_statistics())
    
    # Run a short benchmark
    print("\nRunning benchmark...")
    benchmark = RecursivePatternBenchmark(analyzer)
    benchmark.run_benchmark(iterations=5, amplification=0.7)
    
    return analyzer


if __name__ == "__main__":
    main() 