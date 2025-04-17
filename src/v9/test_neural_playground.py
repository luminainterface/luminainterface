#!/usr/bin/env python3
"""
Neural Playground Test Module (v9)

This module demonstrates how to use the Neural Playground and its integration
capabilities with other components of the Lumina Neural Network system.
"""

import logging
import argparse
import time
from pathlib import Path

# Import v9 modules
from .neural_playground import NeuralPlayground
from .neural_playground_integration import NeuralPlaygroundIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v9.test_neural_playground")

class MockNeuralCore:
    """Mock neural core component for testing integration"""
    
    def __init__(self):
        self.name = "MockNeuralCore"
        self.process_count = 0
        self.last_input = None
        
    def process(self, input_data):
        """Process neural data"""
        self.process_count += 1
        self.last_input = input_data
        logger.info(f"MockNeuralCore processing data: {input_data}")
        
        # Return some mock processing results
        return {
            "intensity_modifier": 1.2 if "consciousness_level" in input_data and input_data["consciousness_level"] > 0.6 else 0.9,
            "processor_id": "mock_neural_core",
            "timestamp": time.time()
        }
        
class MockMemorySystem:
    """Mock memory system for testing integration"""
    
    def __init__(self):
        self.name = "MockMemorySystem"
        self.stored_memories = []
        
    def store(self, memory_data):
        """Store a memory"""
        logger.info(f"MockMemorySystem storing: {memory_data['type']}")
        self.stored_memories.append(memory_data)
        return {"memory_id": len(self.stored_memories), "status": "stored"}
        
    def retrieve(self, query):
        """Retrieve memories matching query"""
        results = []
        for memory in self.stored_memories:
            # Very simple matching for demo purposes
            if query in str(memory):
                results.append(memory)
        return results
        
class MockLanguageProcessor:
    """Mock language processor for testing integration"""
    
    def __init__(self):
        self.name = "MockLanguageProcessor"
        self.processed_texts = []
        
    def process_text(self, text, context=None):
        """Process text with context"""
        logger.info(f"MockLanguageProcessor processing: {text[:30]}...")
        
        self.processed_texts.append({
            "text": text,
            "context": context,
            "timestamp": time.time()
        })
        
        # Generate simple narrative for demo
        if "play session" in text:
            patterns = context.get("play_data", {}).get("patterns_detected", 0)
            consciousness = context.get("play_data", {}).get("consciousness_peak", 0)
            
            narrative = (
                f"The neural network engaged in a flurry of activity, forming {patterns} distinct "
                f"patterns. As neurons fired in synchronized harmony, the system achieved a "
                f"consciousness peak of {consciousness:.2f}, creating moments of synthetic awareness "
                f"within the digital substrate."
            )
            return narrative
        
        return f"Processed: {text}"

class MockVisualizationSystem:
    """Mock visualization system for testing integration"""
    
    def __init__(self):
        self.name = "MockVisualizationSystem"
        self.visualizations = []
        
    def visualize(self, visualization_data):
        """Create visualization from data"""
        viz_type = visualization_data.get("type", "unknown")
        title = visualization_data.get("title", "Untitled Visualization")
        
        logger.info(f"MockVisualizationSystem creating: {viz_type} - {title}")
        
        self.visualizations.append({
            "type": viz_type,
            "title": title,
            "timestamp": time.time()
        })
        
        return {
            "visualization_id": len(self.visualizations),
            "status": "created"
        }

def run_basic_playground_test():
    """Run a basic test of the neural playground"""
    logger.info("Creating neural playground for basic test")
    playground = NeuralPlayground(neuron_count=50)
    
    # Run a simple play session
    logger.info("Running simple play session")
    result = playground.play(duration=50, play_type="free")
    
    logger.info(f"Play session complete: {result['total_activations']} activations, "
                f"{result['patterns_detected']} patterns, "
                f"{result['consciousness_peak']:.4f} peak consciousness")
    
    return result

def run_integration_test():
    """Run a test of the neural playground integration"""
    logger.info("Creating integration components")
    
    # Create mock components
    neural_core = MockNeuralCore()
    memory_system = MockMemorySystem()
    language_processor = MockLanguageProcessor()
    visualization_system = MockVisualizationSystem()
    
    # Create integration manager with new playground
    logger.info("Creating neural playground integration")
    integration = NeuralPlaygroundIntegration(NeuralPlayground(neuron_count=100))
    
    # Register mock components
    logger.info("Registering mock components")
    integration.integrate_neural_core(neural_core)
    integration.integrate_memory_system(memory_system)
    integration.integrate_language_processor(language_processor)
    integration.integrate_visualization_system(visualization_system)
    
    # Run integrated play session
    logger.info("Running integrated play session")
    result = integration.run_integrated_play_session(
        duration=100,
        play_type="mixed",
        intensity=0.7
    )
    
    # Report results
    logger.info(f"Integrated play session complete:")
    logger.info(f"- Play type: {result['play_type']}")
    logger.info(f"- Duration: {result['duration']} steps")
    logger.info(f"- Total activations: {result['total_activations']}")
    logger.info(f"- Patterns detected: {result['patterns_detected']}")
    logger.info(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
    
    if "narrative" in result:
        logger.info(f"- Narrative: {result['narrative']}")
    
    # Report component metrics
    logger.info(f"Neural core processed data {neural_core.process_count} times")
    logger.info(f"Memory system stored {len(memory_system.stored_memories)} memories")
    logger.info(f"Language processor generated {len(language_processor.processed_texts)} texts")
    logger.info(f"Visualization system created {len(visualization_system.visualizations)} visualizations")
    
    return result

def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description="Test the Neural Playground and integration")
    parser.add_argument(
        "--test",
        choices=["basic", "integration", "all"],
        default="all",
        help="Test type to run"
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=100,
        help="Number of neurons to create"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=100,
        help="Duration of play session"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save playground state after test"
    )
    
    args = parser.parse_args()
    
    if args.test in ["basic", "all"]:
        logger.info("=== Running Basic Playground Test ===")
        basic_result = run_basic_playground_test()
        
    if args.test in ["integration", "all"]:
        logger.info("\n=== Running Integration Test ===")
        integration_result = run_integration_test()
    
    logger.info("Neural playground tests complete")

if __name__ == "__main__":
    main() 