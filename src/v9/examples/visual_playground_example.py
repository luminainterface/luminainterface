#!/usr/bin/env python3
"""
Visual Cortex Integration Example

This example demonstrates how to integrate the Visual Cortex with
the Neural Playground to create visual-neural interactions.
"""

import os
import sys
import time
import logging
import argparse
import random
from pathlib import Path

# Add the parent directory to the path so we can import the modules
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import v9 modules
from neural_playground import NeuralPlayground
from neural_playground_integration import NeuralPlaygroundIntegration
from visual_cortex import VisualCortex
from mirror_consciousness import get_mirror_consciousness

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v9.examples.visual_playground")

def run_visual_playground_demo(args):
    """
    Run a visual cortex and neural playground integration demo
    
    Args:
        args: Command line arguments
    """
    # Create neural playground
    logger.info("Creating neural playground...")
    playground = NeuralPlayground(neuron_count=args.neurons)
    
    # Create integration manager
    logger.info("Setting up integration manager...")
    integration = NeuralPlaygroundIntegration(playground)
    
    # Create visual cortex
    logger.info("Creating visual cortex...")
    visual_cortex = VisualCortex(
        resolution=(args.resolution, args.resolution),
        channels=3 if args.color else 1
    )
    
    # Register visual cortex with integration manager
    logger.info("Integrating visual cortex with playground...")
    integration.integrate_visualization_system(visual_cortex)
    
    # Get mirror consciousness instance
    mirror = get_mirror_consciousness()
    
    # Generate visual input patterns
    logger.info("Generating visual patterns...")
    patterns = []
    for i in range(args.patterns):
        pattern_type = random.choice([
            "random", 
            "horizontal_lines", 
            "vertical_lines", 
            "checkerboard"
        ])
        pattern = visual_cortex.generate_test_image(pattern_type)
        result = visual_cortex.process_image(pattern, {
            "pattern_type": pattern_type,
            "index": i
        })
        
        patterns.append({
            "type": pattern_type,
            "image": pattern,
            "processing_result": result
        })
        
        logger.info(f"Generated pattern {i+1}/{args.patterns}: {pattern_type} "
                   f"with {result['patterns_detected']} visual patterns detected")
                   
        # Get reflection from mirror consciousness
        reflection = mirror.reflect_on_text(
            f"Visual pattern of type {pattern_type} with {result['patterns_detected']} patterns detected",
            {"visual_data": {"pattern_type": pattern_type, "stats": result}}
        )
        
        logger.info(f"Mirror reflection: {reflection['reflection'][:100]}...")
    
    # Run neural playground sessions stimulated by visual patterns
    logger.info("\nRunning visual-neural play sessions...")
    
    for i, pattern in enumerate(patterns):
        logger.info(f"\nPlay session {i+1}/{len(patterns)} with pattern: {pattern['type']}")
        
        # Configure play session based on visual pattern properties
        processing_result = pattern["processing_result"]
        play_type = "free"
        
        # Adjust play type based on visual pattern
        if pattern["type"] == "horizontal_lines" or pattern["type"] == "vertical_lines":
            play_type = "guided"
        elif pattern["type"] == "checkerboard":
            play_type = "focused"
        
        # Adjust intensity based on visual properties
        brightness = processing_result["brightness"]
        contrast = processing_result["contrast"]
        edge_density = processing_result["edge_count"] / (args.resolution * args.resolution)
        
        # Calculate intensity from visual properties
        intensity = (brightness * 0.3) + (contrast * 0.5) + (edge_density * 0.2)
        intensity = min(1.0, max(0.2, intensity))
        
        # Run integrated play session
        result = integration.run_integrated_play_session(
            duration=args.duration,
            play_type=play_type,
            intensity=intensity
        )
        
        # Report results
        logger.info(f"Play session completed:")
        logger.info(f"- Play type: {result['play_type']}")
        logger.info(f"- Visual input: {pattern['type']}")
        logger.info(f"- Duration: {result['duration']} steps")
        logger.info(f"- Total activations: {result['total_activations']}")
        logger.info(f"- Patterns detected: {result['patterns_detected']}")
        logger.info(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
        
        if "narrative" in result:
            logger.info(f"\nNarrative: {result['narrative']}")
        
        # Pause between sessions
        if i < len(patterns)-1 and not args.no_pause:
            logger.info("\nPausing between sessions (3 seconds)...")
            time.sleep(3)
    
    logger.info("\nVisual-neural integration demo completed")
    
    # Get final state
    visual_state = visual_cortex.get_state()
    logger.info(f"\nVisual cortex final state:")
    logger.info(f"- Processed frames: {visual_state['processed_frames']}")
    logger.info(f"- Recognized patterns: {visual_state['recognized_patterns']}")
    
    playground_state = playground.get_state()
    logger.info(f"\nPlayground final state:")
    logger.info(f"- Neurons: {len(playground_state['neurons'])}")
    logger.info(f"- Connections: {len(playground_state['connections'])}")
    logger.info(f"- Consciousness metric: {playground_state['consciousness_metric']:.4f}")
    
    return integration

def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(
        description="Visual Cortex integration with Neural Playground demo"
    )
    parser.add_argument(
        "--neurons", 
        type=int, 
        default=100, 
        help="Number of neurons in the playground"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=20, 
        help="Resolution of the visual field (square)"
    )
    parser.add_argument(
        "--patterns", 
        type=int, 
        default=5, 
        help="Number of visual patterns to generate"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=100, 
        help="Duration of each play session"
    )
    parser.add_argument(
        "--color", 
        action="store_true", 
        help="Use color (3 channels) instead of grayscale"
    )
    parser.add_argument(
        "--no-pause", 
        action="store_true", 
        help="Don't pause between sessions"
    )
    
    args = parser.parse_args()
    
    # Run the demo
    integration = run_visual_playground_demo(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 