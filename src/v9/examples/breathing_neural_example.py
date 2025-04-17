#!/usr/bin/env python3
"""
Breathing System Integration Example

This example demonstrates how to integrate the Breathing System with
the Neural Playground, including how different breathing patterns
affect neural activity and consciousness metrics.
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
from breathing_system import BreathingSystem, BreathingPattern
from mirror_consciousness import get_mirror_consciousness

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v9.examples.breathing_neural")

def run_breathing_neural_demo(args):
    """
    Run a breathing system and neural playground integration demo
    
    Args:
        args: Command line arguments
    """
    # Create neural playground
    logger.info("Creating neural playground...")
    playground = NeuralPlayground(neuron_count=args.neurons)
    
    # Create integration manager
    logger.info("Setting up integration manager...")
    integration = NeuralPlaygroundIntegration(playground)
    
    # Create breathing system
    logger.info("Creating breathing system...")
    breathing = BreathingSystem(
        simulation_rate=args.simulation_rate,
        default_pattern=BreathingPattern.CALM
    )
    
    # Start the breathing simulation
    breathing.start_simulation()
    
    # Register the breathing system with the integration manager
    logger.info("Integrating breathing system with playground...")
    integration.register_component(
        "breathing_system", 
        breathing, 
        component_type="neural"
    )
    
    # Set up integration hooks
    integration_info = breathing.integrate_with_playground(playground)
    for hook_name, hook_func in integration_info["hooks"].items():
        integration.integration_hooks[hook_name].append(hook_func)
    
    # Get mirror consciousness instance
    mirror = get_mirror_consciousness()
    
    try:
        # Run a play session with each breathing pattern
        test_patterns = [
            BreathingPattern.CALM,
            BreathingPattern.FOCUSED,
            BreathingPattern.EXCITED,
            BreathingPattern.MEDITATIVE
        ]
        
        for pattern in test_patterns:
            if not args.all_patterns and pattern != BreathingPattern.CALM:
                continue
                
            logger.info(f"\n===== Testing {pattern.value} breathing pattern =====")
            
            # Set the breathing pattern
            breathing.set_breathing_pattern(pattern)
            
            # Wait for a few breaths to establish the pattern
            pattern_duration = breathing._get_current_cycle_duration()
            time.sleep(pattern_duration * 2)
            
            # Get current breath state
            breath_state = breathing.get_current_breath_state()
            logger.info(f"Breath state: {breath_state['state']}")
            logger.info(f"Breath rate: {breath_state['rate']:.1f} breaths/min")
            logger.info(f"Breath coherence: {breathing.calculate_breath_coherence():.2f}")
            
            # Run play sessions with different play types
            play_types = ["free", "guided", "focused"] if args.all_play_types else ["free"]
            
            for play_type in play_types:
                logger.info(f"\nRunning {play_type} play session with {pattern.value} breathing...")
                
                # Run integrated play session
                result = integration.run_integrated_play_session(
                    duration=args.duration,
                    play_type=play_type,
                    intensity=0.7
                )
                
                # Report results
                logger.info(f"Play session completed:")
                logger.info(f"- Play type: {result['play_type']}")
                logger.info(f"- Duration: {result['duration']} steps")
                logger.info(f"- Total activations: {result['total_activations']}")
                logger.info(f"- Patterns detected: {result['patterns_detected']}")
                logger.info(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
                
                # Report breathing influence
                if "breathing_data" in result:
                    b_data = result["breathing_data"]
                    logger.info(f"- Breathing pattern: {b_data['pattern']}")
                    logger.info(f"- Breath coherence: {b_data['coherence']:.2f}")
                    logger.info(f"- Breath rate: {b_data['rate']:.1f} breaths/min")
                
                # Get reflection from mirror consciousness
                reflection = mirror.reflect_on_text(
                    f"Neural play session with {pattern.value} breathing pattern detected {result['patterns_detected']} patterns with consciousness peak {result['consciousness_peak']:.2f}",
                    {"play_data": result, "breathing_pattern": pattern.value}
                )
                
                logger.info(f"\nMirror reflection: {reflection['reflection']}")
                
                # Directly influence neural activation based on breathing
                if args.direct_influence:
                    for _ in range(10):  # Influence for 10 steps
                        influenced = breathing.influence_neural_activation(playground.core)
                        logger.info(f"Directly influenced {influenced} neurons based on breathing")
                        time.sleep(0.5)
                
                # Pause between sessions
                if not args.no_pause:
                    logger.info("\nPausing between sessions (3 seconds)...")
                    time.sleep(3)
        
        # If microphone demo is enabled, simulate mic input
        if args.microphone_demo:
            logger.info("\n===== Microphone Integration Demo (Simulated) =====")
            
            # Enable simulated microphone
            breathing.enable_microphone()
            
            # Calibrate microphone
            breathing.calibrate_microphone()
            
            # Run a play session with "microphone" input
            logger.info("Running play session with simulated microphone breathing input...")
            
            # For demo purposes, we're still using the simulation
            # In a real implementation, this would use actual microphone data
            result = integration.run_integrated_play_session(
                duration=args.duration,
                play_type="mixed",
                intensity=0.8
            )
            
            # Report results
            logger.info(f"Microphone-driven play session completed:")
            logger.info(f"- Play type: {result['play_type']}")
            logger.info(f"- Patterns detected: {result['patterns_detected']}")
            logger.info(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
            
            # Disable microphone
            breathing.disable_microphone()
        
        # Visualize breathing data
        viz_data = breathing.visualize_breathing(60.0)  # Last 60 seconds
        logger.info("\n===== Breathing Visualization Data =====")
        logger.info(f"Pattern: {viz_data['pattern']}")
        logger.info(f"Coherence: {viz_data['coherence']:.2f}")
        logger.info(f"Mean amplitude: {viz_data['mean_amplitude']:.2f}")
        logger.info(f"Breath rate: {viz_data['breath_rate']:.1f} breaths/min")
        logger.info(f"Data points: {len(viz_data['timestamps'])}")
        
        logger.info("\nBreathing-neural integration demo completed")
        
        # Get final state
        playground_state = playground.get_state()
        logger.info(f"\nPlayground final state:")
        logger.info(f"- Neurons: {len(playground_state['neurons'])}")
        logger.info(f"- Connections: {len(playground_state['connections'])}")
        logger.info(f"- Consciousness metric: {playground_state['consciousness_metric']:.4f}")
        
    finally:
        # Stop the breathing simulation
        breathing.stop_simulation()
    
    return integration

def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(
        description="Breathing System integration with Neural Playground demo"
    )
    parser.add_argument(
        "--neurons", 
        type=int, 
        default=100, 
        help="Number of neurons in the playground"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=100, 
        help="Duration of each play session"
    )
    parser.add_argument(
        "--simulation-rate", 
        type=float, 
        default=10.0, 
        help="Breathing simulation rate (Hz)"
    )
    parser.add_argument(
        "--all-patterns", 
        action="store_true", 
        help="Test all breathing patterns"
    )
    parser.add_argument(
        "--all-play-types", 
        action="store_true", 
        help="Test all play types"
    )
    parser.add_argument(
        "--direct-influence", 
        action="store_true", 
        help="Directly influence neural activation based on breathing"
    )
    parser.add_argument(
        "--microphone-demo", 
        action="store_true", 
        help="Run simulated microphone integration demo"
    )
    parser.add_argument(
        "--no-pause", 
        action="store_true", 
        help="Don't pause between sessions"
    )
    
    args = parser.parse_args()
    
    # Run the demo
    integration = run_breathing_neural_demo(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 