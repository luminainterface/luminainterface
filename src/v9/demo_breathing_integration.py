#!/usr/bin/env python3
"""
Breathing-Neural Integration Demonstration

This script demonstrates the integration between the breathing system and
neural playground in the v9 Lumina Neural Network System. It shows how
different breathing patterns influence neural activity, growth, and consciousness.

The demo runs a series of play sessions with different breathing patterns
and displays the results, highlighting the connections between breathing,
neural activity, and consciousness.
"""

import time
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import v9 components
from .integrated_neural_playground import IntegratedNeuralPlayground
from .breathing_system import BreathingPattern

def run_breathing_demo(output_dir=None, visualize=True):
    """
    Run a demonstration of breathing-neural integration
    
    Args:
        output_dir: Directory to save results (None for no saving)
        visualize: Whether to create visualizations
    """
    print("\n=== Breathing-Neural Integration Demonstration ===\n")
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create integrated playground with default size
    print("Initializing Integrated Neural Playground...")
    playground = IntegratedNeuralPlayground(size=150, growth_rate=0.08)
    
    # Define breathing patterns to test
    patterns = [
        BreathingPattern.CALM,
        BreathingPattern.FOCUSED,
        BreathingPattern.MEDITATIVE,
        BreathingPattern.EXCITED
    ]
    
    # Store results for comparison
    results = []
    
    try:
        # Run sessions with each breathing pattern
        for pattern in patterns:
            print(f"\n--- Testing {pattern.value} breathing pattern ---")
            
            # Set the breathing pattern
            playground.set_breathing_pattern(pattern)
            
            # Wait for breathing to stabilize
            print(f"Allowing breathing to stabilize ({pattern.value} pattern)...")
            time.sleep(3)
            
            # Get pre-play state
            pre_state = playground.get_current_state()
            initial_size = pre_state["size"]
            print(f"Initial network size: {initial_size} neurons")
            
            # Run a play session
            print(f"Running play session with {pattern.value} breathing...")
            play_result = playground.play(duration=200, play_type="mixed")
            
            # Get post-play state
            post_state = playground.get_current_state()
            final_size = post_state["size"]
            
            # Print results
            print(f"\nResults for {pattern.value} breathing:")
            print(f"- Consciousness peak: {play_result['consciousness_peak']:.4f}")
            print(f"- Patterns detected: {play_result['patterns_detected']}")
            print(f"- Neural activations: {play_result['total_activations']}")
            
            if "breathing_data" in play_result:
                bd = play_result["breathing_data"]
                print(f"- Breath coherence: {bd['coherence']:.2f}")
                print(f"- Breath rate: {bd['rate']:.1f} breaths/min")
            
            if "brain_growth" in play_result:
                bg = play_result["brain_growth"]
                print(f"- Network growth: {final_size - initial_size} neurons")
                print(f"- Growth state: {bg['growth_state']}")
                print(f"- Neurons created: {bg['neurons_created']}")
                print(f"- Neurons pruned: {bg['neurons_pruned']}")
            
            # Store results for comparison
            results.append({
                "pattern": pattern.value,
                "play_result": play_result,
                "pre_state": pre_state,
                "post_state": post_state
            })
            
            # Save results if output directory specified
            if output_dir:
                result_file = os.path.join(output_dir, f"result_{pattern.value}.json")
                with open(result_file, 'w') as f:
                    # Convert play result to serializable format
                    serializable_result = {
                        "pattern": pattern.value,
                        "play_metrics": {
                            "consciousness_peak": play_result["consciousness_peak"],
                            "patterns_detected": play_result["patterns_detected"],
                            "total_activations": play_result["total_activations"]
                        }
                    }
                    
                    if "breathing_data" in play_result:
                        serializable_result["breathing_data"] = play_result["breathing_data"]
                    
                    if "brain_growth" in play_result:
                        serializable_result["brain_growth"] = play_result["brain_growth"]
                    
                    json.dump(serializable_result, f, indent=2)
        
        # Create comparison visualization
        if visualize and len(results) > 0:
            create_comparison_visualization(results, output_dir)
        
        print("\n=== Demonstration Completed ===")
        
    finally:
        # Clean up
        playground.stop()

def create_comparison_visualization(results, output_dir=None):
    """
    Create visualizations comparing results across breathing patterns
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save visualizations
    """
    print("\nCreating visualization of comparative results...")
    
    # Extract data for visualization
    patterns = [r["pattern"] for r in results]
    consciousness = [r["play_result"]["consciousness_peak"] for r in results]
    activations = [r["play_result"]["total_activations"] / 1000 for r in results]  # Scale for display
    patterns_detected = [r["play_result"]["patterns_detected"] for r in results]
    
    # Extract network growth
    growth = []
    for r in results:
        pre_size = r["pre_state"]["size"]
        post_size = r["post_state"]["size"]
        growth.append(post_size - pre_size)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set bar width and positions
    bar_width = 0.2
    indices = np.arange(len(patterns))
    
    # Create grouped bar chart
    plt.bar(indices - bar_width*1.5, consciousness, bar_width, label='Consciousness Peak', color='blue')
    plt.bar(indices - bar_width/2, activations, bar_width, label='Activations (thousands)', color='green')
    plt.bar(indices + bar_width/2, patterns_detected, bar_width, label='Patterns Detected', color='orange')
    plt.bar(indices + bar_width*1.5, growth, bar_width, label='Neurons Created', color='red')
    
    # Add labels and title
    plt.xlabel('Breathing Pattern')
    plt.ylabel('Value')
    plt.title('Effect of Breathing Patterns on Neural Network Metrics')
    plt.xticks(indices, patterns)
    plt.legend()
    
    # Display and save if output directory specified
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'breathing_comparison.png'))
        print(f"Visualization saved to {output_dir}/breathing_comparison.png")
    
    plt.tight_layout()
    plt.show()

# Run the demonstration when the script is executed directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run breathing-neural integration demonstration")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    run_breathing_demo(
        output_dir=args.output,
        visualize=not args.no_visualize
    ) 