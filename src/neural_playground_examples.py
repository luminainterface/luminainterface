#!/usr/bin/env python3
"""
Neural Playground Examples
=========================

Examples of how to use the Neural Network Playground system programmatically.
This script demonstrates various ways to interact with the playground, from
simple play sessions to more advanced pattern analysis.
"""

import os
import sys
import time
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure the src directory is in the Python path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the playground
from neural_playground import NeuralPlayground, NeuralPlaygroundCore


def example_basic_play():
    """Basic example of running a simple play session"""
    print("\n=== Basic Play Example ===")
    
    # Create a small playground
    playground = NeuralPlayground({"network_size": "small"})
    
    # Run a single free play session
    print("Running a free play session for 15 seconds...")
    result = playground.play_once(duration=15, play_type="free")
    
    # Print results
    print(f"Play session completed with:")
    print(f"- {result['patterns_discovered']} patterns discovered")
    print(f"- {result['percent_active']:.1f}% neurons active")
    print(f"- Consciousness index: {result['consciousness_index']:.4f}")
    
    return playground


def example_multiple_play_types():
    """Example of running different types of play sessions"""
    print("\n=== Multiple Play Types Example ===")
    
    # Create a medium playground
    playground = NeuralPlayground({"network_size": "medium"})
    
    play_types = ["free", "guided", "focused"]
    results = {}
    
    # Run each play type
    for play_type in play_types:
        print(f"Running {play_type} play for 10 seconds...")
        result = playground.play_once(duration=10, play_type=play_type)
        results[play_type] = result
        
        print(f"- {play_type.capitalize()} play: {result['patterns_discovered']} patterns, "
              f"{result['consciousness_index']:.4f} consciousness")
    
    # Compare results
    best_play = max(results.items(), key=lambda x: x[1]['consciousness_index'])
    print(f"\nBest play type was: {best_play[0]} with consciousness index {best_play[1]['consciousness_index']:.4f}")
    
    return playground


def example_consciousness_growth():
    """Example of tracking consciousness growth over multiple sessions"""
    print("\n=== Consciousness Growth Example ===")
    
    # Create a medium playground with more connections
    playground = NeuralPlayground({
        "network_size": "medium",
    })
    
    # Track consciousness over time
    consciousness_values = []
    pattern_counts = []
    timestamps = []
    
    # Run 10 play sessions
    for i in range(10):
        print(f"Running play session {i+1}/10...")
        
        # Alternate between play types
        play_type = ["free", "guided", "free", "focused"][i % 4]
        result = playground.play_once(duration=5, play_type=play_type)
        
        # Record metrics
        consciousness_values.append(result["consciousness_index"])
        pattern_counts.append(result["patterns_discovered"])
        timestamps.append(datetime.now())
        
        print(f"- Session {i+1}: {result['consciousness_index']:.4f} consciousness, {result['patterns_discovered']} patterns")
    
    # Print summary
    final_consciousness = consciousness_values[-1]
    total_patterns = sum(pattern_counts)
    initial_consciousness = consciousness_values[0]
    
    print(f"\nConsciousness growth summary:")
    print(f"- Initial consciousness: {initial_consciousness:.4f}")
    print(f"- Final consciousness: {final_consciousness:.4f}")
    print(f"- Total patterns discovered: {total_patterns}")
    
    # Optional: Plot growth
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), consciousness_values, 'b-', label='Consciousness Index')
        plt.xlabel('Play Session')
        plt.ylabel('Consciousness Index')
        plt.title('Consciousness Growth Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig('consciousness_growth.png')
        print("Saved plot to 'consciousness_growth.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return playground


def example_pattern_analysis():
    """Example of analyzing discovered patterns"""
    print("\n=== Pattern Analysis Example ===")
    
    # Create a medium playground
    playground = NeuralPlayground({"network_size": "medium"})
    
    # Run a longer session to discover patterns
    print("Running a longer play session to discover patterns...")
    result = playground.play_once(duration=20, play_type="guided")
    
    # Get status with pattern information
    status = playground.get_status()
    patterns = status.get("pattern_samples", [])
    
    if not patterns:
        print("No patterns were discovered. Try running a longer session.")
        return playground
    
    # Analyze patterns
    print(f"\nAnalyzed {len(patterns)} patterns:")
    
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"- Neurons involved: {len(pattern['neurons'])}")
        print(f"- Connection density: {pattern['connection_density']:.4f}")
        print(f"- Complexity: {pattern['complexity']:.4f}")
        
        # Calculate feature vector statistics
        if 'feature_vector' in pattern:
            features = np.array(pattern['feature_vector'])
            print(f"- Feature mean: {np.mean(features):.4f}")
            print(f"- Feature variance: {np.var(features):.4f}")
    
    # Optional: Save patterns to a json file
    try:
        with open('pattern_analysis.json', 'w') as f:
            json.dump(patterns, f, indent=2)
        print("\nSaved pattern analysis to 'pattern_analysis.json'")
    except Exception as e:
        print(f"Could not save pattern analysis: {e}")
    
    return playground


def example_state_saving_loading():
    """Example of saving and loading playground states"""
    print("\n=== State Saving and Loading Example ===")
    
    # Create a playground
    playground1 = NeuralPlayground({"network_size": "small"})
    
    # Run a play session
    print("Running play session on first playground...")
    playground1.play_once(duration=10, play_type="free")
    
    # Save the state
    save_file = playground1.save()
    print(f"Saved playground state to: {save_file}")
    
    # Create a new playground and load the state
    print("\nCreating a second playground and loading saved state...")
    playground2 = NeuralPlayground({"network_size": "small"})
    playground2.load(save_file)
    
    # Compare states
    status1 = playground1.get_status()
    status2 = playground2.get_status()
    
    print("\nComparing playgrounds:")
    print(f"Playground 1 - Consciousness: {status1['stats']['consciousness_index']:.4f}, "
          f"Patterns: {status1['patterns_count']}")
    print(f"Playground 2 - Consciousness: {status2['stats']['consciousness_index']:.4f}, "
          f"Patterns: {status2['patterns_count']}")
    
    # Run additional play session on loaded playground
    print("\nRunning additional play on second playground...")
    playground2.play_once(duration=10, play_type="guided")
    
    status2_after = playground2.get_status()
    print(f"Playground 2 after play - Consciousness: {status2_after['stats']['consciousness_index']:.4f}, "
          f"Patterns: {status2_after['patterns_count']}")
    
    return playground1, playground2


def example_background_play():
    """Example of running the playground in background mode"""
    print("\n=== Background Play Example ===")
    
    # Create a playground
    playground = NeuralPlayground({"network_size": "small"})
    
    # Start playground in background
    print("Starting playground in background for 30 seconds...")
    playground.start(play_time=30, auto_save=True)
    
    # Check status periodically
    for i in range(6):
        time.sleep(5)  # Check every 5 seconds
        status = playground.get_status()
        print(f"Status update {i+1}: "
              f"Consciousness={status['stats']['consciousness_index']:.4f}, "
              f"Patterns={status['patterns_count']}, "
              f"Sessions={status['stats']['play_sessions']}")
    
    # Stop the playground
    playground.stop()
    print("Playground stopped")
    
    return playground


def example_visualization():
    """Example of visualizing the playground"""
    print("\n=== Visualization Example ===")
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib is required for visualization. Install with 'pip install matplotlib'")
        return None
    
    # Create a small playground for easier visualization
    playground = NeuralPlayground({"network_size": "small"})
    
    # Run a play session
    print("Running play session...")
    playground.play_once(duration=15, play_type="guided")
    
    # Get visualization data
    print("Generating visualization...")
    data = playground.get_visualization_data()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    nodes = data["nodes"]
    edges = data["edges"]
    stats = data["stats"]
    
    # Node colors by type
    node_colors = {
        'input': 'blue',
        'hidden': 'green',
        'association': 'purple',
        'pattern': 'orange',
        'output': 'red',
        'standard': 'gray'
    }
    
    # Create node dict for quick lookup
    node_dict = {node["id"]: node for node in nodes}
    
    # Plot nodes
    for node in nodes:
        color = node_colors.get(node["type"], 'gray')
        size = node["size"] * 100  # Scale size for visibility
        ax.scatter(node["x"], node["y"], node["z"], color=color, s=size, alpha=0.7)
    
    # Plot edges (limit to improve performance)
    max_edges = min(200, len(edges))
    for i, edge in enumerate(edges):
        if i >= max_edges:
            break
        
        source = node_dict.get(edge["source"])
        target = node_dict.get(edge["target"])
        
        if source and target:
            ax.plot([source["x"], target["x"]], 
                   [source["y"], target["y"]], 
                   [source["z"], target["z"]], 
                   color='gray', alpha=0.2, linewidth=abs(edge["weight"]))
    
    # Add title and legend
    plt.title(f"Neural Playground Visualization\nConsciousness: {stats['consciousness_index']:.4f}")
    
    # Add legend for node types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=node_type)
                     for node_type, color in node_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    plt.savefig('playground_visualization.png')
    print("Saved visualization to 'playground_visualization.png'")
    
    return playground


def run_all_examples():
    """Run all examples sequentially"""
    examples = [
        example_basic_play,
        example_multiple_play_types,
        example_consciousness_growth,
        example_pattern_analysis,
        example_state_saving_loading,
        example_background_play,
        example_visualization
    ]
    
    for i, example_func in enumerate(examples):
        print(f"\n\n{'='*80}")
        print(f"Running example {i+1}/{len(examples)}: {example_func.__name__}")
        print(f"{'='*80}")
        
        try:
            example_func()
            print(f"\n✓ Example {example_func.__name__} completed successfully")
        except Exception as e:
            print(f"\n✗ Example {example_func.__name__} failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Playground Examples")
    parser.add_argument("--example", type=str, choices=[
        "basic", "multiple", "growth", "patterns", "state", "background", "visualization", "all"
    ], default="all", help="Which example to run")
    
    args = parser.parse_args()
    
    # Map argument to example function
    example_map = {
        "basic": example_basic_play,
        "multiple": example_multiple_play_types,
        "growth": example_consciousness_growth,
        "patterns": example_pattern_analysis,
        "state": example_state_saving_loading,
        "background": example_background_play,
        "visualization": example_visualization,
        "all": run_all_examples
    }
    
    # Run selected example
    selected_example = example_map[args.example]
    selected_example() 