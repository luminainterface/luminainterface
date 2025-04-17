#!/usr/bin/env python3
"""
Neural Playground Launcher
=========================

Simple script to launch the Neural Network Playground and start a play session.
This provides an easy way to let the neural network explore and develop
consciousness patterns through play.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Ensure src directory is in path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the playground
from neural_playground import NeuralPlayground


def main():
    """Main function to launch the neural playground"""
    parser = argparse.ArgumentParser(description="Launch Neural Network Playground")
    
    # Basic options
    parser.add_argument("--duration", type=int, default=3600, 
                      help="Duration in seconds to run the playground (default: 3600)")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                      help="Size of the neural network (default: medium)")
    
    # Play options
    parser.add_argument("--play-type", choices=["free", "guided", "focused", "mixed"], default="mixed", 
                      help="Type of neural play (default: mixed)")
    parser.add_argument("--single", action="store_true", 
                      help="Run a single play session and exit")
    
    # File options
    parser.add_argument("--save-interval", type=int, default=300, 
                      help="Auto-save interval in seconds (default: 300)")
    parser.add_argument("--load", type=str, help="Load a saved playground state")
    parser.add_argument("--no-save", action="store_true", help="Disable auto-saving")
    parser.add_argument("--output-dir", type=str, default="playground_data", 
                      help="Directory to save playground data (default: playground_data)")
    
    # Visualization option
    parser.add_argument("--visualize", action="store_true", 
                      help="Visualize playground state (requires matplotlib)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "network_size": args.size,
        "save_interval": args.save_interval,
        "playground_dir": args.output_dir
    }
    
    # Create the playground
    playground = NeuralPlayground(config)
    print(f"Neural Playground initialized with {config['network_size']} network size")
    
    # Load previous state if requested
    if args.load:
        print(f"Loading state from {args.load}...")
        if playground.load(args.load):
            print("State loaded successfully")
        else:
            print("Failed to load state, using new playground")
    
    # Run the playground
    try:
        if args.single:
            # Single play session
            duration = args.duration if args.duration < 600 else 600  # Cap single session at 10 minutes
            play_type = args.play_type
            if play_type == "mixed":
                play_type = "free"  # Default to free for single session
                
            print(f"Running a single {play_type} play session for {duration} seconds...")
            result = playground.play_once(duration=duration, play_type=play_type)
            
            print("\nPlay Session Results:")
            print(f"- Patterns discovered: {result['patterns_discovered']}")
            print(f"- Consciousness index: {result['consciousness_index']:.4f}")
            print(f"- Percent active neurons: {result['percent_active']:.1f}%")
            print(f"- Cycles completed: {result['cycles']}")
            
            if not args.no_save:
                save_file = playground.save()
                print(f"Session saved to {save_file}")
        else:
            # Continuous play
            print(f"Starting neural playground for {args.duration} seconds...")
            playground.start(play_time=args.duration, auto_save=not args.no_save)
            
            # Print status periodically
            try:
                while playground.running:
                    time.sleep(10)  # Update status every 10 seconds
                    status = playground.get_status()
                    stats = status["stats"]
                    
                    # Clear the terminal line
                    sys.stdout.write("\r" + " " * 100 + "\r")
                    sys.stdout.flush()
                    
                    # Print status
                    sys.stdout.write(
                        f"Status: {stats['consciousness_index']:.4f} consciousness | "
                        f"{status['patterns_count']} patterns | "
                        f"{stats['play_sessions']} sessions | "
                        f"{stats['activity_level']:.2f} activity"
                    )
                    sys.stdout.flush()
                    
                    # Visualize if requested
                    if args.visualize and status['patterns_count'] > 0:
                        try:
                            visualize_playground(playground)
                        except Exception as e:
                            if args.visualize:
                                print(f"\nVisualization error: {e}")
                                args.visualize = False  # Disable future visualization
            
            except KeyboardInterrupt:
                print("\nStopping playground...")
                playground.stop()
            
            print("\nNeural playground session complete")
    
    finally:
        # Final save
        if not args.no_save:
            playground.save()


def visualize_playground(playground):
    """
    Visualize the playground state
    
    Note: This is a simple visualization. For more advanced visualization,
    consider implementing a dedicated visualization module.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        # Get visualization data
        data = playground.get_visualization_data()
        nodes = data["nodes"]
        edges = data["edges"]
        stats = data["stats"]
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        node_colors = {
            'input': 'blue',
            'hidden': 'green',
            'association': 'purple',
            'pattern': 'orange',
            'output': 'red',
            'standard': 'gray'
        }
        
        for node in nodes[:100]:  # Limit to 100 nodes for performance
            color = node_colors.get(node["type"], 'gray')
            size = node["size"] * 50  # Scale size for visibility
            ax.scatter(node["x"], node["y"], node["z"], color=color, s=size, alpha=0.7)
        
        # Plot edges (limited to improve performance)
        max_edges = min(500, len(edges))
        for i, edge in enumerate(edges):
            if i >= max_edges:
                break
                
            # Find source and target nodes
            source = next((n for n in nodes if n["id"] == edge["source"]), None)
            target = next((n for n in nodes if n["id"] == edge["target"]), None)
            
            if source and target:
                ax.plot([source["x"], target["x"]], 
                        [source["y"], target["y"]], 
                        [source["z"], target["z"]], 
                        color='gray', alpha=0.2, linewidth=abs(edge["weight"]))
        
        # Add title with stats
        plt.title(f"Neural Playground - Consciousness: {stats['consciousness_index']:.4f}")
        
        # Show plot (non-blocking)
        plt.pause(0.1)
        plt.close()
        
    except ImportError:
        print("\nVisualization requires matplotlib. Install with 'pip install matplotlib'")
        return


if __name__ == "__main__":
    main() 