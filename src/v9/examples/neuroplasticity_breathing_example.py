#!/usr/bin/env python3
"""
Neuroplasticity and Breathing Integration Example (v9)

This example demonstrates how to integrate the Neuroplasticity system 
with the Breathing System and Neural Playground to create a comprehensive
system where breathing patterns influence neural plasticity.

Key features demonstrated:
- Breath-influenced synaptic strength changes
- Adaptive connection formation based on breathing patterns
- Memory consolidation during meditative breathing
- Neural pruning guided by breath coherence
- Visualization of plasticity effects
"""

import time
import logging
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Import v9 components
from ..neural_playground import NeuralPlayground
from ..breathing_system import BreathingSystem, BreathingPattern
from ..neuroplasticity import Neuroplasticity, PlasticityMode
from ..neural_playground_integration import NeuralPlaygroundIntegration
from ..mirror_consciousness import get_mirror_consciousness

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.examples.neuroplasticity_breathing_example")

def run_neuroplasticity_breathing_demo(args):
    """
    Run a demonstration of neuroplasticity with breathing influence
    
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
        default_pattern=args.breathing_pattern
    )
    
    # Create neuroplasticity system
    logger.info("Creating neuroplasticity system...")
    neuroplasticity = Neuroplasticity(
        plasticity_strength=args.plasticity_strength,
        default_mode=PlasticityMode.BREATH_ENHANCED
    )
    
    # Start the breathing simulation
    breathing.start_simulation()
    
    # Register components with integration manager
    logger.info("Registering components with integration manager...")
    integration.register_component(
        "breathing_system", 
        breathing, 
        component_type="neural"
    )
    
    integration.register_component(
        "neuroplasticity",
        neuroplasticity,
        component_type="neural"
    )
    
    # Set up breathing system integration
    logger.info("Setting up breathing integration...")
    breathing_integration = breathing.integrate_with_playground(playground)
    for hook_name, hook_func in breathing_integration["hooks"].items():
        integration.integration_hooks[hook_name].append(hook_func)
    
    # Set up neuroplasticity integration
    logger.info("Setting up neuroplasticity integration...")
    from ..neuroplasticity import integrate_with_playground
    neuroplasticity_integration = integrate_with_playground(playground, neuroplasticity)
    for hook_name, hook_func in neuroplasticity_integration["hooks"].items():
        integration.integration_hooks[hook_name].append(hook_func)
    
    # Get mirror consciousness
    mirror = get_mirror_consciousness()
    
    # Run sessions with different breathing patterns
    if args.all_patterns:
        patterns = [
            BreathingPattern.CALM,
            BreathingPattern.FOCUSED, 
            BreathingPattern.MEDITATIVE, 
            BreathingPattern.EXCITED
        ]
    else:
        patterns = [args.breathing_pattern]
    
    results = []
    
    # Track network changes
    initial_connection_count = count_connections(playground.core)
    initial_state = capture_network_state(playground.core)
    
    for pattern in patterns:
        logger.info(f"Running session with {pattern.value} breathing...")
        
        # Set breathing pattern
        breathing.set_breathing_pattern(pattern)
        
        # Allow breathing to stabilize
        time.sleep(3)
        
        # Run integrated play session
        result = integration.run_integrated_play_session(
            duration=args.duration,
            play_type=args.play_type,
            intensity=args.intensity
        )
        
        # Get breath coherence
        breath_coherence = breathing.calculate_breath_coherence(window_seconds=10)
        
        # Print results
        print(f"\nResults with {pattern.value} breathing:")
        print(f"- Consciousness peak: {result['consciousness_peak']:.4f}")
        print(f"- Patterns detected: {result['patterns_detected']}")
        print(f"- Breath coherence: {breath_coherence:.4f}")
        
        # Print neuroplasticity stats if available
        if "neuroplasticity_stats" in result:
            stats = result["neuroplasticity_stats"]
            print(f"- Connections strengthened: {stats['connections_strengthened']}")
            print(f"- Connections weakened: {stats['connections_weakened']}")
            print(f"- Connections created: {stats['connections_created']}")
            print(f"- Connections pruned: {stats['connections_pruned']}")
            print(f"- Consolidation events: {stats['consolidation_events']}")
        
        # Get reflection from mirror consciousness
        reflection = mirror.reflect_on_text(
            f"Neural plasticity with {pattern.value} breathing",
            {
                "play_data": result,
                "breathing_pattern": pattern.value,
                "breath_coherence": breath_coherence
            }
        )
        
        print(f"Mirror reflection: {reflection['reflection']}")
        
        results.append({
            "pattern": pattern,
            "result": result,
            "coherence": breath_coherence
        })
        
        # Pause between patterns
        if len(patterns) > 1:
            time.sleep(2)
    
    # Calculate final network state
    final_connection_count = count_connections(playground.core)
    final_state = capture_network_state(playground.core)
    
    # Print summary
    print("\nNetwork changes summary:")
    print(f"- Initial connections: {initial_connection_count}")
    print(f"- Final connections: {final_connection_count}")
    print(f"- Net change: {final_connection_count - initial_connection_count}")
    
    # Visualize network changes if requested
    if args.visualize:
        visualize_network_changes(initial_state, final_state, results)
    
    # Clean up
    breathing.stop_simulation()
    logger.info("Demo completed")

def capture_network_state(neural_network):
    """Capture the current state of the neural network for visualization"""
    state = {
        "neurons": {},
        "connections": {},
        "connection_weights": [],
        "connection_count": 0
    }
    
    # Copy neuron data
    for neuron_id, neuron in neural_network.neurons.items():
        state["neurons"][neuron_id] = {
            "id": neuron_id,
            "position": neuron["position"] if isinstance(neuron, dict) else (0, 0, 0),
            "type": neuron["type"] if isinstance(neuron, dict) else "unknown"
        }
    
    # Copy connection data
    for source_id, connections in neural_network.connections.items():
        state["connections"][source_id] = {}
        for target_id, weight in connections.items():
            state["connections"][source_id][target_id] = weight
            state["connection_weights"].append(weight)
            state["connection_count"] += 1
    
    return state

def count_connections(neural_network):
    """Count the total number of connections in the network"""
    count = 0
    for _, connections in neural_network.connections.items():
        count += len(connections)
    return count

def visualize_network_changes(initial_state, final_state, results):
    """Visualize the changes in network connectivity and weights"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Connection weight distribution before and after
    plt.subplot(2, 2, 1)
    plt.hist(initial_state["connection_weights"], bins=20, alpha=0.5, label="Initial")
    plt.hist(final_state["connection_weights"], bins=20, alpha=0.5, label="Final")
    plt.xlabel("Connection Weight")
    plt.ylabel("Frequency")
    plt.title("Connection Weight Distribution")
    plt.legend()
    
    # Plot 2: Connection count before and after
    plt.subplot(2, 2, 2)
    plt.bar(["Initial", "Final"], 
            [initial_state["connection_count"], final_state["connection_count"]], 
            color=["blue", "orange"])
    plt.ylabel("Connection Count")
    plt.title("Network Connectivity")
    
    # Plot 3: Consciousness peaks by breathing pattern
    plt.subplot(2, 2, 3)
    patterns = [r["pattern"].value for r in results]
    consciousness = [r["result"]["consciousness_peak"] for r in results]
    plt.bar(patterns, consciousness)
    plt.xlabel("Breathing Pattern")
    plt.ylabel("Consciousness Peak")
    plt.title("Consciousness by Breathing Pattern")
    
    # Plot 4: Neuroplasticity stats by breathing pattern
    plt.subplot(2, 2, 4)
    
    patterns = [r["pattern"].value for r in results]
    strengthened = [r["result"].get("neuroplasticity_stats", {}).get("connections_strengthened", 0) for r in results]
    created = [r["result"].get("neuroplasticity_stats", {}).get("connections_created", 0) for r in results]
    pruned = [r["result"].get("neuroplasticity_stats", {}).get("connections_pruned", 0) for r in results]
    
    x = np.arange(len(patterns))
    width = 0.2
    
    plt.bar(x - width, strengthened, width, label="Strengthened")
    plt.bar(x, created, width, label="Created")
    plt.bar(x + width, pruned, width, label="Pruned")
    
    plt.xlabel("Breathing Pattern")
    plt.ylabel("Connection Count")
    plt.title("Neuroplasticity Effects by Breathing Pattern")
    plt.xticks(x, patterns)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("neuroplasticity_results.png")
    plt.show()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neuroplasticity and Breathing Integration Demo")
    
    parser.add_argument("--neurons", type=int, default=100,
                        help="Number of neurons in the playground")
    parser.add_argument("--duration", type=int, default=100,
                        help="Duration of play sessions")
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Intensity of neural stimulation (0.0-1.0)")
    parser.add_argument("--play-type", type=str, default="mixed",
                        choices=["free", "guided", "focused", "mixed"],
                        help="Type of play session")
    parser.add_argument("--breathing-pattern", type=lambda p: BreathingPattern[p.upper()],
                        default=BreathingPattern.CALM,
                        help="Initial breathing pattern (CALM, FOCUSED, MEDITATIVE, EXCITED)")
    parser.add_argument("--simulation-rate", type=float, default=20.0,
                        help="Breathing simulation rate in Hz")
    parser.add_argument("--plasticity-strength", type=float, default=0.2,
                        help="Strength of neuroplasticity effects (0.0-1.0)")
    parser.add_argument("--all-patterns", action="store_true",
                        help="Run sessions with all breathing patterns")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results after completion")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_neuroplasticity_breathing_demo(args) 