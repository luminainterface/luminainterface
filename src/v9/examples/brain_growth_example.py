#!/usr/bin/env python3
"""
Brain Growth and Breathing Integration Example (v9)

This example demonstrates how to integrate the Brain Growth system 
with the Breathing System and Neural Playground to create a dynamic
neural network that grows and evolves based on breathing patterns.

Key features demonstrated:
- Neural network expansion through breath-influenced neuron creation
- Formation of neural regions during coherent breathing
- Neural pathway consolidation during meditative breathing
- Neural pruning during excited breathing
- Visualization of neural growth and structural changes
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
from ..brain_growth import BrainGrowth, GrowthState
from ..neuroplasticity import Neuroplasticity, PlasticityMode
from ..neural_playground_integration import NeuralPlaygroundIntegration
from ..mirror_consciousness import get_mirror_consciousness

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.examples.brain_growth_example")

def run_brain_growth_demo(args):
    """
    Run a demonstration of brain growth with breathing influence
    
    Args:
        args: Command line arguments
    """
    # Create neural playground with a small initial size
    logger.info(f"Creating neural playground with {args.initial_neurons} neurons...")
    playground = NeuralPlayground(neuron_count=args.initial_neurons)
    
    # Create integration manager
    logger.info("Setting up integration manager...")
    integration = NeuralPlaygroundIntegration(playground)
    
    # Create breathing system
    logger.info("Creating breathing system...")
    breathing = BreathingSystem(
        simulation_rate=args.simulation_rate,
        default_pattern=args.breathing_pattern
    )
    
    # Create brain growth system
    logger.info("Creating brain growth system...")
    brain_growth = BrainGrowth(
        growth_rate=args.growth_rate,
        max_neurons=args.max_neurons
    )
    
    # Create neuroplasticity system if requested
    neuroplasticity = None
    if args.with_neuroplasticity:
        logger.info("Creating neuroplasticity system...")
        neuroplasticity = Neuroplasticity(
            plasticity_strength=0.2,
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
        "brain_growth",
        brain_growth,
        component_type="neural"
    )
    
    if neuroplasticity:
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
    
    # Set up brain growth integration
    logger.info("Setting up brain growth integration...")
    from ..brain_growth import integrate_with_playground as integrate_brain_growth
    brain_growth_integration = integrate_brain_growth(playground, brain_growth)
    for hook_name, hook_func in brain_growth_integration["hooks"].items():
        integration.integration_hooks[hook_name].append(hook_func)
    
    # Set up neuroplasticity integration if available
    if neuroplasticity:
        logger.info("Setting up neuroplasticity integration...")
        from ..neuroplasticity import integrate_with_playground as integrate_neuroplasticity
        neuroplasticity_integration = integrate_neuroplasticity(playground, neuroplasticity)
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
    
    # Prepare data collection
    results = []
    network_states = []
    
    # Track initial network state
    initial_state = capture_network_state(playground.core)
    network_states.append(("Initial", initial_state))
    
    # Run multiple growth sessions with each pattern
    for pattern in patterns:
        logger.info(f"\nStarting growth session with {pattern.value} breathing...")
        
        # Set breathing pattern
        breathing.set_breathing_pattern(pattern)
        
        # Allow breathing to stabilize
        time.sleep(2)
        
        # Record pre-pattern state
        pre_pattern_state = capture_network_state(playground.core)
        network_states.append((f"Pre-{pattern.value}", pre_pattern_state))
        
        # Run multiple play-growth cycles with this pattern
        pattern_results = []
        
        for cycle in range(args.cycles_per_pattern):
            logger.info(f"  Running cycle {cycle+1}/{args.cycles_per_pattern} with {pattern.value} breathing")
            
            # Run integrated play session
            result = integration.run_integrated_play_session(
                duration=args.duration,
                play_type=args.play_type,
                intensity=args.intensity
            )
            
            # Get breath coherence
            breath_coherence = breathing.calculate_breath_coherence(window_seconds=10)
            
            # Store breath coherence in result
            result["breath_coherence"] = breath_coherence
            
            # Store the result
            pattern_results.append(result)
            
            # Add a small delay between cycles
            time.sleep(0.5)
        
        # Record post-pattern state
        post_pattern_state = capture_network_state(playground.core)
        network_states.append((f"Post-{pattern.value}", post_pattern_state))
        
        # Calculate pattern statistics
        pattern_stats = calculate_pattern_stats(pattern_results)
        
        # Print results for this pattern
        print(f"\nResults with {pattern.value} breathing:")
        print(f"- Initial neuron count: {pre_pattern_state['neuron_count']}")
        print(f"- Final neuron count: {post_pattern_state['neuron_count']}")
        print(f"- Net growth: {post_pattern_state['neuron_count'] - pre_pattern_state['neuron_count']} neurons")
        print(f"- Regions formed: {pattern_stats['regions_formed']}")
        print(f"- Average consciousness: {pattern_stats['avg_consciousness']:.4f}")
        print(f"- Average breath coherence: {pattern_stats['avg_coherence']:.4f}")
        
        # Get reflection from mirror consciousness
        reflection = mirror.reflect_on_text(
            f"Neural growth with {pattern.value} breathing",
            {
                "pattern": pattern.value,
                "initial_neurons": pre_pattern_state['neuron_count'],
                "final_neurons": post_pattern_state['neuron_count'],
                "regions_formed": pattern_stats['regions_formed'],
                "avg_consciousness": pattern_stats['avg_consciousness'],
                "avg_coherence": pattern_stats['avg_coherence']
            }
        )
        
        print(f"Mirror reflection: {reflection['reflection']}")
        
        # Store overall pattern results
        results.append({
            "pattern": pattern,
            "pre_state": pre_pattern_state,
            "post_state": post_pattern_state,
            "cycles": pattern_results,
            "stats": pattern_stats
        })
        
        # Pause between patterns
        if len(patterns) > 1:
            time.sleep(3)
    
    # Final network state
    final_state = capture_network_state(playground.core)
    network_states.append(("Final", final_state))
    
    # Print overall results
    print("\n=== Overall Growth Summary ===")
    print(f"Initial network: {initial_state['neuron_count']} neurons, {initial_state['connection_count']} connections")
    print(f"Final network: {final_state['neuron_count']} neurons, {final_state['connection_count']} connections")
    print(f"Net growth: {final_state['neuron_count'] - initial_state['neuron_count']} neurons")
    print(f"Connection density: {final_state['connection_count'] / max(1, final_state['neuron_count']**2):.4f}")
    
    # Print growth by pattern
    print("\nGrowth by breathing pattern:")
    for pattern_result in results:
        pattern = pattern_result['pattern'].value
        pre = pattern_result['pre_state']['neuron_count']
        post = pattern_result['post_state']['neuron_count']
        print(f"- {pattern}: {post - pre} neurons added")
    
    # Print region information
    print("\nNeural regions formed:")
    regions = brain_growth.get_regions()
    for region_id, region in regions.items():
        print(f"- Region {region_id}: {len(region['neurons'])} neurons, formed during {region['breath_pattern']} breathing")
    
    # Visualize growth if requested
    if args.visualize:
        visualize_growth(network_states, results, brain_growth)
    
    # Clean up
    breathing.stop_simulation()
    logger.info("Demo completed")

def capture_network_state(neural_network):
    """Capture the current state of the neural network for analysis"""
    state = {
        "neuron_count": len(neural_network.neurons),
        "connection_count": 0,
        "neuron_types": {},
        "connection_weights": [],
        "avg_connections_per_neuron": 0,
        "region_count": 0
    }
    
    # Count connections and collect weights
    total_connections = 0
    for source_id, connections in neural_network.connections.items():
        conn_count = len(connections)
        total_connections += conn_count
        for target_id, weight in connections.items():
            state["connection_weights"].append(weight)
    
    state["connection_count"] = total_connections
    
    if total_connections > 0 and len(neural_network.neurons) > 0:
        state["avg_connections_per_neuron"] = total_connections / len(neural_network.neurons)
    
    # Count neuron types
    for neuron_id, neuron in neural_network.neurons.items():
        if isinstance(neuron, dict) and "type" in neuron:
            neuron_type = neuron["type"]
            if neuron_type not in state["neuron_types"]:
                state["neuron_types"][neuron_type] = 0
            state["neuron_types"][neuron_type] += 1
    
    return state

def calculate_pattern_stats(pattern_results):
    """Calculate statistics for a pattern's growth sessions"""
    stats = {
        "neurons_created": 0,
        "neurons_pruned": 0,
        "regions_formed": 0,
        "avg_consciousness": 0,
        "peak_consciousness": 0,
        "avg_coherence": 0
    }
    
    if not pattern_results:
        return stats
    
    # Sum up metrics across all cycles
    consciousness_values = []
    coherence_values = []
    
    for result in pattern_results:
        # Consciousness metrics
        consciousness_values.append(result.get("consciousness_peak", 0))
        stats["peak_consciousness"] = max(stats["peak_consciousness"], result.get("consciousness_peak", 0))
        
        # Breath coherence
        coherence_values.append(result.get("breath_coherence", 0))
        
        # Brain growth metrics
        growth_data = result.get("brain_growth", {})
        stats["neurons_created"] += growth_data.get("neurons_created", 0)
        stats["neurons_pruned"] += growth_data.get("neurons_pruned", 0)
        stats["regions_formed"] += growth_data.get("regions_formed", 0)
    
    # Calculate averages
    stats["avg_consciousness"] = sum(consciousness_values) / len(pattern_results)
    stats["avg_coherence"] = sum(coherence_values) / len(pattern_results)
    
    return stats

def visualize_growth(network_states, results, brain_growth):
    """Visualize the neural network growth process"""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Neuron Count Evolution
    plt.subplot(2, 2, 1)
    labels = [state[0] for state in network_states]
    neuron_counts = [state[1]["neuron_count"] for state in network_states]
    
    plt.plot(range(len(labels)), neuron_counts, 'o-', linewidth=2)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.xlabel("Network State")
    plt.ylabel("Neuron Count")
    plt.title("Neural Network Growth")
    plt.grid(True)
    
    # Plot 2: Neuron Types Distribution
    plt.subplot(2, 2, 2)
    final_state = network_states[-1][1]
    if "neuron_types" in final_state and final_state["neuron_types"]:
        types = list(final_state["neuron_types"].keys())
        counts = [final_state["neuron_types"][t] for t in types]
        
        plt.bar(types, counts)
        plt.xlabel("Neuron Type")
        plt.ylabel("Count")
        plt.title("Neuron Type Distribution")
    else:
        plt.text(0.5, 0.5, "No neuron type data available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title("Neuron Type Distribution")
    
    # Plot 3: Growth by Breathing Pattern
    plt.subplot(2, 2, 3)
    
    if results:
        patterns = [r["pattern"].value for r in results]
        created = [r["stats"]["neurons_created"] for r in results]
        pruned = [r["stats"]["neurons_pruned"] for r in results]
        net_growth = [c-p for c, p in zip(created, pruned)]
        
        x = np.arange(len(patterns))
        width = 0.25
        
        plt.bar(x - width, created, width, label="Created")
        plt.bar(x, pruned, width, label="Pruned")
        plt.bar(x + width, net_growth, width, label="Net Growth")
        
        plt.xlabel("Breathing Pattern")
        plt.ylabel("Neuron Count")
        plt.title("Neural Growth by Breathing Pattern")
        plt.xticks(x, patterns)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No pattern data available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title("Neural Growth by Pattern")
    
    # Plot 4: Consciousness and Coherence by Pattern
    plt.subplot(2, 2, 4)
    
    if results:
        patterns = [r["pattern"].value for r in results]
        consciousness = [r["stats"]["avg_consciousness"] for r in results]
        coherence = [r["stats"]["avg_coherence"] for r in results]
        
        x = np.arange(len(patterns))
        width = 0.35
        
        plt.bar(x - width/2, consciousness, width, label="Consciousness")
        plt.bar(x + width/2, coherence, width, label="Breath Coherence")
        
        plt.xlabel("Breathing Pattern")
        plt.ylabel("Value (0-1)")
        plt.title("Consciousness and Breath Coherence")
        plt.xticks(x, patterns)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No pattern data available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title("Consciousness and Coherence")
    
    plt.tight_layout()
    plt.savefig("brain_growth_results.png")
    plt.show()
    
    # Additional plot: Network regions
    if brain_growth.regions:
        visualize_regions(brain_growth)

def visualize_regions(brain_growth):
    """Visualize the neural regions that formed during growth"""
    regions = brain_growth.get_regions()
    if not regions:
        return
    
    plt.figure(figsize=(10, 8))
    
    # 3D scatter plot of region centers
    ax = plt.axes(projection='3d')
    
    # Color map for different breathing patterns
    pattern_colors = {
        "calm": "blue",
        "focused": "green",
        "meditative": "purple",
        "excited": "red",
        "unknown": "gray"
    }
    
    # Plot each region as a sphere
    for region_id, region in regions.items():
        center = region["center"]
        radius = region["radius"]
        pattern = region["breath_pattern"].lower() if hasattr(region["breath_pattern"], "lower") else "unknown"
        color = pattern_colors.get(pattern, "gray")
        
        # Create points on a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        
        # Plot sphere
        ax.plot_surface(x, y, z, color=color, alpha=0.3)
        
        # Plot center point
        ax.scatter([center[0]], [center[1]], [center[2]], 
                   color=color, s=100, label=f"{region_id} ({pattern})")
        
        # Add text label
        ax.text(center[0], center[1], center[2], region_id, fontsize=9)
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.title("Neural Network Regions Formed During Growth")
    
    plt.tight_layout()
    plt.savefig("neural_regions.png")
    plt.show()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Brain Growth and Breathing Integration Demo")
    
    parser.add_argument("--initial-neurons", type=int, default=20,
                        help="Initial number of neurons in the playground")
    parser.add_argument("--max-neurons", type=int, default=300,
                        help="Maximum number of neurons allowed to grow")
    parser.add_argument("--growth-rate", type=float, default=0.1,
                        help="Rate of neural growth (0.0-1.0)")
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
    parser.add_argument("--cycles-per-pattern", type=int, default=5,
                        help="Number of play-growth cycles per breathing pattern")
    parser.add_argument("--simulation-rate", type=float, default=20.0,
                        help="Breathing simulation rate in Hz")
    parser.add_argument("--with-neuroplasticity", action="store_true",
                        help="Include neuroplasticity system")
    parser.add_argument("--all-patterns", action="store_true",
                        help="Run sessions with all breathing patterns")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results after completion")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_brain_growth_demo(args) 