#!/usr/bin/env python3
"""
Interactive Neural Playground Command-line Utility (v9)

This script provides an interactive command-line interface for experimenting
with the Integrated Neural Playground system. It allows users to run play sessions,
change breathing patterns, and observe the effects on neural activity and growth.
"""

import cmd
import time
import os
import json
from pathlib import Path
import logging

# Import v9 components
from .integrated_neural_playground import IntegratedNeuralPlayground
from .breathing_system import BreathingPattern
from .demo_breathing_integration import run_breathing_demo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.interactive")

class InteractivePlayground(cmd.Cmd):
    """Interactive command-line interface for the Integrated Neural Playground"""
    
    intro = """
=== Lumina Neural Network - Interactive Playground (v9) ===

This interface allows you to experiment with the Integrated Neural Playground.
Type 'help' or '?' to list available commands.
Type 'quit' or 'exit' to exit.
    """
    prompt = "playground> "
    
    def __init__(self):
        super().__init__()
        self.playground = None
        self.results_dir = "playground_results"
        self.setup_environment()
        
    def setup_environment(self):
        """Set up the environment and create results directory"""
        try:
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Results will be saved to: {self.results_dir}")
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
    
    def do_init(self, arg):
        """
        Initialize the neural playground with specified parameters
        
        Usage: init [size] [growth_rate] [breathing_pattern] [seed]
        Example: init 150 0.08 meditative 42
        """
        try:
            args = arg.split()
            size = int(args[0]) if len(args) > 0 and args[0] else 150
            growth_rate = float(args[1]) if len(args) > 1 and args[1] else 0.05
            pattern_str = args[2] if len(args) > 2 and args[2] else "calm"
            seed = int(args[3]) if len(args) > 3 and args[3] else None
            
            # Find the breathing pattern
            pattern = BreathingPattern.CALM  # Default
            for p in BreathingPattern:
                if p.value.lower() == pattern_str.lower():
                    pattern = p
                    break
            
            # Clean up existing playground if any
            if self.playground:
                self.playground.stop()
                
            # Create a new playground
            print(f"Initializing Neural Playground with {size} neurons...")
            print(f"Growth rate: {growth_rate}")
            print(f"Breathing pattern: {pattern.value}")
            print(f"Random seed: {seed if seed is not None else 'None (random)'}")
            
            self.playground = IntegratedNeuralPlayground(
                size=size,
                breathing_pattern=pattern,
                growth_rate=growth_rate,
                random_seed=seed
            )
            
            print(f"Neural Playground initialized with {size} neurons")
            
        except Exception as e:
            print(f"Error initializing playground: {e}")
    
    def do_play(self, arg):
        """
        Run a play session with the neural playground
        
        Usage: play [duration] [play_type] [intensity]
        Example: play 200 mixed 0.7
        
        Play types: free, guided, focused, mixed
        """
        if not self.check_initialized():
            return
            
        try:
            args = arg.split()
            duration = int(args[0]) if len(args) > 0 and args[0] else 100
            play_type = args[1] if len(args) > 1 and args[1] else "mixed"
            intensity = float(args[2]) if len(args) > 2 and args[2] else 0.5
            
            valid_play_types = ["free", "guided", "focused", "mixed"]
            if play_type not in valid_play_types:
                print(f"Invalid play type: {play_type}")
                print(f"Valid play types: {', '.join(valid_play_types)}")
                return
            
            print(f"Running play session: duration={duration}, type={play_type}, intensity={intensity}")
            
            # Get pre-play state
            pre_state = self.playground.get_current_state()
            initial_size = pre_state["size"]
            print(f"Initial network size: {initial_size} neurons")
            
            # Run play session
            start_time = time.time()
            result = self.playground.play(
                duration=duration,
                play_type=play_type,
                intensity=intensity
            )
            elapsed_time = time.time() - start_time
            
            # Get post-play state
            post_state = self.playground.get_current_state()
            final_size = post_state["size"]
            
            # Print results
            print("\nPlay session completed:")
            print(f"- Elapsed time: {elapsed_time:.2f} seconds")
            print(f"- Play type: {result['play_type']}")
            print(f"- Duration: {result['duration']} steps")
            print(f"- Total activations: {result['total_activations']}")
            print(f"- Patterns detected: {result['patterns_detected']}")
            print(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
            
            if "breathing_data" in result:
                bd = result["breathing_data"]
                print(f"\nBreathing data:")
                print(f"- Pattern: {bd['pattern']}")
                print(f"- Coherence: {bd['coherence']:.2f}")
                print(f"- Rate: {bd['rate']:.1f} breaths/min")
            
            if "brain_growth" in result:
                bg = result["brain_growth"]
                print(f"\nBrain growth:")
                print(f"- Network growth: {final_size - initial_size} neurons")
                print(f"- Growth state: {bg['growth_state']}")
                print(f"- Neurons created: {bg['neurons_created']}")
                print(f"- Neurons pruned: {bg['neurons_pruned']}")
            
            # Save results
            timestamp = int(time.time())
            result_file = os.path.join(self.results_dir, f"play_result_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "play_metrics": {
                        "play_type": result["play_type"],
                        "duration": result["duration"],
                        "consciousness_peak": result["consciousness_peak"],
                        "patterns_detected": result["patterns_detected"],
                        "total_activations": result["total_activations"]
                    },
                    "breathing_data": result.get("breathing_data"),
                    "brain_growth": result.get("brain_growth"),
                    "network_size": {
                        "initial": initial_size,
                        "final": final_size,
                        "difference": final_size - initial_size
                    }
                }, f, indent=2)
            
            print(f"\nResults saved to {result_file}")
            
        except Exception as e:
            print(f"Error running play session: {e}")
    
    def do_breathing(self, arg):
        """
        Change the current breathing pattern
        
        Usage: breathing [pattern]
        Example: breathing meditative
        
        Available patterns: calm, focused, meditative, excited, custom
        """
        if not self.check_initialized():
            return
            
        try:
            pattern_str = arg.strip().lower() if arg else ""
            
            if not pattern_str:
                # Print current pattern
                current_state = self.playground.breathing.get_current_breath_state()
                print(f"Current breathing pattern: {current_state['pattern']}")
                print(f"Available patterns: {', '.join([p.value for p in BreathingPattern])}")
                return
            
            # Find the breathing pattern
            pattern = None
            for p in BreathingPattern:
                if p.value.lower() == pattern_str:
                    pattern = p
                    break
            
            if pattern is None:
                print(f"Invalid breathing pattern: {pattern_str}")
                print(f"Available patterns: {', '.join([p.value for p in BreathingPattern])}")
                return
            
            # Change the pattern
            self.playground.set_breathing_pattern(pattern)
            print(f"Breathing pattern changed to: {pattern.value}")
            
        except Exception as e:
            print(f"Error changing breathing pattern: {e}")
    
    def do_status(self, arg):
        """
        Show the current status of the neural playground
        
        Usage: status
        """
        if not self.check_initialized():
            return
            
        try:
            state = self.playground.get_current_state()
            
            print("\n=== Neural Playground Status ===")
            print(f"Network size: {state['size']} neurons")
            print(f"Consciousness level: {state['neural']['consciousness_level']:.4f}")
            print(f"Play sessions: {state['neural']['play_sessions']}")
            print(f"Total activations: {state['neural']['total_activations']}")
            print(f"Patterns detected: {state['neural']['patterns_detected']}")
            print(f"Consciousness peaks: {state['neural']['consciousness_peaks']}")
            
            print("\n=== Breathing System Status ===")
            print(f"Current pattern: {state['breathing']['pattern']}")
            print(f"Current state: {state['breathing']['state']}")
            print(f"Breath amplitude: {state['breathing']['amplitude']:.2f}")
            print(f"Breath rate: {state['breathing']['rate']:.1f} breaths/min")
            print(f"Breath coherence: {state['breathing']['coherence']:.2f}")
            
            print("\n=== Brain Growth Status ===")
            print(f"Neurons created: {state['growth']['neurons_created_total']}")
            print(f"Neurons pruned: {state['growth']['neurons_pruned_total']}")
            print(f"Regions formed: {state['growth']['regions_formed_total']}")
            print(f"Growth cycles: {state['growth']['growth_cycles_total']}")
            
            # Print pattern-specific growth
            print("\nNeuron Growth by Pattern:")
            for pattern, count in state['growth']['neuron_growth_by_pattern'].items():
                print(f"- {pattern}: {count} neurons")
            
        except Exception as e:
            print(f"Error getting status: {e}")
    
    def do_save(self, arg):
        """
        Save the current state of the neural playground
        
        Usage: save [filename]
        Example: save my_session
        
        If no filename is provided, a timestamp will be used.
        """
        if not self.check_initialized():
            return
            
        try:
            filename = arg.strip() if arg else f"session_{int(time.time())}"
            
            # Ensure it has the proper directory
            if not os.path.dirname(filename):
                filepath = os.path.join(self.results_dir, filename)
            else:
                filepath = filename
            
            # Save the state
            self.playground.save_state(filepath)
            print(f"Neural playground state saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def do_load(self, arg):
        """
        Load a previously saved state
        
        Usage: load [filename]
        Example: load my_session
        """
        try:
            filename = arg.strip()
            
            if not filename:
                print("Please provide a filename to load")
                return
            
            # Check if file exists in results directory if no directory specified
            if not os.path.dirname(filename):
                filepath = os.path.join(self.results_dir, filename)
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    return
            else:
                filepath = filename
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                    return
            
            # Initialize a new playground if needed
            if not self.playground:
                self.playground = IntegratedNeuralPlayground()
            
            # Load the state
            result = self.playground.load_state(filepath)
            
            if result:
                print(f"Neural playground state loaded from: {filepath}")
                # Show current status
                self.do_status("")
            else:
                print(f"Failed to load state from: {filepath}")
            
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def do_demo(self, arg):
        """
        Run the breathing integration demonstration
        
        Usage: demo [output_dir] [no_visualize]
        Example: demo my_results
        Example: demo my_results no_vis
        """
        try:
            args = arg.split()
            output_dir = args[0] if len(args) > 0 and args[0] else "demo_results"
            visualize = "no_vis" not in args and "no_visualize" not in args
            
            # Stop current playground if running
            if self.playground:
                self.playground.stop()
                self.playground = None
            
            print(f"Running breathing integration demonstration...")
            print(f"Output directory: {output_dir}")
            print(f"Visualization: {'enabled' if visualize else 'disabled'}")
            
            # Run the demo
            run_breathing_demo(output_dir=output_dir, visualize=visualize)
            
        except Exception as e:
            print(f"Error running demonstration: {e}")
    
    def do_quit(self, arg):
        """Exit the interactive playground"""
        print("Closing Neural Playground...")
        if self.playground:
            self.playground.stop()
        print("Thank you for using the Lumina Neural Playground!")
        return True
        
    def do_exit(self, arg):
        """Exit the interactive playground"""
        return self.do_quit(arg)
    
    def check_initialized(self):
        """Check if playground is initialized and prompt if not"""
        if not self.playground:
            print("Neural playground not initialized. Use 'init' command first.")
            return False
        return True

def run_interactive_playground():
    """Run the interactive playground interface"""
    InteractivePlayground().cmdloop()

if __name__ == "__main__":
    run_interactive_playground() 