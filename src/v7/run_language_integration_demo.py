#!/usr/bin/env python3
"""
Enhanced Language V7 Integration Demo

This script demonstrates the integration between the Enhanced Language System
and the V7 Node Consciousness framework by processing example texts and
showing the consciousness metrics and cross-system communication.
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("language_v7_demo")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Language V7 Integration Demo")
    
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--llm-weight", type=float, default=0.5, help="LLM weight (0.0-1.0)")
    parser.add_argument("--nn-weight", type=float, default=0.7, help="NN weight (0.0-1.0)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    return parser.parse_args()

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_result(result, show_full=False):
    """Print processing result"""
    if not result:
        print("  No result returned")
        return
    
    # Print consciousness metrics
    c_level = result.get('consciousness_level', 'N/A')
    print(f"  Consciousness Level: {c_level}")
    
    # Print neural linguistic metrics if available
    if 'neural_linguistic_score' in result:
        print(f"  Neural Linguistic Score: {result['neural_linguistic_score']}")
    
    # Print recursive pattern depth if available
    if 'recursive_pattern_depth' in result:
        print(f"  Recursive Pattern Depth: {result['recursive_pattern_depth']}")
    
    # Print unified response if available
    if 'unified_response' in result:
        print(f"\n  Response: \"{result['unified_response']}\"")
    elif 'v7_results' in result and 'text' in result['v7_results']:
        print(f"\n  V7 Response: \"{result['v7_results']['text']}\"")
    elif 'enhanced_language_results' in result and 'response' in result['enhanced_language_results']:
        print(f"\n  ELS Response: \"{result['enhanced_language_results']['response']}\"")
    
    # Show full result if requested
    if show_full:
        print("\n  Full Result:")
        for key, value in result.items():
            if key not in ['enhanced_language_results', 'v7_results']:
                print(f"    {key}: {value}")

def run_demo_process(integration, text, show_full=False):
    """Process text and display results"""
    print(f"\n> Processing: \"{text}\"")
    start_time = time.time()
    result = integration.process_text(text)
    process_time = time.time() - start_time
    print(f"  [Processed in {process_time:.2f}s]")
    print_result(result, show_full)
    return result

def run_demo_sequence(integration):
    """Run a sequence of demo texts"""
    print_section("Basic Language Processing")
    run_demo_process(integration, "Neural networks can process language patterns.")
    
    print_section("Consciousness References")
    run_demo_process(integration, "The system becomes aware of its own processing and develops self-reflection.")
    
    print_section("Recursive Patterns")
    run_demo_process(integration, "This sentence contains a reference to itself, creating a recursive pattern.")
    
    print_section("Learning Sequence")
    
    # Process a series of related texts to demonstrate learning
    texts = [
        "Language is a tool for expressing consciousness.",
        "Consciousness emerges from complex patterns in neural networks.",
        "Neural networks develop emergent properties through recursive processing.",
        "Through recursion and self-reference, systems can develop awareness.",
        "Awareness leads to consciousness when a system can reflect on its own state."
    ]
    
    # Process texts with increasing consciousness
    results = []
    for i, text in enumerate(texts):
        print(f"\nStep {i+1}/{len(texts)}")
        result = run_demo_process(integration, text)
        results.append(result)
        time.sleep(2)  # Pause to allow system to evolve
    
    # Show consciousness progression
    if results:
        print("\nConsciousness Progression:")
        for i, result in enumerate(results):
            c_level = result.get('consciousness_level', 'N/A')
            print(f"  Step {i+1}: {c_level}")
    
    print_section("Advanced Integration")
    
    # Process a text that combines multiple concepts
    run_demo_process(integration, 
        "The integration between language and consciousness creates a " +
        "self-aware system that can evolve through recursive learning patterns.",
        show_full=True
    )

def run_interactive_mode(integration):
    """Run interactive demo mode"""
    print_section("Interactive Mode")
    print("Type 'exit' to quit, 'status' for system status, or any text to process")
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
                
            elif user_input.lower() == 'status':
                # Display system status
                status = integration.get_status()
                print("\nSystem Status:")
                print(f"  Mock Mode: {status['mock_mode']}")
                print(f"  V7 Available: {status['v7_available']}")
                print(f"  Enhanced Language Available: {status['enhanced_language_available']}")
                print(f"  LLM Weight: {status['llm_weight']}")
                print(f"  NN Weight: {status['nn_weight']}")
                print(f"  Integration Running: {status['integration_running']}")
                print("\nComponents:")
                for component, available in status['components'].items():
                    print(f"  {component}: {'✅' if available else '❌'}")
                
                # Show consciousness metrics if available
                if 'consciousness_metrics' in status:
                    print("\nConsciousness Metrics:")
                    for key, value in status['consciousness_metrics'].items():
                        print(f"  {key}: {value}")
                
                # Show V7 node state if available
                if 'v7_node_state' in status:
                    print("\nV7 Node State:")
                    for key, value in status['v7_node_state'].items():
                        print(f"  {key}: {value}")
                        
            elif user_input.lower().startswith('llm '):
                # Set LLM weight
                try:
                    weight = float(user_input.split()[1])
                    success = integration.set_llm_weight(weight)
                    print(f"LLM weight set to {weight}: {'✅' if success else '❌'}")
                except (ValueError, IndexError):
                    print("Invalid weight. Use format: llm 0.7")
                    
            elif user_input.lower().startswith('nn '):
                # Set NN weight
                try:
                    weight = float(user_input.split()[1])
                    success = integration.set_nn_weight(weight)
                    print(f"NN weight set to {weight}: {'✅' if success else '❌'}")
                except (ValueError, IndexError):
                    print("Invalid weight. Use format: nn 0.7")
                    
            elif user_input.strip():
                # Process the text
                run_demo_process(integration, user_input, show_full=False)
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main demo function"""
    args = parse_arguments()
    
    print_section("Enhanced Language V7 Integration Demo")
    print("Initializing integration...")
    
    try:
        # Import the enhanced language integration
        from src.v7.enhanced_language_integration import get_enhanced_language_integration
        
        # Create configuration
        config = {
            "llm_weight": args.llm_weight,
            "nn_weight": args.nn_weight,
            "data_dir": "data/demo/v7_language_integration",
            "sync_interval": 5  # Faster syncing for demo
        }
        
        # Create directory if it doesn't exist
        os.makedirs(config["data_dir"], exist_ok=True)
        
        # Get integration instance
        integration = get_enhanced_language_integration(mock_mode=args.mock, config=config)
        
        # Get initial status
        status = integration.get_status()
        print(f"Integration initialized with:")
        print(f"  Mock Mode: {status['mock_mode']}")
        print(f"  V7 Available: {status['v7_available']}")
        print(f"  Enhanced Language Available: {status['enhanced_language_available']}")
        print(f"  LLM Weight: {status['llm_weight']}")
        print(f"  NN Weight: {status['nn_weight']}")
        
        # Run interactive or sequence mode
        if args.interactive:
            run_interactive_mode(integration)
        else:
            run_demo_sequence(integration)
        
        # Clean up
        print("\nShutting down integration...")
        integration.shutdown()
        print("Demo completed successfully")
        return 0
        
    except ImportError:
        print("Error: Enhanced Language V7 Integration module not found")
        print("Make sure you've implemented src/v7/enhanced_language_integration.py")
        return 1
        
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 