#!/usr/bin/env python3
"""
Enhanced Language Mistral Integration Demo

This script demonstrates the integration between the Enhanced Language System
and Mistral AI language models, combining neural consciousness processing
with Mistral's language capabilities.
"""

import os
import sys
import time
import logging
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
logger = logging.getLogger("enhanced_mistral_demo")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Language Mistral Integration Demo")
    
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--model", type=str, default="mistral-medium", 
                        choices=["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"],
                        help="Mistral model to use")
    parser.add_argument("--llm-weight", type=float, default=0.7, help="LLM weight (0.0-1.0)")
    parser.add_argument("--nn-weight", type=float, default=0.6, help="NN weight (0.0-1.0)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--learning", action="store_true", help="Enable learning with autowiki")
    parser.add_argument("--api-key", type=str, help="Mistral API key (overrides environment variable)")
    
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
    
    # Print response
    print(f"\n  Response: \"{result.get('response', 'No response')}\"")
    
    # Show full result if requested
    if show_full:
        print("\n  Full Result:")
        for key, value in result.items():
            if key not in ['enhanced_language_results', 'mistral_results']:
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

def stream_demo_process(integration, text):
    """Process text with streaming and display results"""
    print(f"\n> Processing (streaming): \"{text}\"")
    
    # Callback function to handle streaming chunks
    def chunk_callback(chunk, metrics):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    
    # Process with streaming
    print("\n  Response: \"", end="")
    start_time = time.time()
    full_response = integration.process_text_streaming(text, chunk_callback)
    process_time = time.time() - start_time
    print("\"")
    print(f"  [Streamed in {process_time:.2f}s]")
    
    return full_response

def add_demo_autowiki_entries(integration):
    """Add demo entries to the autowiki system"""
    print_section("Adding Autowiki Entries")
    
    # Add some information about neural networks
    success = integration.add_autowiki_entry(
        topic="Neural Networks",
        content="Neural networks are computing systems inspired by biological neural networks in animal brains.",
        source="https://en.wikipedia.org/wiki/Neural_network"
    )
    print(f"  Added 'Neural Networks' entry: {'✓' if success else '✗'}")
    
    # Add information about consciousness
    success = integration.add_autowiki_entry(
        topic="Consciousness",
        content="Consciousness is the state of being aware of one's surroundings and internal states.",
        source="https://en.wikipedia.org/wiki/Consciousness"
    )
    print(f"  Added 'Consciousness' entry: {'✓' if success else '✗'}")
    
    # Add information about V7 system
    success = integration.add_autowiki_entry(
        topic="V7 System",
        content="V7 is an advanced neural network system with consciousness-level processing capabilities.",
        source="Internal documentation"
    )
    print(f"  Added 'V7 System' entry: {'✓' if success else '✗'}")
    
    # Add information about integration
    success = integration.add_autowiki_entry(
        topic="Neural-Linguistic Integration",
        content="The integration of neural networks with language models creates systems capable of deeper understanding.",
        source="Research papers"
    )
    print(f"  Added 'Neural-Linguistic Integration' entry: {'✓' if success else '✗'}")

def run_demo_sequence(integration):
    """Run a sequence of demo scenarios"""
    print_section("Basic Language Processing")
    run_demo_process(integration, "Neural networks can process language patterns.")
    
    print_section("Consciousness References")
    run_demo_process(integration, "The system becomes aware of its own processing and develops self-reflection.")
    
    print_section("Mistral Integration")
    run_demo_process(integration, "How does Mistral AI enhance the language capabilities of the system?")
    
    print_section("Streaming Demo")
    stream_demo_process(integration, "Demonstrate how consciousness emerges in neural networks.")
    
    # If learning is enabled, add autowiki entries and test
    metrics = integration.get_metrics()
    if metrics.get("learning_dict_size", 0) == 0:
        add_demo_autowiki_entries(integration)
    
    print_section("Advanced Processing with Autowiki")
    run_demo_process(integration, 
        "Explain how neural networks and consciousness are related in the V7 system.",
        show_full=True
    )
    
    # Show metrics
    print_section("System Metrics")
    metrics = integration.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

def run_interactive_mode(integration):
    """Run interactive demo mode"""
    print_section("Interactive Mode")
    print("Type 'exit' to quit, 'status' for system status, 'metrics' for metrics.")
    print("Type 'wiki <topic>' to add an entry, 'search <query>' to search the autowiki.")
    print("Type anything else to process as text.")
    
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
                print(f"  Mistral Model: {status['mistral_model']}")
                print(f"  Enhanced Language Available: {status['enhanced_language_available']}")
                print(f"  Mistral Available: {status['mistral_available']}")
                print(f"  LLM Weight: {status['llm_weight']}")
                print(f"  NN Weight: {status['nn_weight']}")
                print(f"  Learning Enabled: {status['learning_enabled']}")
                
                # Show consciousness metrics if available
                if 'consciousness_metrics' in status:
                    print("\nConsciousness Metrics:")
                    for key, value in status['consciousness_metrics'].items():
                        print(f"  {key}: {value}")
                
            elif user_input.lower() == 'metrics':
                # Display system metrics
                metrics = integration.get_metrics()
                print("\nSystem Metrics:")
                for key, value in metrics.items():
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
            
            elif user_input.lower().startswith('wiki '):
                # Add to autowiki
                if not integration.learning_enabled:
                    print("Learning is disabled. Cannot add to autowiki.")
                else:
                    parts = user_input.split(' ', 2)
                    if len(parts) < 3:
                        print("Usage: wiki <topic> <content>")
                    else:
                        topic = parts[1]
                        content = parts[2]
                        success = integration.add_autowiki_entry(topic, content)
                        print(f"Added '{topic}' to autowiki: {'✅' if success else '❌'}")
            
            elif user_input.lower().startswith('search '):
                # Search autowiki
                query = user_input[7:].strip()
                if not query:
                    print("Usage: search <query>")
                else:
                    results = integration.search_autowiki(query)
                    if not results:
                        print("No results found.")
                    else:
                        print(f"Found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            print(f"  {i}. {result['topic']} (Relevance: {result['relevance']:.2f})")
                            print(f"     {result['content'][:100]}...")
                
            elif user_input.lower().startswith('stream '):
                # Process with streaming
                text = user_input[7:].strip()
                if text:
                    stream_demo_process(integration, text)
                else:
                    print("Usage: stream <text>")
                
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
    
    print_section("Enhanced Language Mistral Integration Demo")
    print("Initializing integration...")
    
    try:
        # Import the enhanced language mistral integration
        from src.v7.enhanced_language_mistral_integration import get_enhanced_language_integration
        
        # Get API key from command line or environment
        api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
        
        # Create configuration
        config = {
            "api_key": api_key,
            "model": args.model,
            "llm_weight": args.llm_weight,
            "nn_weight": args.nn_weight,
            "learning_enabled": args.learning,
            "data_dir": "data/demo/enhanced_mistral_integration",
            "learning_dict_path": "data/demo/enhanced_mistral_integration/learning_dict.json"
        }
        
        # Create directories if they don't exist
        os.makedirs(config["data_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(config["learning_dict_path"]), exist_ok=True)
        
        # Get integration instance
        integration = get_enhanced_language_integration(mock_mode=args.mock, config=config)
        
        # Get initial status
        status = integration.get_status()
        print(f"Integration initialized with:")
        print(f"  Mock Mode: {status['mock_mode']}")
        print(f"  Mistral Model: {status['mistral_model']}")
        print(f"  Enhanced Language Available: {status['enhanced_language_available']}")
        print(f"  Mistral Available: {status['mistral_available']}")
        print(f"  LLM Weight: {status['llm_weight']}")
        print(f"  NN Weight: {status['nn_weight']}")
        print(f"  Learning Enabled: {status['learning_enabled'] if 'learning_enabled' in status else args.learning}")
        
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
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure you've implemented the Enhanced Language Mistral Integration module.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 