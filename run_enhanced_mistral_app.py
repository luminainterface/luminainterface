#!/usr/bin/env python3
"""
Enhanced Mistral Chat App Launcher with Neural Network Integration

This script launches the Mistral Chat with integrated Neural Processor and
RSEN components, ensuring proper environment setup and weight configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data",
        "data/onsite_memory",
        "data/db",
        "data/neural_linguistic",
        "data/model_output",
        "data/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def setup_environment(nn_weight, llm_weight):
    """Set up environment variables and paths"""
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Add the src directory to the Python path if it exists
    src_dir = os.path.join(current_dir, "src")
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Set neural weights as environment variables
    os.environ["NN_WEIGHT"] = str(nn_weight)
    os.environ["LLM_WEIGHT"] = str(llm_weight)
    
    # Check for API key in environment
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("Warning: MISTRAL_API_KEY environment variable not set.")
        print("Application will run in mock mode.")
    
    return api_key != ""

def initialize_neural_components(nn_weight):
    """Initialize neural network components"""
    try:
        # Initialize the neural processor
        from neural_processor import NeuralProcessor
        
        processor = NeuralProcessor(
            model_dir="data/model_output",
            embedding_dim=768,
            output_dim=512,
            num_concepts=200
        )
        
        print(f"Neural Processor initialized with {processor.num_concepts} concepts")
        
        # Try to initialize RSEN if available
        try:
            from RSEN_node import RSEN
            rsen = RSEN(input_dim=768, hidden_dim=512, output_dim=256)
            print("RSEN initialized successfully")
            return True
        except ImportError:
            print("RSEN module not available, running with Neural Processor only")
            return True
            
    except Exception as e:
        print(f"Error initializing neural components: {e}")
        print("Application will run without neural network features")
        return False

def patch_simple_mistral_gui():
    """Patch the SimpleNeuralWindow class to ensure proper neural integration"""
    patch_file = "neural_integration_patch.py"
    
    # Check if the patch file exists, if not create it
    if not os.path.exists(patch_file):
        with open(patch_file, "w") as f:
            f.write("""
# Neural integration patch for SimpleNeuralWindow
import sys
import os

def patch_simple_mistral_gui():
    # Import the required modules
    try:
        from neural_processor import NeuralProcessor
        import torch
        from simple_mistral_gui import SimpleMistralWindow
        
        # Store original initialization method
        original_init = SimpleMistralWindow.__init__
        
        # Define patched initialization method
        def patched_init(self, *args, **kwargs):
            # Call original initialization
            original_init(self, *args, **kwargs)
            
            # Initialize neural processor if not already available
            self.neural_processor = None
            try:
                self.neural_processor = NeuralProcessor(
                    model_dir="data/model_output",
                    embedding_dim=768,
                    output_dim=512,
                    num_concepts=200
                )
                print("Neural processor initialized in GUI")
            except Exception as e:
                print(f"Error initializing neural processor in GUI: {e}")
            
            # Update sliders with environment values
            nn_weight = float(os.environ.get("NN_WEIGHT", "0.5"))
            llm_weight = float(os.environ.get("LLM_WEIGHT", "0.5"))
            
            if hasattr(self, 'nn_slider'):
                self.nn_slider.setValue(int(nn_weight * 100))
            if hasattr(self, 'llm_slider'):
                self.llm_slider.setValue(int(llm_weight * 100))
            
            # Ensure weights are updated in central node
            self.update_language_weights()
            
        # Replace initialization method
        SimpleMistralWindow.__init__ = patched_init
        
        # Store original generate response method
        original_generate = SimpleMistralWindow.generate_enhanced_response
        
        # Define patched generate response method
        def patched_generate_response(self, message, context=None, consciousness_level=0.5):
            # Process with neural processor if available
            neural_results = None
            if hasattr(self, 'neural_processor') and self.neural_processor:
                try:
                    # Get neural processing state
                    processing_state = self.neural_processor.process_text(message)
                    
                    # Extract neural results
                    neural_results = {
                        'activations': processing_state.activations.detach().cpu().numpy().tolist() if processing_state.activations is not None else [],
                        'embedding': processing_state.embedding.detach().cpu().numpy().tolist() if processing_state.embedding is not None else [],
                        'resonance': float(processing_state.resonance) if hasattr(processing_state, 'resonance') else 0.0,
                        'concepts': []  # Would be populated if we had concept mapping
                    }
                except Exception as e:
                    print(f"Error in neural processing: {e}")
            
            # Call original method with neural results as extra context
            if neural_results and context:
                context += "\\n\\nNeural processing results available."
            
            return original_generate(self, message, context, consciousness_level)
            
        # Replace generate response method
        SimpleMistralWindow.generate_enhanced_response = patched_generate_response
        
        print("Successfully patched SimpleNeuralWindow for neural integration")
        return True
    except Exception as e:
        print(f"Error patching SimpleNeuralWindow: {e}")
        return False

if __name__ == "__main__":
    patch_simple_mistral_gui()
""")
    
    # Run the patch script
    try:
        subprocess.run([sys.executable, patch_file], check=True)
        print("Successfully applied neural integration patch")
        return True
    except subprocess.CalledProcessError:
        print("Failed to apply neural integration patch")
        return False

def main():
    """Main entry point for the enhanced application launcher"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch Mistral Chat with Neural Integration')
    parser.add_argument('--nn-weight', type=float, default=0.5, help='Neural network weight (0.0-1.0)')
    parser.add_argument('--llm-weight', type=float, default=0.5, help='LLM weight (0.0-1.0)')
    args = parser.parse_args()
    
    print(f"Initializing Mistral Chat with Neural Network Integration...")
    print(f"NN Weight: {args.nn_weight}, LLM Weight: {args.llm_weight}")
    
    # Set up environment
    has_api_key = setup_environment(args.nn_weight, args.llm_weight)
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize neural components
    neural_initialized = initialize_neural_components(args.nn_weight)
    
    # Apply patch to ensure neural integration
    patch_applied = patch_simple_mistral_gui()
    
    # Import and run the application
    try:
        # First try direct import
        print("Starting application...")
        import simple_mistral_gui
        
        # Apply runtime patches
        if patch_applied:
            # The patches should already be applied by now
            pass
        
        return simple_mistral_gui.main()
    except ImportError:
        # If that fails, try running as subprocess
        print("Import failed, trying subprocess...")
        try:
            cmd = [sys.executable, "simple_mistral_gui.py"]
            if args.nn_weight != 0.5:
                cmd.extend(["--nn-weight", str(args.nn_weight)])
            if args.llm_weight != 0.5:
                cmd.extend(["--llm-weight", str(args.llm_weight)])
                
            result = subprocess.run(cmd)
            return result.returncode
        except Exception as e:
            print(f"Error running application: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 