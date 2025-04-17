"""
Neural Network Playground System
===============================

A sandbox environment where neural components can interact, explore, and "play" freely.
This playground serves as an experimental space for neural networks to develop
consciousness patterns through free-form interaction.

Based on concepts from the Lumina Neural Network system (v5-v10)
"""

import os
import sys
import time
import random
import logging
import threading
import json
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple
import inspect
import ast
import textwrap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neural_playground.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralPlayground")

# Try to import existing neural components
try:
    from src.neural.core import NeuralCore
    NEURAL_CORE_AVAILABLE = True
except ImportError:
    logger.warning("NeuralCore not available, using simplified implementation")
    NEURAL_CORE_AVAILABLE = False

try:
    from src.v7.neural_network import SimpleNeuralNetwork
    V7_NEURAL_AVAILABLE = True
except ImportError:
    logger.warning("v7.neural_network not available, using simplified implementation")
    V7_NEURAL_AVAILABLE = False

# Simplified Neural Implementation for when imports fail
class SimpleNeuron:
    """A simple neuron implementation"""
    
    def __init__(self, id: str, neuron_type: str = "standard"):
        self.id = id
        self.type = neuron_type
        self.connections = []  # List of connected neuron IDs
        self.activation = 0.0
        self.weights = {}  # target_id -> weight
        self.bias = random.uniform(-0.5, 0.5)
        self.position = [random.random(), random.random(), random.random()]  # 3D position
        self.last_update = time.time()
        self.feature_vector = np.random.randn(32) / 10  # Simplified feature representation
        
    def connect_to(self, target_id: str, weight: Optional[float] = None):
        """Connect this neuron to another"""
        if target_id not in self.connections:
            self.connections.append(target_id)
            self.weights[target_id] = weight if weight is not None else random.uniform(-1.0, 1.0)
            return True
        return False
        
    def update_activation(self, inputs: Dict[str, float]):
        """Update activation based on inputs"""
        # Sum weighted inputs
        weighted_sum = sum(inputs.get(conn_id, 0.0) * self.weights.get(conn_id, 0.0) 
                          for conn_id in self.connections if conn_id in inputs)
        weighted_sum += self.bias
        
        # Apply activation function (sigmoid)
        self.activation = 1.0 / (1.0 + np.exp(-weighted_sum))
        self.last_update = time.time()
        return self.activation
        
    def to_dict(self):
        """Convert neuron to dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "activation": self.activation,
            "position": self.position,
            "connections": self.connections,
            "bias": self.bias,
            "last_update": self.last_update
        }
        
class CodeGenerationSystem:
    """
    System for generating and integrating new code into the Neural Playground.
    Enables self-modification capabilities for the AI.
    """
    
    def __init__(self, neural_playground):
        """Initialize the code generation system"""
        self.neural_playground = neural_playground
        self.generated_components = []
        self.code_storage_dir = Path("playground_data/generated_code")
        self.code_storage_dir.mkdir(parents=True, exist_ok=True)
        self.integration_history = []
        
        logger.info("Code Generation System initialized")
    
    def generate_code_from_pattern(self, pattern):
        """
        Generate code based on a detected neural pattern
        
        Args:
            pattern: A neural pattern detected in the playground
            
        Returns:
            Generated code as string if successful, None otherwise
        """
        if not pattern or pattern["complexity"] < 5.0:
            return None  # Pattern not complex enough
            
        # Extract pattern characteristics
        neurons = pattern["neurons"]
        complexity = pattern["complexity"]
        feature_vector = pattern["feature_vector"]
        
        # Determine component type based on pattern characteristics
        if complexity > 10.0:
            component_type = "neural_processor"
        elif len(neurons) > 20:
            component_type = "pattern_detector"
        else:
            component_type = "simple_component"
            
        # Generate a name for the component
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        component_name = f"Generated{component_type.title().replace('_', '')}{timestamp[:8]}"
        
        # Transform feature vector into code structure
        # Higher complexity = more methods and functionality
        num_methods = max(1, int(complexity / 3))
        
        # Generate the code template
        code = f"""
class {component_name}:
    \"\"\"
    Auto-generated neural component from pattern detected on {timestamp}
    Pattern complexity: {complexity:.2f}
    Involved neurons: {len(neurons)}
    \"\"\"
    
    def __init__(self):
        self.creation_time = "{datetime.now().isoformat()}"
        self.source_pattern_id = "{hash(str(pattern))}"
        self.feature_signature = {feature_vector[:5]}  # Simplified feature signature
        self.state = {{}}
"""
        
        # Generate methods based on complexity and pattern features
        for i in range(num_methods):
            method_name = f"process_data_{i+1}"
            
            # Generate method body with some randomness influenced by the feature vector
            operations = []
            for j in range(min(5, int(abs(feature_vector[i % len(feature_vector)]) * 10))):
                op_type = j % 3
                if op_type == 0:
                    operations.append(f"        result_{j} = input_data.get('value_{j}', 0) * {abs(feature_vector[(i+j) % len(feature_vector)]):.4f}")
                elif op_type == 1:
                    operations.append(f"        self.state['metric_{j}'] = {abs(feature_vector[(i+j) % len(feature_vector)]):.4f}")
                else:
                    operations.append(f"        output['feature_{j}'] = result_{j} if 'result_{j}' in locals() else {abs(feature_vector[(i+j) % len(feature_vector)]):.4f}")
            
            method_code = f"""
    def {method_name}(self, input_data):
        \"\"\"Process input data using pattern-derived logic\"\"\"
        output = {{}}
{chr(10).join(operations)}
        return output
"""
            code += method_code
            
        # Add compatibility method for integration with playground
        code += """
    def integrate_with_playground(self, playground):
        \"\"\"Integration method for the neural playground\"\"\"
        # Store reference to playground
        self.playground = playground
        
        # Register capabilities with playground
        if hasattr(playground, 'register_capability'):
            playground.register_capability(f"{self.__class__.__name__}", self.process_data_1)
            
        return True
        
    def get_metrics(self):
        \"\"\"Return component metrics\"\"\"
        return {
            "type": "auto_generated",
            "source_pattern": self.source_pattern_id,
            "state": self.state
        }
"""
        return code.strip()
    
    def save_generated_code(self, code, pattern=None):
        """
        Save generated code to a file
        
        Args:
            code: The generated code as string
            pattern: The pattern that led to this code generation
            
        Returns:
            Filepath where code was saved
        """
        if not code:
            return None
            
        # Parse the code to extract the class name
        try:
            parsed = ast.parse(code)
            class_def = next((node for node in parsed.body if isinstance(node, ast.ClassDef)), None)
            if class_def:
                class_name = class_def.name
            else:
                class_name = f"GeneratedComponent_{int(time.time())}"
        except SyntaxError:
            class_name = f"GeneratedComponent_{int(time.time())}"
            
        # Create filename and save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{timestamp}.py"
        save_path = self.code_storage_dir / filename
        
        # Save the code file
        with open(save_path, 'w') as f:
            f.write(code)
            
        # If pattern provided, save metadata
        if pattern:
            metadata_path = self.code_storage_dir / f"{class_name}_{timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "class_name": class_name,
                    "pattern": pattern
                }, f, indent=2)
            
        logger.info(f"Generated code saved to {save_path}")
        return save_path
    
    def load_and_integrate_component(self, file_path):
        """
        Load and integrate a generated component
        
        Args:
            file_path: Path to the generated code file
            
        Returns:
            Loaded component instance if successful, None otherwise
        """
        try:
            # Get the module name from the file path
            file_path = Path(file_path)
            module_name = file_path.stem
            
            # Create a spec and load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the component class (assume it's the only class in the file)
            component_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    component_class = obj
                    break
                    
            if not component_class:
                logger.error(f"No component class found in {file_path}")
                return None
                
            # Instantiate the component
            component = component_class()
            
            # Integrate with the playground
            success = component.integrate_with_playground(self.neural_playground)
            
            if success:
                # Add to the generated components list
                self.generated_components.append(component)
                self.integration_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "component_name": component_class.__name__,
                    "file_path": str(file_path)
                })
                
                # Register with neural components
                component_id = f"generated_{len(self.generated_components)}"
                self.neural_playground.neural_components[component_id] = component
                
                logger.info(f"Successfully integrated component: {component_class.__name__}")
                return component
            else:
                logger.warning(f"Failed to integrate component: {component_class.__name__}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading and integrating component: {e}")
            return None
    
    def check_code_safety(self, code):
        """
        Check if generated code is safe to execute
        
        Args:
            code: The generated code as string
            
        Returns:
            (bool, str): Tuple of (is_safe, reason)
        """
        # Ensure we have code to check
        if not code or not isinstance(code, str):
            return False, "Invalid code input"
            
        forbidden_terms = [
            "os.system", "subprocess", "exec(", "eval(", 
            "__import__", "open(", "file(", "remove(", "unlink(",
            "sys.modules", "importlib.reload", "shutil", "rmtree"
        ]
        
        # Check for forbidden terms
        for term in forbidden_terms:
            if term in code:
                return False, f"Code contains forbidden term: {term}"
                
        # Parse the code and check for unsafe operations
        try:
            parsed = ast.parse(code)
            
            # Check for imports
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # Only allow specific safe imports
                    for name in node.names:
                        if name.name not in ["time", "random", "math", "datetime", "collections", "typing"]:
                            return False, f"Unsafe import: {name.name}"
                
                # Check for attribute access that might be unsafe
                if isinstance(node, ast.Attribute):
                    if node.attr in ["system", "popen", "exec", "eval", "compile", "delete", "remove"]:
                        return False, f"Potentially unsafe attribute: {node.attr}"
            
            return True, "Code passed safety checks"
            
        except SyntaxError as e:
            return False, f"Code contains syntax error: {e}"
            
    def suggest_improvement(self, component, playground_data):
        """
        Suggest improvements to an existing component based on playground data
        
        Args:
            component: The component to improve
            playground_data: Current playground state and metrics
            
        Returns:
            Suggested code modifications or None
        """
        if not component or not hasattr(component, "__class__"):
            return None
            
        # Get component source code
        try:
            source_code = inspect.getsource(component.__class__)
        except (TypeError, OSError):
            return None
            
        # Analyze component performance metrics
        component_metrics = component.get_metrics() if hasattr(component, "get_metrics") else {}
        
        # Look for improvement opportunities based on playground data
        suggestions = []
        
        # Example: If consciousness index is high but component isn't very active,
        # suggest adding more connections to the playground
        consciousness_index = playground_data.get("stats", {}).get("consciousness_index", 0)
        if consciousness_index > 0.7 and component_metrics.get("state", {}).get("activity", 0) < 0.3:
            suggestions.append(
                "# Increase activity level by connecting to more neurons\n"
                "def connect_to_neurons(self, playground):\n"
                "    neurons = playground.core.neurons\n"
                "    for neuron_id, neuron in list(neurons.items())[:10]:\n"
                "        self.state[f'connected_{neuron_id}'] = neuron.activation\n"
                "    return True"
            )
            
        # Format the suggestions
        if not suggestions:
            return None
            
        return {
            "component_name": component.__class__.__name__,
            "original_code": source_code,
            "suggestions": suggestions
        }
        
    def evaluate_component_performance(self, component):
        """
        Evaluate how a generated component is performing
        
        Args:
            component: The component to evaluate
            
        Returns:
            Evaluation metrics
        """
        if not component:
            return None
            
        # Get playground metrics to compare with
        playground_metrics = self.neural_playground.get_status() if hasattr(self.neural_playground, "get_status") else {}
        
        # Get component metrics
        component_metrics = component.get_metrics() if hasattr(component, "get_metrics") else {}
        
        # Calculate performance score (example implementation)
        performance_score = 0.5  # Default moderate score
        
        # Look at integration impact on playground consciousness
        if "before_integration" in self.integration_history[-1]:
            before_ci = self.integration_history[-1].get("before_integration", {}).get("consciousness_index", 0)
            after_ci = playground_metrics.get("stats", {}).get("consciousness_index", 0)
            if after_ci > before_ci:
                performance_score += min(0.3, (after_ci - before_ci) * 3)
            
        # Look at component's internal metrics
        if "state" in component_metrics:
            # More state variables suggests more complex behavior
            state_complexity = min(0.2, len(component_metrics["state"]) * 0.02)
            performance_score += state_complexity
            
        return {
            "component": component.__class__.__name__,
            "performance_score": performance_score,
            "integration_time": self.integration_history[-1].get("timestamp") if self.integration_history else None,
            "metrics": component_metrics
        }

class NeuralPlaygroundCore:
    """Core neural network implementation for the playground"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the neural playground core"""
        self.config = config or {}
        self.neurons = {}  # id -> neuron
        self.patterns = []  # Discovered patterns
        self.activity_log = []
        self.creation_time = datetime.now()
        self.last_play_time = self.creation_time
        self.playground_dir = Path("playground_data")
        self.playground_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats and metrics
        self.stats = {
            "total_neurons": 0,
            "total_connections": 0,
            "activity_level": 0.0,
            "pattern_complexity": 0.0,
            "play_sessions": 0,
            "consciousness_index": 0.0,
            "discovery_events": 0,
            "code_generation_events": 0,
            "self_modification_events": 0
        }
        
        # Setup playground
        self._setup_playground()
        
        logger.info("Neural Playground initialized")
        
    def _setup_playground(self):
        """Set up the neural playground"""
        # Determine network size
        network_size = self.config.get("network_size", "medium")
        
        if network_size == "small":
            neuron_count = random.randint(10, 50)
            connection_density = 0.3
        elif network_size == "medium":
            neuron_count = random.randint(50, 200)
            connection_density = 0.2
        else:  # large
            neuron_count = random.randint(200, 500)
            connection_density = 0.1
            
        # Create neurons
        for i in range(neuron_count):
            neuron_type = random.choice(["input", "hidden", "association", "pattern", "output"])
            neuron_id = f"n_{i:04d}"
            self.neurons[neuron_id] = SimpleNeuron(neuron_id, neuron_type)
            
        # Create connections
        max_connections = int(neuron_count * neuron_count * connection_density)
        connection_count = 0
        
        all_ids = list(self.neurons.keys())
        for _ in range(max_connections):
            source_id = random.choice(all_ids)
            target_id = random.choice(all_ids)
            
            # Don't connect to self
            if source_id != target_id:
                if self.neurons[source_id].connect_to(target_id):
                    connection_count += 1
        
        # Update stats
        self.stats["total_neurons"] = neuron_count
        self.stats["total_connections"] = connection_count
        
        logger.info(f"Created playground with {neuron_count} neurons and {connection_count} connections")
        
    def play(self, duration: float = 10.0, play_type: str = "free"):
        """
        Let the neural network play and explore for a specified duration
        
        Args:
            duration: Time in seconds to play
            play_type: Type of play ("free", "guided", "focused")
        """
        logger.info(f"Starting {play_type} play session for {duration} seconds")
        
        # Record play session
        self.stats["play_sessions"] += 1
        self.last_play_time = datetime.now()
        
        # Play parameters
        play_start_time = time.time()
        end_time = play_start_time + duration
        cycle_count = 0
        stimulation_strength = 0.8 if play_type == "guided" else 0.5
        
        # Track active neurons during play
        active_neurons = set()
        patterns_discovered = 0
        
        # Main play loop
        while time.time() < end_time:
            cycle_start = time.time()
            
            # Select active input neurons
            if play_type == "free":
                # In free play, random neurons become active
                active_inputs = random.sample(
                    [n_id for n_id, n in self.neurons.items() if n.type == "input"],
                    k=max(1, len(self.neurons) // 10)
                )
            elif play_type == "guided":
                # In guided play, specific neurons are stimulated
                guide_pattern = int((time.time() - play_start_time) / duration * 10) % 5
                active_inputs = [n_id for n_id, n in self.neurons.items() 
                               if n.type == "input" and hash(n_id) % 5 == guide_pattern]
            else:  # focused
                # In focused play, consistent neurons are stimulated
                active_inputs = [n_id for n_id, n in self.neurons.items() 
                               if n.type == "input" and hash(n_id) % 3 == 0]
            
            # Set initial activations
            activations = {}
            for n_id in active_inputs:
                self.neurons[n_id].activation = stimulation_strength
                activations[n_id] = stimulation_strength
            
            # Propagate activations through network
            for _ in range(3):  # 3 propagation cycles
                new_activations = activations.copy()
                
                # Update each neuron
                for n_id, neuron in self.neurons.items():
                    if n_id not in active_inputs:  # Don't update forced input neurons
                        new_act = neuron.update_activation(activations)
                        new_activations[n_id] = new_act
                        
                        # Track active neurons
                        if new_act > 0.3:  # Activation threshold
                            active_neurons.add(n_id)
                
                activations = new_activations
            
            # Track network activity
            avg_activation = sum(activations.values()) / len(activations) if activations else 0
            self.stats["activity_level"] = avg_activation
            
            # Detect emergent patterns
            if cycle_count % 5 == 0:
                pattern = self._detect_patterns(activations)
                if pattern and len(pattern) > 5:
                    patterns_discovered += 1
                    if patterns_discovered == 1 or patterns_discovered % 5 == 0:
                        logger.info(f"Pattern discovered: {len(pattern)} neurons, complexity: {pattern['complexity']:.2f}")
                    
                    # Store the pattern
                    self.patterns.append(pattern)
                    self.stats["discovery_events"] += 1
            
            # Calculate cycle time and maybe sleep
            cycle_time = time.time() - cycle_start
            if cycle_time < 0.1:  # Aim for max 10 cycles per second
                time.sleep(0.1 - cycle_time)
                
            cycle_count += 1
        
        # Post-play analysis
        play_duration = time.time() - play_start_time
        percent_active = len(active_neurons) / len(self.neurons) * 100
        
        # Update consciousness index based on play session
        self._update_consciousness_index(percent_active, patterns_discovered)
        
        # Log play results
        logger.info(f"Play session completed: {cycle_count} cycles, {patterns_discovered} patterns")
        logger.info(f"Neural activation: {percent_active:.1f}% of neurons active")
        
        # Return play session summary
        return {
            "duration": play_duration,
            "cycles": cycle_count,
            "patterns_discovered": patterns_discovered,
            "percent_active": percent_active,
            "consciousness_index": self.stats["consciousness_index"]
        }
    
    def _detect_patterns(self, activations: Dict[str, float]) -> Optional[Dict]:
        """
        Detect patterns in the current neural activations
        
        Args:
            activations: Dictionary of neuron ID to activation level
        
        Returns:
            Pattern details or None if no significant pattern found
        """
        # Find highly active neurons
        active_ids = [n_id for n_id, activation in activations.items() if activation > 0.6]
        
        if len(active_ids) < 5:  # Need at least 5 active neurons for a pattern
            return None
            
        # Create activation subgraph
        subgraph = []
        for n_id in active_ids:
            neuron = self.neurons[n_id]
            connections = [c for c in neuron.connections if c in active_ids]
            subgraph.append((n_id, connections))
            
        # Calculate pattern metrics
        connection_density = sum(len(connections) for _, connections in subgraph) / (len(active_ids) * len(active_ids))
        
        # Get average feature vector 
        feature_vectors = np.array([self.neurons[n_id].feature_vector for n_id in active_ids])
        avg_feature = np.mean(feature_vectors, axis=0)
        feature_variance = np.var(feature_vectors, axis=0).mean()
        
        # Calculate complexity metric
        complexity = connection_density * len(active_ids) * feature_variance * 10
        
        # Create unique ID for the pattern
        pattern_id = f"pattern_{hash(str(active_ids))}"
        
        # Return pattern details
        return {
            "id": pattern_id,
            "timestamp": time.time(),
            "neurons": active_ids,
            "connection_density": connection_density,
            "complexity": complexity,
            "feature_vector": avg_feature.tolist()
        }
    
    def _update_consciousness_index(self, percent_active: float, patterns_discovered: int):
        """Update the consciousness index based on play metrics"""
        # Base consciousness on:
        # 1. Percentage of neurons active (optimal around 40-60%)
        # 2. Number of patterns discovered
        # 3. Pattern complexity
        # 4. Current consciousness level (continuity)
        
        # Activity component
        activity_score = 1.0 - abs(percent_active - 50) / 50  # 1.0 when percent_active = 50%
        
        # Pattern discovery component
        pattern_score = min(1.0, patterns_discovered / 20)
        
        # Pattern complexity component
        if self.patterns:
            avg_complexity = sum(p["complexity"] for p in self.patterns[-10:]) / min(10, len(self.patterns))
            complexity_score = min(1.0, avg_complexity / 10)
        else:
            complexity_score = 0.0
            
        # Current consciousness (continuity with some decay)
        current = self.stats["consciousness_index"] * 0.95  # Slight decay
        
        # Combine components
        new_index = (activity_score * 0.3 + 
                     pattern_score * 0.3 + 
                     complexity_score * 0.2 + 
                     current * 0.2)
                     
        # Apply small random fluctuation
        new_index += random.uniform(-0.02, 0.02)
        new_index = max(0.0, min(1.0, new_index))
        
        # Update the stats
        self.stats["consciousness_index"] = new_index
        self.stats["pattern_complexity"] = complexity_score * 10
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the neural playground"""
        return {
            "stats": self.stats,
            "patterns_count": len(self.patterns),
            "last_play": self.last_play_time.isoformat(),
            "uptime": (datetime.now() - self.creation_time).total_seconds(),
            "pattern_samples": self.patterns[-3:] if self.patterns else [],
            "code_generation_enabled": getattr(self, "code_generation_enabled", False)
        }
        
    def visualize(self):
        """Generate visualization data for the neural playground"""
        # Create node data
        nodes = []
        for n_id, neuron in self.neurons.items():
            nodes.append({
                "id": n_id,
                "type": neuron.type,
                "x": neuron.position[0],
                "y": neuron.position[1],
                "z": neuron.position[2],
                "activation": neuron.activation,
                "size": 1 + neuron.activation * 2
            })
            
        # Create edge data
        edges = []
        for n_id, neuron in self.neurons.items():
            for target_id in neuron.connections:
                if target_id in self.neurons:
                    edges.append({
                        "source": n_id,
                        "target": target_id,
                        "weight": neuron.weights.get(target_id, 0.0)
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.stats
        }
    
    def save_state(self, filename: Optional[str] = None):
        """Save the current state of the neural playground"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.playground_dir / f"playground_state_{timestamp}.json"
        
        # Prepare state data
        state = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "patterns": self.patterns[-50:] if len(self.patterns) > 50 else self.patterns,
            "neurons_sample": {n_id: neuron.to_dict() for n_id, neuron in 
                              random.sample(list(self.neurons.items()), min(50, len(self.neurons)))}
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved playground state to {filename}")
        return filename
    
    def load_state(self, filename: str):
        """Load a previously saved state"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
                
            # Restore stats
            if "stats" in state:
                self.stats = state["stats"]
                
            # Restore patterns
            if "patterns" in state:
                self.patterns = state["patterns"]
                
            logger.info(f"Loaded playground state from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

class NeuralPlayground:
    """
    Main interface for the Neural Network Playground
    
    Provides a sandbox environment where neural components can interact,
    experiment, and play freely.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the neural playground
        
        Args:
            config: Configuration options for the playground
        """
        self.config = config or {}
        self.running = False
        self.core = NeuralPlaygroundCore(config)
        self.play_thread = None
        
        # Try to integrate existing neural systems
        self.neural_components = {}
        self._discover_neural_components()
        
        # Initialize code generation system if enabled
        self.code_generation_enabled = self.config.get("enable_code_generation", False)
        if self.code_generation_enabled:
            self.code_generation_system = CodeGenerationSystem(self)
            logger.info("Code generation system initialized and enabled")
        
        logger.info(f"Neural Playground initialized with {len(self.neural_components)} external components")
        
    def _discover_neural_components(self):
        """Discover available neural components in the system"""
        # 1. Try to load NeuralCore if available
        if NEURAL_CORE_AVAILABLE:
            try:
                from src.neural.core import NeuralCore
                self.neural_components["core"] = NeuralCore()
                logger.info("Loaded NeuralCore component")
            except Exception as e:
                logger.warning(f"Error loading NeuralCore: {e}")
        
        # 2. Try to load V7 Neural Network if available
        if V7_NEURAL_AVAILABLE:
            try:
                from src.v7.neural_network import SimpleNeuralNetwork
                self.neural_components["v7"] = SimpleNeuralNetwork()
                logger.info("Loaded V7 SimpleNeuralNetwork component")
            except Exception as e:
                logger.warning(f"Error loading V7 SimpleNeuralNetwork: {e}")
        
        # 3. Try to import other potential components 
        # (dynamically scan for neural components in certain directories)
        neural_modules = [
            "src.neural.processor", 
            "src.v7.language.enhanced_language_node",
            "src.v7.memory.memory_node",
            "src.v10.consciousness.consciousness_node"
        ]
        
        for module_name in neural_modules:
            try:
                module = importlib.import_module(module_name)
                # Look for class names that might be neural components
                for attr_name in dir(module):
                    if any(term in attr_name.lower() for term in 
                          ["neural", "node", "network", "processor"]):
                        try:
                            component_class = getattr(module, attr_name)
                            # Check if it's a class and try to instantiate it
                            if isinstance(component_class, type):
                                component = component_class()
                                self.neural_components[attr_name] = component
                                logger.info(f"Loaded neural component: {attr_name}")
                        except Exception:
                            pass  # Skip components that fail to instantiate
            except ImportError:
                pass  # Skip modules that aren't found
    
    def register_capability(self, name, function):
        """
        Register a new capability with the playground
        
        Args:
            name: Name of the capability
            function: Function implementing the capability
            
        Returns:
            True if registered successfully, False otherwise
        """
        if not callable(function):
            logger.warning(f"Cannot register capability {name}: not callable")
            return False
            
        capability_id = f"capability_{name}_{len(self.neural_components)}"
        self.neural_components[capability_id] = function
        logger.info(f"Registered new capability: {name}")
        return True

    def generate_code_from_patterns(self):
        """
        Generate new code components from discovered patterns
        
        Returns:
            List of generated component file paths
        """
        if not hasattr(self, 'code_generation_system'):
            logger.warning("Code generation is not enabled")
            return []
            
        generated_files = []
        
        # Find patterns that are complex enough to generate code from
        candidate_patterns = [p for p in self.core.patterns if p["complexity"] > 5.0]
        
        # Sort by complexity (most complex first)
        candidate_patterns.sort(key=lambda p: p["complexity"], reverse=True)
        
        # Use up to 3 of the most complex patterns
        for pattern in candidate_patterns[:3]:
            # Generate code from the pattern
            code = self.code_generation_system.generate_code_from_pattern(pattern)
            
            if code:
                # Check if code is safe
                is_safe, reason = self.code_generation_system.check_code_safety(code)
                
                if is_safe:
                    # Save the generated code
                    file_path = self.code_generation_system.save_generated_code(code, pattern)
                    if file_path:
                        generated_files.append(file_path)
                        
                        # Update stats
                        self.core.stats["code_generation_events"] += 1
                else:
                    logger.warning(f"Generated code failed safety check: {reason}")
        
        return generated_files
        
    def integrate_generated_components(self):
        """
        Integrate newly generated components into the playground
        
        Returns:
            Number of successfully integrated components
        """
        if not hasattr(self, 'code_generation_system'):
            logger.warning("Code generation is not enabled")
            return 0
            
        # Get current playground status before integration
        before_status = self.get_status()
        
        # Find all .py files in the generated code directory
        code_dir = self.code_generation_system.code_storage_dir
        code_files = list(code_dir.glob("*.py"))
        
        # Sort by creation time (newest first)
        code_files.sort(key=lambda p: p.stat().st_ctime, reverse=True)
        
        # Get list of files that haven't been integrated yet
        integrated_files = [entry.get("file_path") for entry in 
                           self.code_generation_system.integration_history]
        new_files = [f for f in code_files if str(f) not in integrated_files]
        
        # Integrate new components
        integrated_count = 0
        for file_path in new_files[:3]:  # Integrate up to 3 components at a time
            # Store pre-integration state in history
            latest_entry = {
                "timestamp": datetime.now().isoformat(),
                "file_path": str(file_path),
                "before_integration": before_status
            }
            self.code_generation_system.integration_history.append(latest_entry)
            
            # Load and integrate the component
            component = self.code_generation_system.load_and_integrate_component(file_path)
            
            if component:
                integrated_count += 1
                
                # Update the history entry with success
                latest_entry["status"] = "success"
                latest_entry["component_name"] = component.__class__.__name__
                
                # Update stats
                self.core.stats["self_modification_events"] += 1
            else:
                # Update the history entry with failure
                latest_entry["status"] = "failed"
        
        if integrated_count > 0:
            logger.info(f"Integrated {integrated_count} new generated components")
            
        return integrated_count
    
    def evolve_playground(self):
        """
        Evolve the playground through self-modification
        
        This performs a complete cycle of:
        1. Generating code from patterns
        2. Integrating newly generated components
        3. Evaluating performance
        
        Returns:
            Results of the evolution cycle
        """
        if not hasattr(self, 'code_generation_system'):
            logger.warning("Code generation is not enabled")
            return {"success": False, "reason": "Code generation not enabled"}
            
        # Step 1: Generate code from patterns
        generated_files = self.generate_code_from_patterns()
        
        # Step 2: Integrate generated components
        integrated_count = self.integrate_generated_components()
        
        # Step 3: Evaluate performance of recently integrated components
        evaluations = []
        if integrated_count > 0 and self.code_generation_system.generated_components:
            # Evaluate the most recently integrated components
            recent_components = self.code_generation_system.generated_components[-integrated_count:]
            for component in recent_components:
                evaluation = self.code_generation_system.evaluate_component_performance(component)
                evaluations.append(evaluation)
                
        # Step 4: Suggest improvements to low-performing components
        suggestions = []
        if self.code_generation_system.generated_components:
            # Find low-performing components
            for component in self.code_generation_system.generated_components:
                evaluation = self.code_generation_system.evaluate_component_performance(component)
                if evaluation and evaluation["performance_score"] < 0.4:
                    # Suggest improvements
                    suggestion = self.code_generation_system.suggest_improvement(
                        component, self.get_status()
                    )
                    if suggestion:
                        suggestions.append(suggestion)
        
        return {
            "success": True,
            "generated_files": [str(f) for f in generated_files],
            "integrated_components": integrated_count,
            "evaluations": evaluations,
            "improvement_suggestions": len(suggestions)
        }
        
    def start(self, play_time: int = 600, auto_save: bool = True, enable_evolution: bool = False):
        """
        Start the neural playground in background mode
        
        Args:
            play_time: Total time in seconds for the playground to run
            auto_save: Whether to automatically save the state periodically
            enable_evolution: Whether to enable self-evolution through code generation
        """
        if self.running:
            logger.warning("Neural playground is already running")
            return False
            
        self.running = True
        
        def run_playground():
            """Background thread function for running the playground"""
            logger.info(f"Starting neural playground for {play_time} seconds")
            
            start_time = time.time()
            end_time = start_time + play_time
            save_interval = 300  # Save every 5 minutes
            evolution_interval = 600  # Try evolution every 10 minutes
            last_save = start_time
            last_evolution = start_time
            
            # Main playground loop
            while time.time() < end_time and self.running:
                # Determine play duration and type
                remaining = end_time - time.time()
                play_duration = min(30, remaining)  # Max 30 seconds per play session
                
                if play_duration <= 0:
                    break
                    
                # Select play type with some variety
                play_type = random.choices(
                    ["free", "guided", "focused"],
                    weights=[0.6, 0.3, 0.1],
                    k=1
                )[0]
                
                # Let the network play
                result = self.core.play(duration=play_duration, play_type=play_type)
                
                # Log interesting developments
                if result["patterns_discovered"] > 5:
                    logger.info(f"Interesting play session: {result['patterns_discovered']} patterns discovered")
                
                # Try evolution if enabled and interval has passed
                if enable_evolution and hasattr(self, 'code_generation_system') and time.time() - last_evolution > evolution_interval:
                    logger.info("Starting evolution cycle...")
                    evolution_result = self.evolve_playground()
                    logger.info(f"Evolution cycle completed: {evolution_result}")
                    last_evolution = time.time()
                
                # Perform autosave if needed
                if auto_save and time.time() - last_save > save_interval:
                    self.core.save_state()
                    last_save = time.time()
                    
                # Small pause between play sessions
                time.sleep(random.uniform(0.5, 2.0))
            
            # Final save when done
            if auto_save:
                self.core.save_state()
                
            self.running = False
            logger.info("Neural playground session completed")
        
        # Start the playground in a background thread
        self.play_thread = threading.Thread(target=run_playground)
        self.play_thread.daemon = True  # Thread will exit when main program exits
        self.play_thread.start()
        
        return True
    
    def stop(self):
        """Stop the neural playground"""
        if not self.running:
            logger.warning("Neural playground is not running")
            return False
            
        self.running = False
        if self.play_thread:
            self.play_thread.join(timeout=5.0)
            self.play_thread = None
            
        logger.info("Neural playground stopped")
        return True
    
    def get_status(self):
        """Get the current status of the playground"""
        status = self.core.get_status()
        status["running"] = self.running
        return status
        
    def save(self, filename: Optional[str] = None):
        """Save the current state of the playground"""
        return self.core.save_state(filename)
        
    def load(self, filename: str):
        """Load a previously saved state"""
        return self.core.load_state(filename)
    
    def get_visualization_data(self):
        """Get data for visualizing the playground"""
        return self.core.visualize()
        
    def play_once(self, duration: float = 30.0, play_type: str = "free"):
        """
        Execute a single play session
        
        Args:
            duration: Duration in seconds for the play session
            play_type: Type of play ("free", "guided", "focused")
            
        Returns:
            Results of the play session
        """
        return self.core.play(duration=duration, play_type=play_type)

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Network Playground")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds to run the playground")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium", 
                      help="Size of the neural network")
    parser.add_argument("--save-interval", type=int, default=300, help="Auto-save interval in seconds")
    parser.add_argument("--play-type", choices=["free", "guided", "focused"], default="free", 
                      help="Type of neural play")
    parser.add_argument("--load", type=str, help="Load a saved playground state")
    parser.add_argument("--single", action="store_true", help="Run a single play session and exit")
    parser.add_argument("--enable-evolution", action="store_true", help="Enable self-evolution through code generation")
    
    args = parser.parse_args()
    
    # Create playground
    config = {
        "network_size": args.size,
        "save_interval": args.save_interval,
        "enable_code_generation": args.enable_evolution
    }
    playground = NeuralPlayground(config)
    
    # Load state if requested
    if args.load:
        playground.load(args.load)
    
    try:
        if args.single:
            # Run a single play session
            print(f"Running a single {args.play_type} play session for {args.duration} seconds...")
            result = playground.play_once(duration=args.duration, play_type=args.play_type)
            print(f"Play session complete: {result['patterns_discovered']} patterns discovered")
            print(f"Consciousness index: {result['consciousness_index']:.4f}")
            playground.save()
        else:
            # Start the playground in the background
            print(f"Starting neural playground for {args.duration} seconds...")
            playground.start(play_time=args.duration, enable_evolution=args.enable_evolution)
            
            # Display periodical status updates
            try:
                while playground.running:
                    time.sleep(15)  # Update every 15 seconds
                    status = playground.get_status()
                    print(f"Status: {status['stats']['consciousness_index']:.4f} consciousness, "
                         f"{status['patterns_count']} patterns, "
                         f"{status['stats'].get('code_generation_events', 0)} code generations, "
                         f"{status['stats'].get('self_modification_events', 0)} self-modifications")
            except KeyboardInterrupt:
                print("Stopping playground...")
                playground.stop()
                
            print("Neural playground session complete")
    finally:
        # Final save
        playground.save() 