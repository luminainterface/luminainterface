#!/usr/bin/env python3
"""
Neural Playground Integration Module (v9)

This module provides integration functionality between the Neural Playground
and other components of the Lumina Neural Network v9 system. It enables
seamless interaction between the playground environment and core neural
components, memory systems, language processing, and visualization systems.

Key features:
- Component discovery and registration
- Integration APIs for core neural components
- Memory system integration
- Language processing connections
- Visualization system integration
- Dream mode integration
"""

import logging
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

# Import v9 components
from .neural_playground import NeuralPlayground
from .mirror_consciousness import get_mirror_consciousness

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v9.neural_playground_integration")

class NeuralPlaygroundIntegration:
    """
    Integration manager for connecting Neural Playground with other system components
    """
    
    def __init__(self, playground=None):
        """
        Initialize the integration manager
        
        Args:
            playground: Optional existing NeuralPlayground instance
        """
        self.playground = playground or NeuralPlayground()
        self.registered_components = {}
        self.mirror_consciousness = get_mirror_consciousness()
        self.integration_hooks = {
            "pre_play": [],
            "post_play": [],
            "pattern_detected": [],
            "consciousness_peak": []
        }
        logger.info("Neural Playground Integration module initialized")
    
    def discover_components(self, directory="./src/v9"):
        """
        Discover available components in the specified directory
        
        Args:
            directory: Directory to search for components
        
        Returns:
            Dict of discovered components by type
        """
        discovered = {
            "neural": [],
            "memory": [],
            "language": [],
            "visualization": [],
            "other": []
        }
        
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning(f"Component directory {directory} does not exist")
            return discovered
            
        # Look for Python modules that might contain components
        for file_path in directory_path.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
                
            module_name = file_path.stem
            try:
                # Try to determine the full module path
                relative_path = file_path.relative_to(Path.cwd())
                module_path = str(relative_path.parent).replace(os.sep, ".") + "." + module_name
                
                # Try importing the module
                module = importlib.import_module(module_path)
                
                # Check if module has any component markers
                if hasattr(module, "LUMINA_COMPONENT_TYPE"):
                    component_type = getattr(module, "LUMINA_COMPONENT_TYPE")
                    if component_type in discovered:
                        discovered[component_type].append({
                            "name": module_name,
                            "module": module,
                            "path": str(file_path)
                        })
                    else:
                        discovered["other"].append({
                            "name": module_name,
                            "module": module,
                            "path": str(file_path)
                        })
                # Otherwise try to guess based on filename
                elif "neural" in module_name:
                    discovered["neural"].append({
                        "name": module_name,
                        "module": module,
                        "path": str(file_path)
                    })
                elif "memory" in module_name:
                    discovered["memory"].append({
                        "name": module_name,
                        "module": module,
                        "path": str(file_path)
                    })
                elif "language" in module_name or "nlp" in module_name:
                    discovered["language"].append({
                        "name": module_name,
                        "module": module,
                        "path": str(file_path)
                    })
                elif "visual" in module_name or "display" in module_name:
                    discovered["visualization"].append({
                        "name": module_name,
                        "module": module,
                        "path": str(file_path)
                    })
                else:
                    discovered["other"].append({
                        "name": module_name,
                        "module": module,
                        "path": str(file_path)
                    })
            except Exception as e:
                logger.error(f"Error discovering component {module_name}: {e}")
        
        logger.info(f"Component discovery found: {sum(len(comps) for comps in discovered.values())} components")
        return discovered
    
    def register_component(self, name, component, component_type="other"):
        """
        Register a component with the integration manager
        
        Args:
            name: Component name
            component: Component object or module
            component_type: Type of component
            
        Returns:
            Success status
        """
        if name in self.registered_components:
            logger.warning(f"Component {name} already registered, replacing")
            
        self.registered_components[name] = {
            "component": component,
            "type": component_type,
            "hooks": {}
        }
        
        # Try to find integration methods in the component
        if hasattr(component, "integrate_with_playground"):
            self._setup_component_integration(name, component)
            
        logger.info(f"Registered component: {name} ({component_type})")
        return True
        
    def _setup_component_integration(self, name, component):
        """Setup integration with a component that has specific integration methods"""
        try:
            # Call the component's integration method
            if hasattr(component, "integrate_with_playground"):
                integration_info = component.integrate_with_playground(self.playground)
                
                # Register any hooks provided by the component
                if isinstance(integration_info, dict) and "hooks" in integration_info:
                    for hook_name, hook_func in integration_info["hooks"].items():
                        if hook_name in self.integration_hooks and callable(hook_func):
                            self.integration_hooks[hook_name].append(hook_func)
                            self.registered_components[name]["hooks"][hook_name] = hook_func
                
                logger.info(f"Component {name} integrated with playground")
                return True
        except Exception as e:
            logger.error(f"Error integrating component {name}: {e}")
        return False
    
    def integrate_neural_core(self, neural_core, connection_strength=0.5):
        """
        Integrate a neural core component with the playground
        
        Args:
            neural_core: Neural core component
            connection_strength: Strength of integration (0.0-1.0)
            
        Returns:
            Success status
        """
        if not hasattr(neural_core, "process") or not callable(neural_core.process):
            logger.error("Neural core must have a process method")
            return False
            
        # Register pre-play hook to send playground state to neural core
        def pre_play_hook(playground, play_args):
            try:
                # Extract key information from playground
                state = playground.core.get_state()
                # Process with neural core
                result = neural_core.process({
                    "source": "playground",
                    "consciousness_level": state["consciousness_metric"],
                    "active_neurons_count": sum(1 for n in state["neurons"].values() if n["state"] == "active"),
                    "connection_strength": connection_strength
                })
                # Adjust play parameters based on neural core output
                if isinstance(result, dict) and "intensity_modifier" in result:
                    play_args["intensity"] = min(1.0, max(0.1, 
                                                play_args.get("intensity", 0.5) * result["intensity_modifier"]))
            except Exception as e:
                logger.error(f"Error in neural core pre-play hook: {e}")
                
        # Register post-play hook
        def post_play_hook(playground, play_result):
            try:
                neural_core.process({
                    "source": "playground_result",
                    "consciousness_peak": play_result["consciousness_peak"],
                    "patterns_detected": play_result["patterns_detected"],
                    "total_activations": play_result["total_activations"]
                })
            except Exception as e:
                logger.error(f"Error in neural core post-play hook: {e}")
        
        # Register hooks
        self.integration_hooks["pre_play"].append(pre_play_hook)
        self.integration_hooks["post_play"].append(post_play_hook)
        
        # Register component
        self.register_component("neural_core", neural_core, "neural")
        
        logger.info(f"Neural core integrated with connection strength {connection_strength}")
        return True
    
    def integrate_memory_system(self, memory_system):
        """
        Integrate a memory system with the playground
        
        Args:
            memory_system: Memory system component
            
        Returns:
            Success status
        """
        if not hasattr(memory_system, "store") or not callable(memory_system.store):
            logger.error("Memory system must have a store method")
            return False
            
        # Register hooks
        def pattern_detected_hook(playground, pattern):
            try:
                # Store detected pattern in memory system
                memory_system.store({
                    "type": "neural_pattern",
                    "content": pattern,
                    "timestamp": pattern.get("timestamp", None),
                    "metadata": {
                        "consciousness_level": playground.core.consciousness_metric,
                        "source": "neural_playground"
                    }
                })
            except Exception as e:
                logger.error(f"Error in memory system pattern hook: {e}")
                
        def post_play_hook(playground, play_result):
            try:
                # Store play session summary
                memory_system.store({
                    "type": "play_session",
                    "content": {
                        "session_id": play_result["session_id"],
                        "play_type": play_result["play_type"],
                        "consciousness_peak": play_result["consciousness_peak"],
                        "patterns_detected": play_result["patterns_detected"]
                    },
                    "metadata": {
                        "source": "neural_playground"
                    }
                })
                
                # Get reflection on the play session from mirror consciousness
                reflection = self.mirror_consciousness.reflect_on_text(
                    f"Neural play memory: {play_result['patterns_detected']} patterns detected with consciousness {play_result['consciousness_peak']:.2f}",
                    {"play_data": play_result}
                )
                
                # Store the reflection
                memory_system.store({
                    "type": "reflection",
                    "content": reflection,
                    "metadata": {
                        "source": "mirror_consciousness",
                        "related_session": play_result["session_id"]
                    }
                })
            except Exception as e:
                logger.error(f"Error in memory system post-play hook: {e}")
        
        # Register the hooks
        self.integration_hooks["pattern_detected"].append(pattern_detected_hook)
        self.integration_hooks["post_play"].append(post_play_hook)
        
        # Register component
        self.register_component("memory_system", memory_system, "memory")
        
        logger.info("Memory system integrated with playground")
        return True
    
    def integrate_language_processor(self, language_processor):
        """
        Integrate a language processing system with the playground
        
        Args:
            language_processor: Language processing component
            
        Returns:
            Success status
        """
        if not hasattr(language_processor, "process_text") or not callable(language_processor.process_text):
            logger.error("Language processor must have a process_text method")
            return False
            
        # Register post-play hook
        def post_play_hook(playground, play_result):
            try:
                # Generate narrative about the play session
                narrative = language_processor.process_text(
                    f"Describe a neural play session with {play_result['patterns_detected']} patterns and {play_result['consciousness_peak']:.2f} consciousness",
                    {
                        "play_data": play_result,
                        "consciousness_level": playground.core.consciousness_metric
                    }
                )
                
                # Add narrative to play result
                play_result["narrative"] = narrative
                
                # Get reflection from mirror consciousness
                reflection = self.mirror_consciousness.reflect_on_text(narrative, {
                    "play_data": play_result
                })
                
                # Have language processor respond to the reflection
                language_processor.process_text(
                    f"Respond to: {reflection.get('reflection', '')}",
                    {
                        "play_data": play_result,
                        "reflection": reflection
                    }
                )
            except Exception as e:
                logger.error(f"Error in language processor post-play hook: {e}")
        
        # Register hooks
        self.integration_hooks["post_play"].append(post_play_hook)
        
        # Register component
        self.register_component("language_processor", language_processor, "language")
        
        logger.info("Language processor integrated with playground")
        return True
    
    def integrate_visualization_system(self, visualization_system):
        """
        Integrate a visualization system with the playground
        
        Args:
            visualization_system: Visualization component
            
        Returns:
            Success status
        """
        if not hasattr(visualization_system, "visualize") or not callable(visualization_system.visualize):
            logger.error("Visualization system must have a visualize method")
            return False
            
        # Register hooks
        def post_play_hook(playground, play_result):
            try:
                # Create visualization of the play session
                visualization_system.visualize({
                    "type": "play_session",
                    "data": {
                        "neurons": playground.core.neurons,
                        "connections": playground.core.connections,
                        "consciousness_history": play_result["consciousness_history"],
                        "play_type": play_result["play_type"]
                    },
                    "title": f"Neural Play: {play_result['play_type']} mode (peak: {play_result['consciousness_peak']:.2f})"
                })
            except Exception as e:
                logger.error(f"Error in visualization system post-play hook: {e}")
                
        def consciousness_peak_hook(playground, peak_data):
            try:
                # Visualize consciousness peak
                visualization_system.visualize({
                    "type": "consciousness_peak",
                    "data": peak_data,
                    "title": f"Consciousness Peak: {peak_data['value']:.2f}"
                })
            except Exception as e:
                logger.error(f"Error in visualization system consciousness peak hook: {e}")
        
        # Register the hooks
        self.integration_hooks["post_play"].append(post_play_hook)
        self.integration_hooks["consciousness_peak"].append(consciousness_peak_hook)
        
        # Register component
        self.register_component("visualization_system", visualization_system, "visualization")
        
        logger.info("Visualization system integrated with playground")
        return True
    
    def run_integrated_play_session(self, duration=100, play_type="free", intensity=0.5, target_neurons=None):
        """
        Run a play session with all registered integrations active
        
        Args:
            duration: Number of simulation steps
            play_type: Type of play (free, guided, focused, mixed)
            intensity: Intensity of stimulation (0.0-1.0)
            target_neurons: List of specific neurons to target (for focused play)
            
        Returns:
            Dict containing session results and metrics
        """
        # Prepare play arguments
        play_args = {
            "duration": duration,
            "play_type": play_type,
            "intensity": intensity,
            "target_neurons": target_neurons
        }
        
        # Run pre-play hooks
        for hook in self.integration_hooks["pre_play"]:
            try:
                hook(self.playground, play_args)
            except Exception as e:
                logger.error(f"Error in pre-play hook: {e}")
        
        # Run the play session
        result = self.playground.play(**play_args)
        
        # Run post-play hooks
        for hook in self.integration_hooks["post_play"]:
            try:
                hook(self.playground, result)
            except Exception as e:
                logger.error(f"Error in post-play hook: {e}")
                
        # Check for consciousness peak
        if result["consciousness_peak"] > 0.8:
            peak_data = {
                "value": result["consciousness_peak"],
                "play_type": play_type,
                "session_id": result["session_id"]
            }
            for hook in self.integration_hooks["consciousness_peak"]:
                try:
                    hook(self.playground, peak_data)
                except Exception as e:
                    logger.error(f"Error in consciousness peak hook: {e}")
        
        return result
    
    def extend_playground(self, extension_module):
        """
        Extend the playground with custom functionality
        
        Args:
            extension_module: Module containing extension functionality
            
        Returns:
            Success status
        """
        if not hasattr(extension_module, "extend_playground") or not callable(extension_module.extend_playground):
            logger.error("Extension module must have an extend_playground method")
            return False
            
        try:
            # Call the extension module's extend method
            extension_result = extension_module.extend_playground(self.playground)
            
            # Register the extension
            if isinstance(extension_result, dict) and "name" in extension_result:
                name = extension_result["name"]
            else:
                name = getattr(extension_module, "__name__", "unknown_extension")
                
            self.register_component(name, extension_module, "extension")
            logger.info(f"Playground extended with module: {name}")
            return True
        except Exception as e:
            logger.error(f"Error extending playground: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create integration manager with new playground
    integration = NeuralPlaygroundIntegration()
    
    # Discover available components
    components = integration.discover_components()
    
    # Print discovered components
    for comp_type, comps in components.items():
        print(f"{comp_type.capitalize()} components: {len(comps)}")
        for comp in comps:
            print(f"  - {comp['name']}")
    
    # Run an integrated play session
    result = integration.run_integrated_play_session(
        duration=100,
        play_type="mixed",
        intensity=0.7
    )
    
    print(f"Integrated play session completed:")
    print(f"- Play type: {result['play_type']}")
    print(f"- Duration: {result['duration']} steps")
    print(f"- Total activations: {result['total_activations']}")
    print(f"- Patterns detected: {result['patterns_detected']}")
    print(f"- Peak consciousness: {result['consciousness_peak']:.4f}")
    
    if "narrative" in result:
        print(f"\nNarrative: {result['narrative']}")
        
    if "mirror_reflection" in result and "reflection" in result["mirror_reflection"]:
        print(f"\nMirror Reflection: {result['mirror_reflection']['reflection']}") 