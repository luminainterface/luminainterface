import logging
from typing import Dict, List, Any, Optional, Callable
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseComponent:
    """Base class for all components to extend"""
    def __init__(self):
        self.central_node = None
        self.dependencies = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def set_central_node(self, central_node):
        """Set a reference to the central node"""
        self.central_node = central_node
        
    def add_dependency(self, name, component):
        """Add a dependency component"""
        self.dependencies[name] = component
        
    def get_dependency(self, name):
        """Get a dependency by name"""
        return self.dependencies.get(name)
        
    def process(self, data):
        """Default processing method"""
        return data

class MinimalCentralNode:
    """Minimal version of the central node that doesn't require external dependencies"""
    def __init__(self):
        self.logger = logging.getLogger('MinimalCentralNode')
        self.nodes = {}
        self.processors = {}
        self.component_registry = {}
        self.connections = {}
        self.data_flow_pipeline = {
            'input': self._process_input,
            'resonance_encoder': self._resonance_encoding,
            'fractal_recursive_core': self._fractal_processing,
            'echo_spiral_memory': self._echo_processing,
            'mirror_engine': self._mirror_processing,
            'chronoglyph_decoder': self._chronoglyph_processing,
            'semantic_mapper': self._semantic_mapping,
            'output': self._generate_output
        }

    def register_component(self, name: str, component: Any, component_type: str = None):
        """Register a component with the central node"""
        self.component_registry[name] = component
        
        if component_type == 'node':
            self.nodes[name] = component
        elif component_type == 'processor':
            self.processors[name] = component
            
        # Set central node reference
        if hasattr(component, 'set_central_node'):
            component.set_central_node(self)
            
        self.logger.info(f"Registered component: {name}")
        return component

    def connect_components(self, source: str, target: str):
        """Connect two components"""
        # Create connection entry
        if source not in self.connections:
            self.connections[source] = []
        
        if target not in self.connections[source]:
            self.connections[source].append(target)
            
        # Add direct reference if possible
        source_comp = self.get_component(source)
        target_comp = self.get_component(target)
        
        if source_comp and target_comp and hasattr(source_comp, 'add_dependency'):
            source_comp.add_dependency(target, target_comp)
            
        self.logger.info(f"Connected {source} to {target}")

    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a component by name"""
        return self.component_registry.get(component_name)

    def get_node(self, node_name: str) -> Optional[Any]:
        """Get a node by name"""
        return self.nodes.get(node_name)

    def get_processor(self, processor_name: str) -> Optional[Any]:
        """Get a processor by name"""
        return self.processors.get(processor_name)

    def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input (Symbol / Emotion / Breath / Paradox)"""
        self.logger.info("Processing input data")
        return input_data
        
    def _resonance_encoding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through resonance encoder"""
        self.logger.info("Resonance encoding")
        return data
    
    def _fractal_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through fractal recursive core"""
        self.logger.info("Fractal recursive processing")
        return data
    
    def _echo_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through echo spiral memory"""
        self.logger.info("Echo spiral memory processing")
        return data
    
    def _mirror_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through mirror engine"""
        self.logger.info("Mirror engine processing")
        return data
    
    def _chronoglyph_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through chronoglyph decoder"""
        self.logger.info("Chronoglyph decoder processing")
        return data
    
    def _semantic_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through semantic mapper"""
        self.logger.info("Semantic mapping processing")
        return data
    
    def _generate_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final output (Action, Glyph, Story, Signal)"""
        self.logger.info("Generating output")
        
        output = {
            'action': data.get('action', None),
            'glyph': data.get('glyph', None),
            'story': data.get('story', None),
            'signal': data.get('signal', None)
        }
        
        return output
    
    def process_complete_flow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the complete flow pipeline"""
        self.logger.info("Starting complete flow processing")
        
        # Process through each stage of the pipeline
        data = self._process_input(input_data)
        data = self._resonance_encoding(data)
        data = self._fractal_processing(data)
        
        # Split for parallel processing
        echo_data = self._echo_processing(data.copy())
        mirror_data = self._mirror_processing(data.copy())
        
        # Merge results
        for key, value in mirror_data.items():
            if key not in echo_data:
                echo_data[key] = value
            
        # Continue processing
        data = self._chronoglyph_processing(echo_data)
        data = self._semantic_mapping(data)
        output = self._generate_output(data)
        
        self.logger.info("Completed flow processing")
        return output

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the central node"""
        return {
            'nodes': len(self.nodes),
            'processors': len(self.processors),
            'components': len(self.component_registry),
            'connections': len(self.connections)
        }

# Example mock components
class MockRSEN(BaseComponent):
    def process_data(self, data):
        self.logger.info("MockRSEN processing data")
        data['resonance'] = 0.85
        return data

class MockFractalNodes(BaseComponent):
    def get_patterns(self):
        self.logger.info("MockFractalNodes getting patterns")
        return ["spiral", "loop", "mandelbrot"]
        
    def process_suggestions(self, suggestions):
        self.logger.info(f"MockFractalNodes processing suggestions: {suggestions}")
        return suggestions

class MockConsciousness(BaseComponent):
    def reflect(self, data):
        self.logger.info("MockConsciousness reflecting")
        data['reflection'] = True
        return data

class MockLanguageProcessor(BaseComponent):
    def process(self, data):
        self.logger.info("MockLanguageProcessor processing")
        data['language_processed'] = True
        return data

class MockNeuralProcessor(BaseComponent):
    def process(self, data):
        self.logger.info("MockNeuralProcessor processing")
        data['semantically_mapped'] = True
        data['action'] = "explore"
        data['glyph'] = "⚛"
        data['story'] = "A journey through fractal dimensions"
        data['signal'] = 1.0
        return data

def test_minimal_central_node():
    """Test the minimal central node implementation"""
    print("Testing minimal central node...")
    
    # Create central node
    central = MinimalCentralNode()
    
    # Create and register mock components
    rsen = central.register_component('RSEN', MockRSEN(), 'node')
    fractal = central.register_component('FractalNodes', MockFractalNodes(), 'node')
    consciousness = central.register_component('ConsciousnessNode', MockConsciousness(), 'node')
    language = central.register_component('LanguageProcessor', MockLanguageProcessor(), 'processor')
    neural = central.register_component('NeuralProcessor', MockNeuralProcessor(), 'processor')
    
    # Create connections
    central.connect_components('RSEN', 'NeuralProcessor')
    central.connect_components('FractalNodes', 'NeuralProcessor')
    central.connect_components('ConsciousnessNode', 'LanguageProcessor')
    
    # Override the pipeline methods
    central._resonance_encoding = lambda data: central.get_node('RSEN').process_data(data)
    central._fractal_processing = lambda data: data.update({'patterns': central.get_node('FractalNodes').get_patterns()}) or data
    central._mirror_processing = lambda data: central.get_node('ConsciousnessNode').reflect(data)
    central._chronoglyph_processing = lambda data: central.get_processor('LanguageProcessor').process(data)
    central._semantic_mapping = lambda data: central.get_processor('NeuralProcessor').process(data)
    
    # Test the flow
    input_data = {
        'symbol': 'infinity',
        'emotion': 'wonder',
        'breath': 'deep',
        'paradox': 'existence'
    }
    
    # Process through pipeline
    output = central.process_complete_flow(input_data)
    
    # Print results
    print("\n========================================")
    print("       CENTRAL NODE TEST RESULTS        ")
    print("========================================")
    print(f"System status: {central.get_system_status()}")
    print("\nOutput from pipeline:")
    for key, value in output.items():
        print(f"  - {key}: {value}")
    print("========================================")
    
    return central, output

if __name__ == "__main__":
    test_minimal_central_node() 