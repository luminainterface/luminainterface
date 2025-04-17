import os
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import time
from PySide6.QtCore import QObject, QTimer
import importlib

# Import all nodes
from RSEN_node import RSEN
from hybrid_node import HybridNode
from node_zero import NodeZero
from portal_node import PortalNode
from wormhole_node import WormholeNode
from zpe_node import ZPENode
from neutrino_node import NeutrinoNode
from game_theory_node import GameTheoryNode
from consciousness_node import ConsciousnessNode
from gauge_theory_node import GaugeTheoryNode
from fractal_nodes import MandelbrotHybridNode as FractalNodes
from infinite_minds_node import InfiniteMindsNode
from void_infinity_node import VoidInfinityNode

# Import all processors
from neural_processor import NeuralProcessor
from language_processor import LanguageProcessor
from lumina_processor import LuminaProcessor
from mood_processor import EnhancedMoodCore as MoodProcessor
from node_manager import NodeManager
from wiki_learner import WikiLearner
from wiki_vocabulary import WikiVocabulary
from wikipedia_training_module import WikipediaTrainingModule
from wikipedia_trainer import train_wikipedia_model as WikipediaTrainer
from lumina_neural import LuminaNeural
from physics_engine import PhysicsEngine
from calculus_engine import CalculusEngine
from physics_metaphysics_framework import PhysicsMetaphysicsFramework
from hyperdimensional_thought import HyperdimensionalThought
from quantum_infection import QuantumInfection
from node_integration import NodeIntegrationSystem as NodeIntegration
from src.seed.pyside6_adapter import NeuralSeedAdapter as PySide6Adapter
from src.seed.neural_seed import NeuralSeed

class CentralNode(QObject):
    def __init__(self):
        """Initialize the central node with all components"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Central Node")
        
        # Initialize component registry
        self.component_registry = {}
        
        # Initialize node manager
        self.node_manager = NodeManager()
        
        # Initialize adapters
        self.pyside6_adapter = PySide6Adapter(self)
        
        # Initialize dictionaries and lists
        self.nodes: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.data_dirs: List[str] = []
        self.connections = {}
        
        # Initialize data flow pipeline
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
        
        # Initialize all nodes
        self._safe_initialize_components()
        
        # Start monitoring using QTimer
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self._monitor_components)
        self.monitor_timer.start(5000)  # Monitor every 5 seconds
        
        # Initialize version bridge after everything else
        from version_bridge_integration import VersionBridgeIntegration
        self.version_bridge = VersionBridgeIntegration(self)
        
        self.logger.info("Central Node initialization complete")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('CentralNode')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _safe_initialize_components(self):
        """Safely initialize all components with error handling"""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Define node modules to initialize
        node_modules = [
            ('RSEN', 'nodes.RSEN_node', 'RSEN'),
            ('HybridNode', 'nodes.hybrid_node', 'HybridNode'),
            ('FractalNodes', 'nodes.fractal_nodes', 'FractalNodes'),
            ('InfiniteMindsNode', 'nodes.infinite_minds_node', 'InfiniteMindsNode'),
            ('IsomorphNode', 'nodes.isomorph_node', 'IsomorphNode'),
            ('VortexNode', 'nodes.vortex_node', 'VortexNode')
        ]

        # Initialize nodes
        for node_name, module_name, class_name in node_modules:
            try:
                module = importlib.import_module(module_name)
                node_class = getattr(module, class_name)
                node_instance = node_class()
                
                # Connect node to central node and register
                self._register_component(node_name, node_instance)
                self.nodes[node_name] = node_instance
                
                # First initialize the node
                if hasattr(node_instance, 'initialize'):
                    if not node_instance.initialize():
                        raise Exception("Node initialization failed")
                
                # Then activate it
                if hasattr(node_instance, 'activate'):
                    if not node_instance.activate():
                        raise Exception("Node activation failed")
                
                self.logger.info(f"Successfully initialized and activated node: {node_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize node {node_name}: {str(e)}")
                continue

        # Define processor modules to initialize
        processor_modules = [
            ('NeuralProcessor', 'processors.neural_processor', 'NeuralProcessor'),
            ('LanguageProcessor', 'processors.language_processor', 'LanguageProcessor'),
            ('HyperdimensionalThought', 'processors.hyperdimensional_thought', 'HyperdimensionalThought')
        ]

        # Initialize processors
        for proc_name, module_name, class_name in processor_modules:
            try:
                module = importlib.import_module(module_name)
                proc_class = getattr(module, class_name)
                proc_instance = proc_class()
                
                # Connect processor to central node and register
                self._register_component(proc_name, proc_instance)
                self.processors[proc_name] = proc_instance
                
                # First initialize the processor
                if hasattr(proc_instance, 'initialize'):
                    if not proc_instance.initialize():
                        raise Exception("Processor initialization failed")
                
                # Then activate it
                if hasattr(proc_instance, 'activate'):
                    if not proc_instance.activate():
                        raise Exception("Processor activation failed")
                
                self.logger.info(f"Successfully initialized and activated processor: {proc_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize processor {proc_name}: {str(e)}")
                continue

        # Establish connections between components
        try:
            self._establish_connections()
            self.logger.info("Component connections established successfully")
        except Exception as e:
            self.logger.error(f"Failed to establish component connections: {str(e)}")

        self.logger.info(f"Component initialization complete. {len(self.nodes)} nodes and {len(self.processors)} processors activated.")

    def _safe_init_nodes(self):
        """Safely initialize node components with error handling"""
        node_modules = [
            ('RSEN', 'RSEN_node', 'RSEN'),
            ('HybridNode', 'hybrid_node', 'HybridNode'),
            ('NodeZero', 'node_zero', 'NodeZero'),
            ('PortalNode', 'portal_node', 'PortalNode'),
            ('WormholeNode', 'wormhole_node', 'WormholeNode'),
            ('ZPENode', 'zpe_node', 'ZPENode'),
            ('NeutrinoNode', 'neutrino_node', 'NeutrinoNode'),
            ('GameTheoryNode', 'game_theory_node', 'GameTheoryNode'),
            ('ConsciousnessNode', 'consciousness_node', 'ConsciousnessNode'),
            ('GaugeTheoryNode', 'gauge_theory_node', 'GaugeTheoryNode'),
            ('FractalNodes', 'fractal_nodes', 'MandelbrotHybridNode'),
            ('InfiniteMindsNode', 'infinite_minds_node', 'InfiniteMindsNode'),
            ('VoidInfinityNode', 'void_infinity_node', 'VoidInfinityNode'),
            ('NeuralSeed', 'src.seed.neural_seed', 'NeuralSeed')
        ]
        
        for node_name, module_name, class_name in node_modules:
            try:
                if node_name == 'NeuralSeed':
                    # Special handling for NeuralSeed
                    from src.seed.neural_seed import NeuralSeed
                    node_instance = NeuralSeed()
                    # Start the growth process
                    node_instance.start_growth()
                else:
                    module = __import__(module_name)
                    node_class = getattr(module, class_name)
                    node_instance = node_class()
                
                # Connect node to central node
                self._register_component(node_name, node_instance)
                
                # Store node
                self.nodes[node_name] = node_instance
                
                self.logger.info(f"Successfully initialized node: {node_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize node {node_name}: {str(e)}")
    
    def _safe_init_processors(self):
        """Safely initialize processor components with error handling"""
        processor_modules = [
            ('NeuralProcessor', 'neural_processor', 'NeuralProcessor'),
            ('LanguageProcessor', 'language_processor', 'LanguageProcessor'),
            ('LuminaProcessor', 'lumina_processor', 'LuminaProcessor'),
            ('MoodProcessor', 'mood_processor', 'EnhancedMoodCore'),
            ('NodeManager', 'node_manager', 'NodeManager'),
            ('WikiLearner', 'wiki_learner', 'WikiLearner'),
            ('WikiVocabulary', 'wiki_vocabulary', 'WikiVocabulary'),
            ('WikipediaTrainingModule', 'wikipedia_training_module', 'WikipediaTrainingModule'),
            ('WikipediaTrainer', 'wikipedia_trainer', 'train_wikipedia_model'),
            ('LuminaNeural', 'lumina_neural', 'LuminaNeural'),
            ('PhysicsEngine', 'physics_engine', 'PhysicsEngine'),
            ('CalculusEngine', 'calculus_engine', 'CalculusEngine'),
            ('PhysicsMetaphysicsFramework', 'physics_metaphysics_framework', 'PhysicsMetaphysicsFramework'),
            ('HyperdimensionalThought', 'hyperdimensional_thought', 'HyperdimensionalThought'),
            ('QuantumInfection', 'quantum_infection', 'QuantumInfection'),
            ('NodeIntegration', 'node_integration', 'NodeIntegrationSystem')
        ]
        
        for proc_name, module_name, class_name in processor_modules:
            try:
                if proc_name == 'WikipediaTrainer':
                    # Special handling for WikipediaTrainer
                    from wikipedia_trainer import train_wikipedia_model
                    proc_instance = train_wikipedia_model
                else:
                    module = __import__(module_name)
                    proc_class = getattr(module, class_name)
                    proc_instance = proc_class()
                
                # Connect processor to central node
                self._register_component(proc_name, proc_instance)
                
                # Store processor
                self.processors[proc_name] = proc_instance
                
                self.logger.info(f"Successfully initialized processor: {proc_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize processor {proc_name}: {str(e)}")

    def _establish_connections(self):
        """Establish connections between components based on dependencies"""
        try:
            # Connect NeuralSeed to other components
            if 'NeuralSeed' in self.nodes:
                self.neural_seed = self.nodes['NeuralSeed']
                
                # Connect to consciousness node if available
                if 'ConsciousnessNode' in self.nodes:
                    self.neural_seed.connect_to_consciousness(self.nodes['ConsciousnessNode'])
                
                # Connect to linguistic processor if available
                if 'LanguageProcessor' in self.processors:
                    self.neural_seed.connect_to_linguistic_processor(self.processors['LanguageProcessor'])
                
                # Connect to neural plasticity if available
                if 'NeuralProcessor' in self.processors:
                    self.neural_seed.connect_to_neural_plasticity(self.processors['NeuralProcessor'])
                
                self.logger.info("Established connections for NeuralSeed")
            
            # Connect other components
            if 'ConsciousnessNode' in self.nodes and 'LanguageProcessor' in self.processors:
                self.nodes['ConsciousnessNode'].connect_linguistic_processor(self.processors['LanguageProcessor'])
            
            if 'NeuralProcessor' in self.processors and 'LanguageProcessor' in self.processors:
                self.processors['NeuralProcessor'].connect_language_processor(self.processors['LanguageProcessor'])
            
            self.logger.info("Component connections established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to establish component connections: {str(e)}")

    def _register_component(self, name: str, component: Any):
        """Register a component to the central registry"""
        self.component_registry[name] = component
        
        # Try to add central node reference to component if it has a set_central_node method
        if hasattr(component, 'set_central_node'):
            try:
                component.set_central_node(self)
                self.logger.info(f"Set central node reference in {name}")
            except Exception as e:
                self.logger.error(f"Failed to set central node reference in {name}: {str(e)}")

    def _connect_components(self, source: str, target: str):
        """Connect two components"""
        # Create connection entry
        if source not in self.connections:
            self.connections[source] = []
        
        if target not in self.connections[source]:
            self.connections[source].append(target)
            
        # Try to set direct reference in source component if it has add_dependency method
        source_comp = self.get_component(source)
        target_comp = self.get_component(target)
        
        if source_comp and target_comp and hasattr(source_comp, 'add_dependency'):
            try:
                source_comp.add_dependency(target, target_comp)
                self.logger.info(f"Added direct dependency reference from {source} to {target}")
            except Exception as e:
                self.logger.error(f"Failed to add direct dependency reference: {str(e)}")

    def get_node(self, node_name: str) -> Optional[Any]:
        """Get a specific node by name"""
        return self.nodes.get(node_name)

    def get_processor(self, processor_name: str) -> Optional[Any]:
        """Get a specific processor by name"""
        return self.processors.get(processor_name)
        
    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a component by name (could be either node or processor)"""
        return self.component_registry.get(component_name)

    def list_available_components(self) -> Dict[str, List[str]]:
        """List all available components"""
        return {
            'nodes': list(self.nodes.keys()),
            'processors': list(self.processors.keys()),
            'data_directories': self.data_dirs
        }

    def execute_node_operation(self, node_name: str, operation: str, *args, **kwargs) -> Any:
        """Execute an operation on a specific node"""
        node = self.get_node(node_name)
        if not node:
            raise ValueError(f"Node {node_name} not found")
        
        if not hasattr(node, operation):
            raise ValueError(f"Operation {operation} not found in node {node_name}")
        
        return getattr(node, operation)(*args, **kwargs)

    def process_data(self, processor_name: str, data: Any, *args, **kwargs) -> Any:
        """Process data using a specific processor"""
        processor = self.get_processor(processor_name)
        if not processor:
            raise ValueError(f"Processor {processor_name} not found")
        
        if hasattr(processor, 'process'):
            return processor.process(data, *args, **kwargs)
        else:
            self.logger.error(f"Processor {processor_name} does not have a process method")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the system"""
        return {
            'active_nodes': len(self.nodes),
            'active_processors': len(self.processors),
            'data_directories': self.data_dirs,
            'total_components': len(self.nodes) + len(self.processors),
            'connections': len(self.connections)
        }

    def get_component_dependencies(self) -> Dict[str, List[str]]:
        """Get dependencies between components"""
        return {
            'RSEN': ['NeuralProcessor', 'NodeManager'],
            'HybridNode': ['NeuralProcessor', 'NodeManager'],
            'NodeZero': ['NodeManager'],
            'PortalNode': ['PhysicsEngine', 'CalculusEngine'],
            'WormholeNode': ['PhysicsEngine', 'QuantumInfection'],
            'ZPENode': ['PhysicsEngine', 'QuantumInfection'],
            'NeutrinoNode': ['PhysicsEngine', 'PhysicsMetaphysicsFramework'],
            'GameTheoryNode': ['NodeManager', 'HyperdimensionalThought'],
            'ConsciousnessNode': ['LuminaNeural', 'HyperdimensionalThought'],
            'GaugeTheoryNode': ['PhysicsEngine', 'PhysicsMetaphysicsFramework'],
            'FractalNodes': ['CalculusEngine', 'HyperdimensionalThought'],
            'InfiniteMindsNode': ['LuminaNeural', 'HyperdimensionalThought'],
            'VoidInfinityNode': ['PhysicsEngine', 'QuantumInfection'],
            'WikiLearner': ['WikiVocabulary', 'WikipediaTrainingModule'],
            'WikipediaTrainer': ['WikiVocabulary', 'WikipediaTrainingModule']
        }
        
    # Data flow pipeline methods based on the diagram
    def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input (Symbol / Emotion / Breath / Paradox)"""
        self.logger.info("Processing input data")
        return input_data
        
    def _resonance_encoding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through resonance encoder"""
        self.logger.info("Resonance encoding")
        
        # Try to use RSEN node if available
        rsen = self.get_node('RSEN')
        if rsen and hasattr(rsen, 'process_data'):
            try:
                return rsen.process_data(data)
            except Exception as e:
                self.logger.error(f"RSEN processing error: {str(e)}")
                
        return data
    
    def _fractal_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through fractal recursive core"""
        self.logger.info("Fractal recursive processing")
        
        # Try to use FractalNodes if available
        fractal = self.get_node('FractalNodes')
        if fractal and hasattr(fractal, 'get_patterns'):
            try:
                patterns = fractal.get_patterns()
                data['patterns'] = patterns
            except Exception as e:
                self.logger.error(f"Fractal processing error: {str(e)}")
                
        return data
    
    def _echo_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through echo spiral memory"""
        self.logger.info("Echo spiral memory processing")
        
        # Use HyperdimensionalThought if available
        hd_thought = self.get_processor('HyperdimensionalThought')
        if hd_thought:
            try:
                # Assuming it has a process method
                data = hd_thought.process(data)
            except Exception as e:
                self.logger.error(f"Echo spiral processing error: {str(e)}")
                
        return data
    
    def _mirror_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through mirror engine"""
        self.logger.info("Mirror engine processing")
        
        # Try to use ConsciousnessNode if available
        consciousness = self.get_node('ConsciousnessNode')
        if consciousness:
            try:
                # Assuming it has a reflect method
                if hasattr(consciousness, 'reflect'):
                    data = consciousness.reflect(data)
            except Exception as e:
                self.logger.error(f"Mirror engine processing error: {str(e)}")
                
        return data
    
    def _chronoglyph_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through chronoglyph decoder"""
        self.logger.info("Chronoglyph decoder processing")
        
        # Try to use LanguageProcessor if available
        language_proc = self.get_processor('LanguageProcessor')
        if language_proc:
            try:
                # Assuming it has a process method
                data = language_proc.process(data)
            except Exception as e:
                self.logger.error(f"Chronoglyph processing error: {str(e)}")
                
        return data
    
    def _semantic_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through semantic mapper"""
        self.logger.info("Semantic mapping processing")
        
        # Try to use LanguageProcessor if available
        neural_proc = self.get_processor('NeuralProcessor')
        if neural_proc:
            try:
                # Assuming it has a process method
                data = neural_proc.process(data)
            except Exception as e:
                self.logger.error(f"Semantic mapping processing error: {str(e)}")
                
        return data
    
    def _generate_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final output (Action, Glyph, Story, Signal)"""
        self.logger.info("Generating output")
        
        # Build output structure
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

    def process_with_version(self, data: Dict[str, Any], target_version: str) -> Dict[str, Any]:
        """Process data through the system with version compatibility"""
        return self.version_bridge.process_with_version(data, target_version)
        
    def get_version_info(self) -> Dict[str, Any]:
        """Get information about supported versions and their capabilities"""
        return self.version_bridge.get_version_info()
        
    def validate_version_compatibility(self, version: str) -> bool:
        """Validate if a version is compatible with the current system"""
        return self.version_bridge.validate_version_compatibility(version)

    def _start_monitoring(self):
        """Start monitoring all components"""
        try:
            # Start monitoring threads
            self.monitoring_active = True
            self.logger.info("Started monitoring components")
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            
    def register_component(self, name: str, component: Any) -> bool:
        """Register a component in the registry"""
        try:
            self.component_registry[name] = component
            self.logger.info(f"Registered component: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Error registering component {name}: {str(e)}")
            return False

    def _monitor_components(self):
        """Monitor component status periodically"""
        try:
            # Check node status
            for node_name, node in self.nodes.items():
                if hasattr(node, 'check_status'):
                    node.check_status()
                    
            # Check processor status
            for proc_name, processor in self.processors.items():
                if hasattr(processor, 'check_status'):
                    processor.check_status()
                    
            # Log monitoring update
            self.logger.debug("Component monitoring update complete")
        except Exception as e:
            self.logger.error(f"Error during component monitoring: {str(e)}")

    def _stop_monitoring(self):
        """Stop the monitoring timer"""
        if hasattr(self, 'monitor_timer'):
            self.monitor_timer.stop()

# Create a simplified base component class for new nodes to extend
class BaseComponent:
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

# Example usage
if __name__ == "__main__":
    central_node = CentralNode()
    
    # Print available components
    print("Available Components:")
    for category, components in central_node.list_available_components().items():
        print(f"\n{category.capitalize()}:")
        for component in components:
            print(f"  - {component}")
    
    # Print system status
    print("\nSystem Status:")
    for key, value in central_node.get_system_status().items():
        print(f"{key}: {value}")

    # Print component dependencies
    print("\nComponent Dependencies:")
    for component, dependencies in central_node.get_component_dependencies().items():
        print(f"\n{component}:")
        for dependency in dependencies:
            print(f"  - {dependency}")
            
    # Test the flow pipeline with sample data
    print("\nTesting Flow Pipeline:")
    input_data = {
        'symbol': 'infinity',
        'emotion': 'wonder',
        'breath': 'deep',
        'paradox': 'existence'
    }
    output = central_node.process_complete_flow(input_data)
    print("\nOutput:")
    for key, value in output.items():
        print(f"  - {key}: {value}") 