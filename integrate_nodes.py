import logging
import importlib
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
from minimal_central import MinimalCentralNode, BaseComponent
from component_adapter import adapt_component

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NodeIntegration")

class NodeIntegrator:
    """Class to safely integrate all existing nodes with the central node"""
    def __init__(self):
        self.central_node = MinimalCentralNode()
        self.successfully_loaded = []
        self.failed_loads = []
        
    def discover_components(self):
        """Discover all potential components in the directory"""
        # Define patterns to look for
        node_patterns = ["*_node.py", "node_*.py"]
        processor_patterns = ["*_processor.py", "*_engine.py", "*_learner.py"]
        
        # Get current directory
        current_dir = Path(".")
        
        # Find node files
        node_files = []
        for pattern in node_patterns:
            node_files.extend(list(current_dir.glob(pattern)))
            
        # Find processor files
        processor_files = []
        for pattern in processor_patterns:
            processor_files.extend(list(current_dir.glob(pattern)))
            
        logger.info(f"Discovered {len(node_files)} potential node files")
        logger.info(f"Discovered {len(processor_files)} potential processor files")
        
        return node_files, processor_files
        
    def _safe_import_module(self, file_path: Path) -> Optional[Any]:
        """Safely import a module from file path"""
        try:
            # Get module name (file name without extension)
            module_name = file_path.stem
            
            # Try to import the module
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported module: {module_name}")
            return module
        except Exception as e:
            logger.error(f"Failed to import module {file_path}: {str(e)}")
            self.failed_loads.append((str(file_path), str(e)))
            return None
    
    def _find_component_classes(self, module) -> List[tuple]:
        """Find all potential component classes in a module"""
        potential_components = []
        
        # Look for classes in the module
        for name in dir(module):
            # Skip private/dunder methods
            if name.startswith('_'):
                continue
                
            try:
                obj = getattr(module, name)
                
                # Check if it's a class (not a function or module)
                if isinstance(obj, type):
                    # Basic check - class name suggests it's a component
                    if ("Node" in name or 
                        "Processor" in name or 
                        "Engine" in name or 
                        "Learner" in name or 
                        "Neural" in name):
                        
                        # Check if it has common component methods
                        has_process = hasattr(obj, 'process')
                        has_process_data = hasattr(obj, 'process_data')
                        has_forward = hasattr(obj, 'forward')
                        has_call = hasattr(obj, '__call__')
                        
                        if has_process or has_process_data or has_forward or has_call:
                            potential_components.append((name, obj))
            except Exception as e:
                logger.warning(f"Error inspecting {name}: {str(e)}")
                
        return potential_components
    
    def _safe_instantiate(self, class_name, class_obj) -> Optional[Any]:
        """Safely instantiate a component class"""
        try:
            # Try instantiating with no arguments
            instance = class_obj()
            logger.info(f"Successfully instantiated {class_name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate {class_name}: {str(e)}")
            self.failed_loads.append((class_name, str(e)))
            return None
    
    def integrate_components(self):
        """Integrate all discovered components with the central node"""
        # Discover component files
        node_files, processor_files = self.discover_components()
        
        # Process node files
        for file_path in node_files:
            module = self._safe_import_module(file_path)
            if not module:
                continue
                
            # Find component classes
            potential_components = self._find_component_classes(module)
            
            # Try to instantiate and integrate each component
            for class_name, class_obj in potential_components:
                try:
                    # Create instance
                    instance = self._safe_instantiate(class_name, class_obj)
                    if not instance:
                        continue
                        
                    # Adapt and register with central node
                    adapted_component = adapt_component(instance, class_name)
                    self.central_node.register_component(class_name, adapted_component, 'node')
                    
                    # Add to successful list
                    self.successfully_loaded.append((class_name, 'node'))
                except Exception as e:
                    logger.error(f"Failed to integrate {class_name}: {str(e)}")
                    self.failed_loads.append((class_name, str(e)))
        
        # Process processor files
        for file_path in processor_files:
            module = self._safe_import_module(file_path)
            if not module:
                continue
                
            # Find component classes
            potential_components = self._find_component_classes(module)
            
            # Try to instantiate and integrate each component
            for class_name, class_obj in potential_components:
                try:
                    # Create instance
                    instance = self._safe_instantiate(class_name, class_obj)
                    if not instance:
                        continue
                        
                    # Adapt and register with central node
                    adapted_component = adapt_component(instance, class_name)
                    self.central_node.register_component(class_name, adapted_component, 'processor')
                    
                    # Add to successful list
                    self.successfully_loaded.append((class_name, 'processor'))
                except Exception as e:
                    logger.error(f"Failed to integrate {class_name}: {str(e)}")
                    self.failed_loads.append((class_name, str(e)))
        
        # Establish connections based on a simplified dependency mapping
        self._establish_connections()
        
        # Print summary
        self._print_integration_summary()
        
        return self.central_node
    
    def _establish_connections(self):
        """Establish connections between components based on naming conventions"""
        try:
            # For nodes named like XyzNode, try to connect to matching XyzProcessor
            for component_name, comp_type in self.successfully_loaded:
                if comp_type == 'node' and 'Node' in component_name:
                    # Find base name (remove 'Node' suffix)
                    base_name = component_name.replace('Node', '')
                    
                    # Look for matching processor
                    processor_name = f"{base_name}Processor"
                    if processor_name in self.central_node.component_registry:
                        self.central_node.connect_components(component_name, processor_name)
                        
            # Connect common dependencies
            common_dependencies = {
                'RSEN': ['NeuralProcessor'],
                'HybridNode': ['NeuralProcessor'],
                'ConsciousnessNode': ['HyperdimensionalThought'],
                'PortalNode': ['PhysicsEngine'],
                'WormholeNode': ['PhysicsEngine'],
                'NeutrinoNode': ['PhysicsEngine'],
                'FractalNodes': ['HyperdimensionalThought']
            }
            
            # Connect based on known dependencies
            for source, targets in common_dependencies.items():
                if source in self.central_node.component_registry:
                    for target in targets:
                        if target in self.central_node.component_registry:
                            self.central_node.connect_components(source, target)
        except Exception as e:
            logger.error(f"Error establishing connections: {str(e)}")
    
    def _print_integration_summary(self):
        """Print a summary of the integration process"""
        print("\n========================================")
        print("    NEURAL NETWORK INTEGRATION SUMMARY   ")
        print("========================================")
        print(f"Successfully integrated components: {len(self.successfully_loaded)}")
        print(f"Failed component integrations: {len(self.failed_loads)}")
        
        print("\nSuccessful components:")
        for name, comp_type in self.successfully_loaded:
            print(f"  - {name} ({comp_type})")
            
        if self.failed_loads:
            print("\nFailed components:")
            for name, error in self.failed_loads:
                print(f"  - {name}: {error}")
                
        print("\nComponent connections:")
        for source, targets in self.central_node.connections.items():
            print(f"  - {source} connected to: {', '.join(targets)}")
            
        print("========================================")
    
    def test_integrated_system(self, input_data=None):
        """Test the integrated system with sample input data"""
        if input_data is None:
            input_data = {
                'symbol': 'infinity',
                'emotion': 'wonder',
                'breath': 'deep',
                'paradox': 'existence'
            }
            
        try:
            # Configure pipeline to use available components
            self._configure_pipeline()
            
            # Process input through the central node
            logger.info("Processing data through integrated system")
            output = self.central_node.process_complete_flow(input_data)
            
            print("\n========================================")
            print("         PROCESSING RESULTS             ")
            print("========================================")
            print(f"Input: {input_data}")
            print("\nOutput:")
            for key, value in output.items():
                print(f"  - {key}: {value}")
            print("========================================")
            
            return output
        except Exception as e:
            logger.error(f"Error testing integrated system: {str(e)}")
            return None
            
    def _configure_pipeline(self):
        """Configure the pipeline to use available components"""
        # Override the pipeline methods to use actual components if available
        
        # RSEN for resonance encoding if available
        if 'RSEN' in self.central_node.component_registry:
            self.central_node._resonance_encoding = lambda data: self.central_node.get_component('RSEN').process(data)
            
        # FractalNodes for fractal processing if available
        if 'FractalNodes' in self.central_node.component_registry:
            def fractal_process(data):
                fractal = self.central_node.get_component('FractalNodes')
                if hasattr(fractal, 'get_patterns'):
                    try:
                        data['patterns'] = fractal.get_patterns()
                    except:
                        pass
                return data
            self.central_node._fractal_processing = fractal_process
            
        # LanguageProcessor for chronoglyph processing if available
        if 'LanguageProcessor' in self.central_node.component_registry:
            self.central_node._chronoglyph_processing = lambda data: self.central_node.get_component('LanguageProcessor').process(data)
            
        # NeuralProcessor for semantic mapping if available
        if 'NeuralProcessor' in self.central_node.component_registry:
            self.central_node._semantic_mapping = lambda data: self.central_node.get_component('NeuralProcessor').process(data)
            
        # ConsciousnessNode for mirror processing if available 
        if 'ConsciousnessNode' in self.central_node.component_registry:
            def consciousness_process(data):
                consciousness = self.central_node.get_component('ConsciousnessNode')
                if hasattr(consciousness, 'reflect'):
                    try:
                        return consciousness.reflect(data)
                    except:
                        pass
                return data
            self.central_node._mirror_processing = consciousness_process
            
        # HyperdimensionalThought for echo processing if available
        if 'HyperdimensionalThought' in self.central_node.component_registry:
            self.central_node._echo_processing = lambda data: self.central_node.get_component('HyperdimensionalThought').process(data)

def run_integration():
    """Run the integration process"""
    print("Starting neural network integration...")
    
    # Create integrator
    integrator = NodeIntegrator()
    
    # Integrate components
    central_node = integrator.integrate_components()
    
    # Test the integrated system
    integrator.test_integrated_system()
    
    print("\nIntegration complete!")
    return central_node

if __name__ == "__main__":
    try:
        central_node = run_integration()
    except Exception as e:
        logger.error(f"Integration failed: {str(e)}")
        print(f"Integration failed: {str(e)}") 