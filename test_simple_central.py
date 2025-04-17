import logging
from central_node import CentralNode, BaseComponent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestCentral")

# Create mock components to test with
class MockRSEN(BaseComponent):
    def process_data(self, data):
        logger.info("MockRSEN processing data")
        data['resonance'] = 0.85
        return data

class MockFractalNodes(BaseComponent):
    def get_patterns(self):
        logger.info("MockFractalNodes getting patterns")
        return ["spiral", "loop", "mandelbrot"]
        
    def process_suggestions(self, suggestions):
        logger.info(f"MockFractalNodes processing suggestions: {suggestions}")
        return suggestions

class MockConsciousness(BaseComponent):
    def reflect(self, data):
        logger.info("MockConsciousness reflecting")
        data['reflection'] = True
        return data

class MockLanguageProcessor(BaseComponent):
    def process(self, data):
        logger.info("MockLanguageProcessor processing")
        data['language_processed'] = True
        return data

class MockNeuralProcessor(BaseComponent):
    def process(self, data):
        logger.info("MockNeuralProcessor processing")
        data['semantically_mapped'] = True
        data['action'] = "explore"
        data['glyph'] = "⚛"
        data['story'] = "A journey through fractal dimensions"
        data['signal'] = 1.0
        return data

def test_component_registration():
    """Test component registration with central node"""
    logger.info("Testing component registration")
    
    # Create central node
    central_node = CentralNode()
    
    # Create and register mock components
    rsen = MockRSEN()
    fractal = MockFractalNodes()
    consciousness = MockConsciousness()
    language = MockLanguageProcessor()
    neural = MockNeuralProcessor()
    
    # Register with central node
    central_node._register_component('MockRSEN', rsen)
    central_node._register_component('MockFractalNodes', fractal)
    central_node._register_component('MockConsciousness', consciousness)
    central_node._register_component('MockLanguageProcessor', language)
    central_node._register_component('MockNeuralProcessor', neural)
    
    # Add to appropriate collections
    central_node.nodes['MockRSEN'] = rsen
    central_node.nodes['MockFractalNodes'] = fractal
    central_node.nodes['MockConsciousness'] = consciousness
    central_node.processors['MockLanguageProcessor'] = language
    central_node.processors['MockNeuralProcessor'] = neural
    
    # Test component retrieval
    assert central_node.get_component('MockRSEN') == rsen
    assert central_node.get_node('MockRSEN') == rsen
    assert central_node.get_processor('MockLanguageProcessor') == language
    
    logger.info("✓ Component registration test passed")
    return central_node

def test_component_connections(central_node):
    """Test connections between components"""
    logger.info("Testing component connections")
    
    # Create connections between components
    central_node._connect_components('MockRSEN', 'MockNeuralProcessor')
    central_node._connect_components('MockFractalNodes', 'MockLanguageProcessor')
    
    # Verify connections are established
    assert 'MockRSEN' in central_node.connections
    assert 'MockNeuralProcessor' in central_node.connections['MockRSEN']
    
    logger.info("✓ Component connection test passed")

def test_data_flow_pipeline(central_node):
    """Test the complete data flow pipeline"""
    logger.info("Testing data flow pipeline")
    
    # Override pipeline methods to use our mock components
    central_node._resonance_encoding = lambda data: central_node.get_node('MockRSEN').process_data(data)
    central_node._fractal_processing = lambda data: data.update({'patterns': central_node.get_node('MockFractalNodes').get_patterns()}) or data
    central_node._echo_processing = lambda data: data
    central_node._mirror_processing = lambda data: central_node.get_node('MockConsciousness').reflect(data)
    central_node._chronoglyph_processing = lambda data: central_node.get_processor('MockLanguageProcessor').process(data)
    central_node._semantic_mapping = lambda data: central_node.get_processor('MockNeuralProcessor').process(data)
    
    # Test input data
    input_data = {
        'symbol': 'infinity',
        'emotion': 'wonder',
        'breath': 'deep',
        'paradox': 'existence'
    }
    
    # Process through pipeline
    output = central_node.process_complete_flow(input_data)
    
    # Verify output has expected fields
    assert 'action' in output
    assert 'glyph' in output
    assert 'story' in output
    assert 'signal' in output
    
    # Print output
    logger.info(f"Pipeline output: {output}")
    logger.info("✓ Data flow pipeline test passed")
    
    return output

def run_all_tests():
    """Run all tests"""
    logger.info("Starting central node tests")
    
    # Run component registration test
    central_node = test_component_registration()
    
    # Run component connection test
    test_component_connections(central_node)
    
    # Run data flow pipeline test
    output = test_data_flow_pipeline(central_node)
    
    logger.info("All tests completed successfully!")
    
    # Print final results
    print("\n========================================")
    print("       CENTRAL NODE TEST RESULTS        ")
    print("========================================")
    print(f"Components registered: {len(central_node.component_registry)}")
    print(f"Connections established: {len(central_node.connections)}")
    print("\nOutput from pipeline:")
    for key, value in output.items():
        print(f"  - {key}: {value}")
    print("========================================")

if __name__ == "__main__":
    run_all_tests() 