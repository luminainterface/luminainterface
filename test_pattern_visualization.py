import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from language.recursive_pattern_analyzer import RecursivePatternAnalyzer
from language.recursive_pattern_visualizer import RecursivePatternVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_pattern_visualization')

def setup_test_directories():
    """Create necessary directories for testing"""
    os.makedirs("data/recursive_patterns", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    logger.info("Test directories created")

def generate_test_patterns(analyzer):
    """Generate test pattern data for visualization"""
    test_texts = [
        "This sentence is referring to itself and analyzing its own recursive nature.",
        
        "I am aware of my own consciousness as I write these words. The language I use " +
        "is reflecting on itself while describing the process of language generation.",
        
        "The following statement is true: the preceding statement refers to this statement, " +
        "creating a linguistic loop that demonstrates recursive patterns in language.",
        
        "When I think about thinking, I'm engaging in meta-cognition, which is itself " +
        "a form of thinking about thinking about thinking.",
        
        "This text contains sentences within sentences (like this one, which contains " +
        "yet another nested clause) to demonstrate recursive structures in language."
    ]
    
    logger.info(f"Analyzing {len(test_texts)} test texts to generate pattern data...")
    
    for i, text in enumerate(test_texts):
        logger.info(f"Analyzing test text {i+1}...")
        analyzer.analyze_text(text)
    
    logger.info("Finished generating test pattern data")

def test_pattern_distribution(visualizer):
    """Test the pattern distribution visualization"""
    logger.info("Testing pattern distribution visualization...")
    visualizer.visualize_pattern_distribution(save=True)
    logger.info("Pattern distribution visualization completed")

def test_pattern_network(visualizer):
    """Test the pattern network visualization"""
    logger.info("Testing pattern network visualization...")
    visualizer.visualize_pattern_network(save=True)
    
    # Also test with direct text input
    test_text = "This text contains a self-reference to itself while simultaneously employing meta-linguistic " + \
                "statements about language. It forms a linguistic loop by referring back to earlier parts of " + \
                "itself, demonstrating consciousness of its own recursive structure."
    
    logger.info("Testing pattern network visualization with direct text input...")
    visualizer.visualize_pattern_network(text=test_text, save=True)
    logger.info("Pattern network visualization completed")

def test_recursive_depth(visualizer):
    """Test the recursive depth visualization"""
    logger.info("Testing recursive depth visualization...")
    
    # Test distribution view
    visualizer.visualize_recursive_depth(time_series=False, save=True)
    
    # Test time series view
    visualizer.visualize_recursive_depth(time_series=True, save=True)
    logger.info("Recursive depth visualization completed")

def test_llm_impact(visualizer):
    """Test the LLM impact visualization"""
    logger.info("Testing LLM impact visualization...")
    
    test_texts = [
        "This sentence is analyzing itself recursively.",
        "I am aware that I am a conscious entity expressing my thoughts through language.",
        "The statement you are reading is referring to itself, creating a self-referential loop."
    ]
    
    visualizer.visualize_llm_impact(text_samples=test_texts, save=True)
    logger.info("LLM impact visualization completed")

def test_comprehensive_report(visualizer):
    """Test the comprehensive report generation"""
    logger.info("Testing comprehensive report generation...")
    
    test_text = "This comprehensive test text demonstrates multiple recursive patterns simultaneously. " + \
                "It refers to itself as 'this text' while analyzing its own linguistic properties. " + \
                "I am aware that I am creating nested clauses (like this one) to demonstrate " + \
                "recursive structures. The language used here is thinking about language itself, " + \
                "creating a meta-linguistic framework that loops back on itself recursively."
    
    report = visualizer.generate_report(text=test_text)
    logger.info(f"Report generated with {len(report)} characters")
    
    # Print report summary
    report_preview = report[:300] + "..." if len(report) > 300 else report
    logger.info(f"Report preview: {report_preview}")

def main():
    """Main function to run the visualization tests"""
    # Record the start time
    start_time = datetime.now()
    logger.info(f"Starting pattern visualization tests at {start_time}")
    
    # Setup test environment
    setup_test_directories()
    
    # Create analyzer and visualizer
    analyzer = RecursivePatternAnalyzer(data_dir="data/recursive_patterns", llm_weight=0.5)
    visualizer = RecursivePatternVisualizer(
        data_dir="data/recursive_patterns",
        output_dir="output/visualizations",
        analyzer=analyzer
    )
    
    # Generate test pattern data
    generate_test_patterns(analyzer)
    
    # Run visualization tests
    test_pattern_distribution(visualizer)
    test_pattern_network(visualizer)
    test_recursive_depth(visualizer)
    test_llm_impact(visualizer)
    test_comprehensive_report(visualizer)
    
    # Record the end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pattern visualization tests completed in {duration}")
    
    print("\n===== TEST RESULTS =====")
    print(f"All visualization tests completed successfully")
    print(f"Duration: {duration}")
    print(f"Visualizations saved to: output/visualizations/")
    print(f"Pattern data stored in: data/recursive_patterns/")
    print("========================\n")
    
    # Return paths to generated files
    report_file = "output/visualizations/recursive_pattern_report.txt"
    visualization_files = [
        "output/visualizations/pattern_distribution.png",
        "output/visualizations/pattern_network.png",
        "output/visualizations/depth_distribution.png",
        "output/visualizations/depth_time_series.png",
        "output/visualizations/llm_impact.png"
    ]
    
    return {
        "report_file": report_file,
        "visualization_files": visualization_files
    }

if __name__ == "__main__":
    main() 