"""
Main script to run the semantic validation suite.
"""

import asyncio
from semantic_validation import (
    ValidationRunner,
    DriftScenario,
    UsageScenario,
    RelationshipScenario,
    HealthReporter
)

# Test concepts and their relationships
TEST_CONCEPTS = {
    "machine_learning": {
        "related": ["neural_networks", "deep_learning", "training_data"],
        "importance": 0.9
    },
    "neural_networks": {
        "related": ["machine_learning", "backpropagation", "activation_functions"],
        "importance": 0.8
    },
    "deep_learning": {
        "related": ["machine_learning", "neural_networks", "gpu_computing"],
        "importance": 0.85
    },
    "training_data": {
        "related": ["machine_learning", "data_preprocessing", "validation"],
        "importance": 0.75
    },
    "backpropagation": {
        "related": ["neural_networks", "gradient_descent", "optimization"],
        "importance": 0.7
    }
}

async def main():
    print("Starting Lumina Semantic Validation Suite")
    
    # Initialize validation runner
    runner = ValidationRunner(redis_host='localhost', redis_port=6379)
    
    # Add test scenarios
    runner.add_scenario(DriftScenario, TEST_CONCEPTS)
    runner.add_scenario(UsageScenario, TEST_CONCEPTS)
    runner.add_scenario(RelationshipScenario, TEST_CONCEPTS)
    
    # Run validation
    print("\nRunning validation scenarios...")
    results = await runner.run_validation(duration=300)  # Run for 5 minutes
    
    # Generate reports
    print("\nGenerating validation reports...")
    reporter = HealthReporter(results)
    
    # Save reports
    report_paths = reporter.save_reports("semantic_validation")
    print("\nReports generated:")
    for format, path in report_paths.items():
        print(f"- {format.upper()}: {path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    reporter.plot_drift_trends(save_path="reports/drift_trends.png")
    reporter.plot_relationship_heatmap(save_path="reports/relationship_heatmap.png")
    
    print("\nValidation complete! Check the reports directory for detailed results.")

if __name__ == "__main__":
    asyncio.run(main()) 