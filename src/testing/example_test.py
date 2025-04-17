"""
Example tests demonstrating how to use the testing framework.

This file showcases different types of tests using the Lumina testing framework,
including unit tests, performance tests, integration tests, and regression tests.
"""

import time
import unittest
import random
from typing import List, Dict, Any

from src.testing.test_framework import (
    TestCase, PerformanceTestCase, IntegrationTestCase, RegressionTestCase,
    TestCategory, TestPriority, test_case, create_test_suite, run_tests
)

# Example simple class to test
class Calculator:
    """A simple calculator class for demonstration"""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def complex_calculation(self, numbers: List[float]) -> float:
        """Perform a more complex calculation that takes time"""
        result = 0
        for num in numbers:
            result += num
            # Simulate a calculation that takes time
            time.sleep(0.001)
        return result

# Unit Tests
class CalculatorTests(TestCase):
    """Unit tests for the Calculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.calculator = Calculator()
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.CRITICAL, component="Calculator")
    def test_add(self):
        """Test the add method"""
        self.assertEqual(self.calculator.add(2, 3), 5)
        self.assertEqual(self.calculator.add(-1, 1), 0)
        self.assertEqual(self.calculator.add(0, 0), 0)
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.HIGH, component="Calculator")
    def test_subtract(self):
        """Test the subtract method"""
        self.assertEqual(self.calculator.subtract(5, 3), 2)
        self.assertEqual(self.calculator.subtract(1, 1), 0)
        self.assertEqual(self.calculator.subtract(0, 5), -5)
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.MEDIUM, component="Calculator")
    def test_multiply(self):
        """Test the multiply method"""
        self.assertEqual(self.calculator.multiply(2, 3), 6)
        self.assertEqual(self.calculator.multiply(-1, 1), -1)
        self.assertEqual(self.calculator.multiply(0, 5), 0)
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.HIGH, component="Calculator")
    def test_divide(self):
        """Test the divide method"""
        self.assertEqual(self.calculator.divide(6, 3), 2)
        self.assertEqual(self.calculator.divide(5, 2), 2.5)
        self.assertEqual(self.calculator.divide(0, 5), 0)
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.MEDIUM, component="Calculator", 
               skip=True, skip_reason="Temporarily disabled for debugging")
    def test_divide_by_zero(self):
        """Test division by zero raises an error"""
        with self.assertRaises(ValueError):
            self.calculator.divide(5, 0)

# Performance Tests
class CalculatorPerformanceTests(PerformanceTestCase):
    """Performance tests for the Calculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.calculator = Calculator()
        
        # Set performance thresholds
        self.set_thresholds({
            "execution_time_ms": 100,  # Maximum execution time in milliseconds
            "memory_delta_mb": 1.0     # Maximum memory usage in MB
        })
    
    @test_case(category=TestCategory.PERFORMANCE, priority=TestPriority.HIGH, component="Calculator")
    def test_complex_calculation_performance(self):
        """Test the performance of the complex calculation method"""
        # Generate test data
        test_data = [random.random() for _ in range(50)]
        
        # Measure performance
        metrics = self.measure_performance(
            self.calculator.complex_calculation,
            test_data
        )
        
        # Assert performance meets thresholds
        self.assert_performance(metrics)
        
        # We can also make specific assertions
        self.assertLess(metrics["execution_time_ms"], 100, 
                        "Complex calculation took too long")

# Integration Tests
class CalculatorIntegrationTests(IntegrationTestCase):
    """Integration tests for the Calculator with other components"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.calculator = Calculator()
        
        # Check for required dependencies
        # (example - there are no actual dependencies here)
        try:
            self.require_dependency("math")
        except ImportError:
            self.skipTest("Math library not available")
    
    @test_case(category=TestCategory.INTEGRATION, priority=TestPriority.MEDIUM, 
               component="Calculator", tags=["math", "integration"])
    def test_calculator_with_math_library(self):
        """Test Calculator integration with math library"""
        import math
        
        # Test that our calculator gives same results as math library
        self.assertEqual(self.calculator.add(2, 3), sum([2, 3]))
        
        # Check component health
        self.assertTrue(self.check_component_health("Calculator"))
        
        # More integration tests would go here

# Regression Tests  
class CalculatorRegressionTests(RegressionTestCase):
    """Regression tests for the Calculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.calculator = Calculator()
        
        # Set up baseline data (in a real case, this might be loaded from a file)
        self.save_baseline({
            "add_result": 5,
            "subtract_result": 2,
            "multiply_result": 6,
            "divide_result": 2.5,
            "complex_calc_result": 15.5
        })
    
    @test_case(category=TestCategory.REGRESSION, priority=TestPriority.CRITICAL, 
               component="Calculator", tags=["regression"])
    def test_against_baseline(self):
        """Test calculator results against baseline"""
        # Test basic operations
        self.assert_matches_baseline(self.calculator.add(2, 3), "add_result")
        self.assert_matches_baseline(self.calculator.subtract(5, 3), "subtract_result")
        self.assert_matches_baseline(self.calculator.multiply(2, 3), "multiply_result")
        self.assert_matches_baseline(self.calculator.divide(5, 2), "divide_result")
        
        # Test complex calculation with tolerance
        result = self.calculator.complex_calculation([3.5, 4.0, 8.0])
        self.assert_matches_baseline(result, "complex_calc_result", tolerance=0.1)

def run_example_tests():
    """Run the example tests"""
    # Create a test suite
    suite = create_test_suite("CalculatorTests")
    
    # Add test cases
    suite.add_tests([
        unittest.makeSuite(CalculatorTests),
        unittest.makeSuite(CalculatorPerformanceTests),
        unittest.makeSuite(CalculatorIntegrationTests),
        unittest.makeSuite(CalculatorRegressionTests)
    ])
    
    # Run the tests
    results = run_tests(suite_name="CalculatorTests")
    print(f"Test Results: {results}")
    
    # Run only performance tests
    perf_results = run_tests(category=TestCategory.PERFORMANCE)
    print(f"Performance Test Results: {perf_results}")
    
    # Run tests by tag
    integration_results = run_tests(tags=["integration"])
    print(f"Integration Test Results: {integration_results}")

if __name__ == "__main__":
    run_example_tests() 