"""
Example Test Suite

This module demonstrates how to use the comprehensive test framework
with various test types and configurations.
"""

import unittest
import time
import random
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import the test framework
from testing.test_framework import (
    TestCase, 
    PerformanceTestCase,
    IntegrationTestCase, 
    RegressionTestCase,
    TestCategory,
    TestPriority,
    test_case,
    create_test_suite,
    run_tests
)

# Mock classes for testing
class SimpleCalculator:
    """Simple calculator for testing"""
    
    def add(self, a, b):
        return a + b
        
    def subtract(self, a, b):
        return a - b
        
    def multiply(self, a, b):
        return a * b
        
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
        
    def slow_operation(self, n):
        """Deliberately slow operation for performance testing"""
        result = 0
        for i in range(n):
            result += i
            time.sleep(0.001)  # Simulate work
        return result


class BasicTests(TestCase):
    """Basic unit test examples"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = SimpleCalculator()
        
    @test_case(category=TestCategory.UNIT, priority=TestPriority.CRITICAL, component="Calculator")
    def test_addition(self):
        """Test addition functionality"""
        self.assertEqual(self.calculator.add(2, 3), 5)
        self.assertEqual(self.calculator.add(-1, 1), 0)
        self.assertEqual(self.calculator.add(0, 0), 0)
        
    @test_case(category=TestCategory.UNIT, priority=TestPriority.HIGH, component="Calculator")
    def test_subtraction(self):
        """Test subtraction functionality"""
        self.assertEqual(self.calculator.subtract(5, 3), 2)
        self.assertEqual(self.calculator.subtract(1, 1), 0)
        self.assertEqual(self.calculator.subtract(0, 5), -5)
        
    @test_case(category=TestCategory.UNIT, priority=TestPriority.MEDIUM, component="Calculator")
    def test_multiplication(self):
        """Test multiplication functionality"""
        self.assertEqual(self.calculator.multiply(2, 3), 6)
        self.assertEqual(self.calculator.multiply(-2, 3), -6)
        self.assertEqual(self.calculator.multiply(0, 5), 0)
        
    @test_case(category=TestCategory.UNIT, priority=TestPriority.HIGH, component="Calculator")
    def test_division(self):
        """Test division functionality"""
        self.assertEqual(self.calculator.divide(6, 3), 2)
        self.assertEqual(self.calculator.divide(5, 2), 2.5)
        self.assertEqual(self.calculator.divide(0, 5), 0)
        
        # Test division by zero
        with self.assertRaises(ValueError):
            self.calculator.divide(5, 0)


class PerformanceTests(PerformanceTestCase):
    """Performance test examples"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = SimpleCalculator()
        
        # Set performance thresholds
        self.set_thresholds({
            "execution_time_ms": 200,  # 200ms max
            "memory_delta_mb": 1.0     # 1MB max
        })
        
    @test_case(category=TestCategory.PERFORMANCE, component="Calculator")
    def test_slow_operation_performance(self):
        """Test performance of slow operation"""
        # Measure performance
        metrics = self.measure_performance(self.calculator.slow_operation, 100)
        
        # Assert performance meets thresholds
        self.assert_performance(metrics)
        
        # Add details to test result
        self.add_test_data("operation", "slow_operation")
        self.add_test_data("input_size", 100)
        self.add_test_data("metrics", metrics)


class IntegrationTests(IntegrationTestCase):
    """Integration test examples"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Check if we have required dependencies
        self.require_dependency("random")
        
        # Create components for testing
        self.calculator = SimpleCalculator()
        
    @test_case(category=TestCategory.INTEGRATION, component="CalculatorSystem")
    def test_calculator_integration(self):
        """Test integration between calculator components"""
        # Test a complex calculation that uses multiple operations
        a = 10
        b = 5
        
        # (a + b) * (a - b) / 5
        result = self.calculator.divide(
            self.calculator.multiply(
                self.calculator.add(a, b),
                self.calculator.subtract(a, b)
            ),
            5
        )
        
        # Expected: (10 + 5) * (10 - 5) / 5 = 15 * 5 / 5 = 15
        self.assertEqual(result, 15)


class RegressionTests(RegressionTestCase):
    """Regression test examples"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = SimpleCalculator()
        
        # Set baseline data
        self.baseline_data = {
            "addition": 5,
            "subtraction": 2,
            "multiplication": 6,
            "division": 2
        }
        
        # Set tolerance for numeric comparisons
        self.tolerance = 0.01
        
    @test_case(category=TestCategory.REGRESSION, component="Calculator")
    def test_against_baseline(self):
        """Test calculator operations against baseline"""
        # Test addition
        add_result = self.calculator.add(2, 3)
        self.assert_matches_baseline(add_result, "addition")
        
        # Test subtraction
        sub_result = self.calculator.subtract(5, 3)
        self.assert_matches_baseline(sub_result, "subtraction")
        
        # Test multiplication
        mul_result = self.calculator.multiply(2, 3)
        self.assert_matches_baseline(mul_result, "multiplication")
        
        # Test division
        div_result = self.calculator.divide(6, 3)
        self.assert_matches_baseline(div_result, "division")


def create_test_suites():
    """Create and register test suites"""
    # Create individual test suites
    unit_suite = create_test_suite("UnitTests")
    unit_suite.add_tests([
        BasicTests('test_addition'),
        BasicTests('test_subtraction'),
        BasicTests('test_multiplication'),
        BasicTests('test_division')
    ])
    
    performance_suite = create_test_suite("PerformanceTests")
    performance_suite.add_tests([
        PerformanceTests('test_slow_operation_performance')
    ])
    
    integration_suite = create_test_suite("IntegrationTests")
    integration_suite.add_tests([
        IntegrationTests('test_calculator_integration')
    ])
    
    regression_suite = create_test_suite("RegressionTests")
    regression_suite.add_tests([
        RegressionTests('test_against_baseline')
    ])
    
    # Create a combined suite
    all_tests_suite = create_test_suite("AllTests")
    all_tests_suite.add_tests([
        BasicTests('test_addition'),
        BasicTests('test_subtraction'),
        BasicTests('test_multiplication'),
        BasicTests('test_division'),
        PerformanceTests('test_slow_operation_performance'),
        IntegrationTests('test_calculator_integration'),
        RegressionTests('test_against_baseline')
    ])
    
    return {
        "unit": unit_suite,
        "performance": performance_suite,
        "integration": integration_suite,
        "regression": regression_suite,
        "all": all_tests_suite
    }


if __name__ == "__main__":
    # Create test suites
    test_suites = create_test_suites()
    
    # Run the desired test suite
    import sys
    if len(sys.argv) > 1:
        suite_name = sys.argv[1]
        if suite_name in test_suites:
            print(f"Running {suite_name} test suite")
            summary = run_tests(suite_name)
            print(f"Pass rate: {summary.get('pass_rate', 0):.2f}%")
        else:
            print(f"Unknown test suite: {suite_name}")
            print(f"Available suites: {', '.join(test_suites.keys())}")
    else:
        # Default to running all tests
        print("Running all tests")
        summary = run_tests("AllTests")
        print(f"Pass rate: {summary.get('pass_rate', 0):.2f}%") 