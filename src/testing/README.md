# Lumina Neural Network Testing Framework

A comprehensive testing framework for the Lumina Neural Network system, designed to standardize and simplify the testing process across different components and test categories.

## Features

- **Multiple Test Categories**: Support for unit, integration, performance, and regression testing
- **Test Prioritization**: Categorize tests by priority (Critical, High, Medium, Low)
- **Performance Metrics**: Built-in tools for measuring execution time and memory usage
- **Parallel Test Execution**: Run tests concurrently for faster test runs
- **Test Reporting**: Generate detailed test reports with pass/fail rates and metrics
- **Test Registry**: Centralized registry for test cases and suites
- **Baseline Comparison**: Compare test results against known baselines for regression testing
- **Dependency Management**: Verify dependencies before running integration tests
- **Modular Design**: Easy to extend for custom test types and metrics

## Neural Network Components

The `neural_network_test.py` module includes the `SimpleNeuralNetwork` implementation for testing purposes. For comprehensive documentation of all neural network implementations across the project, please refer to [../../docs/NEURAL_ARCHITECTURE.md](../../docs/NEURAL_ARCHITECTURE.md).

## Getting Started

### Installation

The testing framework is included in the Lumina Neural Network project. No additional installation is required.

### Basic Usage

1. **Import the framework:**

```python
from testing.test_framework import (
    TestCase, 
    TestCategory,
    TestPriority,
    test_case,
    create_test_suite,
    run_tests
)
```

2. **Create test cases:**

```python
class MyTests(TestCase):
    @test_case(category=TestCategory.UNIT, priority=TestPriority.HIGH, component="MyComponent")
    def test_something(self):
        # Your test code here
        self.assertEqual(1 + 1, 2)
```

3. **Create and run a test suite:**

```python
# Create a test suite
my_suite = create_test_suite("MySuite")
my_suite.add_tests([
    MyTests('test_something')
])

# Run the tests
results = run_tests("MySuite")
```

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Performance Tests**: Measure and verify system performance
- **Regression Tests**: Ensure changes don't break existing functionality

## Advanced Usage

### Performance Testing

```python
class MyPerformanceTests(PerformanceTestCase):
    def setUp(self):
        # Set up test fixtures
        self.set_thresholds({
            "execution_time_ms": 100,  # 100ms max
            "memory_delta_mb": 0.5     # 0.5MB max
        })
        
    @test_case(category=TestCategory.PERFORMANCE, component="MyComponent")
    def test_performance(self):
        # Measure performance of a function
        metrics = self.measure_performance(my_function, arg1, arg2)
        
        # Assert performance meets thresholds
        self.assert_performance(metrics)
```

### Integration Testing

```python
class MyIntegrationTests(IntegrationTestCase):
    def setUp(self):
        # Check if we have required dependencies
        self.require_dependency("package_name")
        
    @test_case(category=TestCategory.INTEGRATION, component="MySystem")
    def test_integration(self):
        # Test component integration
        result = component_a.process(component_b.get_data())
        self.assertTrue(result.is_valid())
```

### Regression Testing

```python
class MyRegressionTests(RegressionTestCase):
    def setUp(self):
        # Set baseline data
        self.baseline_data = {
            "key1": expected_value1,
            "key2": expected_value2
        }
        
    @test_case(category=TestCategory.REGRESSION, component="MyComponent")
    def test_regression(self):
        # Compare against baseline
        result = my_function()
        self.assert_matches_baseline(result, "key1")
```

## Running Tests

### Command Line

You can run tests from the command line:

```
python -m testing.test_example [suite_name]
```

If no suite name is provided, all tests will be run.

### Filtering Tests

You can filter tests by category or priority:

```python
# Run only critical tests
run_tests("MySuite", priority=TestPriority.CRITICAL)

# Run only unit tests
run_tests("MySuite", category=TestCategory.UNIT)
```

## Example

See `test_example.py` for a complete example of how to use the testing framework with various test types.

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on the state from other tests
2. **Clear Test Names**: Use descriptive test names that indicate what is being tested
3. **One Assertion Per Test**: Prefer one assertion per test for clearer failure messages
4. **Test Data Management**: Use setUp and tearDown methods to manage test data
5. **Mock External Dependencies**: Use mocking to isolate the system under test
6. **Test Edge Cases**: Include tests for boundary conditions and error cases
7. **Regular Test Runs**: Run tests regularly as part of the development workflow

## Contributing

To add new test types or extend the framework:

1. Create a new test case class that extends one of the existing base classes
2. Implement custom assertions or metrics as needed
3. Register your test cases with the test registry

## License

This project is licensed under the same license as the Lumina Neural Network project. 