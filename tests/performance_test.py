import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from typing import Dict, List, Any
import logging
from central_node import CentralNode

class PerformanceTester:
    def __init__(self):
        self.logger = logging.getLogger('PerformanceTester')
        self.central_node = CentralNode()
        self.results = {
            'response_times': [],
            'memory_usage': [],
            'processing_speeds': [],
            'test_times': []
        }
        self.desired_metrics = {
            'response_time': 100,  # ms
            'memory_usage': 500,   # MB
            'processing_speed': 1000,  # operations/second
            'test_time': 5  # seconds
        }
        
    def run_tests(self, num_iterations: int = 100):
        """Run performance tests for multiple iterations"""
        self.logger.info(f"Starting performance tests with {num_iterations} iterations")
        
        for i in range(num_iterations):
            test_start = time.time()
            
            # Test response time
            response_time = self.test_response_time()
            self.results['response_times'].append(response_time)
            
            # Test memory usage
            memory_usage = self.test_memory_usage()
            self.results['memory_usage'].append(memory_usage)
            
            # Test processing speed
            processing_speed = self.test_processing_speed()
            self.results['processing_speeds'].append(processing_speed)
            
            test_time = time.time() - test_start
            self.results['test_times'].append(test_time)
            
            self.logger.info(f"Iteration {i+1}/{num_iterations} completed")
            
        self.save_results()
        self.generate_visualizations()
        
    def test_response_time(self) -> float:
        """Test system response time for a sample operation"""
        start_time = time.time()
        
        # Test a sample operation (e.g., processing through the neural-linguistic bridge)
        sample_data = {
            'symbol': 'infinity',
            'emotion': 'wonder',
            'breath': 'deep',
            'paradox': 'existence'
        }
        self.central_node.process_complete_flow(sample_data)
        
        return (time.time() - start_time) * 1000  # Convert to milliseconds
        
    def test_memory_usage(self) -> float:
        """Test system memory usage"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
        
    def test_processing_speed(self) -> float:
        """Test system processing speed"""
        start_time = time.time()
        operations = 0
        
        # Perform operations for 1 second
        while time.time() - start_time < 1:
            sample_data = {
                'symbol': random.choice(['infinity', 'circle', 'triangle', 'square']),
                'emotion': random.choice(['wonder', 'joy', 'peace', 'excitement']),
                'breath': random.choice(['deep', 'shallow', 'rapid', 'slow']),
                'paradox': random.choice(['existence', 'time', 'space', 'consciousness'])
            }
            self.central_node.process_complete_flow(sample_data)
            operations += 1
            
        return operations
        
    def save_results(self):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'performance_results_{timestamp}.json'
        
        results = {
            'timestamp': timestamp,
            'results': self.results,
            'desired_metrics': self.desired_metrics,
            'averages': {
                'response_time': np.mean(self.results['response_times']),
                'memory_usage': np.mean(self.results['memory_usage']),
                'processing_speed': np.mean(self.results['processing_speeds']),
                'test_time': np.mean(self.results['test_times'])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        self.logger.info(f"Results saved to {filename}")
        
    def generate_visualizations(self):
        """Generate performance visualization charts"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory for visualizations
        vis_dir = f'performance_visualizations_{timestamp}'
        os.makedirs(vis_dir, exist_ok=True)
        
        # Response Time Chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['response_times'], label='Actual Response Time')
        plt.axhline(y=self.desired_metrics['response_time'], color='r', linestyle='--', label='Desired Response Time')
        plt.title('System Response Time (ms)')
        plt.xlabel('Test Iteration')
        plt.ylabel('Response Time (ms)')
        plt.legend()
        plt.savefig(f'{vis_dir}/response_time.png')
        plt.close()
        
        # Memory Usage Chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['memory_usage'], label='Actual Memory Usage')
        plt.axhline(y=self.desired_metrics['memory_usage'], color='r', linestyle='--', label='Desired Memory Usage')
        plt.title('System Memory Usage (MB)')
        plt.xlabel('Test Iteration')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.savefig(f'{vis_dir}/memory_usage.png')
        plt.close()
        
        # Processing Speed Chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['processing_speeds'], label='Actual Processing Speed')
        plt.axhline(y=self.desired_metrics['processing_speed'], color='r', linestyle='--', label='Desired Processing Speed')
        plt.title('System Processing Speed (ops/sec)')
        plt.xlabel('Test Iteration')
        plt.ylabel('Operations per Second')
        plt.legend()
        plt.savefig(f'{vis_dir}/processing_speed.png')
        plt.close()
        
        # Performance Comparison Chart
        metrics = ['Response Time', 'Memory Usage', 'Processing Speed']
        actual_values = [
            np.mean(self.results['response_times']),
            np.mean(self.results['memory_usage']),
            np.mean(self.results['processing_speeds'])
        ]
        desired_values = [
            self.desired_metrics['response_time'],
            self.desired_metrics['memory_usage'],
            self.desired_metrics['processing_speed']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, actual_values, width, label='Actual')
        plt.bar(x + width/2, desired_values, width, label='Desired')
        plt.title('Performance Metrics Comparison')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(x, metrics)
        plt.legend()
        plt.savefig(f'{vis_dir}/performance_comparison.png')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run performance tests
    tester = PerformanceTester()
    tester.run_tests(num_iterations=100) 