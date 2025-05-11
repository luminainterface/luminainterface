"""
Validation runner for semantic health tests.
Orchestrates multiple test scenarios and collects results.
"""

import asyncio
import json
from typing import Dict, List, Type
import redis.asyncio as redis
from datetime import datetime
from .scenarios import BaseScenario, DriftScenario, UsageScenario, RelationshipScenario

class ValidationRunner:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """Initialize the validation runner"""
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.scenarios = []
        self.results = {}
        
    def add_scenario(self, scenario_class: Type[BaseScenario], concepts: Dict):
        """Add a test scenario to the runner"""
        scenario = scenario_class(self.redis_client, concepts)
        self.scenarios.append(scenario)
        
    async def run_validation(self, duration: int = 300):
        """Run all scenarios for specified duration"""
        print(f"Starting validation suite at {datetime.now().isoformat()}")
        print(f"Running {len(self.scenarios)} scenarios for {duration} seconds each")
        
        # Run all scenarios concurrently
        tasks = [
            asyncio.create_task(scenario.run(duration))
            for scenario in self.scenarios
        ]
        
        # Wait for all scenarios to complete
        scenario_results = await asyncio.gather(*tasks)
        
        # Collect results
        for scenario, results in zip(self.scenarios, scenario_results):
            scenario_name = scenario.__class__.__name__
            self.results[scenario_name] = results
            
        print(f"Validation complete at {datetime.now().isoformat()}")
        return self.results
        
    def save_results(self, filename: str):
        """Save validation results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def get_summary(self) -> Dict:
        """Generate summary of validation results"""
        summary = {}
        
        for scenario_name, results in self.results.items():
            scenario_summary = {
                'total_events': len(results),
                'start_time': results[0]['timestamp'] if results else None,
                'end_time': results[-1]['timestamp'] if results else None
            }
            
            if scenario_name == 'DriftScenario':
                total_drift = sum(abs(r['drift_amount']) for r in results)
                scenario_summary['average_drift'] = total_drift / len(results) if results else 0
                scenario_summary['max_drift'] = max(abs(r['drift_amount']) for r in results) if results else 0
                
            elif scenario_name == 'UsageScenario':
                total_usage_increase = sum(r['new_usage'] - r['old_usage'] for r in results)
                scenario_summary['total_usage_increase'] = total_usage_increase
                scenario_summary['average_usage_increase'] = total_usage_increase / len(results) if results else 0
                
            elif scenario_name == 'RelationshipScenario':
                avg_strength = sum(r['relationship_strength'] for r in results) / len(results) if results else 0
                scenario_summary['average_relationship_strength'] = avg_strength
                
            summary[scenario_name] = scenario_summary
            
        return summary 