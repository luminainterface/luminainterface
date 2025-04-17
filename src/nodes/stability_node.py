from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime

class StabilityNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.stability_state = {}
        self.stability_history = []
        self.max_history = 100
        self.stability_threshold = 0.7
        self.warning_threshold = 0.5
        self.critical_threshold = 0.3
        
    def initialize(self) -> bool:
        """Initialize the stability node"""
        try:
            self.stability_state = {
                'overall_stability': 0.0,
                'component_stability': {},
                'warnings': [],
                'critical_issues': [],
                'last_update': datetime.now().isoformat()
            }
            self.active = True
            logging.info("StabilityNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize StabilityNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for stability analysis"""
        try:
            # Store in stability history
            self.stability_history.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.stability_history) > self.max_history:
                self.stability_history.pop(0)
                
            # Analyze stability
            stability_result = self._analyze_stability(input_data)
            
            # Update stability state
            self.stability_state.update({
                'overall_stability': stability_result['overall_stability'],
                'component_stability': stability_result['component_stability'],
                'warnings': stability_result['warnings'],
                'critical_issues': stability_result['critical_issues'],
                'last_update': datetime.now().isoformat()
            })
            
            return stability_result
        except Exception as e:
            logging.error(f"Error processing stability input: {str(e)}")
            return {'error': str(e)}
            
    def _analyze_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system stability"""
        try:
            stability_result = {
                'overall_stability': 0.0,
                'component_stability': {},
                'warnings': [],
                'critical_issues': [],
                'metrics': self._calculate_stability_metrics(data)
            }
            
            # Analyze component stability
            for component, state in data.items():
                if isinstance(state, dict):
                    stability_score = self._calculate_component_stability(component, state)
                    stability_result['component_stability'][component] = stability_score
                    
                    # Check for issues
                    if stability_score < self.critical_threshold:
                        stability_result['critical_issues'].append({
                            'component': component,
                            'stability': stability_score,
                            'issues': self._identify_issues(component, state)
                        })
                    elif stability_score < self.warning_threshold:
                        stability_result['warnings'].append({
                            'component': component,
                            'stability': stability_score,
                            'issues': self._identify_issues(component, state)
                        })
                        
            # Calculate overall stability
            if stability_result['component_stability']:
                stability_result['overall_stability'] = float(np.mean(list(
                    stability_result['component_stability'].values()
                )))
                
            return stability_result
        except Exception as e:
            logging.error(f"Error analyzing stability: {str(e)}")
            return {}
            
    def _calculate_stability_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate stability metrics"""
        try:
            metrics = {
                'error_rate': 0.0,
                'performance': 0.0,
                'resource_usage': 0.0,
                'response_time': 0.0
            }
            
            # Calculate error rate
            if 'errors' in data:
                error_data = data['errors']
                if isinstance(error_data, list):
                    total_operations = len(self.stability_history)
                    error_rate = len(error_data) / total_operations if total_operations > 0 else 0
                    metrics['error_rate'] = 1.0 - min(1.0, error_rate * 5)  # Normalize
                    
            # Calculate performance
            if 'performance' in data:
                perf_data = data['performance']
                if isinstance(perf_data, dict):
                    perf_values = [v for v in perf_data.values() if isinstance(v, (int, float))]
                    if perf_values:
                        metrics['performance'] = float(np.mean(perf_values))
                        
            # Calculate resource usage
            if 'resources' in data:
                res_data = data['resources']
                if isinstance(res_data, dict):
                    usage_values = [v for v in res_data.values() if isinstance(v, (int, float))]
                    if usage_values:
                        avg_usage = float(np.mean(usage_values))
                        metrics['resource_usage'] = 1.0 - min(1.0, avg_usage)
                        
            # Calculate response time
            if len(self.stability_history) >= 2:
                try:
                    t1 = datetime.fromisoformat(self.stability_history[-2]['timestamp'])
                    t2 = datetime.fromisoformat(self.stability_history[-1]['timestamp'])
                    response_time = (t2 - t1).total_seconds()
                    metrics['response_time'] = 1.0 - min(1.0, response_time / 5.0)  # Normalize
                except:
                    pass
                    
            return metrics
        except Exception as e:
            logging.error(f"Error calculating stability metrics: {str(e)}")
            return {}
            
    def _calculate_component_stability(self, component: str, state: Dict[str, Any]) -> float:
        """Calculate stability score for a component"""
        try:
            scores = []
            
            # Check state completeness
            if state:
                scores.append(min(1.0, len(state) / 5))  # Expect at least 5 state attributes
                
            # Check state consistency
            if len(self.stability_history) >= 2:
                prev_data = self.stability_history[-2]['data']
                if component in prev_data:
                    changes = sum(1 for k, v in state.items()
                                if k in prev_data[component] and v != prev_data[component][k])
                    consistency = 1.0 - min(1.0, changes / len(state))
                    scores.append(consistency)
                    
            # Check error presence
            error_indicators = ['error', 'failure', 'warning']
            error_count = sum(1 for k in state if any(ind in str(k).lower() 
                                                    for ind in error_indicators))
            error_score = 1.0 - min(1.0, error_count / len(state))
            scores.append(error_score)
            
            # Check resource health
            resource_indicators = ['memory', 'cpu', 'load', 'capacity']
            resource_values = []
            for key, value in state.items():
                if any(ind in str(key).lower() for ind in resource_indicators):
                    if isinstance(value, (int, float)):
                        resource_values.append(1.0 - min(1.0, value))
            if resource_values:
                scores.append(float(np.mean(resource_values)))
                
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logging.error(f"Error calculating component stability: {str(e)}")
            return 0.0
            
    def _identify_issues(self, component: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify stability issues in component state"""
        issues = []
        try:
            # Check for error states
            error_indicators = ['error', 'failure', 'warning']
            for key, value in state.items():
                if any(ind in str(key).lower() for ind in error_indicators):
                    issues.append({
                        'type': 'error_state',
                        'component': component,
                        'attribute': key,
                        'value': value,
                        'severity': 'critical' if 'error' in str(key).lower() else 'warning'
                    })
                    
            # Check for resource issues
            resource_indicators = ['memory', 'cpu', 'load', 'capacity']
            for key, value in state.items():
                if any(ind in str(key).lower() for ind in resource_indicators):
                    if isinstance(value, (int, float)) and value > 0.8:
                        issues.append({
                            'type': 'resource_pressure',
                            'component': component,
                            'resource': key,
                            'value': value,
                            'severity': 'critical' if value > 0.9 else 'warning'
                        })
                        
            # Check for performance issues
            if 'performance' in state:
                perf = state['performance']
                if isinstance(perf, (int, float)) and perf < 0.6:
                    issues.append({
                        'type': 'performance_degradation',
                        'component': component,
                        'value': perf,
                        'severity': 'critical' if perf < 0.4 else 'warning'
                    })
                    
            # Check for stale state
            if 'timestamp' in state:
                try:
                    state_time = datetime.fromisoformat(state['timestamp'])
                    if (datetime.now() - state_time).total_seconds() > 300:  # 5 minutes
                        issues.append({
                            'type': 'stale_state',
                            'component': component,
                            'last_update': state['timestamp'],
                            'severity': 'warning'
                        })
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"Error identifying issues: {str(e)}")
            
        return issues
        
    def get_status(self) -> str:
        """Get current status of the stability node"""
        if not self.active:
            return "inactive"
            
        status = f"active (stability: {self.stability_state['overall_stability']:.2f}"
        
        if self.stability_state['critical_issues']:
            status += f", critical issues: {len(self.stability_state['critical_issues'])}"
        if self.stability_state['warnings']:
            status += f", warnings: {len(self.stability_state['warnings'])}"
            
        return status + ")"
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 