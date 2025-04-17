from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime

class AdaptationNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.adaptation_state = {}
        self.learning_rate = 0.01
        self.adaptation_history = []
        self.max_history = 100
        self.adaptation_threshold = 0.7
        
    def initialize(self) -> bool:
        """Initialize the adaptation node"""
        try:
            self.adaptation_state = {
                'current_adaptation': None,
                'adaptation_level': 0.0,
                'adaptations': [],
                'last_update': datetime.now().isoformat()
            }
            self.active = True
            logging.info("AdaptationNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize AdaptationNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for adaptation"""
        try:
            # Store in adaptation history
            self.adaptation_history.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.adaptation_history) > self.max_history:
                self.adaptation_history.pop(0)
                
            # Perform adaptation
            adaptation_result = self._adapt_to_input(input_data)
            adaptation_level = self._calculate_adaptation_level(adaptation_result)
            
            # Update adaptation state
            self.adaptation_state.update({
                'current_adaptation': adaptation_result,
                'adaptation_level': adaptation_level,
                'last_update': datetime.now().isoformat()
            })
            
            return {
                'adaptation': adaptation_result,
                'adaptation_level': adaptation_level,
                'adaptations': self.adaptation_state['adaptations']
            }
        except Exception as e:
            logging.error(f"Error processing adaptation input: {str(e)}")
            return {'error': str(e)}
            
    def _adapt_to_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt system based on input data"""
        try:
            adaptation = {
                'behavioral_changes': self._analyze_behavioral_changes(data),
                'parameter_adjustments': self._adjust_parameters(data),
                'structural_adaptations': self._adapt_structure(data),
                'learning_updates': self._update_learning(data)
            }
            
            # Track adaptations
            self.adaptation_state['adaptations'].append({
                'type': 'system_adaptation',
                'changes': adaptation,
                'timestamp': datetime.now().isoformat()
            })
            
            return adaptation
        except Exception as e:
            logging.error(f"Error adapting to input: {str(e)}")
            return {}
            
    def _analyze_behavioral_changes(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and implement behavioral changes"""
        changes = []
        try:
            # Analyze performance metrics
            if 'performance' in data:
                perf_data = data['performance']
                if isinstance(perf_data, dict):
                    # Identify areas needing improvement
                    for metric, value in perf_data.items():
                        if isinstance(value, (int, float)) and value < 0.6:  # Below threshold
                            changes.append({
                                'type': 'performance_adaptation',
                                'metric': metric,
                                'current_value': value,
                                'target_value': 0.8,
                                'priority': 'high' if value < 0.4 else 'medium'
                            })
                            
            # Analyze error patterns
            if 'errors' in data:
                error_data = data['errors']
                if isinstance(error_data, list):
                    from collections import Counter
                    error_counts = Counter(error_data)
                    for error_type, count in error_counts.items():
                        if count > 3:  # Frequent error
                            changes.append({
                                'type': 'error_handling_adaptation',
                                'error_type': error_type,
                                'frequency': count,
                                'priority': 'high' if count > 5 else 'medium'
                            })
                            
            # Analyze resource usage
            if 'resources' in data:
                res_data = data['resources']
                if isinstance(res_data, dict):
                    for resource, usage in res_data.items():
                        if isinstance(usage, (int, float)) and usage > 0.8:  # High usage
                            changes.append({
                                'type': 'resource_adaptation',
                                'resource': resource,
                                'current_usage': usage,
                                'target_usage': 0.7,
                                'priority': 'high' if usage > 0.9 else 'medium'
                            })
                            
        except Exception as e:
            logging.error(f"Error analyzing behavioral changes: {str(e)}")
            
        return changes
        
    def _adjust_parameters(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adjust system parameters based on input"""
        adjustments = []
        try:
            # Analyze historical data for parameter optimization
            if len(self.adaptation_history) >= 2:
                # Calculate parameter trends
                for param_name in data.keys():
                    if param_name in self.adaptation_history[-2]['data']:
                        old_value = self.adaptation_history[-2]['data'][param_name]
                        new_value = data[param_name]
                        
                        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                            # Calculate adjustment based on trend
                            delta = new_value - old_value
                            adjustment = {
                                'parameter': param_name,
                                'old_value': old_value,
                                'new_value': new_value,
                                'delta': delta,
                                'adjustment': self.learning_rate * delta
                            }
                            adjustments.append(adjustment)
                            
            # Add new parameter adjustments based on current data
            for param_name, value in data.items():
                if isinstance(value, (int, float)):
                    # Check if parameter needs optimization
                    if value > 0.9 or value < 0.1:  # Extreme values
                        adjustment = {
                            'parameter': param_name,
                            'current_value': value,
                            'target_value': 0.5,  # Move toward middle range
                            'adjustment': self.learning_rate * (0.5 - value)
                        }
                        adjustments.append(adjustment)
                        
        except Exception as e:
            logging.error(f"Error adjusting parameters: {str(e)}")
            
        return adjustments
        
    def _adapt_structure(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implement structural adaptations"""
        adaptations = []
        try:
            # Analyze component interactions
            if 'components' in data:
                comp_data = data['components']
                if isinstance(comp_data, dict):
                    # Identify bottlenecks
                    for comp_name, metrics in comp_data.items():
                        if isinstance(metrics, dict):
                            if metrics.get('load', 0) > 0.8:  # High load
                                adaptations.append({
                                    'type': 'load_balancing',
                                    'component': comp_name,
                                    'current_load': metrics['load'],
                                    'action': 'redistribute'
                                })
                                
            # Analyze connection patterns
            if 'connections' in data:
                conn_data = data['connections']
                if isinstance(conn_data, list):
                    from collections import Counter
                    conn_counts = Counter(c['type'] for c in conn_data)
                    
                    # Identify overused connections
                    for conn_type, count in conn_counts.items():
                        if count > len(conn_data) * 0.3:  # More than 30% of connections
                            adaptations.append({
                                'type': 'connection_optimization',
                                'connection_type': conn_type,
                                'frequency': count,
                                'action': 'optimize'
                            })
                            
        except Exception as e:
            logging.error(f"Error adapting structure: {str(e)}")
            
        return adaptations
        
    def _update_learning(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update learning based on input data"""
        updates = []
        try:
            # Analyze learning progress
            if 'learning' in data:
                learning_data = data['learning']
                if isinstance(learning_data, dict):
                    # Update learning parameters
                    for param, value in learning_data.items():
                        if isinstance(value, (int, float)):
                            if value < self.adaptation_threshold:
                                updates.append({
                                    'type': 'learning_rate_adjustment',
                                    'parameter': param,
                                    'current_value': value,
                                    'adjustment': self.learning_rate * (1 - value)
                                })
                                
            # Analyze feedback
            if 'feedback' in data:
                feedback_data = data['feedback']
                if isinstance(feedback_data, list):
                    # Process feedback for learning
                    positive_feedback = sum(1 for f in feedback_data if f.get('score', 0) > 0.7)
                    negative_feedback = sum(1 for f in feedback_data if f.get('score', 0) < 0.3)
                    
                    if positive_feedback + negative_feedback > 0:
                        success_rate = positive_feedback / (positive_feedback + negative_feedback)
                        updates.append({
                            'type': 'feedback_learning',
                            'success_rate': success_rate,
                            'positive_count': positive_feedback,
                            'negative_count': negative_feedback,
                            'adjustment': self.learning_rate * (success_rate - 0.5)
                        })
                        
        except Exception as e:
            logging.error(f"Error updating learning: {str(e)}")
            
        return updates
        
    def _calculate_adaptation_level(self, adaptation_result: Dict[str, Any]) -> float:
        """Calculate overall adaptation level"""
        try:
            scores = []
            
            # Behavioral adaptation score
            if adaptation_result.get('behavioral_changes'):
                changes = adaptation_result['behavioral_changes']
                score = min(1.0, len(changes) * 0.2)
                scores.append(score)
                
            # Parameter adjustment score
            if adaptation_result.get('parameter_adjustments'):
                adjustments = adaptation_result['parameter_adjustments']
                score = min(1.0, len(adjustments) * 0.15)
                scores.append(score)
                
            # Structural adaptation score
            if adaptation_result.get('structural_adaptations'):
                adaptations = adaptation_result['structural_adaptations']
                score = min(1.0, len(adaptations) * 0.25)
                scores.append(score)
                
            # Learning update score
            if adaptation_result.get('learning_updates'):
                updates = adaptation_result['learning_updates']
                score = min(1.0, len(updates) * 0.3)
                scores.append(score)
                
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logging.error(f"Error calculating adaptation level: {str(e)}")
            return 0.0
            
    def get_status(self) -> str:
        """Get current status of the adaptation node"""
        if not self.active:
            return "inactive"
        return (f"active (adaptations: {len(self.adaptation_state['adaptations'])}, "
                f"level: {self.adaptation_state['adaptation_level']:.2f})")
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 