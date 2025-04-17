from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime

class GrowthNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.growth_state = {}
        self.growth_rate = 0.01
        self.growth_history = []
        self.max_history = 100
        self.growth_stages = ['seed', 'sprout', 'sapling', 'mature']
        
    def initialize(self) -> bool:
        """Initialize the growth node"""
        try:
            self.growth_state = {
                'current_stage': 'seed',
                'growth_level': 0.0,
                'stage_progress': 0.0,
                'growth_metrics': {},
                'last_update': datetime.now().isoformat()
            }
            self.active = True
            logging.info("GrowthNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize GrowthNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for growth"""
        try:
            # Store in growth history
            self.growth_history.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.growth_history) > self.max_history:
                self.growth_history.pop(0)
                
            # Update growth metrics
            self._update_growth_metrics(input_data)
            
            # Process growth
            growth_result = self._process_growth(input_data)
            
            # Check for stage transition
            self._check_stage_transition()
            
            # Update growth state
            self.growth_state.update({
                'last_update': datetime.now().isoformat()
            })
            
            return {
                'growth': growth_result,
                'current_stage': self.growth_state['current_stage'],
                'growth_level': self.growth_state['growth_level'],
                'stage_progress': self.growth_state['stage_progress'],
                'metrics': self.growth_state['growth_metrics']
            }
        except Exception as e:
            logging.error(f"Error processing growth input: {str(e)}")
            return {'error': str(e)}
            
    def _update_growth_metrics(self, data: Dict[str, Any]):
        """Update growth metrics based on input data"""
        try:
            metrics = self.growth_state.get('growth_metrics', {})
            
            # Update complexity metric
            if 'complexity' in data:
                complexity = float(data['complexity'])
                metrics['complexity'] = {
                    'current': complexity,
                    'history': metrics.get('complexity', {}).get('history', []) + [complexity],
                    'trend': self._calculate_trend(metrics.get('complexity', {}).get('history', []) + [complexity])
                }
                
            # Update stability metric
            if 'stability' in data:
                stability = float(data['stability'])
                metrics['stability'] = {
                    'current': stability,
                    'history': metrics.get('stability', {}).get('history', []) + [stability],
                    'trend': self._calculate_trend(metrics.get('stability', {}).get('history', []) + [stability])
                }
                
            # Update activity metric
            if 'activity' in data:
                activity = float(data['activity'])
                metrics['activity'] = {
                    'current': activity,
                    'history': metrics.get('activity', {}).get('history', []) + [activity],
                    'trend': self._calculate_trend(metrics.get('activity', {}).get('history', []) + [activity])
                }
                
            # Maintain history size for each metric
            for metric in metrics.values():
                if len(metric['history']) > self.max_history:
                    metric['history'] = metric['history'][-self.max_history:]
                    
            self.growth_state['growth_metrics'] = metrics
            
        except Exception as e:
            logging.error(f"Error updating growth metrics: {str(e)}")
            
    def _calculate_trend(self, values: List[float], window: int = 5) -> str:
        """Calculate trend direction from recent values"""
        try:
            if len(values) < window:
                return 'stable'
                
            recent_values = values[-window:]
            slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logging.error(f"Error calculating trend: {str(e)}")
            return 'unknown'
            
    def _process_growth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process growth based on input data"""
        try:
            growth_result = {
                'growth_rate': self.growth_rate,
                'changes': [],
                'factors': []
            }
            
            # Calculate growth factors
            metrics = self.growth_state['growth_metrics']
            
            # Complexity factor
            if 'complexity' in metrics:
                complexity = metrics['complexity']['current']
                complexity_factor = min(1.0, complexity / 0.8)  # Normalize to 0-1
                growth_result['factors'].append({
                    'type': 'complexity',
                    'value': complexity_factor,
                    'trend': metrics['complexity']['trend']
                })
                
            # Stability factor
            if 'stability' in metrics:
                stability = metrics['stability']['current']
                stability_factor = min(1.0, stability / 0.9)  # Normalize to 0-1
                growth_result['factors'].append({
                    'type': 'stability',
                    'value': stability_factor,
                    'trend': metrics['stability']['trend']
                })
                
            # Activity factor
            if 'activity' in metrics:
                activity = metrics['activity']['current']
                activity_factor = min(1.0, activity / 0.7)  # Normalize to 0-1
                growth_result['factors'].append({
                    'type': 'activity',
                    'value': activity_factor,
                    'trend': metrics['activity']['trend']
                })
                
            # Calculate overall growth
            if growth_result['factors']:
                factor_values = [f['value'] for f in growth_result['factors']]
                growth_rate = self.growth_rate * np.mean(factor_values)
                
                # Update growth level
                current_level = self.growth_state['growth_level']
                new_level = min(1.0, current_level + growth_rate)
                
                growth_result['changes'].append({
                    'type': 'growth_level',
                    'old_value': current_level,
                    'new_value': new_level,
                    'change': growth_rate
                })
                
                self.growth_state['growth_level'] = new_level
                
                # Update stage progress
                stage_index = self.growth_stages.index(self.growth_state['current_stage'])
                stage_range = 1.0 / len(self.growth_stages)
                stage_start = stage_index * stage_range
                stage_progress = (new_level - stage_start) / stage_range
                
                self.growth_state['stage_progress'] = min(1.0, max(0.0, stage_progress))
                
            return growth_result
            
        except Exception as e:
            logging.error(f"Error processing growth: {str(e)}")
            return {}
            
    def _check_stage_transition(self):
        """Check and handle growth stage transitions"""
        try:
            current_stage = self.growth_state['current_stage']
            current_index = self.growth_stages.index(current_stage)
            growth_level = self.growth_state['growth_level']
            
            # Calculate stage thresholds
            stage_range = 1.0 / len(self.growth_stages)
            next_stage_threshold = (current_index + 1) * stage_range
            
            # Check for stage transition
            if growth_level >= next_stage_threshold and current_index < len(self.growth_stages) - 1:
                next_stage = self.growth_stages[current_index + 1]
                logging.info(f"Growth stage transition: {current_stage} -> {next_stage}")
                
                self.growth_state.update({
                    'current_stage': next_stage,
                    'stage_progress': 0.0
                })
                
        except Exception as e:
            logging.error(f"Error checking stage transition: {str(e)}")
            
    def get_status(self) -> str:
        """Get current status of the growth node"""
        if not self.active:
            return "inactive"
        return (f"active (stage: {self.growth_state['current_stage']}, "
                f"progress: {self.growth_state['stage_progress']:.2f})")
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 