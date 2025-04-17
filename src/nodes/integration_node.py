from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime

class IntegrationNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.integration_state = {}
        self.component_states = {}
        self.integration_history = []
        self.max_history = 100
        self.integration_threshold = 0.7
        
    def initialize(self) -> bool:
        """Initialize the integration node"""
        try:
            self.integration_state = {
                'current_integration': None,
                'integration_level': 0.0,
                'active_components': [],
                'last_update': datetime.now().isoformat()
            }
            self.active = True
            logging.info("IntegrationNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize IntegrationNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for integration"""
        try:
            # Store in integration history
            self.integration_history.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.integration_history) > self.max_history:
                self.integration_history.pop(0)
                
            # Update component states
            self._update_component_states(input_data)
            
            # Perform integration
            integration_result = self._integrate_components(input_data)
            integration_level = self._calculate_integration_level(integration_result)
            
            # Update integration state
            self.integration_state.update({
                'current_integration': integration_result,
                'integration_level': integration_level,
                'last_update': datetime.now().isoformat()
            })
            
            return {
                'integration': integration_result,
                'integration_level': integration_level,
                'active_components': self.integration_state['active_components']
            }
        except Exception as e:
            logging.error(f"Error processing integration input: {str(e)}")
            return {'error': str(e)}
            
    def _update_component_states(self, data: Dict[str, Any]):
        """Update states of individual components"""
        try:
            current_time = datetime.now()
            
            # Update component states
            for component, state in data.items():
                if isinstance(state, dict):
                    self.component_states[component] = {
                        'state': state,
                        'last_update': current_time.isoformat(),
                        'active': state.get('active', True)
                    }
                    
            # Update active components list
            active_components = [comp for comp, info in self.component_states.items()
                               if info['active']]
            self.integration_state['active_components'] = active_components
            
            # Remove stale component states (not updated in last hour)
            stale_threshold = current_time.timestamp() - 3600
            stale_components = []
            
            for component, info in self.component_states.items():
                last_update = datetime.fromisoformat(info['last_update'])
                if last_update.timestamp() < stale_threshold:
                    stale_components.append(component)
                    
            for component in stale_components:
                del self.component_states[component]
                
        except Exception as e:
            logging.error(f"Error updating component states: {str(e)}")
            
    def _integrate_components(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate components and their states"""
        try:
            integration = {
                'component_interactions': self._analyze_interactions(data),
                'state_synchronization': self._synchronize_states(data),
                'data_flow': self._analyze_data_flow(data),
                'system_stability': self._assess_stability(data)
            }
            
            return integration
        except Exception as e:
            logging.error(f"Error integrating components: {str(e)}")
            return {}
            
    def _analyze_interactions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze interactions between components"""
        interactions = []
        try:
            components = list(data.keys())
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    # Check for direct interactions
                    if isinstance(data[comp1], dict) and isinstance(data[comp2], dict):
                        shared_keys = set(data[comp1].keys()) & set(data[comp2].keys())
                        if shared_keys:
                            interactions.append({
                                'type': 'direct',
                                'components': [comp1, comp2],
                                'shared_attributes': list(shared_keys),
                                'strength': len(shared_keys) / max(len(data[comp1]), len(data[comp2]))
                            })
                            
            # Analyze historical interactions
            if len(self.integration_history) >= 2:
                prev_data = self.integration_history[-2]['data']
                for comp in set(prev_data.keys()) & set(data.keys()):
                    if isinstance(prev_data[comp], dict) and isinstance(data[comp], dict):
                        changes = {k: data[comp][k] for k in data[comp]
                                 if k in prev_data[comp] and data[comp][k] != prev_data[comp][k]}
                        if changes:
                            interactions.append({
                                'type': 'temporal',
                                'component': comp,
                                'changes': changes,
                                'change_count': len(changes)
                            })
                            
        except Exception as e:
            logging.error(f"Error analyzing interactions: {str(e)}")
            
        return interactions
        
    def _synchronize_states(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize states across components"""
        try:
            synchronization = {
                'synchronized_components': [],
                'conflicts': [],
                'sync_level': 0.0
            }
            
            # Check for state consistency
            state_groups = {}
            for component, state in data.items():
                if isinstance(state, dict):
                    state_hash = hash(frozenset(state.items()))
                    if state_hash not in state_groups:
                        state_groups[state_hash] = []
                    state_groups[state_hash].append(component)
                    
            # Find synchronized components
            for components in state_groups.values():
                if len(components) > 1:
                    synchronization['synchronized_components'].append(components)
                    
            # Detect conflicts
            for comp1 in data:
                for comp2 in data:
                    if comp1 < comp2:  # Avoid duplicate checks
                        if isinstance(data[comp1], dict) and isinstance(data[comp2], dict):
                            conflicts = self._detect_conflicts(data[comp1], data[comp2])
                            if conflicts:
                                synchronization['conflicts'].append({
                                    'components': [comp1, comp2],
                                    'conflicts': conflicts
                                })
                                
            # Calculate synchronization level
            total_pairs = len(data) * (len(data) - 1) / 2
            if total_pairs > 0:
                sync_pairs = sum(len(group) * (len(group) - 1) / 2 
                               for group in synchronization['synchronized_components'])
                synchronization['sync_level'] = sync_pairs / total_pairs
                
            return synchronization
        except Exception as e:
            logging.error(f"Error synchronizing states: {str(e)}")
            return {}
            
    def _detect_conflicts(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between component states"""
        conflicts = []
        try:
            shared_keys = set(state1.keys()) & set(state2.keys())
            for key in shared_keys:
                if state1[key] != state2[key]:
                    conflicts.append({
                        'attribute': key,
                        'values': [state1[key], state2[key]]
                    })
        except Exception as e:
            logging.error(f"Error detecting conflicts: {str(e)}")
        return conflicts
        
    def _analyze_data_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data flow between components"""
        try:
            flow_analysis = {
                'flows': [],
                'bottlenecks': [],
                'efficiency': 0.0
            }
            
            # Analyze data dependencies
            for comp1, state1 in data.items():
                if isinstance(state1, dict):
                    dependencies = []
                    for comp2, state2 in data.items():
                        if comp1 != comp2 and isinstance(state2, dict):
                            if any(v2 in state1.values() for v2 in state2.values()):
                                dependencies.append(comp2)
                                
                    if dependencies:
                        flow_analysis['flows'].append({
                            'component': comp1,
                            'dependencies': dependencies,
                            'flow_type': 'data_dependency'
                        })
                        
            # Identify bottlenecks
            dependency_counts = {}
            for flow in flow_analysis['flows']:
                for dep in flow['dependencies']:
                    dependency_counts[dep] = dependency_counts.get(dep, 0) + 1
                    
            for comp, count in dependency_counts.items():
                if count > len(data) * 0.5:  # More than 50% components depend on it
                    flow_analysis['bottlenecks'].append({
                        'component': comp,
                        'dependency_count': count
                    })
                    
            # Calculate flow efficiency
            if flow_analysis['flows']:
                total_flows = len(flow_analysis['flows'])
                bottleneck_flows = sum(1 for flow in flow_analysis['flows']
                                     if any(b['component'] in flow['dependencies']
                                           for b in flow_analysis['bottlenecks']))
                flow_analysis['efficiency'] = 1.0 - (bottleneck_flows / total_flows)
                
            return flow_analysis
        except Exception as e:
            logging.error(f"Error analyzing data flow: {str(e)}")
            return {}
            
    def _assess_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system stability"""
        try:
            stability = {
                'overall_stability': 0.0,
                'component_stability': {},
                'risk_factors': []
            }
            
            # Assess individual component stability
            for component, state in data.items():
                if isinstance(state, dict):
                    # Calculate component stability score
                    stability_score = self._calculate_component_stability(state)
                    stability['component_stability'][component] = stability_score
                    
                    # Identify risk factors
                    if stability_score < 0.6:
                        risk_factors = self._identify_risk_factors(component, state)
                        if risk_factors:
                            stability['risk_factors'].extend(risk_factors)
                            
            # Calculate overall stability
            if stability['component_stability']:
                stability['overall_stability'] = float(np.mean(list(
                    stability['component_stability'].values()
                )))
                
            return stability
        except Exception as e:
            logging.error(f"Error assessing stability: {str(e)}")
            return {}
            
    def _calculate_component_stability(self, state: Dict[str, Any]) -> float:
        """Calculate stability score for a component"""
        try:
            scores = []
            
            # Check state completeness
            if state:
                scores.append(min(1.0, len(state) / 5))  # Expect at least 5 state attributes
                
            # Check state consistency
            if len(self.integration_history) >= 2:
                prev_data = self.integration_history[-2]['data']
                changes = sum(1 for k, v in state.items()
                            if k in prev_data and v != prev_data[k])
                consistency = 1.0 - min(1.0, changes / len(state))
                scores.append(consistency)
                
            # Check error presence
            error_indicators = ['error', 'failure', 'warning']
            error_count = sum(1 for k in state if any(ind in str(k).lower() 
                                                    for ind in error_indicators))
            error_score = 1.0 - min(1.0, error_count / len(state))
            scores.append(error_score)
            
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logging.error(f"Error calculating component stability: {str(e)}")
            return 0.0
            
    def _identify_risk_factors(self, component: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify risk factors in component state"""
        risk_factors = []
        try:
            # Check for error states
            error_indicators = ['error', 'failure', 'warning']
            for key, value in state.items():
                if any(ind in str(key).lower() for ind in error_indicators):
                    risk_factors.append({
                        'component': component,
                        'type': 'error_state',
                        'attribute': key,
                        'value': value
                    })
                    
            # Check for resource issues
            resource_indicators = ['memory', 'cpu', 'load', 'capacity']
            for key, value in state.items():
                if any(ind in str(key).lower() for ind in resource_indicators):
                    if isinstance(value, (int, float)) and value > 0.8:  # High resource usage
                        risk_factors.append({
                            'component': component,
                            'type': 'resource_pressure',
                            'resource': key,
                            'value': value
                        })
                        
            # Check for stale state
            if 'timestamp' in state:
                try:
                    state_time = datetime.fromisoformat(state['timestamp'])
                    if (datetime.now() - state_time).total_seconds() > 300:  # 5 minutes
                        risk_factors.append({
                            'component': component,
                            'type': 'stale_state',
                            'last_update': state['timestamp']
                        })
                except:
                    pass
                    
        except Exception as e:
            logging.error(f"Error identifying risk factors: {str(e)}")
            
        return risk_factors
        
    def _calculate_integration_level(self, integration_result: Dict[str, Any]) -> float:
        """Calculate overall integration level"""
        try:
            scores = []
            
            # Interaction score
            if integration_result.get('component_interactions'):
                interactions = integration_result['component_interactions']
                interaction_score = min(1.0, len(interactions) * 0.2)
                scores.append(interaction_score)
                
            # Synchronization score
            if integration_result.get('state_synchronization'):
                sync = integration_result['state_synchronization']
                sync_score = sync.get('sync_level', 0.0)
                scores.append(sync_score)
                
            # Data flow score
            if integration_result.get('data_flow'):
                flow = integration_result['data_flow']
                flow_score = flow.get('efficiency', 0.0)
                scores.append(flow_score)
                
            # Stability score
            if integration_result.get('system_stability'):
                stability = integration_result['system_stability']
                stability_score = stability.get('overall_stability', 0.0)
                scores.append(stability_score)
                
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logging.error(f"Error calculating integration level: {str(e)}")
            return 0.0
            
    def get_status(self) -> str:
        """Get current status of the integration node"""
        if not self.active:
            return "inactive"
        return (f"active (components: {len(self.component_states)}, "
                f"integration: {self.integration_state['integration_level']:.2f})")
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 