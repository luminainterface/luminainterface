from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime

class SynthesisNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.synthesis_state = {}
        self.synthesis_history = []
        self.max_history = 100
        self.synthesis_threshold = 0.7
        
    def initialize(self) -> bool:
        """Initialize the synthesis node"""
        try:
            self.synthesis_state = {
                'current_synthesis': None,
                'synthesis_level': 0.0,
                'components': [],
                'last_update': datetime.now().isoformat()
            }
            self.active = True
            logging.info("SynthesisNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize SynthesisNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for synthesis"""
        try:
            # Store in synthesis history
            self.synthesis_history.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.synthesis_history) > self.max_history:
                self.synthesis_history.pop(0)
                
            # Perform synthesis
            synthesis_result = self._synthesize_data(input_data)
            synthesis_level = self._calculate_synthesis_level(synthesis_result)
            
            # Update synthesis state
            self.synthesis_state.update({
                'current_synthesis': synthesis_result,
                'synthesis_level': synthesis_level,
                'last_update': datetime.now().isoformat()
            })
            
            return {
                'synthesis': synthesis_result,
                'synthesis_level': synthesis_level,
                'components': self.synthesis_state['components']
            }
        except Exception as e:
            logging.error(f"Error processing synthesis input: {str(e)}")
            return {'error': str(e)}
            
    def _synthesize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize input data"""
        try:
            synthesis = {
                'patterns': self._extract_patterns(data),
                'relationships': self._analyze_relationships(data),
                'abstractions': self._create_abstractions(data),
                'integrations': self._integrate_components(data)
            }
            
            # Track components
            components = list(set(self.synthesis_state['components'] + list(data.keys())))
            self.synthesis_state['components'] = components
            
            return synthesis
        except Exception as e:
            logging.error(f"Error synthesizing data: {str(e)}")
            return {}
            
    def _extract_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from data"""
        patterns = []
        try:
            # Numerical patterns
            numerical_values = [v for v in data.values() if isinstance(v, (int, float))]
            if numerical_values:
                patterns.append({
                    'type': 'numerical',
                    'mean': float(np.mean(numerical_values)),
                    'std': float(np.std(numerical_values)),
                    'range': (float(min(numerical_values)), float(max(numerical_values)))
                })
                
            # Categorical patterns
            categorical_values = [v for v in data.values() if isinstance(v, str)]
            if categorical_values:
                from collections import Counter
                counts = Counter(categorical_values)
                patterns.append({
                    'type': 'categorical',
                    'frequencies': dict(counts),
                    'unique_count': len(set(categorical_values))
                })
                
            # Temporal patterns
            if len(self.synthesis_history) >= 2:
                time_diffs = []
                for i in range(1, len(self.synthesis_history)):
                    t1 = datetime.fromisoformat(self.synthesis_history[i-1]['timestamp'])
                    t2 = datetime.fromisoformat(self.synthesis_history[i]['timestamp'])
                    time_diffs.append((t2 - t1).total_seconds())
                    
                patterns.append({
                    'type': 'temporal',
                    'mean_interval': float(np.mean(time_diffs)),
                    'regularity': 1.0 - min(1.0, float(np.std(time_diffs)) / float(np.mean(time_diffs)))
                })
                
        except Exception as e:
            logging.error(f"Error extracting patterns: {str(e)}")
            
        return patterns
        
    def _analyze_relationships(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze relationships between components"""
        relationships = []
        try:
            # Analyze correlations between numerical values
            numerical_items = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            if len(numerical_items) >= 2:
                from scipy.stats import pearsonr
                for k1 in numerical_items:
                    for k2 in numerical_items:
                        if k1 < k2:  # Avoid duplicate correlations
                            try:
                                corr, _ = pearsonr([numerical_items[k1]], [numerical_items[k2]])
                                relationships.append({
                                    'type': 'correlation',
                                    'components': [k1, k2],
                                    'strength': float(corr)
                                })
                            except:
                                continue
                                
            # Analyze co-occurrence of categorical values
            categorical_items = {k: v for k, v in data.items() if isinstance(v, str)}
            if len(categorical_items) >= 2:
                from itertools import combinations
                for k1, k2 in combinations(categorical_items.keys(), 2):
                    relationships.append({
                        'type': 'co-occurrence',
                        'components': [k1, k2],
                        'values': [categorical_items[k1], categorical_items[k2]]
                    })
                    
        except Exception as e:
            logging.error(f"Error analyzing relationships: {str(e)}")
            
        return relationships
        
    def _create_abstractions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create higher-level abstractions"""
        abstractions = []
        try:
            # Group related components
            numerical_group = [k for k, v in data.items() if isinstance(v, (int, float))]
            if numerical_group:
                abstractions.append({
                    'type': 'group',
                    'name': 'numerical_metrics',
                    'components': numerical_group
                })
                
            categorical_group = [k for k, v in data.items() if isinstance(v, str)]
            if categorical_group:
                abstractions.append({
                    'type': 'group',
                    'name': 'categorical_features',
                    'components': categorical_group
                })
                
            # Create composite metrics
            if len(numerical_group) >= 2:
                values = [float(data[k]) for k in numerical_group]
                abstractions.append({
                    'type': 'composite',
                    'name': 'aggregate_metric',
                    'value': float(np.mean(values)),
                    'components': numerical_group
                })
                
        except Exception as e:
            logging.error(f"Error creating abstractions: {str(e)}")
            
        return abstractions
        
    def _integrate_components(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Integrate components and their relationships"""
        integrations = []
        try:
            # Analyze component dependencies
            components = list(data.keys())
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    # Check for direct relationships
                    if isinstance(data[comp1], type(data[comp2])):
                        integrations.append({
                            'type': 'direct',
                            'components': [comp1, comp2],
                            'relationship': 'type_match'
                        })
                        
                    # Check for derived relationships
                    if isinstance(data[comp1], (int, float)) and isinstance(data[comp2], (int, float)):
                        ratio = data[comp1] / data[comp2] if data[comp2] != 0 else 0
                        integrations.append({
                            'type': 'derived',
                            'components': [comp1, comp2],
                            'relationship': 'ratio',
                            'value': float(ratio)
                        })
                        
            # Analyze temporal integration
            if len(self.synthesis_history) >= 2:
                prev_data = self.synthesis_history[-2]['data']
                for comp in set(prev_data.keys()) & set(data.keys()):
                    if isinstance(prev_data[comp], (int, float)) and isinstance(data[comp], (int, float)):
                        change = data[comp] - prev_data[comp]
                        integrations.append({
                            'type': 'temporal',
                            'component': comp,
                            'change': float(change),
                            'rate': float(change) / 1.0  # Assume 1 second interval
                        })
                        
        except Exception as e:
            logging.error(f"Error integrating components: {str(e)}")
            
        return integrations
        
    def _calculate_synthesis_level(self, synthesis_result: Dict[str, Any]) -> float:
        """Calculate overall synthesis level"""
        try:
            scores = []
            
            # Pattern score
            if synthesis_result.get('patterns'):
                pattern_score = min(1.0, len(synthesis_result['patterns']) * 0.2)
                scores.append(pattern_score)
                
            # Relationship score
            if synthesis_result.get('relationships'):
                relationship_score = min(1.0, len(synthesis_result['relationships']) * 0.15)
                scores.append(relationship_score)
                
            # Abstraction score
            if synthesis_result.get('abstractions'):
                abstraction_score = min(1.0, len(synthesis_result['abstractions']) * 0.25)
                scores.append(abstraction_score)
                
            # Integration score
            if synthesis_result.get('integrations'):
                integration_score = min(1.0, len(synthesis_result['integrations']) * 0.3)
                scores.append(integration_score)
                
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logging.error(f"Error calculating synthesis level: {str(e)}")
            return 0.0
            
    def get_status(self) -> str:
        """Get current status of the synthesis node"""
        if not self.active:
            return "inactive"
        return (f"active (components: {len(self.synthesis_state['components'])}, "
                f"synthesis: {self.synthesis_state['synthesis_level']:.2f})")
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 