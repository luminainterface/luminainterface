from typing import Dict, Any, List
import logging
from .base_node import BaseNode
import numpy as np
from datetime import datetime

class PatternNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.active = False
        self.pattern_state = {}
        self.pattern_history = []
        self.max_history = 1000
        self.pattern_threshold = 0.6
        
    def initialize(self) -> bool:
        """Initialize the pattern node"""
        try:
            self.pattern_state = {
                'current_patterns': [],
                'pattern_strength': 0.0,
                'active_patterns': {},
                'last_update': datetime.now().isoformat()
            }
            self.active = True
            logging.info("PatternNode initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize PatternNode: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input for pattern recognition"""
        try:
            # Store in pattern history
            self.pattern_history.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.pattern_history) > self.max_history:
                self.pattern_history.pop(0)
                
            # Analyze patterns
            temporal_patterns = self._analyze_temporal_patterns()
            spatial_patterns = self._analyze_spatial_patterns(input_data)
            sequence_patterns = self._analyze_sequence_patterns()
            
            # Combine results
            patterns = {
                'temporal': temporal_patterns,
                'spatial': spatial_patterns,
                'sequence': sequence_patterns
            }
            
            # Update pattern state
            pattern_strength = self._calculate_pattern_strength(patterns)
            self.pattern_state.update({
                'current_patterns': patterns,
                'pattern_strength': pattern_strength,
                'last_update': datetime.now().isoformat()
            })
            
            # Track active patterns
            self._update_active_patterns(patterns)
            
            return {
                'patterns': patterns,
                'pattern_strength': pattern_strength,
                'active_patterns': self.pattern_state['active_patterns']
            }
        except Exception as e:
            logging.error(f"Error processing pattern input: {str(e)}")
            return {'error': str(e)}
            
    def _analyze_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in history"""
        patterns = []
        try:
            if len(self.pattern_history) < 3:
                return patterns
                
            # Extract timestamps
            timestamps = [datetime.fromisoformat(entry['timestamp']) 
                        for entry in self.pattern_history]
                        
            # Calculate intervals
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
                        
            # Detect periodic patterns
            if len(intervals) >= 3:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if std_interval / mean_interval < 0.2:  # Regular intervals
                    patterns.append({
                        'type': 'periodic',
                        'interval': mean_interval,
                        'regularity': 1 - (std_interval / mean_interval),
                        'confidence': 0.8
                    })
                    
            # Detect trends
            if len(self.pattern_history) >= 5:
                values = []
                for entry in self.pattern_history[-5:]:
                    data = entry['data']
                    if isinstance(data, dict):
                        nums = [v for v in data.values() if isinstance(v, (int, float))]
                        if nums:
                            values.append(np.mean(nums))
                            
                if len(values) >= 3:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    if abs(trend) > 0.1:
                        patterns.append({
                            'type': 'trend',
                            'direction': 'increasing' if trend > 0 else 'decreasing',
                            'magnitude': abs(trend),
                            'confidence': min(1.0, abs(trend) * 2)
                        })
                        
        except Exception as e:
            logging.error(f"Error analyzing temporal patterns: {str(e)}")
            
        return patterns
        
    def _analyze_spatial_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze spatial patterns in current data"""
        patterns = []
        try:
            # Extract numerical values
            values = [v for v in data.values() if isinstance(v, (int, float))]
            
            if values:
                # Calculate statistical patterns
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Detect clusters
                if len(values) >= 3:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=min(3, len(values)))
                    kmeans.fit(np.array(values).reshape(-1, 1))
                    
                    patterns.append({
                        'type': 'cluster',
                        'centers': kmeans.cluster_centers_.flatten().tolist(),
                        'count': len(values),
                        'confidence': 0.7
                    })
                    
                # Detect outliers
                z_scores = np.abs((values - mean_val) / std_val)
                outliers = [v for v, z in zip(values, z_scores) if z > 2]
                
                if outliers:
                    patterns.append({
                        'type': 'outlier',
                        'values': outliers,
                        'threshold': mean_val + 2*std_val,
                        'confidence': 0.9
                    })
                    
        except Exception as e:
            logging.error(f"Error analyzing spatial patterns: {str(e)}")
            
        return patterns
        
    def _analyze_sequence_patterns(self) -> List[Dict[str, Any]]:
        """Analyze sequential patterns in history"""
        patterns = []
        try:
            if len(self.pattern_history) < 3:
                return patterns
                
            # Extract sequences of values
            sequences = []
            for entry in self.pattern_history:
                data = entry['data']
                if isinstance(data, dict):
                    seq = [v for v in data.values() if isinstance(v, (int, float, str))]
                    if seq:
                        sequences.append(seq)
                        
            if not sequences:
                return patterns
                
            # Find repeating subsequences
            for length in range(2, min(5, len(sequences[0]) + 1)):
                for i in range(len(sequences) - length + 1):
                    subsequence = tuple(sequences[i:i+length])
                    count = sum(1 for j in range(len(sequences) - length + 1)
                              if tuple(sequences[j:j+length]) == subsequence)
                    
                    if count > 1:  # Repeating pattern found
                        patterns.append({
                            'type': 'sequence',
                            'sequence': list(subsequence),
                            'length': length,
                            'occurrences': count,
                            'confidence': min(1.0, count * 0.2)
                        })
                        
        except Exception as e:
            logging.error(f"Error analyzing sequence patterns: {str(e)}")
            
        return patterns
        
    def _calculate_pattern_strength(self, patterns: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall pattern strength"""
        try:
            strengths = []
            
            # Temporal pattern strength
            if patterns.get('temporal'):
                temporal_strength = sum(p.get('confidence', 0) 
                                     for p in patterns['temporal'])
                strengths.append(min(1.0, temporal_strength * 0.3))
                
            # Spatial pattern strength
            if patterns.get('spatial'):
                spatial_strength = sum(p.get('confidence', 0) 
                                    for p in patterns['spatial'])
                strengths.append(min(1.0, spatial_strength * 0.3))
                
            # Sequence pattern strength
            if patterns.get('sequence'):
                sequence_strength = sum(p.get('confidence', 0) 
                                     for p in patterns['sequence'])
                strengths.append(min(1.0, sequence_strength * 0.3))
                
            return float(np.mean(strengths)) if strengths else 0.0
        except Exception as e:
            logging.error(f"Error calculating pattern strength: {str(e)}")
            return 0.0
            
    def _update_active_patterns(self, patterns: Dict[str, List[Dict[str, Any]]]):
        """Update active patterns tracking"""
        try:
            current_time = datetime.now()
            
            # Add new patterns
            for pattern_type, type_patterns in patterns.items():
                for pattern in type_patterns:
                    if pattern.get('confidence', 0) >= self.pattern_threshold:
                        pattern_id = f"{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        self.pattern_state['active_patterns'][pattern_id] = {
                            'type': pattern_type,
                            'pattern': pattern,
                            'start_time': current_time.isoformat(),
                            'last_seen': current_time.isoformat()
                        }
                        
            # Remove inactive patterns (not seen in last hour)
            inactive_threshold = current_time.timestamp() - 3600  # 1 hour
            inactive_patterns = []
            
            for pattern_id, pattern_info in self.pattern_state['active_patterns'].items():
                last_seen = datetime.fromisoformat(pattern_info['last_seen'])
                if last_seen.timestamp() < inactive_threshold:
                    inactive_patterns.append(pattern_id)
                    
            for pattern_id in inactive_patterns:
                del self.pattern_state['active_patterns'][pattern_id]
                
        except Exception as e:
            logging.error(f"Error updating active patterns: {str(e)}")
            
    def get_status(self) -> str:
        """Get current status of the pattern node"""
        if not self.active:
            return "inactive"
        return (f"active (patterns: {len(self.pattern_state['active_patterns'])}, "
                f"strength: {self.pattern_state['pattern_strength']:.2f})")
        
    def is_active(self) -> bool:
        """Check if the node is active"""
        return self.active 