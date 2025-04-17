from typing import Dict, Any, List
import logging
import numpy as np
from datetime import datetime

class PatternProcessor:
    def __init__(self):
        self.active = False
        self.patterns = {}
        self.pattern_threshold = 0.75
        self.recent_data = []
        self.max_history = 1000
        
    def initialize(self) -> bool:
        """Initialize the pattern processor"""
        try:
            self.patterns = {
                'temporal': {},
                'spatial': {},
                'sequential': {},
                'hierarchical': {}
            }
            self.active = True
            logging.info("PatternProcessor initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize PatternProcessor: {str(e)}")
            return False
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data for pattern recognition"""
        try:
            # Store input in recent history
            self.recent_data.append({
                'data': input_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Maintain history size
            if len(self.recent_data) > self.max_history:
                self.recent_data.pop(0)
                
            # Analyze patterns
            temporal_patterns = self._analyze_temporal_patterns()
            spatial_patterns = self._analyze_spatial_patterns(input_data)
            sequential_patterns = self._analyze_sequential_patterns()
            
            # Combine results
            return {
                'temporal_patterns': temporal_patterns,
                'spatial_patterns': spatial_patterns,
                'sequential_patterns': sequential_patterns,
                'pattern_strength': self._calculate_pattern_strength()
            }
        except Exception as e:
            logging.error(f"Error processing pattern input: {str(e)}")
            return {'error': str(e)}
            
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in recent data"""
        try:
            if len(self.recent_data) < 2:
                return {}
                
            patterns = {}
            timestamps = [datetime.fromisoformat(d['timestamp']) for d in self.recent_data]
            
            # Calculate time intervals
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
            
            # Detect periodic patterns
            if len(intervals) >= 3:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if std_interval / mean_interval < 0.2:  # Low variation indicates pattern
                    patterns['periodic'] = {
                        'interval': mean_interval,
                        'confidence': 1 - (std_interval / mean_interval)
                    }
                    
            return patterns
        except Exception as e:
            logging.error(f"Error analyzing temporal patterns: {str(e)}")
            return {}
            
    def _analyze_spatial_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial patterns in input data"""
        try:
            patterns = {}
            
            # Extract numerical values
            values = [v for v in data.values() if isinstance(v, (int, float))]
            
            if values:
                # Calculate statistical patterns
                patterns['distribution'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                
                # Check for clusters
                if len(values) > 3:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=min(3, len(values)))
                    kmeans.fit(np.array(values).reshape(-1, 1))
                    patterns['clusters'] = {
                        'centers': kmeans.cluster_centers_.flatten().tolist(),
                        'labels': kmeans.labels_.tolist()
                    }
                    
            return patterns
        except Exception as e:
            logging.error(f"Error analyzing spatial patterns: {str(e)}")
            return {}
            
    def _analyze_sequential_patterns(self) -> Dict[str, Any]:
        """Analyze sequential patterns in recent data"""
        try:
            if len(self.recent_data) < 3:
                return {}
                
            patterns = {}
            
            # Look for repeating sequences
            data_values = [str(d['data']) for d in self.recent_data]
            sequence_length = min(5, len(data_values) // 2)
            
            for length in range(2, sequence_length + 1):
                sequences = {}
                for i in range(len(data_values) - length + 1):
                    seq = tuple(data_values[i:i+length])
                    sequences[seq] = sequences.get(seq, 0) + 1
                    
                # Find significant sequences
                significant = {seq: count for seq, count in sequences.items() 
                             if count > 1 and count/len(data_values) > 0.1}
                             
                if significant:
                    patterns[f'length_{length}'] = significant
                    
            return patterns
        except Exception as e:
            logging.error(f"Error analyzing sequential patterns: {str(e)}")
            return {}
            
    def _calculate_pattern_strength(self) -> float:
        """Calculate overall pattern strength"""
        try:
            strengths = []
            
            # Temporal pattern strength
            if self.patterns['temporal']:
                strengths.append(
                    sum(p.get('confidence', 0) for p in self.patterns['temporal'].values())
                    / len(self.patterns['temporal'])
                )
                
            # Spatial pattern strength
            if self.patterns['spatial']:
                if 'clusters' in self.patterns['spatial']:
                    strengths.append(0.8)  # Strong pattern if clusters found
                if 'distribution' in self.patterns['spatial']:
                    strengths.append(0.6)  # Moderate pattern if distribution analyzed
                    
            # Sequential pattern strength
            if self.patterns['sequential']:
                strengths.append(
                    min(1.0, len(self.patterns['sequential']) * 0.2)
                )
                
            return np.mean(strengths) if strengths else 0.0
        except Exception as e:
            logging.error(f"Error calculating pattern strength: {str(e)}")
            return 0.0
            
    def get_status(self) -> str:
        """Get current status of the pattern processor"""
        if not self.active:
            return "inactive"
        return f"active (patterns: {sum(len(p) for p in self.patterns.values())})"
        
    def is_active(self) -> bool:
        """Check if the processor is active"""
        return self.active 