import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import time
import hashlib

@dataclass
class NeuralPattern:
    """Represents a neural activity pattern"""
    activation_pattern: np.ndarray
    temporal_sequence: List[np.ndarray]
    strength: float
    frequency: float
    connections: List[Tuple[int, int]]
    metadata: Dict[str, Any]
    pattern_id: str = field(default_factory=lambda: f"neural_{time.time():.6f}")

@dataclass
class LinguisticPattern:
    """Represents a linguistic pattern"""
    text: str
    semantic_vector: np.ndarray
    context: Dict[str, Any]
    associations: List[str]
    confidence: float
    pattern_id: str = field(default_factory=lambda: f"linguistic_{time.time():.6f}")
    metadata: Dict[str, Any] = field(default_factory=dict)

class NeuralLinguisticBridge:
    """Bridge for translating between neural and linguistic patterns"""
    
    def __init__(self, dimension: int = 64):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dimension = dimension
        self.pattern_memory = {}
        self.semantic_space = self._initialize_semantic_space()
        
        # Initialize breath detector and memory system
        self.breath_detector = BreathDetector()
        self.memory_system = AdvancedMemory()
        self.memory_system.set_breath_detector(self.breath_detector)
        
        # Initialize weights
        self.weights = {
            'neural_weight': 0.5,
            'language_weight': 0.3,
            'memory_weight': 0.2
        }
        
    def _initialize_semantic_space(self) -> Dict[str, np.ndarray]:
        """Initialize the semantic vector space"""
        return {
            'activation': self._create_basis_vector(),
            'sequence': self._create_basis_vector(),
            'connection': self._create_basis_vector(),
            'emotion': self._create_basis_vector(),
            'concept': self._create_basis_vector(),
            'temporal': self._create_basis_vector()
        }
        
    def _create_basis_vector(self) -> np.ndarray:
        """Create a normalized random basis vector"""
        vector = np.random.randn(self.dimension)
        return vector / np.linalg.norm(vector)
        
    def process_breath_sample(self, sample: float):
        """Process a breath sample and update weights"""
        try:
            # Process breath sample
            pattern = self.breath_detector.process_sample(sample)
            
            if pattern:
                # Update weights based on breath pattern
                self.weights = self.breath_detector.adjust_neural_weights(self.weights)
                self.logger.info(f"Updated weights based on breath pattern: {self.weights}")
                
        except Exception as e:
            self.logger.error(f"Error processing breath sample: {str(e)}")
            
    def neural_to_linguistic(self, pattern: NeuralPattern) -> LinguisticPattern:
        """Convert neural pattern to linguistic representation"""
        try:
            # Extract and normalize features
            activation_features = self._extract_activation_features(pattern.activation_pattern)
            temporal_features = self._extract_temporal_features(pattern.temporal_sequence)
            connection_features = self._extract_connection_features(pattern.connections)
            
            # Store original pattern for accurate reconstruction
            pattern_id = self._generate_pattern_id(pattern)
            self.pattern_memory[pattern_id] = pattern
            
            # Generate description
            description = self._generate_pattern_description(pattern)
            
            # Create linguistic pattern
            linguistic_pattern = LinguisticPattern(
                text=description,
                semantic_vector=np.concatenate([activation_features, temporal_features, connection_features]),
                context={
                    'strength': pattern.strength,
                    'frequency': pattern.frequency,
                    'complexity': len(pattern.connections) / (pattern.activation_pattern.size ** 2)
                },
                associations=self._extract_pattern_associations(pattern),
                confidence=self._calculate_translation_confidence(pattern),
                metadata={
                    'original_pattern_type': 'neural',
                    'translation_timestamp': time.time(),
                    **pattern.metadata
                }
            )
            
            # Store in memory system with breath context
            self.memory_system.store_memory(
                content=linguistic_pattern,
                confidence=linguistic_pattern.confidence,
                associations=linguistic_pattern.associations,
                neural_pattern=pattern.activation_pattern,
                linguistic_pattern=linguistic_pattern.semantic_vector
            )
            
            return linguistic_pattern
            
        except Exception as e:
            self.logger.error(f"Error converting neural to linguistic pattern: {str(e)}")
            raise
            
    def linguistic_to_neural(self, pattern: LinguisticPattern) -> NeuralPattern:
        """Convert linguistic pattern to neural representation"""
        try:
            # Check memory system first
            memories = self.memory_system.retrieve_memory(pattern.text)
            for memory in memories:
                if memory.neural_pattern is not None:
                    # Found matching neural pattern in memory
                    return NeuralPattern(
                        activation_pattern=memory.neural_pattern,
                        temporal_sequence=[memory.neural_pattern],  # Simple temporal sequence
                        strength=np.mean(memory.neural_pattern),
                        frequency=0.5,
                        connections=self._generate_connections(memory.neural_pattern),
                        metadata={
                            'original_pattern_type': 'linguistic',
                            'translation_timestamp': time.time(),
                            'source_text': pattern.text,
                            'memory_id': memory.memory_id,
                            **pattern.metadata
                        }
                    )
                    
            # If pattern exists in memory, return the original
            if pattern.pattern_id in self.pattern_memory:
                return self.pattern_memory[pattern.pattern_id]
            
            # Otherwise reconstruct from features
            features = pattern.semantic_vector
            feature_size = self.dimension // 3
            
            activation_pattern = self._reconstruct_activation_pattern(features[:feature_size])
            temporal_sequence = self._reconstruct_temporal_sequence(features[feature_size:2*feature_size])
            connections = self._reconstruct_connections(features[2*feature_size:])
            
            neural_pattern = NeuralPattern(
                activation_pattern=activation_pattern,
                temporal_sequence=temporal_sequence,
                strength=np.mean(activation_pattern),
                frequency=0.5,  # Default frequency
                connections=connections,
                metadata={
                    'original_pattern_type': 'linguistic',
                    'translation_timestamp': time.time(),
                    'source_text': pattern.text,
                    **pattern.metadata
                }
            )
            
            # Store in memory system
            self.memory_system.store_memory(
                content=neural_pattern,
                confidence=pattern.confidence,
                associations=pattern.associations,
                neural_pattern=activation_pattern,
                linguistic_pattern=pattern.semantic_vector
            )
            
            return neural_pattern
            
        except Exception as e:
            self.logger.error(f"Error converting linguistic to neural pattern: {str(e)}")
            raise
            
    def _extract_activation_features(self, activation_pattern: np.ndarray) -> np.ndarray:
        """Extract features from activation pattern"""
        # Ensure consistent dimensionality
        target_dim = self.dimension
        features = np.zeros(target_dim)
        
        if activation_pattern.size > target_dim:
            # Use average pooling for dimensionality reduction
            reshaped = activation_pattern.reshape(-1, activation_pattern.size // target_dim)
            features = np.mean(reshaped, axis=1)[:target_dim]
        else:
            # Pad with zeros if needed
            features[:activation_pattern.size] = activation_pattern.flatten()
            
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
        
    def _extract_temporal_features(self, temporal_sequence: List[np.ndarray]) -> np.ndarray:
        """Extract features from temporal sequence"""
        features = np.zeros(self.dimension)
        
        # Process each timestep
        for i, timestep in enumerate(temporal_sequence):
            # Weight more recent timesteps higher
            weight = np.exp(-0.1 * (len(temporal_sequence) - i - 1))
            timestep_features = self._extract_activation_features(timestep)
            features += weight * timestep_features
            
        return features / np.linalg.norm(features)
        
    def _extract_connection_features(self, connections: List[Tuple[int, int]]) -> np.ndarray:
        """Extract features from neural connections"""
        features = np.zeros(self.dimension)
        
        if not connections:
            return features
            
        # Convert connections to adjacency matrix
        size = max(max(i, j) for i, j in connections) + 1
        size = min(size, self.dimension // 4)  # Limit size to prevent dimension overflow
        adj_matrix = np.zeros((size, size))
        for i, j in connections:
            if i < size and j < size:  # Only include connections within size limit
                adj_matrix[i, j] = 1
        
        # Extract graph features
        in_degree = np.sum(adj_matrix, axis=0)
        out_degree = np.sum(adj_matrix, axis=1)
        
        # Normalize and store features
        features[:size] = in_degree / (np.max(in_degree) if np.max(in_degree) > 0 else 1)
        features[size:2*size] = out_degree / (np.max(out_degree) if np.max(out_degree) > 0 else 1)
        
        return features
        
    def _generate_pattern_description(self, pattern: NeuralPattern) -> str:
        """Generate text description of neural pattern"""
        # Analyze activation strength
        mean_activation = np.mean(pattern.activation_pattern)
        activation_desc = "Strong" if mean_activation > 0.6 else "Moderate" if mean_activation > 0.3 else "Weak"
        
        # Analyze temporal characteristics
        temporal_stability = np.mean([np.std(seq) for seq in pattern.temporal_sequence])
        temporal_desc = "stable" if temporal_stability < 0.3 else "variable"
        
        # Analyze connectivity
        connectivity_density = len(pattern.connections) / (len(pattern.activation_pattern) ** 2)
        connectivity_desc = "dense" if connectivity_density > 0.3 else "moderate" if connectivity_density > 0.1 else "sparse"
        
        # Get pattern type and confidence
        pattern_type = pattern.metadata.get('pattern_type', 'unknown')
        confidence = pattern.metadata.get('confidence', 0.5)
        confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return f"{activation_desc} activation pattern with {temporal_desc} temporal state and {connectivity_desc} connectivity showing {pattern_type} characteristics with {confidence_desc} confidence"
        
    def _extract_pattern_associations(self, pattern: NeuralPattern) -> List[str]:
        """Extract associated concepts from neural pattern using sparse representation"""
        associations = []
        
        # Efficient activation analysis using numpy operations
        activation_max = np.max(pattern.activation_pattern)
        activation_std = np.std(pattern.activation_pattern)
        
        if activation_max > 0.8:
            associations.append("high_intensity")
        if activation_std > 0.5:
            associations.append("variable_activation")
            
        # Temporal analysis
        if pattern.temporal_sequence:
            seq_length = len(pattern.temporal_sequence)
            seq_variance = np.mean([np.std(t) for t in pattern.temporal_sequence])
            
            if seq_length > 5:
                associations.append("extended_sequence")
            if seq_variance > 0.3:
                associations.append("dynamic_pattern")
            
        # Efficient sparse connection analysis
        if pattern.connections:
            total_possible = pattern.activation_pattern.size
            connection_density = len(pattern.connections) / total_possible
            
            if connection_density > 0.5:
                associations.append("highly_connected")
            elif connection_density < 0.1:
                associations.append("sparse_connectivity")
                
        if pattern.strength > 0.7:
            associations.append("strong_pattern")
            
        # Add metadata-based associations
        if pattern.metadata.get('pattern_type'):
            associations.append(f"type_{pattern.metadata['pattern_type']}")
            
        return associations
        
    def _calculate_translation_confidence(self, pattern: NeuralPattern) -> float:
        """Calculate confidence score with adaptive weighting"""
        scores = {}
        
        # Activation quality (30%)
        scores['activation'] = np.mean(np.abs(pattern.activation_pattern))
        
        # Temporal quality (25%)
        if pattern.temporal_sequence:
            scores['temporal'] = np.mean([np.std(t) for t in pattern.temporal_sequence])
        
        # Connection quality (25%)
        if pattern.connections:
            total_possible = pattern.activation_pattern.size
            scores['connectivity'] = len(pattern.connections) / total_possible
        
        # Pattern strength (20%)
        scores['strength'] = pattern.strength
        
        # Calculate weighted average with adaptive weights
        weights = {
            'activation': 0.3,
            'temporal': 0.25,
            'connectivity': 0.25,
            'strength': 0.2
        }
        
        # Adjust weights if some scores are missing
        available_weights = {k: v for k, v in weights.items() if k in scores}
        weight_sum = sum(available_weights.values())
        
        if weight_sum > 0:
            # Normalize weights
            available_weights = {k: v/weight_sum for k, v in available_weights.items()}
            
            # Calculate confidence
            confidence = sum(scores[k] * available_weights[k] for k in scores)
        else:
            # Fallback if no scores available
            confidence = 0.5
            
        return float(confidence)
        
    def _generate_pattern_id(self, pattern: Union[NeuralPattern, LinguisticPattern]) -> str:
        """Generate deterministic unique ID for pattern using SHA-256"""
        if isinstance(pattern, NeuralPattern):
            # Create deterministic string from pattern data
            pattern_data = (
                pattern.activation_pattern.tobytes() +  
                b''.join(t.tobytes() for t in pattern.temporal_sequence) +
                str(sorted(pattern.connections)).encode()
            )
        else:
            # For linguistic patterns
            pattern_data = (
                pattern.text.encode() + 
                pattern.semantic_vector.tobytes() +
                str(sorted(pattern.context.items())).encode()
            )
            
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(pattern_data)
        pattern_hash = hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
        
        prefix = "neural" if isinstance(pattern, NeuralPattern) else "linguistic"
        return f"{prefix}_{pattern_hash}"
        
    def _generate_connections(self, pattern: np.ndarray) -> List[Tuple[int, int]]:
        """Generate connections based on pattern activation"""
        connections = []
        size = len(pattern)
        
        # Create connections between active nodes
        for i in range(size):
            if pattern[i] > 0.5:  # Threshold for active nodes
                for j in range(i + 1, size):
                    if pattern[j] > 0.5:
                        connections.append((i, j))
                        
        return connections
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the bridge system"""
        return {
            'weights': self.weights,
            'memory_stats': self.memory_system.get_memory_stats(),
            'breath_state': self.breath_detector.get_current_state(),
            'pattern_memory_size': len(self.pattern_memory)
        } 
 
 
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import time
import hashlib

@dataclass
class NeuralPattern:
    """Represents a neural activity pattern"""
    activation_pattern: np.ndarray
    temporal_sequence: List[np.ndarray]
    strength: float
    frequency: float
    connections: List[Tuple[int, int]]
    metadata: Dict[str, Any]
    pattern_id: str = field(default_factory=lambda: f"neural_{time.time():.6f}")

@dataclass
class LinguisticPattern:
    """Represents a linguistic pattern"""
    text: str
    semantic_vector: np.ndarray
    context: Dict[str, Any]
    associations: List[str]
    confidence: float
    pattern_id: str = field(default_factory=lambda: f"linguistic_{time.time():.6f}")
    metadata: Dict[str, Any] = field(default_factory=dict)

class NeuralLinguisticBridge:
    """Bridge for translating between neural and linguistic patterns"""
    
    def __init__(self, dimension: int = 64):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dimension = dimension
        self.pattern_memory = {}
        self.semantic_space = self._initialize_semantic_space()
        
        # Initialize breath detector and memory system
        self.breath_detector = BreathDetector()
        self.memory_system = AdvancedMemory()
        self.memory_system.set_breath_detector(self.breath_detector)
        
        # Initialize weights
        self.weights = {
            'neural_weight': 0.5,
            'language_weight': 0.3,
            'memory_weight': 0.2
        }
        
    def _initialize_semantic_space(self) -> Dict[str, np.ndarray]:
        """Initialize the semantic vector space"""
        return {
            'activation': self._create_basis_vector(),
            'sequence': self._create_basis_vector(),
            'connection': self._create_basis_vector(),
            'emotion': self._create_basis_vector(),
            'concept': self._create_basis_vector(),
            'temporal': self._create_basis_vector()
        }
        
    def _create_basis_vector(self) -> np.ndarray:
        """Create a normalized random basis vector"""
        vector = np.random.randn(self.dimension)
        return vector / np.linalg.norm(vector)
        
    def process_breath_sample(self, sample: float):
        """Process a breath sample and update weights"""
        try:
            # Process breath sample
            pattern = self.breath_detector.process_sample(sample)
            
            if pattern:
                # Update weights based on breath pattern
                self.weights = self.breath_detector.adjust_neural_weights(self.weights)
                self.logger.info(f"Updated weights based on breath pattern: {self.weights}")
                
        except Exception as e:
            self.logger.error(f"Error processing breath sample: {str(e)}")
            
    def neural_to_linguistic(self, pattern: NeuralPattern) -> LinguisticPattern:
        """Convert neural pattern to linguistic representation"""
        try:
            # Extract and normalize features
            activation_features = self._extract_activation_features(pattern.activation_pattern)
            temporal_features = self._extract_temporal_features(pattern.temporal_sequence)
            connection_features = self._extract_connection_features(pattern.connections)
            
            # Store original pattern for accurate reconstruction
            pattern_id = self._generate_pattern_id(pattern)
            self.pattern_memory[pattern_id] = pattern
            
            # Generate description
            description = self._generate_pattern_description(pattern)
            
            # Create linguistic pattern
            linguistic_pattern = LinguisticPattern(
                text=description,
                semantic_vector=np.concatenate([activation_features, temporal_features, connection_features]),
                context={
                    'strength': pattern.strength,
                    'frequency': pattern.frequency,
                    'complexity': len(pattern.connections) / (pattern.activation_pattern.size ** 2)
                },
                associations=self._extract_pattern_associations(pattern),
                confidence=self._calculate_translation_confidence(pattern),
                metadata={
                    'original_pattern_type': 'neural',
                    'translation_timestamp': time.time(),
                    **pattern.metadata
                }
            )
            
            # Store in memory system with breath context
            self.memory_system.store_memory(
                content=linguistic_pattern,
                confidence=linguistic_pattern.confidence,
                associations=linguistic_pattern.associations,
                neural_pattern=pattern.activation_pattern,
                linguistic_pattern=linguistic_pattern.semantic_vector
            )
            
            return linguistic_pattern
            
        except Exception as e:
            self.logger.error(f"Error converting neural to linguistic pattern: {str(e)}")
            raise
            
    def linguistic_to_neural(self, pattern: LinguisticPattern) -> NeuralPattern:
        """Convert linguistic pattern to neural representation"""
        try:
            # Check memory system first
            memories = self.memory_system.retrieve_memory(pattern.text)
            for memory in memories:
                if memory.neural_pattern is not None:
                    # Found matching neural pattern in memory
                    return NeuralPattern(
                        activation_pattern=memory.neural_pattern,
                        temporal_sequence=[memory.neural_pattern],  # Simple temporal sequence
                        strength=np.mean(memory.neural_pattern),
                        frequency=0.5,
                        connections=self._generate_connections(memory.neural_pattern),
                        metadata={
                            'original_pattern_type': 'linguistic',
                            'translation_timestamp': time.time(),
                            'source_text': pattern.text,
                            'memory_id': memory.memory_id,
                            **pattern.metadata
                        }
                    )
                    
            # If pattern exists in memory, return the original
            if pattern.pattern_id in self.pattern_memory:
                return self.pattern_memory[pattern.pattern_id]
            
            # Otherwise reconstruct from features
            features = pattern.semantic_vector
            feature_size = self.dimension // 3
            
            activation_pattern = self._reconstruct_activation_pattern(features[:feature_size])
            temporal_sequence = self._reconstruct_temporal_sequence(features[feature_size:2*feature_size])
            connections = self._reconstruct_connections(features[2*feature_size:])
            
            neural_pattern = NeuralPattern(
                activation_pattern=activation_pattern,
                temporal_sequence=temporal_sequence,
                strength=np.mean(activation_pattern),
                frequency=0.5,  # Default frequency
                connections=connections,
                metadata={
                    'original_pattern_type': 'linguistic',
                    'translation_timestamp': time.time(),
                    'source_text': pattern.text,
                    **pattern.metadata
                }
            )
            
            # Store in memory system
            self.memory_system.store_memory(
                content=neural_pattern,
                confidence=pattern.confidence,
                associations=pattern.associations,
                neural_pattern=activation_pattern,
                linguistic_pattern=pattern.semantic_vector
            )
            
            return neural_pattern
            
        except Exception as e:
            self.logger.error(f"Error converting linguistic to neural pattern: {str(e)}")
            raise
            
    def _extract_activation_features(self, activation_pattern: np.ndarray) -> np.ndarray:
        """Extract features from activation pattern"""
        # Ensure consistent dimensionality
        target_dim = self.dimension
        features = np.zeros(target_dim)
        
        if activation_pattern.size > target_dim:
            # Use average pooling for dimensionality reduction
            reshaped = activation_pattern.reshape(-1, activation_pattern.size // target_dim)
            features = np.mean(reshaped, axis=1)[:target_dim]
        else:
            # Pad with zeros if needed
            features[:activation_pattern.size] = activation_pattern.flatten()
            
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
        
    def _extract_temporal_features(self, temporal_sequence: List[np.ndarray]) -> np.ndarray:
        """Extract features from temporal sequence"""
        features = np.zeros(self.dimension)
        
        # Process each timestep
        for i, timestep in enumerate(temporal_sequence):
            # Weight more recent timesteps higher
            weight = np.exp(-0.1 * (len(temporal_sequence) - i - 1))
            timestep_features = self._extract_activation_features(timestep)
            features += weight * timestep_features
            
        return features / np.linalg.norm(features)
        
    def _extract_connection_features(self, connections: List[Tuple[int, int]]) -> np.ndarray:
        """Extract features from neural connections"""
        features = np.zeros(self.dimension)
        
        if not connections:
            return features
            
        # Convert connections to adjacency matrix
        size = max(max(i, j) for i, j in connections) + 1
        size = min(size, self.dimension // 4)  # Limit size to prevent dimension overflow
        adj_matrix = np.zeros((size, size))
        for i, j in connections:
            if i < size and j < size:  # Only include connections within size limit
                adj_matrix[i, j] = 1
        
        # Extract graph features
        in_degree = np.sum(adj_matrix, axis=0)
        out_degree = np.sum(adj_matrix, axis=1)
        
        # Normalize and store features
        features[:size] = in_degree / (np.max(in_degree) if np.max(in_degree) > 0 else 1)
        features[size:2*size] = out_degree / (np.max(out_degree) if np.max(out_degree) > 0 else 1)
        
        return features
        
    def _generate_pattern_description(self, pattern: NeuralPattern) -> str:
        """Generate text description of neural pattern"""
        # Analyze activation strength
        mean_activation = np.mean(pattern.activation_pattern)
        activation_desc = "Strong" if mean_activation > 0.6 else "Moderate" if mean_activation > 0.3 else "Weak"
        
        # Analyze temporal characteristics
        temporal_stability = np.mean([np.std(seq) for seq in pattern.temporal_sequence])
        temporal_desc = "stable" if temporal_stability < 0.3 else "variable"
        
        # Analyze connectivity
        connectivity_density = len(pattern.connections) / (len(pattern.activation_pattern) ** 2)
        connectivity_desc = "dense" if connectivity_density > 0.3 else "moderate" if connectivity_density > 0.1 else "sparse"
        
        # Get pattern type and confidence
        pattern_type = pattern.metadata.get('pattern_type', 'unknown')
        confidence = pattern.metadata.get('confidence', 0.5)
        confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return f"{activation_desc} activation pattern with {temporal_desc} temporal state and {connectivity_desc} connectivity showing {pattern_type} characteristics with {confidence_desc} confidence"
        
    def _extract_pattern_associations(self, pattern: NeuralPattern) -> List[str]:
        """Extract associated concepts from neural pattern using sparse representation"""
        associations = []
        
        # Efficient activation analysis using numpy operations
        activation_max = np.max(pattern.activation_pattern)
        activation_std = np.std(pattern.activation_pattern)
        
        if activation_max > 0.8:
            associations.append("high_intensity")
        if activation_std > 0.5:
            associations.append("variable_activation")
            
        # Temporal analysis
        if pattern.temporal_sequence:
            seq_length = len(pattern.temporal_sequence)
            seq_variance = np.mean([np.std(t) for t in pattern.temporal_sequence])
            
            if seq_length > 5:
                associations.append("extended_sequence")
            if seq_variance > 0.3:
                associations.append("dynamic_pattern")
            
        # Efficient sparse connection analysis
        if pattern.connections:
            total_possible = pattern.activation_pattern.size
            connection_density = len(pattern.connections) / total_possible
            
            if connection_density > 0.5:
                associations.append("highly_connected")
            elif connection_density < 0.1:
                associations.append("sparse_connectivity")
                
        if pattern.strength > 0.7:
            associations.append("strong_pattern")
            
        # Add metadata-based associations
        if pattern.metadata.get('pattern_type'):
            associations.append(f"type_{pattern.metadata['pattern_type']}")
            
        return associations
        
    def _calculate_translation_confidence(self, pattern: NeuralPattern) -> float:
        """Calculate confidence score with adaptive weighting"""
        scores = {}
        
        # Activation quality (30%)
        scores['activation'] = np.mean(np.abs(pattern.activation_pattern))
        
        # Temporal quality (25%)
        if pattern.temporal_sequence:
            scores['temporal'] = np.mean([np.std(t) for t in pattern.temporal_sequence])
        
        # Connection quality (25%)
        if pattern.connections:
            total_possible = pattern.activation_pattern.size
            scores['connectivity'] = len(pattern.connections) / total_possible
        
        # Pattern strength (20%)
        scores['strength'] = pattern.strength
        
        # Calculate weighted average with adaptive weights
        weights = {
            'activation': 0.3,
            'temporal': 0.25,
            'connectivity': 0.25,
            'strength': 0.2
        }
        
        # Adjust weights if some scores are missing
        available_weights = {k: v for k, v in weights.items() if k in scores}
        weight_sum = sum(available_weights.values())
        
        if weight_sum > 0:
            # Normalize weights
            available_weights = {k: v/weight_sum for k, v in available_weights.items()}
            
            # Calculate confidence
            confidence = sum(scores[k] * available_weights[k] for k in scores)
        else:
            # Fallback if no scores available
            confidence = 0.5
            
        return float(confidence)
        
    def _generate_pattern_id(self, pattern: Union[NeuralPattern, LinguisticPattern]) -> str:
        """Generate deterministic unique ID for pattern using SHA-256"""
        if isinstance(pattern, NeuralPattern):
            # Create deterministic string from pattern data
            pattern_data = (
                pattern.activation_pattern.tobytes() +  
                b''.join(t.tobytes() for t in pattern.temporal_sequence) +
                str(sorted(pattern.connections)).encode()
            )
        else:
            # For linguistic patterns
            pattern_data = (
                pattern.text.encode() + 
                pattern.semantic_vector.tobytes() +
                str(sorted(pattern.context.items())).encode()
            )
            
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(pattern_data)
        pattern_hash = hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
        
        prefix = "neural" if isinstance(pattern, NeuralPattern) else "linguistic"
        return f"{prefix}_{pattern_hash}"
        
    def _generate_connections(self, pattern: np.ndarray) -> List[Tuple[int, int]]:
        """Generate connections based on pattern activation"""
        connections = []
        size = len(pattern)
        
        # Create connections between active nodes
        for i in range(size):
            if pattern[i] > 0.5:  # Threshold for active nodes
                for j in range(i + 1, size):
                    if pattern[j] > 0.5:
                        connections.append((i, j))
                        
        return connections
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the bridge system"""
        return {
            'weights': self.weights,
            'memory_stats': self.memory_system.get_memory_stats(),
            'breath_state': self.breath_detector.get_current_state(),
            'pattern_memory_size': len(self.pattern_memory)
        } 
 