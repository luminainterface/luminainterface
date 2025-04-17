"""
Spiderweb V3 State Manager
Handles temporal states and state transitions with energy tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple
import json
import math
from datetime import datetime
from .spiderweb_db import SpiderwebDBV3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, db: SpiderwebDBV3):
        """Initialize the state manager."""
        self.db = db
        self.energy_threshold = 0.1  # Minimum energy for state transition
        self.max_transition_probability = 0.95  # Maximum allowed transition probability

    def create_state(self, node_id: str, state_type: str, 
                    state_data: Dict) -> Optional[int]:
        """Create a new temporal state for a node."""
        try:
            # Validate state data
            if not self._validate_state_data(state_type, state_data):
                logger.error(f"Invalid state data for type {state_type}")
                return None

            # Store the state
            state_id = self.db.store_temporal_state(node_id, state_type, state_data)
            
            # Record metrics
            self.db.record_optimization_metric(
                metric_type="state_creation",
                value=1.0,
                context=state_type,
                node_id=node_id
            )
            
            return state_id

        except Exception as e:
            logger.error(f"Error creating state: {e}")
            return None

    def _validate_state_data(self, state_type: str, state_data: Dict) -> bool:
        """Validate state data based on type."""
        try:
            required_fields = {
                'quantum': ['quantum_state', 'phase'],
                'cosmic': ['cosmic_state', 'dimensional_signature'],
                'temporal': ['temporal_state', 'timestamp'],
                'default': ['state_value']
            }

            fields = required_fields.get(state_type, required_fields['default'])
            return all(field in state_data for field in fields)

        except Exception as e:
            logger.error(f"Error validating state data: {e}")
            return False

    def transition_state(self, source_id: int, target_state_data: Dict,
                        transition_type: str) -> Optional[int]:
        """Create a state transition with energy calculations."""
        try:
            # Get source state data
            source_state = self._get_state_by_id(source_id)
            if not source_state:
                logger.error(f"Source state {source_id} not found")
                return None

            # Calculate transition metrics
            probability, energy_delta = self._calculate_transition_metrics(
                source_state['state_data'],
                target_state_data,
                transition_type
            )

            # Check if transition is possible
            if energy_delta < self.energy_threshold:
                logger.warning(f"Insufficient energy for transition: {energy_delta}")
                return None

            # Create target state
            target_id = self.create_state(
                source_state['node_id'],
                source_state['state_type'],
                target_state_data
            )
            if not target_id:
                return None

            # Record the transition
            transition_id = self.db.record_state_transition(
                source_id,
                target_id,
                transition_type,
                probability,
                energy_delta
            )

            # Record metrics
            self.db.record_optimization_metric(
                metric_type="state_transition",
                value=energy_delta,
                context=transition_type,
                node_id=source_state['node_id']
            )

            return transition_id

        except Exception as e:
            logger.error(f"Error in state transition: {e}")
            return None

    def _get_state_by_id(self, state_id: int) -> Optional[Dict]:
        """Retrieve state data by ID."""
        try:
            self.db.cursor.execute("""
                SELECT * FROM temporal_states WHERE id = ?
            """, (state_id,))
            
            result = self.db.cursor.fetchone()
            if not result:
                return None

            columns = [description[0] for description in self.db.cursor.description]
            state_dict = dict(zip(columns, result))
            state_dict['state_data'] = json.loads(state_dict['state_data'])
            
            return state_dict

        except Exception as e:
            logger.error(f"Error retrieving state: {e}")
            return None

    def _calculate_transition_metrics(self, source_data: Dict, 
                                   target_data: Dict,
                                   transition_type: str) -> Tuple[float, float]:
        """Calculate transition probability and energy delta."""
        try:
            probability = 0.0
            energy_delta = 0.0

            if transition_type == 'quantum':
                probability, energy_delta = self._quantum_transition_metrics(
                    source_data, target_data
                )
            elif transition_type == 'cosmic':
                probability, energy_delta = self._cosmic_transition_metrics(
                    source_data, target_data
                )
            else:
                probability, energy_delta = self._default_transition_metrics(
                    source_data, target_data
                )

            # Ensure probability is within bounds
            probability = min(probability, self.max_transition_probability)
            
            return probability, energy_delta

        except Exception as e:
            logger.error(f"Error calculating transition metrics: {e}")
            return 0.0, 0.0

    def _quantum_transition_metrics(self, source_data: Dict,
                                 target_data: Dict) -> Tuple[float, float]:
        """Calculate quantum-specific transition metrics."""
        try:
            # Extract quantum states
            source_state = source_data.get('quantum_state', [])
            target_state = target_data.get('quantum_state', [])
            
            if not source_state or not target_state:
                return 0.0, 0.0

            # Calculate probability based on state overlap
            overlap = sum(s * t for s, t in zip(source_state, target_state))
            probability = abs(overlap) ** 2

            # Calculate energy change
            source_energy = sum(abs(s) ** 2 for s in source_state)
            target_energy = sum(abs(t) ** 2 for t in target_state)
            energy_delta = abs(target_energy - source_energy)

            return probability, energy_delta

        except Exception as e:
            logger.error(f"Error in quantum transition calculation: {e}")
            return 0.0, 0.0

    def _cosmic_transition_metrics(self, source_data: Dict,
                                target_data: Dict) -> Tuple[float, float]:
        """Calculate cosmic-specific transition metrics."""
        try:
            # Extract dimensional signatures
            source_sig = source_data.get('dimensional_signature', '')
            target_sig = target_data.get('dimensional_signature', '')
            
            if not source_sig or not target_sig:
                return 0.0, 0.0

            # Calculate similarity score
            similarity = self._calculate_signature_similarity(source_sig, target_sig)
            probability = similarity

            # Calculate energy based on dimensional complexity
            source_energy = len(source_sig.split(','))
            target_energy = len(target_sig.split(','))
            energy_delta = abs(target_energy - source_energy)

            return probability, energy_delta

        except Exception as e:
            logger.error(f"Error in cosmic transition calculation: {e}")
            return 0.0, 0.0

    def _default_transition_metrics(self, source_data: Dict,
                                 target_data: Dict) -> Tuple[float, float]:
        """Calculate default transition metrics."""
        try:
            # Extract state values
            source_val = float(source_data.get('state_value', 0))
            target_val = float(target_data.get('state_value', 0))

            # Calculate simple probability based on value difference
            diff = abs(target_val - source_val)
            probability = 1.0 / (1.0 + diff)

            # Energy delta is the absolute difference
            energy_delta = diff

            return probability, energy_delta

        except Exception as e:
            logger.error(f"Error in default transition calculation: {e}")
            return 0.0, 0.0

    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between dimensional signatures."""
        try:
            # Split signatures into components
            components1 = set(sig1.split(','))
            components2 = set(sig2.split(','))

            # Calculate Jaccard similarity
            intersection = len(components1.intersection(components2))
            union = len(components1.union(components2))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating signature similarity: {e}")
            return 0.0

    def get_state_chain(self, node_id: str, limit: int = 10) -> List[Dict]:
        """Get the chain of states for a node."""
        try:
            states = self.db.get_state_history(node_id, limit)
            
            # Enrich with transition information
            for i in range(len(states) - 1):
                self.db.cursor.execute("""
                    SELECT transition_type, probability, energy_delta
                    FROM state_transitions
                    WHERE source_state_id = ? AND target_state_id = ?
                """, (states[i+1]['id'], states[i]['id']))
                
                transition = self.db.cursor.fetchone()
                if transition:
                    states[i]['transition_from'] = {
                        'type': transition[0],
                        'probability': transition[1],
                        'energy_delta': transition[2]
                    }
            
            return states

        except Exception as e:
            logger.error(f"Error retrieving state chain: {e}")
            return [] 