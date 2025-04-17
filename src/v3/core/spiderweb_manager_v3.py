"""
Spiderweb V3 Manager
Main interface for the V3 Spiderweb system, integrating database, cache, and state management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from .spiderweb_db import SpiderwebDBV3
from .cache_manager import CacheManager
from .state_manager import StateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiderwebManagerV3:
    def __init__(self, db_path: str = "spiderweb_v3.db", max_cache_mb: int = 100):
        """Initialize the Spiderweb V3 manager."""
        try:
            # Initialize components
            self.db = SpiderwebDBV3(db_path)
            self.cache_manager = CacheManager(self.db, max_cache_mb)
            self.state_manager = StateManager(self.db)
            
            logger.info("Spiderweb V3 manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Spiderweb V3 manager: {e}")
            raise

    def create_node_state(self, node_id: str, state_type: str,
                         state_data: Dict) -> Optional[int]:
        """Create a new state for a node with caching."""
        try:
            # Check cache first
            cache_key = f"state_{node_id}_{state_type}_{hash(str(state_data))}"
            cached_state = self.cache_manager.get(cache_key)
            
            if cached_state:
                logger.info(f"State found in cache for node {node_id}")
                return cached_state.get('id')

            # Create new state
            state_id = self.state_manager.create_state(node_id, state_type, state_data)
            
            if state_id:
                # Cache the result
                self.cache_manager.store(
                    cache_key,
                    {'id': state_id, 'state_data': state_data},
                    priority=1
                )
            
            return state_id

        except Exception as e:
            logger.error(f"Error creating node state: {e}")
            return None

    def transition_node_state(self, node_id: str, source_state_id: int,
                            target_state_data: Dict,
                            transition_type: str) -> Optional[int]:
        """Perform a state transition for a node."""
        try:
            # Attempt the transition
            transition_id = self.state_manager.transition_state(
                source_state_id,
                target_state_data,
                transition_type
            )
            
            if transition_id:
                # Cache the transition result
                cache_key = f"transition_{node_id}_{source_state_id}_{transition_type}"
                self.cache_manager.store(
                    cache_key,
                    {
                        'transition_id': transition_id,
                        'source_id': source_state_id,
                        'target_data': target_state_data,
                        'type': transition_type
                    },
                    priority=2
                )
            
            return transition_id

        except Exception as e:
            logger.error(f"Error in node state transition: {e}")
            return None

    def get_node_state_history(self, node_id: str, limit: int = 10) -> List[Dict]:
        """Get the state history for a node with caching."""
        try:
            # Check cache first
            cache_key = f"history_{node_id}_{limit}"
            cached_history = self.cache_manager.get(cache_key)
            
            if cached_history:
                logger.info(f"State history found in cache for node {node_id}")
                return cached_history

            # Get fresh history
            history = self.state_manager.get_state_chain(node_id, limit)
            
            if history:
                # Cache the result with shorter expiry for state history
                self.cache_manager.store(
                    cache_key,
                    history,
                    priority=1,
                    expiry_hours=1
                )
            
            return history

        except Exception as e:
            logger.error(f"Error retrieving node state history: {e}")
            return []

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            metrics = {
                'cache': self.cache_manager.get_metrics(),
                'timestamp': datetime.now().isoformat()
            }

            # Get optimization metrics
            db_metrics = self.db.get_optimization_metrics(limit=1000)
            
            # Process metrics by type
            processed_metrics = {}
            for metric in db_metrics:
                metric_type = metric['metric_type']
                if metric_type not in processed_metrics:
                    processed_metrics[metric_type] = []
                processed_metrics[metric_type].append(metric['value'])

            # Calculate averages for each metric type
            for metric_type, values in processed_metrics.items():
                if values:
                    metrics[f'avg_{metric_type}'] = sum(values) / len(values)

            return metrics

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}

    def cleanup(self):
        """Perform system cleanup operations."""
        try:
            # Cleanup cache
            deleted_count = self.db.cleanup_expired_cache()
            logger.info(f"Cleaned up {deleted_count} expired cache entries")

            # Record cleanup metrics
            self.db.record_optimization_metric(
                metric_type="cleanup_operation",
                value=deleted_count,
                context="cache_cleanup"
            )

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def close(self):
        """Close all system components."""
        try:
            self.cleanup()
            self.db.close()
            logger.info("Spiderweb V3 manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing Spiderweb V3 manager: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 