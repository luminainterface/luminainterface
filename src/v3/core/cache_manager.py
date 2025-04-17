"""
Spiderweb V3 Cache Manager
Handles advanced caching features with priority-based eviction and monitoring.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import json
from .spiderweb_db import SpiderwebDBV3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, db: SpiderwebDBV3, max_size_mb: int = 100):
        """Initialize the cache manager."""
        self.db = db
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.current_size_bytes = 0
        self._update_current_size()

    def _update_current_size(self):
        """Update the current cache size from the database."""
        try:
            self.db.cursor.execute("""
                SELECT COALESCE(SUM(size_bytes), 0) 
                FROM cache_entries 
                WHERE expiry > datetime('now')
            """)
            self.current_size_bytes = self.db.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error updating cache size: {e}")
            self.current_size_bytes = 0

    def _evict_entries(self, required_bytes: int):
        """Evict cache entries based on priority and access patterns."""
        try:
            while self.current_size_bytes + required_bytes > self.max_size_bytes:
                # First try to remove expired entries
                deleted_count = self.db.cleanup_expired_cache()
                if deleted_count > 0:
                    self._update_current_size()
                    if self.current_size_bytes + required_bytes <= self.max_size_bytes:
                        break

                # If still need space, remove low priority, least accessed entries
                self.db.cursor.execute("""
                    DELETE FROM cache_entries 
                    WHERE id IN (
                        SELECT id FROM cache_entries
                        ORDER BY priority ASC, access_count ASC, last_access ASC
                        LIMIT 10
                    )
                """)
                self.db.conn.commit()
                self._update_current_size()

        except Exception as e:
            logger.error(f"Error during cache eviction: {e}")
            raise

    def store(self, key: str, data: Any, priority: int = 0, 
              expiry_hours: int = 24) -> bool:
        """Store data in cache with priority and expiration."""
        try:
            data_str = json.dumps(data)
            size_bytes = len(data_str.encode())

            # Check if we need to evict entries
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Data size ({size_bytes} bytes) exceeds maximum cache size")
                return False

            self._evict_entries(size_bytes)
            
            # Store the data
            success = self.db.manage_cache(key, data_str, priority, expiry_hours)
            if success:
                self.current_size_bytes += size_bytes
                self._record_metrics("cache_store", size_bytes)
            return success

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache and update access metrics."""
        try:
            self.db.cursor.execute("""
                SELECT data, access_count 
                FROM cache_entries 
                WHERE key_hash = ? AND expiry > datetime('now')
            """, (key,))
            result = self.db.cursor.fetchone()

            if result:
                data_str, access_count = result
                
                # Update access metrics
                self.db.cursor.execute("""
                    UPDATE cache_entries 
                    SET access_count = ?, last_access = datetime('now')
                    WHERE key_hash = ?
                """, (access_count + 1, key))
                self.db.conn.commit()

                self._record_metrics("cache_hit", 1)
                return json.loads(data_str)
            else:
                self._record_metrics("cache_miss", 1)
                return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def _record_metrics(self, metric_type: str, value: float):
        """Record cache-related metrics."""
        try:
            self.db.record_optimization_metric(
                metric_type=metric_type,
                value=value,
                context="cache_operations"
            )
        except Exception as e:
            logger.error(f"Error recording cache metrics: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Get cache performance metrics."""
        try:
            metrics = {
                'current_size_bytes': self.current_size_bytes,
                'utilization_percent': (self.current_size_bytes / self.max_size_bytes) * 100
            }

            # Calculate hit rate
            self.db.cursor.execute("""
                SELECT metric_type, COUNT(*) as count
                FROM optimization_metrics
                WHERE metric_type IN ('cache_hit', 'cache_miss')
                AND timestamp > datetime('now', '-1 hour')
                GROUP BY metric_type
            """)
            
            hits = misses = 0
            for metric_type, count in self.db.cursor.fetchall():
                if metric_type == 'cache_hit':
                    hits = count
                else:
                    misses = count

            total = hits + misses
            metrics['hit_rate'] = (hits / total * 100) if total > 0 else 0
            
            return metrics

        except Exception as e:
            logger.error(f"Error getting cache metrics: {e}")
            return {} 