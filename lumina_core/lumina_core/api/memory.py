"""Memory management module for the API."""

from typing import Dict, Any
import time

class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Any:
        """Get a value from the cache if it exists and hasn't expired."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if entry['expires_at'] and time.time() > entry['expires_at']:
            del self._cache[key]
            return None
            
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set a value in the cache with optional TTL in seconds."""
        expires_at = time.time() + ttl if ttl else None
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
    
    def delete(self, key: str):
        """Remove a key from the cache."""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all entries from the cache."""
        self._cache.clear()

# Global cache instance
cache = Cache() 