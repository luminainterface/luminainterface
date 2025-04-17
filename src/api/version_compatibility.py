"""
API Version Compatibility Framework

This module provides a standardized approach to API versioning and compatibility
checking across the Lumina Neural Network System. It implements semantic versioning
with utilities for version migration and compatibility validation.
"""

import re
import logging
import json
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from enum import Enum
from pathlib import Path
import functools

logger = logging.getLogger(__name__)

# Version format: MAJOR.MINOR.PATCH
VERSION_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')

class VersionType(Enum):
    """Types of version changes"""
    MAJOR = "MAJOR"  # Breaking changes
    MINOR = "MINOR"  # New features, backwards compatible
    PATCH = "PATCH"  # Bugfixes, backwards compatible


class VersionedAPI:
    """Base class for versioned APIs"""
    
    def __init__(self, current_version: str = "1.0.0"):
        """
        Initialize a versioned API.
        
        Args:
            current_version: The current version of the API
        """
        self.current_version = current_version
        self._validate_version(current_version)
        self.version_history = []
        self.version_migrations = {}
        self.deprecated_endpoints = {}
        
    def _validate_version(self, version: str) -> Tuple[int, int, int]:
        """
        Validate that a version string follows semantic versioning format.
        
        Args:
            version: Version string to validate
            
        Returns:
            Tuple of (major, minor, patch) version components
            
        Raises:
            ValueError: If version doesn't match semantic versioning format
        """
        match = VERSION_PATTERN.match(version)
        if not match:
            raise ValueError(f"Invalid version format: {version}. Must be MAJOR.MINOR.PATCH")
        
        major, minor, patch = map(int, match.groups())
        return major, minor, patch
        
    def register_version(self, version: str, 
                         changes: List[Dict[str, Any]],
                         release_date: str) -> None:
        """
        Register a version in the version history.
        
        Args:
            version: Version string
            changes: List of changes introduced in this version
            release_date: Date of release (YYYY-MM-DD)
        """
        self._validate_version(version)
        
        self.version_history.append({
            "version": version,
            "release_date": release_date,
            "changes": changes
        })
        
        logger.info(f"Registered API version {version} released on {release_date}")
    
    def register_migration(self, from_version: str, to_version: str, 
                          migration_fn: Callable) -> None:
        """
        Register a migration function between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            migration_fn: Function that handles the migration
        """
        self._validate_version(from_version)
        self._validate_version(to_version)
        
        key = f"{from_version}_to_{to_version}"
        self.version_migrations[key] = migration_fn
        
        logger.info(f"Registered migration from {from_version} to {to_version}")
    
    def deprecate_endpoint(self, endpoint: str, 
                          deprecated_in: str,
                          removed_in: Optional[str] = None,
                          alternative: Optional[str] = None) -> None:
        """
        Mark an endpoint as deprecated.
        
        Args:
            endpoint: The endpoint path
            deprecated_in: Version where deprecation started
            removed_in: Version where endpoint will be removed
            alternative: Alternative endpoint to use
        """
        self._validate_version(deprecated_in)
        if removed_in:
            self._validate_version(removed_in)
            
        self.deprecated_endpoints[endpoint] = {
            "deprecated_in": deprecated_in,
            "removed_in": removed_in,
            "alternative": alternative
        }
        
        logger.info(f"Marked endpoint {endpoint} as deprecated in version {deprecated_in}")
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        v1_parts = self._validate_version(version1)
        v2_parts = self._validate_version(version2)
        
        for i in range(3):
            if v1_parts[i] < v2_parts[i]:
                return -1
            if v1_parts[i] > v2_parts[i]:
                return 1
                
        return 0
    
    def is_compatible(self, client_version: str) -> bool:
        """
        Check if a client version is compatible with the current API version.
        
        Args:
            client_version: The client version to check
            
        Returns:
            Boolean indicating compatibility
        """
        current_major = self._validate_version(self.current_version)[0]
        client_major = self._validate_version(client_version)[0]
        
        # Major version must match for compatibility
        return current_major == client_major
    
    def get_migration_path(self, from_version: str, 
                          to_version: str) -> List[Tuple[str, str, Callable]]:
        """
        Get a series of migrations to move from one version to another.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of (from_version, to_version, migration_function) tuples
        """
        if self.compare_versions(from_version, to_version) == 0:
            return []  # No migration needed
            
        # Check if direct migration exists
        key = f"{from_version}_to_{to_version}"
        if key in self.version_migrations:
            return [(from_version, to_version, self.version_migrations[key])]
            
        # Look for indirect migrations
        result = []
        current = from_version
        
        while self.compare_versions(current, to_version) != 0:
            next_step = None
            
            for key, function in self.version_migrations.items():
                source, target = key.split("_to_")
                
                if source == current and self.compare_versions(target, to_version) <= 0:
                    if next_step is None or self.compare_versions(target, next_step[1]) > 0:
                        next_step = (source, target, function)
            
            if next_step is None:
                raise ValueError(f"No migration path from {current} to {to_version}")
                
            result.append(next_step)
            current = next_step[1]
            
        return result
        
    def migrate_data(self, data: Any, from_version: str, 
                    to_version: str) -> Any:
        """
        Migrate data from one API version to another.
        
        Args:
            data: The data to migrate
            from_version: Source version
            to_version: Target version
            
        Returns:
            The migrated data
        """
        if self.compare_versions(from_version, to_version) == 0:
            return data  # No migration needed
            
        migration_path = self.get_migration_path(from_version, to_version)
        
        result = data
        for step_from, step_to, migration_fn in migration_path:
            result = migration_fn(result)
            logger.debug(f"Migrated data from {step_from} to {step_to}")
            
        return result
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        Get information about the API version.
        
        Returns:
            Dictionary with version information
        """
        return {
            "current_version": self.current_version,
            "version_history": self.version_history,
            "deprecated_endpoints": self.deprecated_endpoints
        }


def version_endpoint(min_version: str = None, max_version: str = None):
    """
    Decorator for versioned API endpoints.
    
    Args:
        min_version: Minimum supported version
        max_version: Maximum supported version
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            api_version = getattr(self, 'api_version', None)
            
            if api_version is None:
                logger.warning(f"No API version found when calling {func.__name__}")
                return func(self, *args, **kwargs)
                
            # Check version constraints
            if min_version and self.compare_versions(api_version, min_version) < 0:
                raise ValueError(f"API version {api_version} is below minimum {min_version}")
                
            if max_version and self.compare_versions(api_version, max_version) > 0:
                raise ValueError(f"API version {api_version} exceeds maximum {max_version}")
                
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# Global version registry
_version_registry = {}

def register_api(api_id: str, api: VersionedAPI) -> None:
    """
    Register a versioned API in the global registry.
    
    Args:
        api_id: Unique identifier for the API
        api: VersionedAPI instance
    """
    _version_registry[api_id] = api
    logger.info(f"Registered API '{api_id}' at version {api.current_version}")

def get_api(api_id: str) -> Optional[VersionedAPI]:
    """
    Get a registered API by ID.
    
    Args:
        api_id: The API identifier
        
    Returns:
        VersionedAPI instance or None if not found
    """
    return _version_registry.get(api_id)

def get_apis() -> Dict[str, VersionedAPI]:
    """
    Get all registered APIs.
    
    Returns:
        Dictionary of API_ID -> VersionedAPI
    """
    return _version_registry.copy() 