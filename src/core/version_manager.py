"""
Version manager for the Lumina Frontend System.
Handles version compatibility, upgrades, and feature availability.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal

@dataclass
class VersionInfo:
    """Information about a specific version."""
    version: str
    features: List[str]
    compatible_versions: List[str]
    required_components: List[str]

class VersionManager(QObject):
    """Manages version compatibility and feature availability."""
    
    # Signals
    version_changed = Signal(str)
    compatibility_check = Signal(bool)
    feature_available = Signal(str, bool)
    
    def __init__(self):
        super().__init__()
        self._current_version = "v7.5"
        self._versions: Dict[str, VersionInfo] = {}
        self._initialize_versions()
    
    def _initialize_versions(self) -> None:
        """Initialize version information."""
        self._versions = {
            "v5": VersionInfo(
                version="v5",
                features=["basic_visualization", "neural_metrics"],
                compatible_versions=["v4", "v6"],
                required_components=["visualization", "metrics"]
            ),
            "v6": VersionInfo(
                version="v6",
                features=["spatial_interface", "3d_visualization"],
                compatible_versions=["v5", "v7"],
                required_components=["visualization", "spatial"]
            ),
            "v7": VersionInfo(
                version="v7",
                features=["advanced_visualization", "quantum_effects"],
                compatible_versions=["v6", "v8"],
                required_components=["visualization", "quantum"]
            ),
            "v7.5": VersionInfo(
                version="v7.5",
                features=["unified_interface", "enhanced_metrics"],
                compatible_versions=["v7", "v8"],
                required_components=["visualization", "metrics", "quantum"]
            )
        }
    
    def get_current_version(self) -> str:
        """Get the current version."""
        return self._current_version
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        return self._versions.get(version)
    
    def is_compatible(self, version1: str, version2: str) -> bool:
        """Check if two versions are compatible."""
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            return False
            
        return version2 in info1.compatible_versions and version1 in info2.compatible_versions
    
    def has_feature(self, version: str, feature: str) -> bool:
        """Check if a version has a specific feature."""
        info = self.get_version_info(version)
        return info is not None and feature in info.features
    
    def get_required_components(self, version: str) -> List[str]:
        """Get required components for a version."""
        info = self.get_version_info(version)
        return info.required_components if info else []
    
    def set_version(self, version: str) -> bool:
        """Set the current version."""
        if version not in self._versions:
            return False
            
        self._current_version = version
        self.version_changed.emit(version)
        return True 
Version manager for the Lumina Frontend System.
Handles version compatibility, upgrades, and feature availability.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal

@dataclass
class VersionInfo:
    """Information about a specific version."""
    version: str
    features: List[str]
    compatible_versions: List[str]
    required_components: List[str]

class VersionManager(QObject):
    """Manages version compatibility and feature availability."""
    
    # Signals
    version_changed = Signal(str)
    compatibility_check = Signal(bool)
    feature_available = Signal(str, bool)
    
    def __init__(self):
        super().__init__()
        self._current_version = "v7.5"
        self._versions: Dict[str, VersionInfo] = {}
        self._initialize_versions()
    
    def _initialize_versions(self) -> None:
        """Initialize version information."""
        self._versions = {
            "v5": VersionInfo(
                version="v5",
                features=["basic_visualization", "neural_metrics"],
                compatible_versions=["v4", "v6"],
                required_components=["visualization", "metrics"]
            ),
            "v6": VersionInfo(
                version="v6",
                features=["spatial_interface", "3d_visualization"],
                compatible_versions=["v5", "v7"],
                required_components=["visualization", "spatial"]
            ),
            "v7": VersionInfo(
                version="v7",
                features=["advanced_visualization", "quantum_effects"],
                compatible_versions=["v6", "v8"],
                required_components=["visualization", "quantum"]
            ),
            "v7.5": VersionInfo(
                version="v7.5",
                features=["unified_interface", "enhanced_metrics"],
                compatible_versions=["v7", "v8"],
                required_components=["visualization", "metrics", "quantum"]
            )
        }
    
    def get_current_version(self) -> str:
        """Get the current version."""
        return self._current_version
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        return self._versions.get(version)
    
    def is_compatible(self, version1: str, version2: str) -> bool:
        """Check if two versions are compatible."""
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            return False
            
        return version2 in info1.compatible_versions and version1 in info2.compatible_versions
    
    def has_feature(self, version: str, feature: str) -> bool:
        """Check if a version has a specific feature."""
        info = self.get_version_info(version)
        return info is not None and feature in info.features
    
    def get_required_components(self, version: str) -> List[str]:
        """Get required components for a version."""
        info = self.get_version_info(version)
        return info.required_components if info else []
    
    def set_version(self, version: str) -> bool:
        """Set the current version."""
        if version not in self._versions:
            return False
            
        self._current_version = version
        self.version_changed.emit(version)
        return True 
 