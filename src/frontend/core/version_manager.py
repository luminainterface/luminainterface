from PySide6.QtCore import QObject, Signal
from typing import Dict, List, Optional
import logging

class VersionManager(QObject):
    """Manages version compatibility and switching for the Lumina frontend."""
    
    # Signals
    version_switched = Signal(str)
    compatibility_error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.current_version = "v5"
        self.available_versions = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]
        self.compatibility_matrix = self._build_compatibility_matrix()
        
    def _build_compatibility_matrix(self) -> Dict[str, List[str]]:
        """Build the version compatibility matrix."""
        matrix = {}
        for i, version in enumerate(self.available_versions):
            # Implement "2-version proximity" rule
            compatible = []
            if i > 0:
                compatible.append(self.available_versions[i-1])
            if i < len(self.available_versions) - 1:
                compatible.append(self.available_versions[i+1])
            matrix[version] = compatible
        return matrix
        
    def get_compatible_versions(self, version: str) -> List[str]:
        """Get list of versions compatible with the given version."""
        return self.compatibility_matrix.get(version, [])
        
    def can_switch_to(self, target_version: str) -> bool:
        """Check if we can switch to the target version."""
        return target_version in self.get_compatible_versions(self.current_version)
        
    def switch_version(self, target_version: str) -> bool:
        """Switch to the target version if compatible."""
        try:
            if not self.can_switch_to(target_version):
                error_msg = f"Cannot switch from {self.current_version} to {target_version}"
                self.logger.error(error_msg)
                self.compatibility_error.emit(error_msg)
                return False
                
            self.logger.info(f"Switching from {self.current_version} to {target_version}")
            # Version switching logic here
            self.current_version = target_version
            self.version_switched.emit(target_version)
            return True
            
        except Exception as e:
            self.logger.error(f"Error switching versions: {str(e)}")
            self.compatibility_error.emit(str(e))
            return False
            
    def get_current_version(self) -> str:
        """Get the current version."""
        return self.current_version
        
    def get_available_versions(self) -> List[str]:
        """Get all available versions."""
        return self.available_versions 