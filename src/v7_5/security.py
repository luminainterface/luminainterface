#!/usr/bin/env python3
"""
LUMINA v7.5 Security Module
Handles access control, encryption, and security features
"""

import os
import json
import logging
import hashlib
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "security.log"))
    ]
)
logger = logging.getLogger("Security")

class SecurityManager:
    """Manages security features for LUMINA v7.5"""
    
    def __init__(self):
        self._permissions: Dict[str, Dict[str, bool]] = {}
        self._encryption_key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None
        self._load_security_config()
        
    def _load_security_config(self):
        """Load security configuration"""
        try:
            config_path = os.path.join("data", "security_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self._permissions = config.get("permissions", {})
                    self._load_encryption_key(config.get("encryption_key"))
            else:
                self._initialize_default_config()
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            self._initialize_default_config()
            
    def _initialize_default_config(self):
        """Initialize default security configuration"""
        self._permissions = {
            "v7.5": {
                "read": True,
                "write": True,
                "execute": True
            },
            "v7.0": {
                "read": True,
                "write": False,
                "execute": False
            },
            "v6.0": {
                "read": True,
                "write": False,
                "execute": False
            },
            "v5.0": {
                "read": True,
                "write": False,
                "execute": False
            }
        }
        self._generate_encryption_key()
        self._save_security_config()
        
    def _generate_encryption_key(self):
        """Generate encryption key"""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        self._encryption_key = kdf.derive(os.urandom(32))
        self._fernet = Fernet(Fernet.generate_key())
        
    def _save_security_config(self):
        """Save security configuration"""
        try:
            config = {
                "permissions": self._permissions,
                "encryption_key": self._encryption_key.hex() if self._encryption_key else None
            }
            os.makedirs("data", exist_ok=True)
            with open(os.path.join("data", "security_config.json"), "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save security config: {e}")
            
    def check_permission(self, version: str, permission: str) -> bool:
        """Check if a version has a specific permission"""
        return self._permissions.get(version, {}).get(permission, False)
        
    def encrypt_message(self, message: str) -> bytes:
        """Encrypt a message"""
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        return self._fernet.encrypt(message.encode())
        
    def decrypt_message(self, encrypted_message: bytes) -> str:
        """Decrypt a message"""
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        return self._fernet.decrypt(encrypted_message).decode()
        
    def hash_data(self, data: str) -> str:
        """Hash data for integrity checking"""
        return hashlib.sha256(data.encode()).hexdigest()
        
    def verify_hash(self, data: str, hash_value: str) -> bool:
        """Verify data integrity using hash"""
        return self.hash_data(data) == hash_value
        
    def audit_log(self, action: str, version: str, details: Dict):
        """Log security-related actions"""
        log_entry = {
            "timestamp": logging.Formatter.formatTime(logging.Formatter(), logging.Formatter().converter()),
            "action": action,
            "version": version,
            "details": details
        }
        logger.info(f"Security audit: {json.dumps(log_entry)}")
        
    def update_permissions(self, version: str, permissions: Dict[str, bool]):
        """Update permissions for a version"""
        self._permissions[version] = permissions
        self._save_security_config()
        self.audit_log("permissions_update", version, {"new_permissions": permissions})
        
    def rotate_encryption_key(self):
        """Rotate the encryption key"""
        self._generate_encryption_key()
        self._save_security_config()
        self.audit_log("key_rotation", "system", {}) 