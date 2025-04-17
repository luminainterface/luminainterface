#!/usr/bin/env python3
"""
Security Tests for LUMINA v7.5
"""

import unittest
import os
import json
from src.v7_5.security import SecurityManager

class TestSecurity(unittest.TestCase):
    """Test cases for security features"""
    
    def setUp(self):
        self.security = SecurityManager()
        
    def test_permission_checks(self):
        """Test permission checking"""
        # Test v7.5 permissions
        self.assertTrue(self.security.check_permission("v7.5", "read"))
        self.assertTrue(self.security.check_permission("v7.5", "write"))
        self.assertTrue(self.security.check_permission("v7.5", "execute"))
        
        # Test v7.0 permissions
        self.assertTrue(self.security.check_permission("v7.0", "read"))
        self.assertFalse(self.security.check_permission("v7.0", "write"))
        self.assertFalse(self.security.check_permission("v7.0", "execute"))
        
    def test_encryption(self):
        """Test message encryption and decryption"""
        test_message = "Test message for encryption"
        
        # Encrypt message
        encrypted = self.security.encrypt_message(test_message)
        self.assertIsNotNone(encrypted)
        self.assertNotEqual(encrypted, test_message.encode())
        
        # Decrypt message
        decrypted = self.security.decrypt_message(encrypted)
        self.assertEqual(decrypted, test_message)
        
    def test_data_integrity(self):
        """Test data integrity checking"""
        test_data = "Test data for integrity check"
        
        # Generate hash
        hash_value = self.security.hash_data(test_data)
        self.assertIsNotNone(hash_value)
        
        # Verify hash
        self.assertTrue(self.security.verify_hash(test_data, hash_value))
        
        # Test with modified data
        modified_data = "Modified test data"
        self.assertFalse(self.security.verify_hash(modified_data, hash_value))
        
    def test_permission_updates(self):
        """Test permission updates"""
        # Update permissions for v7.0
        new_permissions = {
            "read": True,
            "write": True,
            "execute": False
        }
        self.security.update_permissions("v7.0", new_permissions)
        
        # Verify updated permissions
        self.assertTrue(self.security.check_permission("v7.0", "read"))
        self.assertTrue(self.security.check_permission("v7.0", "write"))
        self.assertFalse(self.security.check_permission("v7.0", "execute"))
        
    def test_key_rotation(self):
        """Test encryption key rotation"""
        # Store current key
        old_key = self.security._encryption_key
        
        # Rotate key
        self.security.rotate_encryption_key()
        
        # Verify new key
        self.assertNotEqual(self.security._encryption_key, old_key)
        
    def test_config_persistence(self):
        """Test security configuration persistence"""
        # Update configuration
        self.security.update_permissions("v6.0", {
            "read": True,
            "write": True,
            "execute": True
        })
        
        # Create new instance to load saved config
        new_security = SecurityManager()
        
        # Verify loaded configuration
        self.assertTrue(new_security.check_permission("v6.0", "read"))
        self.assertTrue(new_security.check_permission("v6.0", "write"))
        self.assertTrue(new_security.check_permission("v6.0", "execute"))
        
    def test_audit_logging(self):
        """Test audit logging"""
        # Perform an action that should be logged
        self.security.update_permissions("v5.0", {
            "read": True,
            "write": False,
            "execute": False
        })
        
        # Verify log file exists and contains the action
        log_path = os.path.join("logs", "security.log")
        self.assertTrue(os.path.exists(log_path))
        
        with open(log_path, "r") as f:
            log_content = f.read()
            self.assertIn("permissions_update", log_content)
            self.assertIn("v5.0", log_content)

if __name__ == '__main__':
    unittest.main() 