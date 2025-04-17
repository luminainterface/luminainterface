#!/usr/bin/env python3
"""
LUMINA v7.5 Message Transformer
Handles message transformation between different versions
"""

import logging
from typing import Dict, Any, Tuple, Optional
from packaging import version

class MessageTransformer:
    """Handles message transformation between different versions"""
    
    def __init__(self):
        """Initialize the message transformer"""
        self.logger = logging.getLogger("MessageTransformer")
        self._transformers: Dict[str, Dict[str, Any]] = {}
        self._setup_transformers()
        
    def _setup_transformers(self):
        """Set up the transformation rules"""
        # V7.5 to V7.0 transformations
        self._transformers["7.5"] = {
            "7.0": {
                "version.status": self._transform_status_v75_to_v70,
                "version.message": self._transform_message_v75_to_v70,
                "version.error": self._transform_error_v75_to_v70
            }
        }
        
        # V7.5 to V6.0 transformations
        self._transformers["7.5"]["6.0"] = {
            "version.status": self._transform_status_v75_to_v60,
            "version.message": self._transform_message_v75_to_v60,
            "version.error": self._transform_error_v75_to_v60
        }
        
        # V7.5 to V5.0 transformations
        self._transformers["7.5"]["5.0"] = {
            "version.status": self._transform_status_v75_to_v50,
            "version.message": self._transform_message_v75_to_v50,
            "version.error": self._transform_error_v75_to_v50
        }
        
    async def transform_message(self, source_version: str, target_version: str, message: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Transform a message from source version to target version"""
        try:
            # Validate versions
            source_v = version.parse(source_version)
            target_v = version.parse(target_version)
            
            # Get major versions
            source_major = str(source_v.major) + "." + str(source_v.minor)
            target_major = str(target_v.major) + "." + str(target_v.minor)
            
            # Check if transformation exists
            if source_major not in self._transformers or target_major not in self._transformers[source_major]:
                return None, f"No transformer available for {source_version} -> {target_version}"
                
            # Get message type
            msg_type = message.get("type")
            if not msg_type:
                return None, "Message type not specified"
                
            # Get transformer for message type
            transformer = self._transformers[source_major][target_major].get(msg_type)
            if not transformer:
                # If no specific transformer, use default pass-through
                return message, None
                
            # Transform message
            transformed = await transformer(message)
            return transformed, None
            
        except Exception as e:
            error = f"Error transforming message: {str(e)}"
            self.logger.error(error)
            return None, error
            
    async def _transform_status_v75_to_v70(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform status message from v7.5 to v7.0"""
        return {
            "type": "version.status",
            "status": message.get("status", {}),
            "version": "7.0",
            "timestamp": message.get("timestamp")
        }
        
    async def _transform_message_v75_to_v70(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform general message from v7.5 to v7.0"""
        return {
            "type": "version.message",
            "content": message.get("content", {}),
            "version": "7.0",
            "timestamp": message.get("timestamp")
        }
        
    async def _transform_error_v75_to_v70(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform error message from v7.5 to v7.0"""
        return {
            "type": "version.error",
            "error": message.get("error", "Unknown error"),
            "details": message.get("details", {}),
            "version": "7.0",
            "timestamp": message.get("timestamp")
        }
        
    async def _transform_status_v75_to_v60(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform status message from v7.5 to v6.0"""
        # Simplify status for v6.0
        status = message.get("status", {})
        simplified = {k: bool(v) for k, v in status.items()}
        return {
            "type": "version.status",
            "status": simplified,
            "version": "6.0"
        }
        
    async def _transform_message_v75_to_v60(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform general message from v7.5 to v6.0"""
        # Simplify content for v6.0
        content = message.get("content", {})
        if isinstance(content, dict):
            content = str(content)
        return {
            "type": "version.message",
            "content": content,
            "version": "6.0"
        }
        
    async def _transform_error_v75_to_v60(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform error message from v7.5 to v6.0"""
        return {
            "type": "version.error",
            "error": str(message.get("error", "Unknown error")),
            "version": "6.0"
        }
        
    async def _transform_status_v75_to_v50(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform status message from v7.5 to v5.0"""
        # Basic status for v5.0
        return {
            "type": "status",
            "ok": True,
            "version": "5.0"
        }
        
    async def _transform_message_v75_to_v50(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform general message from v7.5 to v5.0"""
        # Basic message format for v5.0
        return {
            "type": "message",
            "text": str(message.get("content", "")),
            "version": "5.0"
        }
        
    async def _transform_error_v75_to_v50(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transform error message from v7.5 to v5.0"""
        return {
            "type": "error",
            "text": str(message.get("error", "Unknown error")),
            "version": "5.0"
        } 