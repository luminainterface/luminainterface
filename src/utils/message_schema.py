"""
Standard Message Format Schema for Lumina Neural Network Project

This module defines the standard message format and validation for
communication between different components of the system.
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

class MessageType(Enum):
    """Types of messages in the system"""
    MEMORY_QUERY = "memory_query"
    MEMORY_RESPONSE = "memory_response"
    TOPIC_QUERY = "topic_query"
    TOPIC_RESPONSE = "topic_response"
    ASSOCIATION_QUERY = "association_query"
    ASSOCIATION_RESPONSE = "association_response"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"

class MessageSchema:
    """Base message schema with validation"""
    
    def __init__(self):
        self.schema = {
            "type": "object",
            "required": ["message_id", "timestamp", "type", "version"],
            "properties": {
                "message_id": {"type": "string", "format": "uuid"},
                "timestamp": {"type": "string", "format": "date-time"},
                "type": {"type": "string", "enum": [t.value for t in MessageType]},
                "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "metadata": {"type": "object"},
                "data": {"type": "object"}
            }
        }
    
    def create_message(self, 
                      message_type: MessageType,
                      data: Dict[str, Any],
                      source: str,
                      destination: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new message with the given parameters"""
        message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "type": message_type.value,
            "version": "1.0.0",
            "source": source,
            "data": data
        }
        
        if destination:
            message["destination"] = destination
        
        if metadata:
            message["metadata"] = metadata
        
        return message
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate a message against the schema"""
        try:
            # Check required fields
            for field in self.schema["required"]:
                if field not in message:
                    return False
            
            # Validate message type
            if message["type"] not in [t.value for t in MessageType]:
                return False
            
            # Validate version format
            import re
            if not re.match(self.schema["properties"]["version"]["pattern"], message["version"]):
                return False
            
            return True
            
        except Exception:
            return False
    
    def to_json(self, message: Dict[str, Any]) -> str:
        """Convert message to JSON string"""
        return json.dumps(message)
    
    def from_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Parse JSON string to message"""
        try:
            message = json.loads(json_str)
            if self.validate_message(message):
                return message
            return None
        except Exception:
            return None

# Create global instance
message_schema = MessageSchema()

def create_message(message_type: MessageType,
                 data: Dict[str, Any],
                 source: str,
                 destination: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new message using the global schema"""
    return message_schema.create_message(
        message_type=message_type,
        data=data,
        source=source,
        destination=destination,
        metadata=metadata
    )

def validate_message(message: Dict[str, Any]) -> bool:
    """Validate a message using the global schema"""
    return message_schema.validate_message(message) 