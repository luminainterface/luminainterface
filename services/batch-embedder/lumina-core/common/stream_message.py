from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime

class StreamMessage(BaseModel):
    """Base class for stream messages with common fields."""
    id: str = Field(..., description="Unique message ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    type: str = Field(..., description="Message type/event name")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source: str = Field(..., description="Source service name")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 