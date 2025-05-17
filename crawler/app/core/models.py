"""Models for the crawler service."""
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import time
import json

class CrawlItem(BaseModel):
    """Model for crawl items."""
    source_type: str = Field(..., description="Type of source (git, pdf, url, graph)")
    source: str = Field(..., description="URL, file path, or graph node ID")
    priority: float = Field(default=0.5, description="Priority of the crawl item")
    metadata: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="Additional metadata")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of creation")
    queued_at: float = Field(default_factory=time.time, description="When the item was queued")
    attempts: int = Field(default=0, description="Number of processing attempts")

    @validator('metadata', pre=True)
    def parse_metadata(cls, v):
        """Parse metadata from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v

class CrawlResult(BaseModel):
    """Model for crawl results."""
    item_id: str = Field(..., description="Unique identifier for the crawl result")
    source_type: str = Field(..., description="Type of source that was crawled")
    content: Any = Field(..., description="Crawled content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Content embedding")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of processing") 