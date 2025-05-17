from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class Concept(BaseModel):
    term: str
    definition: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    last_updated: Optional[str] = None
    license_type: Optional[str] = None
    usage_count: Optional[int] = 0

    def to_dict(self) -> Dict:
        """Convert concept to dictionary format."""
        return {
            "term": self.term,
            "definition": self.definition,
            "embedding": self.embedding,
            "metadata": self.metadata or {},
            "last_updated": self.last_updated or datetime.utcnow().isoformat(),
            "license_type": self.license_type,
            "usage_count": self.usage_count or 0
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Concept':
        """Create a Concept instance from a dictionary."""
        return cls(**data) 