from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None

    def get_text_for_embedding(self) -> str:
        """Generate text for embedding from node properties"""
        text_parts = [
            self.type,
            *[str(v) for v in self.properties.values()],
            *[str(v) for v in self.metadata.values()]
        ]
        return " ".join(text_parts)

class Edge(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class GraphResponse(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchOperation(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

class TraversalConfig(BaseModel):
    max_depth: int = Field(default=3, ge=1, le=10)
    node_types: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    direction: str = Field(default="both", pattern="^(both|incoming|outgoing)$")
    include_metadata: bool = Field(default=True)

class BulkImportRequest(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    source: str = "bulk_import"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BulkExportRequest(BaseModel):
    node_types: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    include_metadata: bool = True
    format: str = Field(default="json", pattern="^(json|csv)$")

class ConceptSyncRequest(BaseModel):
    concept_ids: List[str]
    sync_type: str = Field(default="full", pattern="^(full|incremental)$")
    include_relationships: bool = True

class TrainingDataRequest(BaseModel):
    node_types: Optional[List[str]] = None
    min_relationships: int = 1
    max_samples: int = 1000
    include_metadata: bool = True

class ChatRequest(BaseModel):
    query: str
    context_nodes: Optional[List[str]] = None
    include_graph_context: bool = True
    max_context_nodes: int = 5
    chat_history: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    response: str
    relevant_nodes: List[Node]
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LearningPathOptimizationRequest(BaseModel):
    start_concept: str
    target_concept: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    optimization_criteria: List[str] = Field(default_factory=list)
    max_path_length: int = 10
    include_metadata: bool = True

class LearningPathOptimizationResponse(BaseModel):
    path: List[Dict[str, Any]]
    total_cost: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    alternatives: Optional[List[List[Dict[str, Any]]]] = None

class ConceptAnalysisRequest(BaseModel):
    concept_id: str
    analysis_type: str = Field(default="full", pattern="^(full|basic|relationships|similarity)$")
    include_metadata: bool = True
    max_similar_concepts: int = 5

class ConceptAnalysisResponse(BaseModel):
    concept: Node
    relationships: Dict[str, List[Edge]]
    similar_concepts: List[Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecommendationRequest(BaseModel):
    user_id: str
    context_nodes: Optional[List[str]] = None
    recommendation_type: str = Field(default="concept", pattern="^(concept|path|resource)$")
    constraints: Dict[str, Any] = Field(default_factory=dict)
    max_recommendations: int = 10
    include_metadata: bool = True

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    confidence_scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None

class ClusteringRequest(BaseModel):
    node_types: Optional[List[str]] = None
    min_cluster_size: int = 3
    max_clusters: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True

class ClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphAnalyticsRequest(BaseModel):
    analysis_type: str = Field(default="full", pattern="^(full|centrality|communities|connectivity)$")
    node_types: Optional[List[str]] = None
    include_metadata: bool = True
    max_depth: int = 5

class GraphAnalyticsResponse(BaseModel):
    metrics: Dict[str, Any]
    communities: Optional[List[Dict[str, Any]]] = None
    central_nodes: Optional[List[Dict[str, Any]]] = None
    connectivity: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CrawlerRequest(BaseModel):
    url: str
    depth: int = 1
    extract_concepts: bool = True

class ConceptUpdate(BaseModel):
    concept_id: str
    properties: Dict[str, Any]
    metadata: Dict[str, Any]
    source: str = "graph_api"

class LearningPathRequest(BaseModel):
    start_concept: str
    target_concept: str
    constraints: Optional[Dict[str, Any]] = None 