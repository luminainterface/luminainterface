import json
import asyncio
from redis.asyncio import Redis
import os
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from loguru import logger
from prometheus_client import Counter, Histogram

# Metrics
EMBEDDING_UPDATES = Counter(
    'embedding_updates_total',
    'Total number of embedding updates',
    ['type']
)

MODEL_LATENCY = Histogram(
    'model_update_latency_seconds',
    'Time taken to update model',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return self.classifier(x)

class NNConsumer:
    def __init__(self):
        self.redis = Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8",
            decode_responses=True
        )
        self.stream_name = "graph_stream"
        self.consumer_group = "nn_consumer"
        self.consumer_name = "nn_consumer_1"
        
        # Initialize model
        self.input_dim = 768  # Default embedding size
        self.model = GraphNeuralNetwork(self.input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        # Store graph state
        self.node_embeddings: Dict[str, List[float]] = {}
        self.edge_index: List[List[int]] = []
        self.node_mapping: Dict[str, int] = {}

    async def setup(self):
        """Initialize Redis streams and consumer group."""
        try:
            await self.redis.xgroup_create(
                self.stream_name,
                self.consumer_group,
                mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _get_node_index(self, node_id: str) -> int:
        """Get or create node index."""
        if node_id not in self.node_mapping:
            self.node_mapping[node_id] = len(self.node_mapping)
        return self.node_mapping[node_id]

    async def process_event(self, event: Dict[str, Any]):
        """Process a graph event and update the model."""
        with MODEL_LATENCY.time():
            event_type = event.get('type')
            payload = event.get('payload', {})

            if event_type == 'node.add':
                # Add new node embedding
                if 'embedding' in payload:
                    node_id = payload.get('id')
                    self.node_embeddings[node_id] = payload['embedding']
                    EMBEDDING_UPDATES.labels(type='node').inc()

            elif event_type == 'edge.add':
                # Add new edge
                source = payload.get('source')
                target = payload.get('target')
                if source and target:
                    source_idx = self._get_node_index(source)
                    target_idx = self._get_node_index(target)
                    self.edge_index.append([source_idx, target_idx])
                    EMBEDDING_UPDATES.labels(type='edge').inc()

            # Update model if we have enough data
            if len(self.node_embeddings) > 1 and len(self.edge_index) > 0:
                await self._update_model()

    async def _update_model(self):
        """Update the GNN model with new graph data."""
        try:
            # Convert graph data to PyTorch tensors
            x = torch.tensor([self.node_embeddings[n] for n in self.node_mapping.keys()])
            edge_index = torch.tensor(self.edge_index).t()

            # Create graph data
            data = Data(x=x, edge_index=edge_index)

            # Forward pass
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            
            # Simple loss (can be customized)
            loss = torch.mean(out)
            loss.backward()
            self.optimizer.step()

            logger.info(f"Model updated with {len(self.node_embeddings)} nodes and {len(self.edge_index)} edges")

        except Exception as e:
            logger.error(f"Error updating model: {e}")

    async def start_consumer(self):
        """Start consuming events from Redis stream."""
        while True:
            try:
                # Read new events
                events = await self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: '>'},
                    count=100,
                    block=1000
                )

                for _, messages in events:
                    for msg_id, data in messages:
                        try:
                            event = json.loads(data['event'])
                            await self.process_event(event)
                            # Acknowledge processed message
                            await self.redis.xack(
                                self.stream_name,
                                self.consumer_group,
                                msg_id
                            )
                        except Exception as e:
                            logger.error(f"Error processing event: {e}")

            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

# Create consumer instance
consumer = NNConsumer()

async def main():
    """Main entry point."""
    await consumer.setup()
    await consumer.start_consumer()

if __name__ == "__main__":
    asyncio.run(main()) 