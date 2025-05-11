from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
qdrant_client = QdrantClient(host='qdrant', port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

class BatchEmbedRequest(BaseModel):
    texts: list[str]
    collection_name: str
    metadata: dict = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/embed_batch")
async def embed_batch(request: BatchEmbedRequest):
    try:
        # Generate embeddings
        embeddings = model.encode(request.texts, batch_size=32)
        
        # Prepare points for Qdrant
        points = []
        for i, (text, embedding) in enumerate(zip(request.texts, embeddings)):
            points.append({
                "id": i,
                "vector": embedding.tolist(),
                "payload": {
                    "text": text,
                    **request.metadata
                }
            })
        
        # Upsert to Qdrant
        qdrant_client.upsert(
            collection_name=request.collection_name,
            points=points
        )
        
        # Publish batch completion event
        event = {
            "type": "batch_embedded",
            "collection": request.collection_name,
            "batch_size": len(points)
        }
        redis_client.publish("lumina.embedding", json.dumps(event))
        
        return {
            "success": True,
            "embedded_count": len(points)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800) 