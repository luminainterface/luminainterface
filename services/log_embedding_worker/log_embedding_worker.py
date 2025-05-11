from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
qdrant_client = QdrantClient(host='qdrant', port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

class LogEntry(BaseModel):
    id: str
    text: str
    timestamp: str
    source: str = "chat"

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process_logs")
async def process_logs():
    try:
        # Fetch new logs
        response = requests.get("http://localhost:9000/logs")
        if response.status_code != 200:
            return {"processed": 0, "error": "Failed to fetch logs"}
        
        logs = response.json()
        if not logs:
            return {"processed": 0, "message": "No new logs to process"}
        
        # Generate embeddings
        texts = [log["text"] for log in logs]
        embeddings = model.encode(texts, batch_size=32)
        
        # Prepare points for Qdrant
        points = []
        for log, embedding in zip(logs, embeddings):
            points.append({
                "id": log["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "text": log["text"],
                    "timestamp": log["timestamp"],
                    "source": log.get("source", "chat")
                }
            })
        
        # Upsert to Qdrant
        qdrant_client.upsert(
            collection_name="lumina_embeddings",
            points=points
        )
        
        # Publish Redis events
        for log in logs:
            event = {
                "type": "embedding_update",
                "concept": log["id"],
                "timestamp": log["timestamp"]
            }
            redis_client.publish("lumina.rag_update", json.dumps(event))
        
        return {
            "processed": len(points),
            "message": f"Successfully processed {len(points)} logs"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8600) 