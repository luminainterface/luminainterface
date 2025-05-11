from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
from qdrant_client import QdrantClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
qdrant_client = QdrantClient(host='qdrant', port=6333)

class DriftMetrics(BaseModel):
    collection_name: str
    threshold: float = 0.85

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/check_drift")
async def check_drift(metrics: DriftMetrics):
    try:
        # Get recent vectors from Qdrant
        recent_vectors = qdrant_client.scroll(
            collection_name=metrics.collection_name,
            limit=100
        )[0]
        
        if not recent_vectors:
            return {"drift_detected": False, "message": "No vectors to analyze"}
        
        # Calculate cosine similarity matrix
        vectors = [point.vector for point in recent_vectors]
        similarity_matrix = cosine_similarity(vectors)
        
        # Check for drift (if average similarity is below threshold)
        avg_similarity = np.mean(similarity_matrix)
        drift_detected = avg_similarity < metrics.threshold
        
        # Publish drift event if detected
        if drift_detected:
            event = {
                "type": "drift_detected",
                "collection": metrics.collection_name,
                "avg_similarity": float(avg_similarity),
                "threshold": metrics.threshold
            }
            redis_client.publish("lumina.drift", json.dumps(event))
        
        return {
            "drift_detected": drift_detected,
            "avg_similarity": float(avg_similarity),
            "threshold": metrics.threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200) 