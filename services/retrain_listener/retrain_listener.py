from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
from qdrant_client import QdrantClient
import uvicorn

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
qdrant_client = QdrantClient(host='qdrant', port=6333)

class RetrainRequest(BaseModel):
    collection_name: str
    min_samples: int = 100

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/trigger_retrain")
async def trigger_retrain(request: RetrainRequest):
    try:
        # Check if we have enough samples
        collection_info = qdrant_client.get_collection(request.collection_name)
        if collection_info.points_count < request.min_samples:
            return {
                "retrain_triggered": False,
                "message": f"Not enough samples. Need {request.min_samples}, have {collection_info.points_count}"
            }
        
        # Publish retrain event
        event = {
            "type": "retrain_triggered",
            "collection": request.collection_name,
            "samples_count": collection_info.points_count
        }
        redis_client.publish("lumina.retrain", json.dumps(event))
        
        return {
            "retrain_triggered": True,
            "samples_count": collection_info.points_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8700) 