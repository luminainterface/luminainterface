from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
from typing import Dict, Any
import os

app = FastAPI()

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

class Feedback(BaseModel):
    model_id: str
    prediction_id: str
    feedback: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/feedback")
async def log_feedback(feedback: Feedback):
    try:
        # Store feedback in Redis
        key = f"feedback:{feedback.model_id}:{feedback.prediction_id}"
        redis_client.set(key, json.dumps(feedback.dict()))
        return {"status": "success", "message": "Feedback logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/{model_id}/{prediction_id}")
async def get_feedback(model_id: str, prediction_id: str):
    try:
        key = f"feedback:{model_id}:{prediction_id}"
        feedback = redis_client.get(key)
        if feedback:
            return json.loads(feedback)
        raise HTTPException(status_code=404, detail="Feedback not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 