from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel
import redis
import json
from typing import Dict, Any, Optional
import os

app = FastAPI()

# Initialize Redis connection
redis_url = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
redis_client = redis.from_url(redis_url, decode_responses=True)

# API Key authentication
API_KEY = os.getenv("FEEDBACK_LOGGER_API_KEY", "changeme")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class Feedback(BaseModel):
    model_id: str
    prediction_id: str
    feedback: Dict[str, Any]
    class Config:
        protected_namespaces = ()

@app.get("/health")
async def health_check():
    try:
        # Test Redis connection
        redis_client.ping()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def log_feedback(feedback: Feedback, api_key: str = Depends(verify_api_key)):
    try:
        # Store feedback in Redis
        key = f"feedback:{feedback.model_id}:{feedback.prediction_id}"
        redis_client.set(key, json.dumps(feedback.dict()))
        return {"status": "success", "message": "Feedback logged successfully"}
    except Exception as e:
        print(f"[ERROR] Exception in /feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/{model_id}/{prediction_id}")
async def get_feedback(model_id: str, prediction_id: str, api_key: str = Depends(verify_api_key)):
    try:
        key = f"feedback:{model_id}:{prediction_id}"
        feedback = redis_client.get(key)
        if feedback:
            return json.loads(feedback)
        raise HTTPException(status_code=404, detail="Feedback not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/headers")
async def debug_headers(request: Request):
    return {"headers": dict(request.headers)} 