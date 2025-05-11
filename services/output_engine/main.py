from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import redis
from qdrant_client import QdrantClient
import json
from typing import Dict, Any, List
import os
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import numpy as np
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../concept-trainer-growable')))
from model import GrowableConceptNet
import torch

app = FastAPI()

# Setup logging for JSON flow logs
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("output_engine")

# Prometheus metrics
OUTPUT_REQUESTS = Counter("output_requests_total", "Total output requests")
OUTPUT_SUCCESSES = Counter("output_success_total", "Total successful outputs")
OUTPUT_ERRORS = Counter("output_error_total", "Total output errors")
OUTPUT_LATENCY = Histogram("output_latency_seconds", "Output generation latency")

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://qdrant:6333")
)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "output_vectors")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../concept-trainer-growable/data/'))

def find_newest_model_file():
    files = glob.glob(os.path.join(MODEL_DIR, '*.pt')) + glob.glob(os.path.join(MODEL_DIR, '*.pth'))
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    return newest

# Initialize the ML model (random weights for now)
OUTPUT_VECTOR_SIZE = 384
output_model = GrowableConceptNet(input_size=OUTPUT_VECTOR_SIZE)

# Always try to load the newest model weights at startup
model_file = find_newest_model_file()
if model_file:
    try:
        output_model.load_state_dict(torch.load(model_file, map_location='cpu'))
        logger.info(json.dumps({
            "event": "model_load",
            "source": "output_engine",
            "model_file": model_file,
            "status": "success"
        }))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "model_load",
            "source": "output_engine",
            "model_file": model_file,
            "status": "error",
            "details": str(e)
        }))
else:
    logger.warning(json.dumps({
        "event": "model_load",
        "source": "output_engine",
        "model_file": None,
        "status": "no_model_found"
    }))

# ML inference function
def ml_generate_output(vector):
    x = torch.tensor([vector], dtype=torch.float32)
    with torch.no_grad():
        output = output_model(x)
    # For demonstration, convert output to a string
    # (In practice, you might map this to a label or generate text)
    pred = torch.exp(output)[0].tolist()  # Undo log, get probabilities
    return f"ML Output: {pred}"

class OutputRequest(BaseModel):
    model_id: str
    query: str
    parameters: Dict[str, Any] = {}

class OutputResponse(BaseModel):
    prediction_id: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/output")
async def generate_output(request: OutputRequest):
    OUTPUT_REQUESTS.inc()
    start = time.time()
    try:
        prediction_id = f"{request.model_id}_{int(time.time())}"
        key = f"output:{prediction_id}"
        # Simulate inference
        # If the request includes a vector, use it; else, generate one
        if "vector" in request.parameters:
            vector = request.parameters["vector"]
        else:
            vector = np.random.randn(OUTPUT_VECTOR_SIZE).tolist()
        ml_output = ml_generate_output(vector)
        result = {"text": ml_output, "vector": vector}
        # Upsert vector to Qdrant
        try:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[{
                    "id": prediction_id,
                    "vector": result["vector"],
                    "payload": {"query": request.query, "model_id": request.model_id}
                }]
            )
            logger.info(json.dumps({
                "event": "data_flow",
                "source": "output_engine",
                "destination": "qdrant",
                "protocol": "REST VEC",
                "action": "upsert",
                "status": "success",
                "details": {"collection": QDRANT_COLLECTION, "id": prediction_id}
            }))
        except Exception as qe:
            logger.error(json.dumps({
                "event": "data_flow",
                "source": "output_engine",
                "destination": "qdrant",
                "protocol": "REST VEC",
                "action": "upsert",
                "status": "error",
                "details": {"collection": QDRANT_COLLECTION, "id": prediction_id, "error": str(qe)}
            }))
        # Store request and result in Redis
        redis_client.set(key, json.dumps({**request.dict(), **result}))
        logger.info(json.dumps({
            "event": "data_flow",
            "source": "output_engine",
            "destination": "redis",
            "protocol": "SET JSON",
            "action": "store_output",
            "status": "success",
            "details": {"key": key}
        }))
        response = OutputResponse(
            prediction_id=prediction_id,
            results=[{"text": result["text"]}],
            metadata={"model_id": request.model_id}
        )
        OUTPUT_SUCCESSES.inc()
        OUTPUT_LATENCY.observe(time.time() - start)
        return response
    except Exception as e:
        OUTPUT_ERRORS.inc()
        logger.error(json.dumps({
            "event": "data_flow",
            "source": "output_engine",
            "destination": "redis",
            "protocol": "SET JSON",
            "action": "store_output",
            "status": "error",
            "details": {"error": str(e)}
        }))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/output/{prediction_id}")
async def get_output(prediction_id: str):
    try:
        key = f"output:{prediction_id}"
        output = redis_client.get(key)
        if output:
            logger.info(json.dumps({
                "event": "data_flow",
                "source": "redis",
                "destination": "output_engine",
                "protocol": "GET JSON",
                "action": "fetch_output",
                "status": "success",
                "details": {"key": key}
            }))
            return json.loads(output)
        logger.error(json.dumps({
            "event": "data_flow",
            "source": "redis",
            "destination": "output_engine",
            "protocol": "GET JSON",
            "action": "fetch_output",
            "status": "error",
            "details": {"key": key, "error": "not found"}
        }))
        raise HTTPException(status_code=404, detail="Output not found")
    except Exception as e:
        logger.error(json.dumps({
            "event": "data_flow",
            "source": "redis",
            "destination": "output_engine",
            "protocol": "GET JSON",
            "action": "fetch_output",
            "status": "error",
            "details": {"key": key, "error": str(e)}
        }))
        raise HTTPException(status_code=500, detail=str(e)) 