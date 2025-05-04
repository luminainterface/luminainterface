from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as aioredis
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json
import time
import uuid

app = FastAPI(title="Lumina Core API")

# Initialize connections
redis = aioredis.from_url("redis://redis:6379")
qdrant = QdrantClient("vector-db", port=6333)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

class ChatMessage(BaseModel):
    message: str
    session: str = "default"

class ChatResponse(BaseModel):
    reply: str
    confidence: float
    cite_ids: list[str]

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    # Generate embedding for the message
    vec = model.encode(message.message)
    
    # Search similar past interactions
    hits = qdrant.search(
        collection_name="chat_long",
        query_vector=vec.tolist(),
        limit=3
    )
    
    # Get recent context from Redis
    recent = await redis.lrange(f"chat:{message.session}", 0, 3)
    recent = [json.loads(r) for r in recent]
    
    # TODO: Implement proper LLM call to Ollama
    # For now, echo the message
    response = {
        "reply": f"Echo: {message.message}",
        "confidence": 0.8,
        "cite_ids": [h.id for h in hits]
    }
    
    # Store the interaction
    turn = {
        "id": str(uuid.uuid4()),
        "session": message.session,
        "ts": time.time(),
        "user": message.message,
        "assistant": response["reply"],
        "confidence": response["confidence"],
        "cite_ids": response["cite_ids"],
        "embedding": vec.tolist()
    }
    
    # Store in Redis for recent context
    await redis.lpush(f"chat:{message.session}", json.dumps(turn))
    await redis.ltrim(f"chat:{message.session}", 0, 9)  # Keep last 10
    
    # Store in Qdrant for long-term memory
    qdrant.upsert(
        collection_name="chat_long",
        points=[{
            "id": turn["id"],
            "vector": turn["embedding"],
            "payload": {
                "ts": turn["ts"],
                "session": turn["session"],
                "confidence": turn["confidence"],
                "text": turn["assistant"]
            }
        }]
    )
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 